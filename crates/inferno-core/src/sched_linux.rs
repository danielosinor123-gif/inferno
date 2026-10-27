//! Foresight-sched BPF client (Linux only): raw bpf(2) syscalls against the
//! pins created by foresight_sched's loader. No libbpf dependency — the
//! syscall surface we need is five commands:
//!   BPF_OBJ_GET(7) pin -> fd
//!   BPF_MAP_LOOKUP_ELEM(1) / BPF_MAP_UPDATE_ELEM(2)
//!   BPF_MAP_GET_NEXT_KEY(4) for HASH iteration (demotions)
//!
//! Reuses foresight's OWN pid_probe map to translate namespace pids into the
//! kernel-global pid space the maps are keyed by (the WSL2 offset-drift
//! problem foresight solved with birth-cohort sticky offsets — we inherit
//! the map and the technique instead of reinventing it). The offset is
//! discovered ONCE per process and cached (sticky), then re-derived only on
//! explicit `rediscover()` — matching foresight's drift policy.

use std::fs;
use std::process::Command;
use std::sync::atomic::{AtomicI64, Ordering};
use std::sync::{Mutex, OnceLock};

use libc::{SYS_bpf, syscall};

use crate::sched::{Admission, AdmissionClass, DemotionEvent, ForesightKnobs, ForesightSnapshot};

const PIN_DIR: &str = "/sys/fs/bpf/foresight";

const BPF_MAP_LOOKUP_ELEM: u64 = 1;
const BPF_MAP_UPDATE_ELEM: u64 = 2;
const BPF_MAP_GET_NEXT_KEY: u64 = 4;
const BPF_OBJ_GET: u64 = 7;
const BPF_ANY: u64 = 0;

#[repr(C)]
struct ObjGetAttr {
    pathname: u64,
    bpf_fd: u32,
    file_flags: u32,
}

/// Kernel bpf_attr map_op layout: u32 map_fd; __aligned_u64 key/value/flags.
/// repr(C) gives the same padding (map_fd@0, pad 4, key@8, value@16, flags@24).
#[repr(C)]
struct ElemAttr {
    map_fd: u32,
    _pad: u32,
    key: u64,
    value: u64,
    flags: u64,
}

#[repr(C)]
struct NextKeyAttr {
    map_fd: u32,
    _pad: u32,
    key: u64,
    next_key: u64,
}

fn sys_bpf(cmd: u64, attr: *mut std::ffi::c_void, size: usize) -> i64 {
    unsafe {
        syscall(
            SYS_bpf,
            cmd as std::ffi::c_ulong,
            attr,
            size as std::ffi::c_ulong,
        )
    }
}

fn obj_get(path: &str) -> Option<i32> {
    let c = std::ffi::CString::new(path).ok()?;
    let mut attr = ObjGetAttr {
        pathname: c.as_ptr() as u64,
        bpf_fd: 0,
        file_flags: 0,
    };
    let fd = sys_bpf(
        BPF_OBJ_GET,
        &mut attr as *mut _ as *mut _,
        std::mem::size_of::<ObjGetAttr>(),
    );
    if fd < 0 { None } else { Some(fd as i32) }
}

fn lookup(fd: i32, key: &[u8], value: &mut [u8]) -> bool {
    let mut attr = ElemAttr {
        map_fd: fd as u32,
        _pad: 0,
        key: key.as_ptr() as u64,
        value: value.as_mut_ptr() as u64,
        flags: 0,
    };
    sys_bpf(
        BPF_MAP_LOOKUP_ELEM,
        &mut attr as *mut _ as *mut _,
        std::mem::size_of::<ElemAttr>(),
    ) == 0
}

fn update(fd: i32, key: &[u8], value: &[u8]) -> bool {
    let mut attr = ElemAttr {
        map_fd: fd as u32,
        _pad: 0,
        key: key.as_ptr() as u64,
        value: value.as_ptr() as u64,
        flags: BPF_ANY,
    };
    sys_bpf(
        BPF_MAP_UPDATE_ELEM,
        &mut attr as *mut _ as *mut _,
        std::mem::size_of::<ElemAttr>(),
    ) == 0
}

/// Iterate a HASH map's keys (BPF_MAP_GET_NEXT_KEY). `start` = None begins
/// the iteration. Returns keys until ENOENT, capped at `max`.
fn next_keys(fd: i32, start: Option<&[u8]>, klen: usize, max: usize) -> Vec<Vec<u8>> {
    let mut out = Vec::new();
    let mut cur: Vec<u8> = match start {
        Some(k) => k.to_vec(),
        None => vec![0u8; klen], // zero key = "start" sentinel
    };
    let mut first = start.is_none();
    for _ in 0..max {
        let mut nk = vec![0u8; klen];
        let mut attr = NextKeyAttr {
            map_fd: fd as u32,
            _pad: 0,
            key: cur.as_ptr() as u64,
            next_key: nk.as_mut_ptr() as u64,
        };
        if sys_bpf(
            BPF_MAP_GET_NEXT_KEY,
            &mut attr as *mut _ as *mut _,
            std::mem::size_of::<NextKeyAttr>(),
        ) != 0
        {
            break; // ENOENT = end of map
        }
        if !first {
            out.push(cur.clone());
        }
        cur = nk;
        first = false;
    }
    out
}

struct Fds {
    task_class: i32,
    stats: i32,
    knobs: i32,
    demotions: i32,
    pid_probe: i32,
}

impl Drop for Fds {
    fn drop(&mut self) {
        for fd in [self.task_class, self.stats, self.knobs, self.demotions, self.pid_probe] {
            unsafe { libc::close(fd) };
        }
    }
}

static FDS: OnceLock<Mutex<Option<Fds>>> = OnceLock::new();
static OFFSET: AtomicI64 = AtomicI64::new(i64::MIN);

fn fds_lock() -> Option<std::sync::MutexGuard<'static, Option<Fds>>> {
    FDS.get_or_init(|| Mutex::new(None)).lock().ok()
}

fn probe_name() -> String {
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.subsec_nanos() as u64)
        .unwrap_or(0);
    let pid = std::process::id();
    format!("ifno_{:02x}{:04x}", (nanos & 0xff) as u8, (pid & 0xffff) as u16)
}

/// Discover the WSL2 pid-namespace offset by reusing foresight's own
/// pid_probe map: exec a uniquely-named tiny binary; the kernel's exec
/// tracepoint records comm -> global pid; we know the child's namespace pid
/// from spawn(). Sticky for the process lifetime (birth-cohort lesson).
fn probe_offset(pid_probe_fd: i32) -> Option<i64> {
    let name = probe_name();
    let path = format!("/tmp/{name}");
    fs::copy("/bin/true", &path).ok()?;
    let child = Command::new(&path).arg("--inferno-probe").spawn().ok();
    let ns_pid = child.as_ref().map(|c| c.id());
    let status = child.and_then(|mut c| c.wait().ok());
    let _ = fs::remove_file(&path);
    status?; // child must have actually exec'ed
    let ns_pid = ns_pid?;
    let mut comm = [0u8; 16]; // TASK_COMM_LEN, NUL-padded
    let bytes = name.as_bytes();
    comm[..bytes.len().min(15)].copy_from_slice(&bytes[..bytes.len().min(15)]);
    let mut out = [0u8; 4];
    if !lookup(pid_probe_fd, &comm, &mut out) {
        return None;
    }
    Some(u32::from_le_bytes(out) as i64 - ns_pid as i64)
}

fn parse_demotion(buf: &[u8; 16]) -> DemotionEvent {
    DemotionEvent {
        ts_ns: u64::from_le_bytes(buf[0..8].try_into().unwrap()),
        from_class: buf[8],
        to_class: buf[9],
        streak: u16::from_le_bytes(buf[10..12].try_into().unwrap()),
    }
}

pub struct ForesightAdmission {
    _marker: (),
}

impl ForesightAdmission {
    /// Detect a live foresight-sched: bpffs mounted + all five pins present.
    pub fn detect() -> Option<Self> {
        if !fs::metadata(PIN_DIR).map(|m| m.is_dir()).unwrap_or(false) {
            return None;
        }
        let f = Fds {
            task_class: obj_get(&format!("{PIN_DIR}/task_class"))?,
            stats: obj_get(&format!("{PIN_DIR}/stats"))?,
            knobs: obj_get(&format!("{PIN_DIR}/knobs"))?,
            demotions: obj_get(&format!("{PIN_DIR}/demotions"))?,
            pid_probe: obj_get(&format!("{PIN_DIR}/pid_probe"))?,
        };
        let lock = FDS.get_or_init(|| Mutex::new(None));
        let mut guard = lock.lock().ok()?;
        *guard = Some(f);
        Some(Self { _marker: () })
    }

    fn with_fds<T>(&self, f: impl FnOnce(&Fds) -> T) -> Option<T> {
        let guard = fds_lock()?;
        let fds = guard.as_ref()?;
        Some(f(fds))
    }

    /// Cached namespace->global offset; discovers on first call.
    pub fn offset(&self) -> Option<i64> {
        let cached = OFFSET.load(Ordering::Relaxed);
        if cached != i64::MIN {
            return Some(cached);
        }
        let off = self.with_fds(|fds| probe_offset(fds.pid_probe))??;
        OFFSET.store(off, Ordering::Relaxed);
        Some(off)
    }

    /// Re-derive the offset after a session restart (drift self-heal).
    pub fn rediscover(&self) -> Option<i64> {
        let off = self.with_fds(|fds| probe_offset(fds.pid_probe))??;
        OFFSET.store(off, Ordering::Relaxed);
        Some(off)
    }

    pub fn read_knobs(&self) -> Option<ForesightKnobs> {
        self.with_fds(|fds| {
            let mut buf = [0u8; 24];
            if !lookup(fds.knobs, &0u32.to_le_bytes(), &mut buf) {
                return None;
            }
            Some(ForesightKnobs {
                interactive_slice_ns: u64::from_le_bytes(buf[0..8].try_into().ok()?),
                batch_slice_ns: u64::from_le_bytes(buf[8..16].try_into().ok()?),
                demote_streak: u32::from_le_bytes(buf[16..20].try_into().ok()?),
                batch_duty_pct: u32::from_le_bytes(buf[20..24].try_into().ok()?),
            })
        })?
    }

    pub fn read_stats(&self) -> Option<[u64; 3]> {
        self.with_fds(|fds| {
            let mut out = [0u64; 3];
            for (i, o) in out.iter_mut().enumerate() {
                let mut buf = [0u8; 8];
                if !lookup(fds.stats, &(i as u32).to_le_bytes(), &mut buf) {
                    return None;
                }
                *o = u64::from_le_bytes(buf);
            }
            Some(out)
        })?
    }

    /// Recent demotions via proper HASH iteration (next-key walk).
    pub fn read_demotions(&self, max: usize) -> Vec<(u32, DemotionEvent)> {
        self.with_fds(|fds| {
            let keys = next_keys(fds.demotions, None, 4, max);
            let mut out = Vec::with_capacity(keys.len());
            for k in keys {
                if out.len() >= max {
                    break;
                }
                let pid = u32::from_le_bytes(k[..4].try_into().unwrap());
                let mut buf = [0u8; 16];
                if lookup(fds.demotions, &pid.to_le_bytes(), &mut buf) {
                    out.push((pid, parse_demotion(&buf)));
                }
            }
            out
        })
        .unwrap_or_default()
    }
}

impl Admission for ForesightAdmission {
    fn is_live(&self) -> bool {
        self.read_knobs().is_some()
    }

    fn snapshot(&self) -> Option<ForesightSnapshot> {
        let knobs = self.read_knobs()?;
        let stats = self.read_stats()?;
        Some(ForesightSnapshot {
            live: true,
            knobs,
            stats,
            demotions: self.read_demotions(8),
        })
    }

    fn tag_thread(&self, ns_tid: u32, class: AdmissionClass) -> bool {
        let Some(off) = self.offset() else { return false };
        let global = (ns_tid as i64 + off) as u32;
        self.with_fds(|fds| update(fds.task_class, &global.to_le_bytes(), &[class as u8]))
            .unwrap_or(false)
    }

    fn global_pid(&self, ns_pid: u32) -> Option<u32> {
        let off = self.offset()?;
        Some((ns_pid as i64 + off) as u32)
    }
}
