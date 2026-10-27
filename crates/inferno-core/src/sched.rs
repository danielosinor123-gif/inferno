//! CPU-side admission onto foresight-sched — THE integration point with the
//! existing project. inferno does not replace the kernel scheduler; it sits
//! on top of it and uses its pinned BPF maps (see docs/FORESIGHT-INTEGRATION.md):
//!
//!   /sys/fs/bpf/foresight/task_class   HASH  u32 pid -> u8 class
//!   /sys/fs/bpf/foresight/pid_probe    HASH  char[16] comm -> u32 global pid
//!   /sys/fs/bpf/foresight/stats        ARRAY [3] u64 per-class enqueues
//!   /sys/fs/bpf/foresight/knobs        ARRAY [1] sched_knobs (24B)
//!   /sys/fs/bpf/foresight/demotions    HASH  u32 pid -> demotion_event (16B)
//!
//! OWNERSHIP CONTRACT (learned the hard way in foresight-sched review):
//!   * knobs belongs to the foresight daemon (KnobController). inferno is
//!     READ-ONLY on knobs and stats and demotions.
//!   * inferno WRITES task_class ONLY for its own worker threads' pids —
//!     never for foreign processes.
//!
//! On non-Linux (or when the scheduler isn't loaded) everything degrades to
//! NullAdmission: identical behavior, zero foresight coupling, and the bench
//! reports `sched=off` honestly instead of pretending integration happened.

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AdmissionClass {
    Normal = 0,
    Interactive = 1,
    Batch = 2,
}

impl AdmissionClass {
    pub fn from_u8(v: u8) -> Self {
        match v {
            1 => AdmissionClass::Interactive,
            2 => AdmissionClass::Batch,
            _ => AdmissionClass::Normal,
        }
    }
}

/// Byte-exact mirror of `struct sched_knobs` in foresight_sched.bpf.c
/// (8+8+4+4 = 24 bytes, hole-free).
#[derive(Clone, Copy, Debug, Default)]
pub struct ForesightKnobs {
    pub interactive_slice_ns: u64,
    pub batch_slice_ns: u64,
    pub demote_streak: u32,
    pub batch_duty_pct: u32,
}

/// Byte-exact mirror of `struct demotion_event` (16 bytes incl. tail pad).
#[derive(Clone, Copy, Debug, Default)]
pub struct DemotionEvent {
    pub ts_ns: u64,
    pub from_class: u8,
    pub to_class: u8,
    pub streak: u16,
}

/// Read-only view of foresight-sched state for /metrics export.
#[derive(Clone, Debug, Default)]
pub struct ForesightSnapshot {
    pub live: bool,
    pub knobs: ForesightKnobs,
    /// per-class enqueue counters [normal, interactive, batch]
    pub stats: [u64; 3],
    /// up to 8 most recent demotion events (pid -> event)
    pub demotions: Vec<(u32, DemotionEvent)>,
}

/// Admission control seam. Implementations:
/// * `NullAdmission` — portable default, no kernel dependency.
/// * `ForesightAdmission` (linux module) — talks to the pinned maps.
pub trait Admission: Send + Sync {
    fn is_live(&self) -> bool {
        false
    }
    fn snapshot(&self) -> Option<ForesightSnapshot> {
        None
    }
    /// Tag one of OUR OWN threads (namespace tid) into a foresight class.
    /// Returns false when unsupported / not live.
    fn tag_thread(&self, _ns_tid: u32, _class: AdmissionClass) -> bool {
        false
    }
    /// Translate a namespace-local pid/tid to the kernel-global pid space
    /// foresight's maps are keyed by. None when not discoverable.
    fn global_pid(&self, _ns_pid: u32) -> Option<u32> {
        None
    }
}

/// Current thread's namespace-local tid (what task_class keys must derive
/// from). 0 on platforms without a tid notion.
pub fn current_tid() -> u32 {
    #[cfg(target_os = "linux")]
    unsafe {
        libc::gettid() as u32
    }
    #[cfg(not(target_os = "linux"))]
    {
        0
    }
}

#[derive(Default)]
pub struct NullAdmission;

impl Admission for NullAdmission {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn null_admission_is_inert() {
        let n = NullAdmission;
        assert!(!n.is_live());
        assert!(n.snapshot().is_none());
        assert!(!n.tag_thread(1, AdmissionClass::Interactive));
        assert!(n.global_pid(1).is_none());
        assert_eq!(AdmissionClass::from_u8(2), AdmissionClass::Batch);
    }
}
