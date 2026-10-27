# foresight-infer ↔ foresight-sched integration contract

inferno is a layer ON TOP OF foresight. It does not replace the kernel
scheduler; it consumes it. This document is the contract.

## What foresight-sched provides (existing, unchanged)

Custom WSL2 kernel 6.18.33.2 with `CONFIG_SCHED_CLASS_EXT=y`, running the
`foresight_sched` BPF scheduler. Its loader pins five maps:

```
/sys/fs/bpf/foresight/task_class   HASH    u32 kernel-global pid -> u8 class
/sys/fs/bpf/foresight/pid_probe    HASH    char[16] comm -> u32 global pid
/sys/fs/bpf/foresight/stats        ARRAY   [3] u64  (normal/interactive/batch enqueues)
/sys/fs/bpf/foresight/knobs        ARRAY   [1] struct sched_knobs (24B, hole-free)
/sys/fs/bpf/foresight/demotions    HASH    u32 pid -> struct demotion_event (16B)
```

`struct sched_knobs` (byte-exact, from `foresight_sched.bpf.c`):

```c
u64 interactive_slice_ns;   // 0 -> SCX_SLICE_DFL fallback
u64 batch_slice_ns;         // 0 -> SCX_SLICE_DFL fallback
u32 demote_streak;          // 0 -> demotion disabled
u32 batch_duty_pct;         // 0 = unlimited; 1-99 = CONSERVE duty window
```

## Ownership rules (hard contract)

1. **knobs belongs to the foresight daemon.** inferno is READ-ONLY on
   `knobs`, `stats`, and `demotions`. There is exactly one writer per map,
   ever — this is the review discipline foresight-sched itself adopted.
2. **inferno writes `task_class` ONLY for its own threads.** Worker threads
   (`inferno_wN`) are tagged with their namespace tid → kernel-global tid
   translation. Never foreign pids.
3. **task_meta is kernel-private.** inferno never touches it (it is not
   pinned).

## Class mapping (v1 policy)

| inferno work | foresight class | why |
|---|---|---|
| decode-step workers (latency-critical) | `Interactive` (1) | short bursts, latency-sensitive — the interactive lane's small slices fit |
| prefill-chunk workers (compute-heavy) | `Batch` (2) or `Normal` (0) | long compute bursts; the batch lane + duty gate (CONSERVE) protects interactivity |
| load generator / tokenizer fan-out | `Normal` (0) | plumbing |

v1 bench exposes `--tag-interactive` to pin all workers into the
interactive lane for A/B comparison. Per-phase dynamic retagging
(decode=1 during decode-heavy stretches, prefill=2 during prefill bursts)
lands with the HTTP plane.

## Pid-space translation (WSL2 specifics)

WSL2 distros run in a pid namespace: `/proc` and `gettid()` show
namespace-local pids, while foresight's maps are keyed by kernel-GLOBAL
pids. The offset (~+160-190) DRIFTS mid-session. foresight's answer
(birth-cohort sticky offsets + the `pid_probe` exec-tracepoint map) is
REUSED here: inferno copies `/bin/true` to a uniquely-named `/tmp/ifno_*`,
execs it, and reads the comm → global-pid entry `pid_probe` recorded —
the kernel tells us the offset directly. Discovered once, cached for the
process lifetime (sticky), re-derivable via `rediscover()` after a
restart. We inherit the technique instead of reinventing it.

## Runbook (WSL2)

```bash
# inside WSL (root), per wsl.exe session — bpffs is NOT persistent:
mount -t bpf bpf /sys/fs/bpf

# start foresight-sched (if not running):
/root/WSL2-Linux-Kernel/tools/sched_ext/build/bin/foresight_sched &
# the foresight daemon owns knobs:
python3 /root/foresight/foresight_daemon.py ... &   # (as in foresight-sched docs)

# verify pins:
ls /sys/fs/bpf/foresight/
# task_class  demotions  knobs  pid_probe  stats

# inferno bench (built on Windows or in WSL):
./inferno-bench load          # reports "foresight-live" + knobs/stats
./inferno-bench load --tag-interactive
```

If pins are invisible: `nsenter -t $(pgrep -x foresight_sched) -m -- ls /sys/fs/bpf/foresight/`.

## What inferno does NOT do

* It does not write knobs (daemon's job).
* It does not classify foreign processes (daemon's job).
* It does not bypass the scheduler for its own threads — that would make the
  whole layering dishonest.

## Failure mode

When foresight-sched is absent (non-Linux, bpffs unmounted, scheduler not
loaded), `ForesightAdmission::detect()` returns None and everything runs on
`NullAdmission` — behavior identical, and the bench prints `sched=off`
honestly. No integration, no pretending.
