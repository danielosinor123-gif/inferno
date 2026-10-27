//! inferno-core: a concurrent LLM inference runtime layered ON TOP OF foresight.
//!
//! Layering contract (see docs/FORESIGHT-INTEGRATION.md):
//!
//!   foresight-infer request plane (HTTP, admission, SLOs)   [next PR]
//!   ┌───────────────────────────────────────────────────────┐
//!   │ inferno-core (this crate)                             │
//!   │   threads     : work-stealing worker pool             │
//!   │   memory      : block pool + arena                    │
//!   │   kv_cache    : paged KV cache w/ prefix sharing      │
//!   │   batching    : iteration-level continuous batching   │
//!   │   kernels     : CPU reference kernels (GPU next)      │
//!   │   sched       : foresight-sched CPU-side admission    │  ← reads/writes
//!   └───────────────────────────────────────────────────────┘     /sys/fs/bpf/foresight
//!   foresight-sched (kernel BPF: lanes, demotion, duty gate)
//!
//! Research grounding per module: see docs/RESEARCH.md.

pub mod batching;
pub mod kernels;
pub mod kv_cache;
pub mod memory;
pub mod pseudo;
pub mod sched;
pub mod threads;

#[cfg(target_os = "linux")]
pub mod sched_linux;

pub use batching::{Batcher, Sequence, StepReport};
pub use kv_cache::KvCache;
pub use sched::{Admission, AdmissionClass, ForesightSnapshot, NullAdmission};
pub use threads::ThreadPool;
