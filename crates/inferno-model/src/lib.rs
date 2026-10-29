//! Real-weights backend: Qwen2.5-0.5B-Instruct (safetensors + HF tokenizer)
//! wired into inferno's paged KV cache and `Engine` trait.
//!
//! Design notes (honest, v1):
//! * Weights convert fp16 -> f32 at load (CPU reference math is f32).
//!   ~2.0 GB resident for the 0.5B model; the fp16 file is mmap'd during
//!   load and unmapped after conversion (peak ~3 GB, fits the 4 GB WSL VM).
//! * All projections are row-dot gemv over the native HF `[out, in]`
//!   layout — no transposed weight copies (would cost another ~2 GB).
//! * Exactly ONE forward pass runs per generated token, inside `decode_kv`:
//!   the batcher asks for the just-sampled token's K/V, which requires
//!   forwarding that token; the logits that pass produces are stashed and
//!   the NEXT `decode_step` is a pure argmax. No double forward per step.
//! * Every attention call uses the self-extension kernel
//!   (`kernels::paged_attention_self`): the current token's own K/V is not
//!   in the cache yet at forward time, so it is appended as one virtual
//!   extra slot. Prefill also uses this (its own KV is written once per
//!   token, after the full layer chain).
//! * v1 sampling is greedy argmax — deterministic and reproducible.
//!   Temperature/top-p land with the benchmark PR.

pub mod config;
pub mod engine;
pub mod registry;
pub mod weights;

pub use config::ModelCfg;
pub use engine::QwenEngine;
pub use weights::QwenWeights;
