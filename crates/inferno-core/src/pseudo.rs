//! Pseudo-model engine — a DETERMINISTIC workload simulator used to
//! benchmark the scheduler layer (threads, KV, batching, admission) without
//! model weights.
//!
//! Honest labeling: this is NOT a language model. It reproduces the
//! *scheduling-relevant* cost shape of an LLM forward pass:
//!   prefill cost ~ linear in prompt tokens (compute-bound),
//!   decode cost ~ constant per token (memory-bound),
//! and produces deterministic tokens via a stateful xorshift, so runs are
//! fully reproducible. It also writes REAL K/V vectors through the paged
//! cache so memory traffic is modeled, not hand-waved.
//!
//! The real-model backend (weights via safetensors, HF tokenizer) plugs into
//! the same `Engine` trait in the next PR.

use crate::batching::{Engine, Sequence};
use crate::kv_cache::KvCache;

pub struct PseudoEngine {
    /// Simulated microseconds per prefill token.
    pub prefill_cost_us_per_tok: f64,
    /// Simulated microseconds per decode step (all layers).
    pub decode_cost_us: f64,
    pub layers: usize,
    pub kv_dim: usize,
    pub eos_token: u32,
}

impl PseudoEngine {
    pub fn new(layers: usize, kv_dim: usize) -> Self {
        Self {
            prefill_cost_us_per_tok: 12.0,
            decode_cost_us: 60.0,
            layers,
            kv_dim,
            eos_token: 0,
        }
    }

    fn xorshift(state: u64) -> u32 {
        let mut x = state | 1;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        (x & 0x7fff_ffff) as u32
    }
}

impl Engine for PseudoEngine {
    fn prefill_chunk(&mut self, seq: &mut Sequence, cache: &mut KvCache, budget: usize) -> usize {
        let end = (seq.pos + budget).min(seq.prompt.len());
        let d = self.kv_dim;
        let mut processed = 0usize;
        while seq.pos < end {
            let tok = seq.prompt[seq.pos];
            let kv = token_kv(tok, seq.pos, self.layers, d);
            if cache.write_token(seq.id, seq.pos, &kv).is_err() {
                break; // CoW/pool failure: stop; batcher preempts
            }
            seq.pos += 1;
            processed += 1;
        }
        // simulate compute cost deterministically (no sleeping in tests:
        // cost accounting is exported in the report instead of wall-clock spin)
        seq.simulated_compute_us += self.prefill_cost_us_per_tok * processed as f64;
        processed
    }

    fn decode_step(&mut self, seq: &mut Sequence, _cache: &KvCache) -> u32 {
        seq.simulated_compute_us += self.decode_cost_us;
        let state = seq.id.wrapping_mul(0x9E3779B97F4A7C15)
            ^ (seq.generated.len() as u64).wrapping_mul(0xD1B54A32D192ED03);
        let tok = Self::xorshift(state);
        if tok == 0 {
            1 // never emit EOS accidentally: keep deterministic length control
        } else {
            tok
        }
    }

    fn decode_kv(
        &mut self,
        _seq: &Sequence,
        tok: u32,
        pos: usize,
        _cache: &KvCache,
    ) -> Vec<f32> {
        token_kv(tok, pos, self.layers, self.kv_dim)
    }
}

/// Deterministic K/V for a (token, position): models the write traffic shape.
pub fn token_kv(tok: u32, pos: usize, layers: usize, d: usize) -> Vec<f32> {
    let mut v = Vec::with_capacity(layers * 2 * d);
    for l in 0..layers {
        for w in 0..2 {
            for i in 0..d {
                let x = (tok as usize)
                    .wrapping_add(pos.wrapping_mul(7919))
                    .wrapping_add(l * 31 + w * 17 + i);
                v.push((x % 4096) as f32 * 0.001 - 2.0);
            }
        }
    }
    v
}
