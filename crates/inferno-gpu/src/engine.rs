//! The hybrid engine: ggml for commodity dense ops, inferno-core's paged
//! self-extension attention for the moat layer (docs/G1-GPU.md).
//!
//! Scope of THIS slice (pre-registered): hermetic conformance — the full
//! forward pass on the tiny model must match the scalar `QwenEngine` within
//! the reference tolerance. Weights convert per op into ggml tensors; the
//! persistent-weights design (graphs built once, tensors resident) is the
//! NEXT pre-registered step — this slice proves the arithmetic, not the
//! performance.

use inferno_core::kernels::{paged_attention_self, rope};
use inferno_core::kv_cache::KvCache;
use inferno_model::config::ModelCfg;
use inferno_model::engine::QwenEngine;
use inferno_model::weights::WeightMat;

use crate::{gemm_naive, rms_norm, rope as ggml_rope, swiglu, GgmlCtx, arena_bytes_for};

pub struct GgmlEngine {
    pub cfg: ModelCfg,
    pub w: inferno_model::weights::QwenWeights,
}

impl GgmlEngine {
    pub fn new(cfg: ModelCfg, w: inferno_model::weights::QwenWeights) -> Self {
        Self { cfg, w }
    }

    pub fn weight_bytes(&self) -> usize {
        self.w.weight_bytes()
    }

    /// Stage a WeightMat as f32 for the bridge (per-op convert). The real
    /// path keeps ggml tensors resident — documented next step.
    fn staged(&self, mat: &WeightMat, k: usize) -> Vec<f32> {
        let rows = mat.elem_len() / k;
        let mut out = Vec::with_capacity(rows * k);
        for r in 0..rows {
            out.extend_from_slice(&mat.row_to_vec(r, k));
        }
        out
    }

    fn gemv(
        &self,
        ctx: &GgmlCtx,
        mat: &WeightMat,
        x: &[f32],
        rows: usize,
        out: &mut [f32],
    ) -> Result<(), String> {
        let k = x.len();
        let staged = self.staged(mat, k);
        gemm_naive(ctx, &staged, x, rows, out)
    }

    /// The hybrid forward: same structure as the scalar engine, dense ops
    /// through ggml, attention through inferno-core's kernel.
    pub fn forward_token(
        &self,
        tok: u32,
        pos: usize,
        cache: &KvCache,
        seq_id: u64,
        kv_out: &mut Vec<f32>,
    ) -> Result<Vec<f32>, String> {
        let h = self.cfg.hidden;
        let l = self.cfg.num_layers;
        let d = self.cfg.head_dim;
        let qd = self.cfg.num_q_heads * d;
        let kvd = self.cfg.num_kv_heads * d;
        let inter = self.cfg.intermediate;

        let mut x: Vec<f32> = self.w.embed.row_to_vec(tok as usize, h);
        let mut h1 = vec![0.0f32; h];
        let mut q = vec![0.0f32; qd];
        let mut k = vec![0.0f32; kvd];
        let mut v = vec![0.0f32; kvd];
        let mut attn = vec![0.0f32; qd];
        let mut gate = vec![0.0f32; inter];
        let mut up = vec![0.0f32; inter];
        kv_out.clear();
        kv_out.resize(l * 2 * kvd, 0.0);

        for li in 0..l {
            let w = &self.w.layers[li];
            {
                let ctx = GgmlCtx::new(arena_bytes_for(2, h))?;
                rms_norm(&ctx, &x, &w.norm_in, self.cfg.rms_eps, &mut h1)?;
            }
            {
                // q/k/v (fused input, three calls in this slice)
                let ctx = GgmlCtx::new(arena_bytes_for(1, qd * h).max(arena_bytes_for(1, kvd * h)))?;
                self.gemv(&ctx, &w.q, &h1, qd, &mut q)?;
                self.gemv(&ctx, &w.k, &h1, kvd, &mut k)?;
                self.gemv(&ctx, &w.v, &h1, kvd, &mut v)?;
            }
            // Qwen2.5 QKV bias (same as the scalar engine)
            for (qv, &b) in q.iter_mut().zip(w.qb.iter()) {
                *qv += b;
            }
            for (kv, &b) in k.iter_mut().zip(w.kb.iter()) {
                *kv += b;
            }
            for (vv, &b) in v.iter_mut().zip(w.vb.iter()) {
                *vv += b;
            }
            {
                let ctx = GgmlCtx::new(arena_bytes_for(2, qd))?;
                // the bridge takes separate in/out (no aliasing in safe
                // Rust) — scratch + copy back
                let mut qr = vec![0.0f32; qd];
                let mut kr = vec![0.0f32; kvd];
                ggml_rope(&ctx, &q, self.cfg.num_q_heads, d, pos, self.cfg.rope_theta, &mut qr)?;
                ggml_rope(&ctx, &k, self.cfg.num_kv_heads, d, pos, self.cfg.rope_theta, &mut kr)?;
                q.copy_from_slice(&qr);
                k.copy_from_slice(&kr);
            }
            let base = li * 2 * kvd;
            kv_out[base..base + kvd].copy_from_slice(&k);
            kv_out[base + kvd..base + 2 * kvd].copy_from_slice(&v);

            // THE MOAT: our paged self-extension attention, unchanged.
            paged_attention_self(
                cache, seq_id, li, &q, self.cfg.num_q_heads, d, pos, &k, &v, &mut attn,
            );
            {
                let ctx = GgmlCtx::new(arena_bytes_for(1, h * qd))?;
                self.gemv(&ctx, &w.o, &attn, h, &mut h1)?;
            }
            for i in 0..h {
                x[i] += h1[i];
            }
            {
                let ctx = GgmlCtx::new(arena_bytes_for(2, h))?;
                rms_norm(&ctx, &x, &w.norm_post, self.cfg.rms_eps, &mut h1)?;
            }
            {
                let ctx = GgmlCtx::new(arena_bytes_for(1, inter * h))?;
                self.gemv(&ctx, &w.gate, &h1, inter, &mut gate)?;
                self.gemv(&ctx, &w.up, &h1, inter, &mut up)?;
            }
            {
                let ctx = GgmlCtx::new(arena_bytes_for(2, inter))?;
                let mut gs = vec![0.0f32; inter];
                swiglu(&ctx, &gate, &up, &mut gs)?;
                gate.copy_from_slice(&gs);
            }
            {
                let ctx = GgmlCtx::new(arena_bytes_for(1, h * inter))?;
                self.gemv(&ctx, &w.down, &gate, h, &mut h1)?;
            }
            for i in 0..h {
                x[i] += h1[i];
            }
        }
        Ok(x)
    }

    /// Final norm + lm_head through ggml (the [vocab, h] mat is the biggest
    /// single op).
    pub fn logits_of(&self, x: &[f32]) -> Result<Vec<f32>, String> {
        let h = self.cfg.hidden;
        let mut xn = vec![0.0f32; h];
        {
            let ctx = GgmlCtx::new(arena_bytes_for(2, h))?;
            rms_norm(&ctx, x, &self.w.norm, self.cfg.rms_eps, &mut xn)?;
        }
        let mut logits = vec![0.0f32; self.cfg.vocab];
        let ctx = GgmlCtx::new(arena_bytes_for(1, self.cfg.vocab * h))?;
        self.gemv(&ctx, &self.w.embed, &xn, self.cfg.vocab, &mut logits)?;
        Ok(logits)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use inferno_model::engine::demo::{tiny_cfg, tiny_weights};
    use inferno_model::engine::QwenEngine;

    fn argmax(v: &[f32]) -> u32 {
        let mut best = 0usize;
        let mut bv = f32::NEG_INFINITY;
        for (i, &x) in v.iter().enumerate() {
            if x > bv {
                bv = x;
                best = i;
            }
        }
        best as u32
    }

    #[test]
    fn hybrid_forward_matches_scalar_engine() {
        let cfg = tiny_cfg();
        let cache_cfg = inferno_core::kv_cache::KvConfig {
            block_size: 4,
            num_blocks: 8,
            num_layers: cfg.num_layers,
            num_kv_heads: cfg.num_kv_heads,
            head_dim: cfg.head_dim,
        };
        // both engines build from the SAME deterministic tiny generator —
        // identical weights by construction
        let scalar = QwenEngine::new(cfg.clone(), tiny_weights());
        let hybrid = GgmlEngine::new(cfg.clone(), tiny_weights());

        let prompt: Vec<u32> = vec![3, 11, 5, 21, 8];
        let mut cache_a = inferno_core::kv_cache::KvCache::new(cache_cfg);
        let mut cache_b = inferno_core::kv_cache::KvCache::new(cache_cfg);
        assert!(cache_a.reserve(7, 6, &[]).is_some());
        assert!(cache_b.reserve(9, 6, &[]).is_some());
        let mut last_scalar = Vec::new();
        let mut last_hybrid = Vec::new();
        for (p, &t) in prompt.iter().enumerate() {
            // scalar: one forward, capture x + kv, write kv
            let mut kv_a = Vec::new();
            let x = scalar.forward_token_pub(t, p, &cache_a, 7, &mut kv_a);
            cache_a.write_token(7, p, &kv_a).unwrap();
            // hybrid: same shape, ops through ggml
            let mut kv_b = Vec::new();
            let xh = hybrid
                .forward_token(t, p, &cache_b, 9, &mut kv_b)
                .expect("hybrid forward");
            cache_b.write_token(9, p, &kv_b).unwrap();
            last_scalar = x;
            last_hybrid = xh;
        }
        // final logits of the last prompt token, both engines
        let ls = scalar.logits_of_pub(&last_scalar);
        let lh = hybrid.logits_of(&last_hybrid).expect("hybrid logits");
        assert_eq!(argmax(&ls), argmax(&lh), "greedy ids must agree");
        for (i, (a, b)) in ls.iter().zip(&lh).enumerate() {
            let rel = (a - b).abs() / (1.0 + b.abs());
            assert!(rel < 1e-3, "logit {i}: scalar {a} vs hybrid {b} (rel {rel})");
        }
    }
}
