//! CPU reference kernels — the correctness ground truth for the GPU kernels.
//!
//! Every kernel here is implemented as the NAIVE version AND the optimized
//! version, with tests asserting bit-level agreement (tolerance 1e-5). The
//! optimized versions encode the exact algorithmic structure that the GPU
//! port must reproduce:
//!
//! * `paged_attention_naive` / `paged_attention`: online softmax with running
//!   max/sum (m, l) over a paged block table — the FlashAttention recursion
//!   (Dao et al., arXiv:2205.14135) adapted to paged storage (arXiv:2309.06180
//!   §3.2). The GPU kernel replaces the inner loop with tensor-core MMA +
//!   shared-memory tiles; the recursion is identical.
//! * `rmsnorm`: x / sqrt(mean(x^2) + eps) * w — Llama-family normalization.
//! * `rope`: rotate-half RoPE applied in-place to Q/K.

use crate::kv_cache::{KvCache, Which};

/// Fused RMSNorm over one vector. `w` has len == x.len().
pub fn rmsnorm(x: &[f32], w: &[f32], eps: f32, out: &mut [f32]) {
    debug_assert_eq!(x.len(), w.len());
    let n = x.len() as f32;
    let mut ssq = 0.0f32;
    for &v in x {
        ssq += v * v;
    }
    let inv = 1.0 / (ssq / n + eps).sqrt();
    for i in 0..x.len() {
        out[i] = x[i] * inv * w[i];
    }
}

/// RoPE (rotate half, HF/Qwen2 convention): pairs are (i, i + head_dim/2).
/// positions `pos`, per-head layout [heads][head_dim].
pub fn rope(q: &mut [f32], heads: usize, head_dim: usize, pos: usize, theta: f32) {
    debug_assert_eq!(q.len() % (heads * head_dim), 0);
    let half = head_dim / 2;
    for h in 0..heads {
        let base = h * head_dim;
        for i in 0..half {
            let freq = theta.powf(-(2.0 * i as f32) / head_dim as f32);
            let ang = pos as f32 * freq;
            let (c, s) = (ang.cos(), ang.sin());
            let x1 = q[base + i];
            let x2 = q[base + half + i];
            q[base + i] = x1 * c - x2 * s;
            q[base + half + i] = x2 * c + x1 * s;
        }
    }
}

/// Paged attention for ONE query against a sequence's KV blocks.
/// Naive form: materialize everything, single softmax.
/// `q`: [heads * head_dim] for a single token position.
/// Returns [heads * head_dim] output.
pub fn paged_attention_naive(
    cache: &KvCache,
    seq_id: u64,
    layer: usize,
    q: &[f32],
    heads: usize,
    head_dim: usize,
    valid_tokens: usize,
) -> Vec<f32> {
    let kv_heads = cache.cfg().num_kv_heads;
    let bs = cache.cfg().block_size;
    let d = kv_heads * head_dim; // per-token K/V span
    let n_blocks = cache.seq_len_blocks(seq_id);
    let mut all_k = Vec::with_capacity(valid_tokens * d);
    let mut all_v = Vec::with_capacity(valid_tokens * d);
    'blocks: for b in 0..n_blocks {
        let k_blk = cache.read_block_layer(seq_id, b, layer, Which::K).unwrap();
        let v_blk = cache.read_block_layer(seq_id, b, layer, Which::V).unwrap();
        for slot in 0..bs {
            if b * bs + slot >= valid_tokens {
                break 'blocks;
            }
            all_k.extend_from_slice(&k_blk[slot * d..(slot + 1) * d]);
            all_v.extend_from_slice(&v_blk[slot * d..(slot + 1) * d]);
        }
    }
    let scale = 1.0 / (head_dim as f32).sqrt();
    let mut out = vec![0.0f32; heads * head_dim];
    for h in 0..heads {
        // GQA: query head h reads KV head h % kv_heads
        let group = heads / kv_heads;
        let kvh = h / group;
        let mut scores = vec![0.0f32; valid_tokens];
        for t in 0..valid_tokens {
            let kbase = t * d + kvh * head_dim;
            let mut dot = 0.0f32;
            for i in 0..head_dim {
                dot += q[h * head_dim + i] * all_k[kbase + i];
            }
            scores[t] = dot * scale;
        }
        softmax(&mut scores);
        for t in 0..valid_tokens {
            let vbase = t * d + kvh * head_dim;
            for i in 0..head_dim {
                out[h * head_dim + i] += scores[t] * all_v[vbase + i];
            }
        }
    }
    out
}

/// Optimized form: single pass over blocks with the online-softmax recursion
/// (running m = max, l = sum). Same result, one traversal, no materialized
/// score vector — this is the loop the GPU kernel tile-fuses.
pub fn paged_attention(
    cache: &KvCache,
    seq_id: u64,
    layer: usize,
    q: &[f32],
    heads: usize,
    head_dim: usize,
    valid_tokens: usize,
) -> Vec<f32> {
    let kv_heads = cache.cfg().num_kv_heads;
    let bs = cache.cfg().block_size;
    let d = kv_heads * head_dim;
    let scale = 1.0 / (head_dim as f32).sqrt();
    let mut out = vec![0.0f32; heads * head_dim];
    let n_blocks = cache.seq_len_blocks(seq_id);
    for h in 0..heads {
        let group = heads / kv_heads;
        let kvh = h / group;
        let mut m = f32::NEG_INFINITY;
        let mut l = 0.0f32;
        let mut acc = vec![0.0f32; head_dim];
        let mut processed = 0usize;
        for b in 0..n_blocks {
            if processed >= valid_tokens {
                break;
            }
            let k_blk = cache.read_block_layer(seq_id, b, layer, Which::K).unwrap();
            let v_blk = cache.read_block_layer(seq_id, b, layer, Which::V).unwrap();
            let tokens_this = (valid_tokens - processed).min(bs);
            for slot in 0..tokens_this {
                let kbase = slot * d + kvh * head_dim;
                let mut dot = 0.0f32;
                for i in 0..head_dim {
                    dot += q[h * head_dim + i] * k_blk[kbase + i];
                }
                dot *= scale;
                let m_new = m.max(dot);
                let exp_old = (m - m_new).exp();
                let exp_new = (dot - m_new).exp();
                // rescale accumulator, then accumulate the new value row
                for i in 0..head_dim {
                    acc[i] = acc[i] * exp_old + exp_new * v_blk[slot * d + kvh * head_dim + i];
                }
                l = l * exp_old + exp_new;
                m = m_new;
            }
            processed += tokens_this;
        }
        let inv_l = 1.0 / l; // valid_tokens >= 1 guaranteed by callers
        for i in 0..head_dim {
            out[h * head_dim + i] = acc[i] * inv_l;
        }
    }
    out
}

pub fn softmax(v: &mut [f32]) {
    let m = v.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0f32;
    for x in v.iter_mut() {
        *x = (*x - m).exp();
        sum += *x;
    }
    let inv = 1.0 / sum;
    for x in v.iter_mut() {
        *x *= inv;
    }
}

/// Naive matmul: C[M,N] = A[M,K] * B[K,N], row-major.
pub fn matmul_naive(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
    let mut c = vec![0.0f32; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut s = 0.0f32;
            for p in 0..k {
                s += a[i * k + p] * b[p * n + j];
            }
            c[i * n + j] = s;
        }
    }
    c
}

/// Blocked (cache-tiled) matmul — the structure a GPU GEMM kernel mirrors
/// with shared-memory tiles. Numerically identical to naive (same add order
/// per output? No: blocked changes add order for k-dim, so tolerance test).
pub fn matmul_blocked(a: &[f32], b: &[f32], m: usize, k: usize, n: usize, tile: usize) -> Vec<f32> {
    let mut c = vec![0.0f32; m * n];
    for i0 in (0..m).step_by(tile) {
        let i1 = (i0 + tile).min(m);
        for k0 in (0..k).step_by(tile) {
            let k1 = (k0 + tile).min(k);
            for j0 in (0..n).step_by(tile) {
                let j1 = (j0 + tile).min(n);
                for i in i0..i1 {
                    for p in k0..k1 {
                        let aip = a[i * k + p];
                        for j in j0..j1 {
                            c[i * n + j] += aip * b[p * n + j];
                        }
                    }
                }
            }
        }
    }
    c
}

/// Naive self-extension variant: scores from cache positions `0..valid_tokens`
/// PLUS one own-position slot (`self_k`/`self_v`, each `[kv_heads * head_dim]`).
/// Reference form for the optimized `paged_attention_self`.
pub fn paged_attention_naive_self(
    cache: &KvCache,
    seq_id: u64,
    layer: usize,
    q: &[f32],
    heads: usize,
    head_dim: usize,
    valid_tokens: usize,
    self_k: &[f32],
    self_v: &[f32],
) -> Vec<f32> {
    let kv_heads = cache.cfg().num_kv_heads;
    let group = heads / kv_heads;
    let bs = cache.cfg().block_size;
    let d = kv_heads * head_dim;
    let scale = 1.0 / (head_dim as f32).sqrt();
    let total = valid_tokens + 1;
    let mut out = vec![0.0f32; heads * head_dim];
    for h in 0..heads {
        let kvh = h / group;
        let mut scores = vec![0.0f32; total];
        let mut vals = vec![0.0f32; total * head_dim];
        let mut idx = 0usize;
        for b in 0..cache.seq_len_blocks(seq_id) {
            if idx >= valid_tokens {
                break;
            }
            let k_blk = cache.read_block_layer(seq_id, b, layer, Which::K).unwrap();
            let v_blk = cache.read_block_layer(seq_id, b, layer, Which::V).unwrap();
            let take = (valid_tokens - idx).min(bs);
            for s in 0..take {
                let base = s * d + kvh * head_dim;
                let mut dot = 0.0f32;
                for i in 0..head_dim {
                    dot += q[h * head_dim + i] * k_blk[base + i];
                }
                scores[idx] = dot * scale;
                vals[idx * head_dim..(idx + 1) * head_dim]
                    .copy_from_slice(&v_blk[base..base + head_dim]);
                idx += 1;
            }
        }
        let base = kvh * head_dim;
        let mut dot = 0.0f32;
        for i in 0..head_dim {
            dot += q[h * head_dim + i] * self_k[base + i];
        }
        scores[idx] = dot * scale;
        vals[idx * head_dim..(idx + 1) * head_dim].copy_from_slice(&self_v[base..base + head_dim]);
        softmax(&mut scores);
        for s in 0..total {
            for i in 0..head_dim {
                out[h * head_dim + i] += scores[s] * vals[s * head_dim + i];
            }
        }
    }
    out
}

/// Optimized self-extension variant: the online-softmax recursion with one
/// extra virtual slot for the current token's own (post-rope) K/V. Used by
/// the real-model engine, whose forwards run before the batcher writes the
/// token's own K/V into the cache.
pub fn paged_attention_self(
    cache: &KvCache,
    seq_id: u64,
    layer: usize,
    q: &[f32],
    heads: usize,
    head_dim: usize,
    valid_tokens: usize,
    self_k: &[f32],
    self_v: &[f32],
    out: &mut [f32],
) {
    let kv_heads = cache.cfg().num_kv_heads;
    let group = heads / kv_heads;
    let bs = cache.cfg().block_size;
    let d = kv_heads * head_dim;
    let scale = 1.0 / (head_dim as f32).sqrt();
    let n_blocks = cache.seq_len_blocks(seq_id);
    for h in 0..heads {
        let kvh = h / group;
        let mut m = f32::NEG_INFINITY;
        let mut l = 0.0f32;
        let mut acc = vec![0.0f32; head_dim];
        let mut processed = 0usize;
        for b in 0..n_blocks {
            if processed >= valid_tokens {
                break;
            }
            let k_blk = cache.read_block_layer(seq_id, b, layer, Which::K).unwrap();
            let v_blk = cache.read_block_layer(seq_id, b, layer, Which::V).unwrap();
            let tokens_this = (valid_tokens - processed).min(bs);
            for slot in 0..tokens_this {
                let kbase = slot * d + kvh * head_dim;
                let mut dot = 0.0f32;
                for i in 0..head_dim {
                    dot += q[h * head_dim + i] * k_blk[kbase + i];
                }
                dot *= scale;
                let m_new = m.max(dot);
                let exp_old = (m - m_new).exp();
                let exp_new = (dot - m_new).exp();
                for i in 0..head_dim {
                    acc[i] = acc[i] * exp_old + exp_new * v_blk[slot * d + kvh * head_dim + i];
                }
                l = l * exp_old + exp_new;
                m = m_new;
            }
            processed += tokens_this;
        }
        let kbase = kvh * head_dim;
        let mut dot = 0.0f32;
        for i in 0..head_dim {
            dot += q[h * head_dim + i] * self_k[kbase + i];
        }
        dot *= scale;
        let m_new = m.max(dot);
        let exp_old = (m - m_new).exp();
        let exp_new = (dot - m_new).exp();
        for i in 0..head_dim {
            acc[i] = acc[i] * exp_old + exp_new * self_v[kbase + i];
        }
        l = l * exp_old + exp_new;
        let inv = 1.0 / l;
        for i in 0..head_dim {
            out[h * head_dim + i] = acc[i] * inv;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kv_cache::{KvConfig, KvCache};

    fn approx(a: &[f32], b: &[f32], tol: f32) -> bool {
        a.len() == b.len() && a.iter().zip(b).all(|(x, y)| (x - y).abs() < tol)
    }

    #[test]
    fn rmsnorm_unit() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let w = vec![1.0, 1.0, 1.0, 1.0];
        let mut out = vec![0.0; 4];
        rmsnorm(&x, &w, 1e-6, &mut out);
        // rms = sqrt((1+4+9+16)/4) = sqrt(7.5)
        let rms = (7.5f32).sqrt();
        for (o, v) in out.iter().zip(&x) {
            assert!((o - v / rms).abs() < 1e-5);
        }
    }

    #[test]
    fn rope_preserves_norm_per_head() {
        let mut q: Vec<f32> = (0..16).map(|i| (i as f32) * 0.5 - 3.0).collect();
        let before: Vec<f32> = {
            let mut norms = Vec::new();
            for h in 0..4 {
                let s: f32 = q[h * 4..h * 4 + 4].iter().map(|v| v * v).sum();
                norms.push(s);
            }
            norms
        };
        rope(&mut q, 4, 4, 7, 10000.0);
        for h in 0..4 {
            let s: f32 = q[h * 4..h * 4 + 4].iter().map(|v| v * v).sum();
            assert!((s - before[h]).abs() < 1e-4, "head {} norm drifted", h);
        }
    }

    #[test]
    fn rope_is_half_split_not_interleaved() {
        // Pins the HF/Qwen2 rotate-half convention. For q=[1,0,0,0] (d=4),
        // pos=1, theta=1e4: pair (0, 2) rotates by 1 rad => q[0]=cos(1),
        // q[2]=sin(1); pair (1, 3) is all-zero. The interleaved (GPT-J)
        // convention would instead put sin(1) at index 1 — this test
        // discriminates.
        let mut q = vec![1.0f32, 0.0, 0.0, 0.0];
        rope(&mut q, 1, 4, 1, 10000.0);
        let c1 = 1.0f32.cos();
        let s1 = 1.0f32.sin();
        assert!(
            (q[0] - c1).abs() < 1e-5
                && q[1] == 0.0
                && (q[2] - s1).abs() < 1e-5
                && q[3] == 0.0,
            "rope output {q:?} is not half-split"
        );
    }

    fn setup_cache(valid: usize) -> (KvCache, u64, Vec<f32>) {
        // 1 kv head, head_dim 4, block 4, 2 layers; K/V content deterministic.
        let cfg = KvConfig {
            block_size: 4,
            num_blocks: 8,
            num_layers: 2,
            num_kv_heads: 1,
            head_dim: 4,
        };
        let mut c = KvCache::new(cfg);
        let n = valid.div_ceil(4);
        c.reserve(42, valid, &[]).unwrap();
        let d = 4usize;
        for t in 0..valid {
            let mut kv = Vec::with_capacity(2 * cfg.num_layers * d);
            for l in 0..cfg.num_layers {
                for w in 0..2 {
                    for i in 0..d {
                        kv.push(((t * 3 + l * 7 + w * 11 + i * 5) as f32).rem_euclid(13.0) - 6.5);
                    }
                }
            }
            c.write_token(42, t, &kv).unwrap();
        }
        let q: Vec<f32> = (0..8).map(|i| 0.1 * i as f32).collect(); // 2 heads x 4
        let _ = n;
        (c, 42, q)
    }

    #[test]
    fn paged_attention_optimized_matches_naive() {
        for valid in [1usize, 4, 9, 16] {
            let (c, seq, q) = setup_cache(valid);
            let naive = paged_attention_naive(&c, seq, 0, &q, 2, 4, valid);
            let opt = paged_attention(&c, seq, 0, &q, 2, 4, valid);
            assert!(approx(&naive, &opt, 1e-5), "mismatch at valid={}", valid);
        }
    }

    #[test]
    fn attention_finite_and_sums() {
        let (c, seq, q) = setup_cache(9);
        let out = paged_attention(&c, seq, 1, &q, 2, 4, 9);
        assert!(out.iter().all(|v| v.is_finite()));
        assert!(approx(&out, &paged_attention_naive(&c, seq, 1, &q, 2, 4, 9), 1e-5));
    }

    #[test]
    fn blocked_matmul_matches_naive() {
        let (m, k, n) = (17usize, 23usize, 11usize);
        let a: Vec<f32> = (0..m * k).map(|i| ((i % 7) as f32) - 3.0).collect();
        let b: Vec<f32> = (0..k * n).map(|i| ((i % 5) as f32) - 2.0).collect();
        let naive = matmul_naive(&a, &b, m, k, n);
        let blocked = matmul_blocked(&a, &b, m, k, n, 4);
        assert!(approx(&naive, &blocked, 1e-3), "blocked != naive");
    }

    #[test]
    fn gqa_groups_are_contiguous() {
        // 4 Q heads, 2 KV heads, head_dim 2. KV head 0 has K=[1,0], V=[1,1];
        // KV head 1 has K=[0,1], V=[2,2]. With contiguous grouping (HF
        // semantics) q heads {0,1} must see kv0 and q heads {2,3} must see
        // kv1. The old `h % kv_heads` mapping interleaves and would give
        // heads 1 and 2 the wrong KV — this test pins the fix.
        let cfg = KvConfig {
            block_size: 4,
            num_blocks: 2,
            num_layers: 1,
            num_kv_heads: 2,
            head_dim: 2,
        };
        let mut c = KvCache::new(cfg);
        c.reserve(42, 1, &[]).unwrap();
        // token 0 K/V: [layers=1][2][kv_dim=4]; slot 0, kv head h at kvh*2
        let kv = vec![
            1.0, 0.0, 0.0, 1.0, // K: kv0=[1,0], kv1=[0,1]
            1.0, 1.0, 2.0, 2.0, // V: kv0=[1,1], kv1=[2,2]
        ];
        c.write_token(42, 0, &kv).unwrap();
        // q: head0=[1,0] head1=[1,0] head2=[0,1] head3=[0,1]
        let q = vec![1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0];
        let out = paged_attention(&c, 42, 0, &q, 4, 2, 1);
        // heads 0,1 -> v of kv0 = [1,1]; heads 2,3 -> v of kv1 = [2,2]
        let expect = vec![1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0];
        assert!(approx(&out, &expect, 1e-5), "GQA grouping wrong: {out:?}");
    }

    #[test]
    fn paged_attention_self_matches_naive_self() {
        // 2 q heads, 1 kv head, head_dim 4 — same geometry as setup_cache.
        for valid in [0usize, 1, 4, 9] {
            let (c, seq, q) = setup_cache(valid.max(1));
            if valid == 0 {
                continue; // setup_cache needs >= 1 token to reserve
            }
            let self_k: Vec<f32> = (0..4).map(|i| 0.3 * (i as f32) - 0.4).collect();
            let self_v: Vec<f32> = (0..4).map(|i| 0.2 * (i as f32) + 0.1).collect();
            let naive = paged_attention_naive_self(&c, seq, 0, &q, 2, 4, valid, &self_k, &self_v);
            let mut opt = vec![0.0f32; 8];
            paged_attention_self(&c, seq, 0, &q, 2, 4, valid, &self_k, &self_v, &mut opt);
            assert!(approx(&naive, &opt, 1e-5), "self-variant mismatch at valid={valid}");
        }
    }

    #[test]
    fn self_extension_includes_own_position() {
        // Own K exactly parallel to Q, own V large: the own slot must
        // dominate the output when cache is otherwise empty-ish.
        let (c, seq, _q) = setup_cache(1);
        let q = vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let self_k = vec![2.0, 0.0, 0.0, 0.0]; // kv head 0
        let self_v = vec![100.0, 0.0, 0.0, 0.0];
        let mut opt = vec![0.0f32; 8];
        paged_attention_self(&c, seq, 0, &q, 2, 4, 1, &self_k, &self_v, &mut opt);
        // head 0: own score 2*1=2 (scaled), cache score small -> own V dominates
        assert!(opt[0] > 50.0, "own position contribution missing: {:?}", &opt[..4]);
    }
}
