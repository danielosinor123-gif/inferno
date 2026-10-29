//! Qwen2 CPU engine behind inferno's `Engine` trait — the first real-weights
//! backend.
//!
//! Single-threaded v1: this is the correctness gate. Threading the gemv row
//! loop through the ThreadPool is the NEXT pre-registered step (measure
//! before and after; no silent concurrency changes).
//!
//! Flow (matches the batcher contract in batching.rs):
//!   prefill_chunk: per prompt token — full layer chain, own K/V via the
//!     self-extension kernel, one `write_token` per token; the LAST prompt
//!     token also stashes logits.
//!   decode_step: pure argmax of the stashed logits (no matmul work).
//!   decode_kv: full forward of the just-sampled token at its position;
//!     returns its K/V for the batcher to write, stashes logits for the
//!     next decode_step. ONE forward per generated token total.
//!
//! Known v1 limitation (documented, loud): the logits stash is invalidated
//! by mid-decode preemption. If it is ever stale, decode_step panics with
//! an explicit message rather than silently attending over missing K/V.
//! Proper fix (queued): batcher re-reserves retained decode blocks by hash.

use std::collections::HashMap;

use inferno_core::batching::{Engine, Sequence};
use inferno_core::kernels::{paged_attention_self, rmsnorm, rope};
use inferno_core::kv_cache::KvCache;

use crate::config::ModelCfg;
use crate::weights::{QwenWeights, WeightMat};

/// Contiguous-row gemv: `out[j] = dot(w_row_j, x)` for j in 0..rows.
/// `w` is `[rows, x.len()]` row-major (native HF layout) — the inner loop is
/// a straight contiguous read of both operands, no transposed copies.
pub fn gemv(w: &[f32], x: &[f32], rows: usize, out: &mut [f32]) {
    debug_assert_eq!(out.len(), rows);
    let k = x.len();
    debug_assert!(w.len() >= rows * k);
    for j in 0..rows {
        let row = &w[j * k..j * k + k];
        let mut s = 0.0f32;
        for i in 0..k {
            s += row[i] * x[i];
        }
        out[j] = s;
    }
}

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

/// Logits of the most recent forward, tagged with the position they were
/// computed at (staleness detector).
struct Stash {
    logits: Vec<f32>,
    logits_pos: usize,
}

pub struct QwenEngine {
    pub cfg: ModelCfg,
    pub w: QwenWeights,
    tok: Option<tokenizers::Tokenizer>,
    stash: HashMap<u64, Stash>,
    /// Row-parallel gemv pool (threading step, pre-registered in
    /// docs/THREADING.md). None = scalar path (tests, tiny models).
    pool: Option<inferno_core::threads::ThreadPool>,
}

fn make_pool() -> Option<inferno_core::threads::ThreadPool> {
    let n = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1);
    if n <= 1 {
        return None;
    }
    inferno_core::threads::ThreadPool::new(Some(n)).ok()
}

impl QwenEngine {
    pub fn new(cfg: ModelCfg, w: QwenWeights) -> Self {
        Self {
            cfg,
            w,
            tok: None,
            stash: HashMap::new(),
            // Pool by default so tests exercise the parallel path (the
            // reference test's bit-identical gate then covers threading).
            pool: make_pool(),
        }
    }

    /// Full load from a model directory: config.json + model.safetensors +
    /// tokenizer.json. bf16-native storage (exact).
    pub fn load(dir: &std::path::Path) -> Result<Self, String> {
        Self::load_with_storage(dir, crate::weights::Storage::Bf16)
    }

    /// Full load with an explicit matrix storage dtype (docs/WEIGHTS8.md).
    pub fn load_with_storage(
        dir: &std::path::Path,
        storage: crate::weights::Storage,
    ) -> Result<Self, String> {
        let (cfg, w) = QwenWeights::load_with_storage(dir, storage)?;
        let tok_path = dir.join("tokenizer.json");
        let tok = tokenizers::Tokenizer::from_file(&tok_path)
            .map_err(|e| format!("loading {tok_path:?}: {e}"))?;
        Ok(Self {
            cfg,
            w,
            tok: Some(tok),
            stash: HashMap::new(),
            pool: make_pool(),
        })
    }

    /// Resident weight bytes (registry budget accounting).
    pub fn weight_bytes(&self) -> usize {
        self.w.weight_bytes()
    }

    /// Fused multi-matrix gemv: one worker dispatch for several matrices
    /// that share the SAME input vector (q/k/v on the attention norm;
    /// gate/up on the MLP norm). Cuts per-token dispatches 7/layer to
    /// 4/layer (169 -> 97 calls/token) without touching any dot —
    /// bit-identical by construction (docs/WEIGHTS16.md next candidate).
    /// `tasks[k]` is (matrix, rows); `outs[k]` receives its result; the
    /// reassembly maps each returned value back by flat position, so chunk
    /// boundaries never matter.
    fn gemv_many(
        &self,
        tasks: &[(&WeightMat, usize)],
        x: &[f32],
        outs: &mut [&mut [f32]],
    ) {
        debug_assert_eq!(tasks.len(), outs.len());
        let k = x.len();
        let Some(pool) = self.pool.as_ref() else {
            for (ti, (mat, rows)) in tasks.iter().enumerate() {
                for j in 0..*rows {
                    outs[ti][j] = mat.dot_row(j, x, k);
                }
            }
            return;
        };
        // Flat (task, row) list in submission order; results return in the
        // same order, so position i in the reassembly maps to flat[i].
        let flat: Vec<(usize, usize)> = tasks
            .iter()
            .enumerate()
            .flat_map(|(t, (_, rows))| (0..*rows).map(move |j| (t, j)))
            .collect();
        // 'static captures: clone the matrix handles (refcount bumps) + x.
        let mats: Vec<WeightMat> = tasks
            .iter()
            .map(|(m, _)| match m {
                WeightMat::F32(v) => WeightMat::F32(std::sync::Arc::clone(v)),
                WeightMat::BF16(v) => WeightMat::BF16(std::sync::Arc::clone(v)),
                WeightMat::F16(v) => WeightMat::F16(std::sync::Arc::clone(v)),
                WeightMat::INT8 { q, scale } => WeightMat::INT8 {
                    q: std::sync::Arc::clone(q),
                    scale: std::sync::Arc::clone(scale),
                },
            })
            .collect();
        let xc = x.to_vec();
        // clone for the workers; the local copy drives result reassembly.
        let flat_for_workers = flat.clone();
        let parts: Vec<Vec<f32>> = pool.map_chunks(flat_for_workers, move |items, _id| {
            items.iter().map(|&(t, j)| mats[t].dot_row(j, &xc, k)).collect()
        });
        let mut pos = 0usize;
        for p in parts {
            for (i_local, v) in p.iter().enumerate() {
                let (t, j) = flat[pos + i_local];
                outs[t][j] = *v;
            }
            pos += p.len();
        }
    }

    /// Public wrapper over the one-token forward (tests, diagnostics; the
    /// batcher is the real caller). Builds nothing — the caller owns the cache.
    pub fn forward_token_pub(
        &self,
        tok: u32,
        pos: usize,
        cache: &KvCache,
        seq_id: u64,
        kv_out: &mut Vec<f32>,
    ) -> Vec<f32> {
        self.forward_token(tok, pos, cache, seq_id, kv_out)
    }

    /// Public wrapper over final-norm + lm_head (tests, diagnostics).
    pub fn logits_of_pub(&self, x: &[f32]) -> Vec<f32> {
        self.logits_of(x)
    }

    /// One-shot logits digest for bit-identity fingerprints (registry
    /// evict/reload gate): private geometry-matched cache, one forward,
    /// FNV-1a over the logit bits. Deterministic by construction.
    pub fn logits_digest_pub(&self, tok: u32, pos: usize) -> u64 {
        let mut cache = KvCache::new(inferno_core::kv_cache::KvConfig {
            block_size: 16,
            num_blocks: 2,
            num_layers: self.cfg.num_layers,
            num_kv_heads: self.cfg.num_kv_heads,
            head_dim: self.cfg.head_dim,
        });
        assert!(cache.reserve(1, 1, &[]).is_some(), "digest cache reserve");
        let mut kv = Vec::new();
        let x = self.forward_token(tok, pos, &cache, 1, &mut kv);
        let lg = self.logits_of(&x);
        let mut h: u64 = 0xcbf29ce484222325;
        for v in lg {
            h ^= v.to_bits() as u64;
            h = h.wrapping_mul(0x100000001b3);
        }
        h
    }

    /// Workers in the gemv pool (0 = scalar path). Diagnostics.
    pub fn pool_workers(&self) -> usize {
        self.pool.as_ref().map(|p| p.workers()).unwrap_or(0)
    }

    /// Contiguous-row gemv with the pool when available. Row-parallel by
    /// construction: each worker computes whole rows with the SAME per-row
    /// summation order as the scalar path, and `map_chunks` returns results
    /// in chunk (= row) order — bit-identical outputs, no reduction-order
    /// changes (the threading step's correctness gate). Weights stay in
    /// their native dtype (bf16 halved bytes, docs/WEIGHTS16.md); conversion
    /// happens per element inside the unchanged dot.
    fn gemv_rows<'a>(&self, mat: &'a WeightMat, x: &[f32], rows: usize, out: &mut [f32]) {
        debug_assert_eq!(out.len(), rows);
        let Some(pool) = self.pool.as_ref() else {
            for j in 0..rows {
                out[j] = mat.dot_row(j, x, x.len());
            }
            return;
        };
        let k = x.len();
        // 'static captures for map_chunks: Arc-clone the matrix (bump a
        // refcount) and copy x (<= 4864 floats — noise next to the gemv).
        let mat: WeightMat = match mat {
            WeightMat::F32(v) => WeightMat::F32(std::sync::Arc::clone(v)),
            WeightMat::BF16(v) => WeightMat::BF16(std::sync::Arc::clone(v)),
            WeightMat::F16(v) => WeightMat::F16(std::sync::Arc::clone(v)),
            WeightMat::INT8 { q, scale } => WeightMat::INT8 {
                q: std::sync::Arc::clone(q),
                scale: std::sync::Arc::clone(scale),
            },
        };
        let xc = x.to_vec();
        let idx: Vec<usize> = (0..rows).collect();
        let parts: Vec<Vec<f32>> = pool.map_chunks(idx, move |js, _id| {
            js.iter().map(|&j| mat.dot_row(j, &xc, k)).collect::<Vec<f32>>()
        });
        let mut o = 0usize;
        for p in parts {
            out[o..o + p.len()].copy_from_slice(&p);
            o += p.len();
        }
        debug_assert_eq!(o, rows);
    }

    /// Tokenize a prompt (special tokens per the tokenizer config).
    pub fn encode_text(&self, text: &str) -> Vec<u32> {
        self.tok
            .as_ref()
            .expect("encode_text: engine built via new() has no tokenizer")
            .encode(text, true)
            .expect("tokenization failed")
            .get_ids()
            .to_vec()
    }

    /// Detokenize generated ids (skip_special_tokens = false, so EOS shows).
    pub fn decode_text(&self, ids: &[u32]) -> String {
        self.tok
            .as_ref()
            .expect("decode_text: engine built via new() has no tokenizer")
            .decode(ids, false)
            .unwrap_or_else(|e| format!("<decode error: {e}>"))
    }

    fn q_dim(&self) -> usize {
        self.cfg.num_q_heads * self.cfg.head_dim
    }

    fn kv_dim(&self) -> usize {
        self.cfg.num_kv_heads * self.cfg.head_dim
    }

    /// Full forward of ONE token at absolute position `pos`.
    ///
    /// Reads prior positions from the paged cache (positions `0..pos`) and
    /// appends the token's own K/V as the self-extension slot — the token's
    /// own K/V is NOT assumed to be in the cache yet. RoPE applies to Q and
    /// K only (never V). `kv_out` receives `[layers][2][kv_dim]`
    /// (layer-major, K then V) with post-rope K/V, ready for `write_token`.
    /// Returns the final hidden state (pre-norm).
    fn forward_token(
        &self,
        tok: u32,
        pos: usize,
        cache: &KvCache,
        seq_id: u64,
        kv_out: &mut Vec<f32>,
    ) -> Vec<f32> {
        let h = self.cfg.hidden;
        let l = self.cfg.num_layers;
        let d = self.cfg.head_dim;
        let qd = self.q_dim();
        let kvd = self.kv_dim();
        let inter = self.cfg.intermediate;

        let e = tok as usize;
        let mut x: Vec<f32> = self.w.embed.row_to_vec(e, h);
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
            rmsnorm(&x, &w.norm_in, self.cfg.rms_eps, &mut h1);
            // q/k/v share the attention-norm input — ONE dispatch.
            self.gemv_many(
                &[(&w.q, qd), (&w.k, kvd), (&w.v, kvd)],
                &h1,
                &mut [&mut q[..], &mut k[..], &mut v[..]],
            );
            // Qwen2.5 QKV bias (tensors exist in the checkpoint — verified
            // against the safetensors header)
            for (qv, &b) in q.iter_mut().zip(w.qb.iter()) {
                *qv += b;
            }
            for (kv, &b) in k.iter_mut().zip(w.kb.iter()) {
                *kv += b;
            }
            for (vv, &b) in v.iter_mut().zip(w.vb.iter()) {
                *vv += b;
            }
            rope(&mut q, self.cfg.num_q_heads, d, pos, self.cfg.rope_theta);
            rope(&mut k, self.cfg.num_kv_heads, d, pos, self.cfg.rope_theta);
            // stash post-rope K,V (K first, then V — write_token layout)
            let base = li * 2 * kvd;
            kv_out[base..base + kvd].copy_from_slice(&k);
            kv_out[base + kvd..base + 2 * kvd].copy_from_slice(&v);

            paged_attention_self(
                cache, seq_id, li, &q, self.cfg.num_q_heads, d, pos, &k, &v, &mut attn,
            );
            self.gemv_rows(&w.o, &attn, h, &mut h1); // o_proj output
            for i in 0..h {
                x[i] += h1[i];
            }

            rmsnorm(&x, &w.norm_post, self.cfg.rms_eps, &mut h1);
            // gate/up share the MLP-norm input — ONE dispatch.
            self.gemv_many(
                &[(&w.gate, inter), (&w.up, inter)],
                &h1,
                &mut [&mut gate[..], &mut up[..]],
            );
            // SwiGLU: down(silu(gate) * up)
            for i in 0..inter {
                let g = gate[i];
                gate[i] = g / (1.0 + (-g).exp()); // silu in place
            }
            for i in 0..inter {
                gate[i] *= up[i];
            }
            self.gemv_rows(&w.down, &gate, h, &mut h1);
            for i in 0..h {
                x[i] += h1[i];
            }
        }
        x
    }

    /// Final RMSNorm + lm_head (row-dot over the embedding matrix when tied).
    /// The lm_head is the largest gemv in the stack ([vocab, hidden] — 151936x896
    /// for the 0.5B), so it goes through the same row-parallel path.
    fn logits_of(&self, x: &[f32]) -> Vec<f32> {
        debug_assert!(self.w.tied_head, "v1 supports tied lm_head only");
        let h = self.cfg.hidden;
        let mut xn = vec![0.0f32; h];
        rmsnorm(x, &self.w.norm, self.cfg.rms_eps, &mut xn);
        let mut logits = vec![0.0f32; self.cfg.vocab];
        self.gemv_rows(&self.w.embed, &xn, self.cfg.vocab, &mut logits);
        logits
    }

    /// Top-k of the logits stashed by the most recent forward for `seq_id`
    /// (diagnostics only — the batcher owns the real decode flow).
    pub fn stash_topk(&self, seq_id: u64, k: usize) -> Option<Vec<(u32, f32)>> {
        let s = self.stash.get(&seq_id)?;
        let mut idx: Vec<usize> = (0..s.logits.len()).collect();
        idx.sort_unstable_by(|&a, &b| s.logits[b].partial_cmp(&s.logits[a]).unwrap());
        idx.truncate(k);
        Some(idx.iter().map(|&i| (i as u32, s.logits[i])).collect())
    }

    /// Diagnostic probe, independent of the batcher: prefill `prompt` into a
    /// private cache through the normal forward path, then return the top-k
    /// (token, logit) pairs of the next-token distribution. Used to separate
    /// forward-path bugs from batcher wiring bugs.
    pub fn diagnostic_topk(&self, prompt: &[u32], k: usize) -> Vec<(u32, f32)> {
        let cache = KvCache::new(inferno_core::kv_cache::KvConfig {
            block_size: 16,
            num_blocks: 8,
            num_layers: self.cfg.num_layers,
            num_kv_heads: self.cfg.num_kv_heads,
            head_dim: self.cfg.head_dim,
        });
        let mut cache = cache;
        assert!(cache.reserve(1, prompt.len(), &[]).is_some(), "diag reserve");
        let mut kv = Vec::new();
        let mut x = vec![0.0f32; 0];
        for (p, &t) in prompt.iter().enumerate() {
            x = self.forward_token(t, p, &cache, 1, &mut kv);
            cache.write_token(1, p, &kv).expect("diag write");
        }
        let logits = self.logits_of(&x);
        let mut idx: Vec<usize> = (0..logits.len()).collect();
        idx.sort_unstable_by(|&a, &b| logits[b].partial_cmp(&logits[a]).unwrap());
        idx.truncate(k);
        idx.iter().map(|&i| (i as u32, logits[i])).collect()
    }
}

/// Test-only builders shared by engine and registry tests (hermetic: no
/// files, no disk — tiny in-memory models).
#[cfg(test)]
pub(crate) mod testutil {
    use super::*;
    use crate::config::ModelCfg;
    use crate::weights::{bf16_from_f32, LayerWeights};

    pub fn tiny_cfg() -> ModelCfg {
        ModelCfg {
            num_layers: 2,
            num_q_heads: 2,
            num_kv_heads: 1,
            head_dim: 4,
            hidden: 8,
            intermediate: 6,
            vocab: 32,
            rms_eps: 1e-6,
            rope_theta: 10_000.0,
            tie_word_embeddings: true,
            eos_token: 1,
            max_ctx: 64,
        }
    }

    pub fn tiny_weights() -> QwenWeights {
        tiny_weights_storage(false)
    }

    /// Same tiny model in INT8 storage (quantized once from the f32 fill) —
    /// exercises the int8 path through the same reference gate
    /// (docs/WEIGHTS8.md gate 1c).
    pub fn tiny_weights_int8() -> QwenWeights {
        tiny_weights_storage(true)
    }

    fn tiny_weights_storage(int8: bool) -> QwenWeights {
        let cfg = tiny_cfg();
        let f = |i: usize, j: usize| ((i * 31 + j * 17) % 11) as f32 * 0.04 - 0.2;
        let fill = |rows: usize, cols: usize, salt: usize| -> Vec<f32> {
            (0..rows * cols).map(|k| f(k + salt, salt * 7 + k / cols.max(1))).collect()
        };
        // Matrices in the requested storage; norms/biases stay f32 (exact).
        let m = |v: Vec<f32>, cols: usize| -> crate::weights::WeightMat {
            if int8 {
                crate::weights::int8_from_block(&v, cols)
            } else {
                bf16_from_f32(v)
            }
        };
        QwenWeights {
            // bf16 storage (the real checkpoint's dtype): the reference gate
            // now compares the PARALLEL BF16 engine vs a scalar BF16 reference.
            embed: m(fill(cfg.vocab, cfg.hidden, 0), cfg.hidden),
            layers: (0..cfg.num_layers)
                .map(|l| LayerWeights {
                    q: m(
                        fill(cfg.num_q_heads * cfg.head_dim, cfg.hidden, 10 + l),
                        cfg.hidden,
                    ),
                    k: m(
                        fill(cfg.num_kv_heads * cfg.head_dim, cfg.hidden, 20 + l),
                        cfg.hidden,
                    ),
                    v: m(
                        fill(cfg.num_kv_heads * cfg.head_dim, cfg.hidden, 30 + l),
                        cfg.hidden,
                    ),
                    o: m(
                        fill(cfg.hidden, cfg.num_q_heads * cfg.head_dim, 40 + l),
                        cfg.num_q_heads * cfg.head_dim,
                    ),
                    qb: std::sync::Arc::new(
                        (0..cfg.num_q_heads * cfg.head_dim)
                            .map(|i| f(i, 91 + l) * 0.05)
                            .collect(),
                    ),
                    kb: std::sync::Arc::new(
                        (0..cfg.num_kv_heads * cfg.head_dim)
                            .map(|i| f(i, 92 + l) * 0.05)
                            .collect(),
                    ),
                    vb: std::sync::Arc::new(
                        (0..cfg.num_kv_heads * cfg.head_dim)
                            .map(|i| f(i, 93 + l) * 0.05)
                            .collect(),
                    ),
                    gate: m(fill(cfg.intermediate, cfg.hidden, 50 + l), cfg.hidden),
                    up: m(fill(cfg.intermediate, cfg.hidden, 60 + l), cfg.hidden),
                    down: m(fill(cfg.hidden, cfg.intermediate, 70 + l), cfg.intermediate),
                    norm_in: std::sync::Arc::new(
                        (0..cfg.hidden).map(|i| 1.0 + f(i, l) * 0.1).collect(),
                    ),
                    norm_post: std::sync::Arc::new(
                        (0..cfg.hidden).map(|i| 1.0 + f(i, l + 3) * 0.1).collect(),
                    ),
                })
                .collect(),
            norm: std::sync::Arc::new(
                (0..cfg.hidden).map(|i| 1.0 + f(i, 99) * 0.1).collect(),
            ),
            tied_head: true,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::testutil::{tiny_cfg, tiny_weights, tiny_weights_int8};
    use super::*;
    use crate::weights::WeightMat;

    /// Independent reference forward (test-local, no engine code): dense
    /// history arrays, hand-rolled rmsnorm/rope/attention/silu. Any wiring
    /// bug in the engine shows up as a mismatch here.
    fn ref_forward(
        cfg: &ModelCfg,
        w: &QwenWeights,
        tok: u32,
        pos: usize,
        hist_k: &[Vec<f32>], // per layer: [pos][head_dim] flattened kv-head-major? NO: [pos][kv_dim]
        hist_v: &[Vec<f32>],
        kv_out: &mut Vec<f32>,
    ) -> Vec<f32> {
        let (h, l, d, qh, kvh_n) =
            (cfg.hidden, cfg.num_layers, cfg.head_dim, cfg.num_q_heads, cfg.num_kv_heads);
        let qd = qh * d;
        let kvd = kvh_n * d;
        let group = qh / kvh_n;
        let mut x: Vec<f32> = w.embed.row_to_vec(tok as usize, h);
        kv_out.clear();
        for li in 0..l {
            let lw = &w.layers[li];
            // rmsnorm
            let rms: f32 =
                (x.iter().map(|v| v * v).sum::<f32>() / h as f32 + cfg.rms_eps).sqrt();
            let hn: Vec<f32> =
                x.iter().zip(lw.norm_in.iter()).map(|(v, g)| v / rms * g).collect();
            // projections
            let proj = |mat: &WeightMat, inp: &[f32], rows: usize| -> Vec<f32> {
                (0..rows).map(|j| mat.dot_row(j, inp, inp.len())).collect()
            };
            let projb = |mat: &WeightMat, bias: &[f32], inp: &[f32], rows: usize| -> Vec<f32> {
                let mut v = proj(mat, inp, rows);
                for (o, &b) in v.iter_mut().zip(bias.iter()) {
                    *o += b;
                }
                v
            };
            let mut q = projb(&lw.q, &lw.qb, &hn, qd);
            let mut k = projb(&lw.k, &lw.kb, &hn, kvd);
            let v = projb(&lw.v, &lw.vb, &hn, kvd);
            // rope half-split on q and k only
            let do_rope = |vec: &mut Vec<f32>, heads: usize| {
                for hh in 0..heads {
                    let base = hh * d;
                    for i in 0..d / 2 {
                        let freq = cfg.rope_theta.powf(-(2.0 * i as f32) / d as f32);
                        let ang = pos as f32 * freq;
                        let (c, s) = (ang.cos(), ang.sin());
                        let x1 = vec[base + i];
                        let x2 = vec[base + d / 2 + i];
                        vec[base + i] = x1 * c - x2 * s;
                        vec[base + d / 2 + i] = x2 * c + x1 * s;
                    }
                }
            };
            do_rope(&mut q, qh);
            do_rope(&mut k, kvh_n);
            kv_out.extend_from_slice(&k);
            kv_out.extend_from_slice(&v);
            // attention: scores over history positions + own (post-rope k)
            let mut attn = vec![0.0f32; qd];
            for hh in 0..qh {
                let kh = hh / group;
                let mut scores: Vec<f32> = (0..pos)
                    .map(|p| {
                        let row = &hist_k[li][p * kvd..p * kvd + kvd];
                        let s: f32 = (0..d).map(|i| q[hh * d + i] * row[kh * d + i]).sum::<f32>();
                        s / (d as f32).sqrt()
                    })
                    .collect();
                let own: f32 = (0..d)
                    .map(|i| q[hh * d + i] * k[kh * d + i])
                    .sum::<f32>()
                    / (d as f32).sqrt();
                scores.push(own);
                let mx = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let ex: Vec<f32> = scores.iter().map(|s| (s - mx).exp()).collect();
                let sum: f32 = ex.iter().sum();
                for i in 0..d {
                    let mut s = 0.0f32;
                    for p in 0..pos {
                        s += ex[p] * hist_v[li][p * kvd + kh * d + i];
                    }
                    s += ex[pos] * v[kh * d + i];
                    attn[hh * d + i] = s / sum;
                }
            }
            let o = proj(&lw.o, &attn, h);
            for i in 0..h {
                x[i] += o[i];
            }
            let rms2: f32 =
                (x.iter().map(|v| v * v).sum::<f32>() / h as f32 + cfg.rms_eps).sqrt();
            let hn2: Vec<f32> =
                x.iter().zip(lw.norm_post.iter()).map(|(v, g)| v / rms2 * g).collect();
            let gate = proj(&lw.gate, &hn2, cfg.intermediate);
            let up = proj(&lw.up, &hn2, cfg.intermediate);
            let acted: Vec<f32> = gate
                .iter()
                .zip(&up)
                .map(|(g, u)| (g / (1.0 + (-g).exp())) * u)
                .collect();
            let down = proj(&lw.down, &acted, h);
            for i in 0..h {
                x[i] += down[i];
            }
        }
        x
    }

    fn ref_logits(cfg: &ModelCfg, w: &QwenWeights, x: &[f32]) -> Vec<f32> {
        let h = cfg.hidden;
        let rms: f32 = (x.iter().map(|v| v * v).sum::<f32>() / h as f32 + cfg.rms_eps).sqrt();
        let xn: Vec<f32> = x.iter().zip(w.norm.iter()).map(|(v, g)| v / rms * g).collect();
        (0..cfg.vocab)
            .map(|j| w.embed.row_to_vec(j, h).iter().zip(&xn).map(|(a, b)| a * b).sum::<f32>())
            .collect()
    }

    fn approx(a: &[f32], b: &[f32], tol: f32) -> bool {
        a.len() == b.len()
            && a.iter().zip(b).all(|(x, y)| (x - y).abs() / (1.0 + y.abs()) < tol)
    }

    #[test]
    fn engine_matches_independent_reference_int8() {
        // Same gate, INT8 storage (docs/WEIGHTS8.md gate 1c): the reference
        // forward reads the same quantized storage, so agreement still
        // catches wiring/order bugs; accuracy of the quantization itself is
        // bounded by the weights.rs int8 test + the real-model probe.
        let cfg = tiny_cfg();
        let w = tiny_weights_int8();
        let cache_cfg = inferno_core::kv_cache::KvConfig {
            block_size: 4,
            num_blocks: 8,
            num_layers: cfg.num_layers,
            num_kv_heads: cfg.num_kv_heads,
            head_dim: cfg.head_dim,
        };
        let mut cache = inferno_core::kv_cache::KvCache::new(cache_cfg);
        cache.reserve(7, 6, &[]).unwrap();
        let eng = QwenEngine::new(cfg.clone(), w);

        let prompt: Vec<u32> = vec![3, 11, 5, 21, 8];
        let mut ref_k: Vec<Vec<f32>> = vec![Vec::new(); cfg.num_layers];
        let mut ref_v: Vec<Vec<f32>> = vec![Vec::new(); cfg.num_layers];
        let mut kv = Vec::new();
        for (p, &t) in prompt.iter().enumerate() {
            eng.forward_token(t, p, &cache, 7, &mut kv);
            cache.write_token(7, p, &kv).unwrap();
            let mut rkv = Vec::new();
            let _ = ref_forward(&cfg, &eng.w, t, p, &ref_k, &ref_v, &mut rkv);
            for li in 0..cfg.num_layers {
                let off = li * 2 * (cfg.num_kv_heads * cfg.head_dim);
                ref_k[li].extend_from_slice(
                    &rkv[off..off + cfg.num_kv_heads * cfg.head_dim],
                );
                ref_v[li].extend_from_slice(
                    &rkv[off + cfg.num_kv_heads * cfg.head_dim
                        ..off + 2 * cfg.num_kv_heads * cfg.head_dim],
                );
            }
            assert!(
                approx(&kv, &rkv, 1e-4),
                "kv mismatch at pos {p} (int8)"
            );
        }
        let mut kv2 = Vec::new();
        let x_last = eng.forward_token(*prompt.last().unwrap(), prompt.len() - 1, &cache, 7, &mut kv2);
        let got = eng.logits_of(&x_last);
        let mut rkv2 = Vec::new();
        let rx = ref_forward(&cfg, &eng.w, *prompt.last().unwrap(), prompt.len() - 1, &ref_k, &ref_v, &mut rkv2);
        let want = ref_logits(&cfg, &eng.w, &rx);
        assert!(approx(&got, &want, 1e-3), "logits mismatch (int8)");
    }

    #[test]
    fn engine_matches_independent_reference_single_and_multi_token() {
        let cfg = tiny_cfg();
        let w = tiny_weights();
        let cache_cfg = inferno_core::kv_cache::KvConfig {
            block_size: 4,
            num_blocks: 8,
            num_layers: cfg.num_layers,
            num_kv_heads: cfg.num_kv_heads,
            head_dim: cfg.head_dim,
        };
        let mut cache = inferno_core::kv_cache::KvCache::new(cache_cfg);
        cache.reserve(7, 6, &[]).unwrap();
        let eng = QwenEngine::new(cfg.clone(), w);

        let prompt: Vec<u32> = vec![3, 11, 5, 21, 8];
        let mut ref_k: Vec<Vec<f32>> = vec![Vec::new(); cfg.num_layers];
        let mut ref_v: Vec<Vec<f32>> = vec![Vec::new(); cfg.num_layers];
        let mut kv = Vec::new();
        for (p, &t) in prompt.iter().enumerate() {
            eng.forward_token(t, p, &cache, 7, &mut kv);
            cache.write_token(7, p, &kv).unwrap();
            // reference at the same position, against history that does NOT
            // yet include this token (the engine self-extends instead)
            let mut rkv = Vec::new();
            let _ = ref_forward(&cfg, &eng.w, t, p, &ref_k, &ref_v, &mut rkv);
            // ref_forward appends k then v per layer into rkv: [l][k][d] then [l][v][d]
            for li in 0..cfg.num_layers {
                let off = li * 2 * (cfg.num_kv_heads * cfg.head_dim);
                ref_k[li].extend_from_slice(
                    &rkv[off..off + cfg.num_kv_heads * cfg.head_dim],
                );
                ref_v[li].extend_from_slice(
                    &rkv[off + cfg.num_kv_heads * cfg.head_dim
                        ..off + 2 * cfg.num_kv_heads * cfg.head_dim],
                );
            }
            // engine kv layout must equal reference
            assert!(
                approx(&kv, &rkv, 1e-4),
                "kv mismatch at pos {p}: eng={kv:?} ref={rkv:?}"
            );
        }
        // final logits of the last prompt token
        let mut kv2 = Vec::new();
        let x_last = eng.forward_token(*prompt.last().unwrap(), prompt.len() - 1, &cache, 7, &mut kv2);
        let got = eng.logits_of(&x_last);
        let mut rkv2 = Vec::new();
        let rx = ref_forward(&cfg, &eng.w, *prompt.last().unwrap(), prompt.len() - 1, &ref_k, &ref_v, &mut rkv2);
        let want = ref_logits(&cfg, &eng.w, &rx);
        assert!(approx(&got, &want, 1e-3), "logits mismatch: got[:6]={:?} want[:6]={:?}", &got[..6], &want[..6]);
    }
}

impl Engine for QwenEngine {
    fn prefill_chunk(&mut self, seq: &mut Sequence, cache: &mut KvCache, budget: usize) -> usize {
        let end = (seq.pos + budget).min(seq.prompt.len());
        let mut processed = 0usize;
        // Fully-reused prompt (every block prefix-cached): no token needs a
        // forward for KV, but the next-token logits for the LAST prompt token
        // are transient — they exist only in the round that computed them.
        // Re-forward that one token to restash logits. The KV this produces is
        // bit-identical to what the cache already holds, so the batcher skips
        // the write (no CoW on the shared blocks).
        if seq.pos >= end && end == seq.prompt.len() && !seq.prompt.is_empty() {
            let pos = seq.prompt.len() - 1;
            let tok = seq.prompt[pos];
            let mut kv = Vec::new();
            let x = self.forward_token(tok, pos, cache, seq.id, &mut kv);
            let logits = self.logits_of(&x);
            self.stash.insert(seq.id, Stash { logits, logits_pos: pos });
            return 0;
        }
        while seq.pos < end {
            let tok = seq.prompt[seq.pos];
            let pos = seq.pos;
            let mut kv = Vec::new();
            let x = self.forward_token(tok, pos, cache, seq.id, &mut kv);
            if pos + 1 == seq.prompt.len() {
                let logits = self.logits_of(&x);
                self.stash.insert(seq.id, Stash { logits, logits_pos: pos });
            }
            if cache.write_token(seq.id, pos, &kv).is_err() {
                break; // pool/CoW failure: batcher preempts
            }
            seq.pos += 1;
            processed += 1;
        }
        processed
    }

    fn decode_step(&mut self, seq: &mut Sequence, _cache: &KvCache) -> u32 {
        match self.stash.get(&seq.id) {
            Some(s) if s.logits_pos + 1 == seq.decode_pos() => argmax(&s.logits),
            Some(_) => panic!(
                "decode_step: stale logits stash for seq {} (stash pos {} vs decode_pos {}); \
                 mid-decode preemption is unsupported in v1 — see engine.rs header",
                seq.id,
                self.stash.get(&seq.id).unwrap().logits_pos,
                seq.decode_pos()
            ),
            None => panic!(
                "decode_step: no logits stash for seq {} — prefill must run first",
                seq.id
            ),
        }
    }

    fn decode_kv(
        &mut self,
        seq: &Sequence,
        tok: u32,
        pos: usize,
        cache: &KvCache,
    ) -> Vec<f32> {
        let mut kv = Vec::new();
        let x = self.forward_token(tok, pos, cache, seq.id, &mut kv);
        let logits = self.logits_of(&x);
        self.stash.insert(seq.id, Stash { logits, logits_pos: pos });
        kv
    }

    fn eos(&self) -> Option<u32> {
        Some(self.cfg.eos_token)
    }
}
