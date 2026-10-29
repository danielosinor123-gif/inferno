//! safetensors loading for Qwen2 checkpoints: fp16 -> f32 conversion,
//! native HF weight layout, no transposed copies.

use memmap2::Mmap;
use safetensors::{Dtype, SafeTensors};
use std::fs::File;
use std::path::Path;
use std::sync::Arc;

use crate::config::ModelCfg;

/// A weight matrix in the checkpoint's native dtype, converted per row at
/// use time. bf16 → f32 is exact (`bits << 16`) and halves the bytes streamed
/// per token — the dominant memory traffic of the forward pass
/// (docs/WEIGHTS16.md). Arc'd for 'static captures into the gemv pool.
pub enum WeightMat {
    F32(Arc<Vec<f32>>),
    BF16(Arc<Vec<u16>>),
    F16(Arc<Vec<u16>>),
}

impl WeightMat {
    pub fn elem_len(&self) -> usize {
        match self {
            WeightMat::F32(v) => v.len(),
            WeightMat::BF16(v) => v.len(),
            WeightMat::F16(v) => v.len(),
        }
    }

    /// Element count (rows*cols) for shape validation.
    pub fn len(&self) -> usize {
        self.elem_len()
    }

    pub fn is_empty(&self) -> bool {
        self.elem_len() == 0
    }

    /// f32 value of element `idx` (per-row conversion).
    #[inline]
    pub fn at(&self, idx: usize) -> f32 {
        match self {
            WeightMat::F32(v) => v[idx],
            WeightMat::BF16(v) => bf16_bits_to_f32(v[idx]),
            WeightMat::F16(v) => f16_bits_to_f32(v[idx]),
        }
    }

    /// Row j of a [rows, k] matrix as f32 — the per-gemv-row accessor.
    /// The dtype dispatch happens ONCE PER ROW (not per element — a per-
    /// element match blocks auto-vectorization of the inner loop, measured
    /// 12-50% slower in W6). Each inner loop stays sequential in i, so
    /// results are bit-identical to the f32 path on identical values.
    #[inline]
    pub fn dot_row(&self, j: usize, x: &[f32], k: usize) -> f32 {
        let base = j * k;
        match self {
            WeightMat::F32(v) => {
                let row = &v[base..base + k];
                let mut s = 0.0f32;
                for i in 0..k {
                    s += row[i] * x[i];
                }
                s
            }
            WeightMat::BF16(v) => {
                let row = &v[base..base + k];
                let mut s = 0.0f32;
                for i in 0..k {
                    s += bf16_bits_to_f32(row[i]) * x[i];
                }
                s
            }
            WeightMat::F16(v) => {
                let row = &v[base..base + k];
                let mut s = 0.0f32;
                for i in 0..k {
                    s += f16_bits_to_f32(row[i]) * x[i];
                }
                s
            }
        }
    }

    /// Storage dtype tag (diagnostics/tests).
    pub fn tag(&self) -> &'static str {
        match self {
            WeightMat::F32(_) => "f32",
            WeightMat::BF16(_) => "bf16",
            WeightMat::F16(_) => "f16",
        }
    }

    /// Resident bytes for this matrix (the registry's budget accounting).
    pub fn bytes(&self) -> usize {
        match self {
            WeightMat::F32(v) => 4 * v.len(),
            WeightMat::BF16(v) | WeightMat::F16(v) => 2 * v.len(),
        }
    }
}

/// f32 → bf16 bits, round-to-nearest-even on the truncated mantissa.
/// bf16 shares f32's exponent, so this is exact up to mantissa rounding.
pub fn f32_to_bf16_bits(x: f32) -> u16 {
    let bits = x.to_bits();
    // NaN handling: keep it a NaN (preserve sign, saturate exponent).
    if x.is_nan() {
        return ((bits >> 16) as u16) | 0x0200;
    }
    let lsb = (bits >> 16) & 1;
    let rounded = bits.wrapping_add(0x7FFF + lsb);
    (rounded >> 16) as u16
}

/// Build bf16 storage from f32 values (tests: exact-by-construction gate).
pub fn bf16_from_f32(v: Vec<f32>) -> WeightMat {
    let u: Vec<u16> = v.iter().map(|&x| f32_to_bf16_bits(x)).collect();
    WeightMat::BF16(Arc::new(u))
}

pub struct LayerWeights {
    /// self_attn projections, native [out, in] row-major layout, bf16-stored.
    pub q: WeightMat,
    pub k: WeightMat,
    pub v: WeightMat,
    pub o: WeightMat,
    /// QKV bias — Qwen2.5 adds bias to q/k/v projections (72 bias tensors
    /// confirmed in the checkpoint header). o_proj and the MLP have none.
    /// Small: kept f32 (exact, read once per layer per forward).
    pub qb: Arc<Vec<f32>>,
    pub kb: Arc<Vec<f32>>,
    pub vb: Arc<Vec<f32>>,
    /// MLP (SwiGLU): down(silu(gate(x)) * up(x))
    pub gate: WeightMat,
    pub up: WeightMat,
    pub down: WeightMat,
    /// RMSNorm gains.
    pub norm_in: Arc<Vec<f32>>,
    pub norm_post: Arc<Vec<f32>>,
}

pub struct QwenWeights {
    /// Token embedding [vocab, hidden]; doubles as lm_head when weights are
    /// tied (Qwen2.5-0.5B: tie_word_embeddings = true). bf16-stored.
    pub embed: WeightMat,
    pub layers: Vec<LayerWeights>,
    /// Final RMSNorm gain [hidden].
    pub norm: Arc<Vec<f32>>,
    pub tied_head: bool,
}

/// bf16/f16 (little-endian u16) -> f32, no external crate.
/// bf16 is a truncated f32: same 8-bit exponent as f32, 7-bit mantissa —
/// conversion is exactly `bits << 16`.
pub fn bf16_bits_to_f32(bits: u16) -> f32 {
    f32::from_bits((bits as u32) << 16)
}

/// f16 (IEEE 754 binary16) -> f32: different exponent width/bias, so the
/// expansion is manual.
pub fn f16_bits_to_f32(bits: u16) -> f32 {
    let sign = ((bits & 0x8000) as u32) << 16;
    let exp = ((bits & 0x7C00) >> 10) as u32;
    let man = (bits & 0x03FF) as u32;
    let bits32 = if exp == 0 {
        if man == 0 {
            sign // zero
        } else {
            // subnormal: renormalize
            let mut e = 127 - 15 + 1;
            let mut m = man;
            while m & 0x400 == 0 {
                m <<= 1;
                e -= 1;
            }
            m &= 0x3FF;
            sign | (e << 23) | (m << 13)
        }
    } else if exp == 0x1F {
        // inf / nan
        sign | 0x7F80_0000 | (man << 13)
    } else {
        sign | ((exp + 112) << 23) | (man << 13)
    };
    f32::from_bits(bits32)
}

/// Read one tensor into dtype-preserving `WeightMat` storage, validating the
/// exact safetensors shape. Shape validation matters because several weight
/// matrices are square ([h, h], [h, qd] vs [qd, h]) — a length-only check
/// would silently accept a transposed checkpoint and scramble the math.
/// A bf16 checkpoint becomes a pure bit-copy (no conversion at load; the
/// unchanged `bits << 16` conversion happens per row at use time).
fn tensor_to_mat(st: &SafeTensors, name: &str, want: &[usize]) -> Result<WeightMat, String> {
    let t = st
        .tensor(name)
        .map_err(|e| format!("missing tensor {name}: {e}"))?;
    let sh = t.shape();
    if sh != want {
        return Err(format!(
            "tensor {name}: shape {sh:?} != expected {want:?}"
        ));
    }
    let n: usize = want.iter().product();
    let bytes = t.data();
    let copy_u16 = || -> Vec<u16> {
        // Exact-size preallocation: push into a reserved buffer (no collect
        // growth transients — those used to cost ~1.4 GB on a 4 GB VM).
        let mut out: Vec<u16> = Vec::with_capacity(n);
        for c in bytes.chunks_exact(2) {
            out.push(u16::from_le_bytes([c[0], c[1]]));
        }
        out
    };
    let mat = match t.dtype() {
        Dtype::BF16 => WeightMat::BF16(Arc::new(copy_u16())),
        Dtype::F16 => WeightMat::F16(Arc::new(copy_u16())),
        Dtype::F32 => {
            let mut out: Vec<f32> = Vec::with_capacity(n);
            for c in bytes.chunks_exact(4) {
                out.push(f32::from_le_bytes([c[0], c[1], c[2], c[3]]));
            }
            WeightMat::F32(Arc::new(out))
        }
        other => {
            return Err(format!(
                "tensor {name}: unsupported dtype {other:?} (need bf16/fp16/fp32)"
            ))
        }
    };
    debug_assert_eq!(mat.elem_len(), n);
    Ok(mat)
}
/// exact safetensors shape. Writes DIRECTLY into an exact-size buffer — no
/// `.collect()` (iterator collect cannot pre-size, so a 544 MB tensor
/// transiently costs up to ~1.4 GB in growth-copies; on a 4 GB VM under host
/// memory pressure that spike is what gets the whole machine killed).
fn tensor_to_f32(st: &SafeTensors, name: &str, want: &[usize]) -> Result<Vec<f32>, String> {
    let t = st
        .tensor(name)
        .map_err(|e| format!("missing tensor {name}: {e}"))?;
    let sh = t.shape();
    if sh != want {
        return Err(format!(
            "tensor {name}: shape {sh:?} != expected {want:?}"
        ));
    }
    let n: usize = want.iter().product();
    let mut out = vec![0.0f32; n];
    match t.dtype() {
        Dtype::BF16 => {
            for (c, o) in t.data().chunks_exact(2).zip(out.iter_mut()) {
                *o = bf16_bits_to_f32(u16::from_le_bytes([c[0], c[1]]));
            }
        }
        Dtype::F16 => {
            for (c, o) in t.data().chunks_exact(2).zip(out.iter_mut()) {
                *o = f16_bits_to_f32(u16::from_le_bytes([c[0], c[1]]));
            }
        }
        Dtype::F32 => {
            for (c, o) in t.data().chunks_exact(4).zip(out.iter_mut()) {
                *o = f32::from_le_bytes([c[0], c[1], c[2], c[3]]);
            }
        }
        other => {
            return Err(format!(
                "tensor {name}: unsupported dtype {other:?} (need bf16/fp16/fp32)"
            ))
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bf16_roundtrip_exact_and_bounded() {
        // bf16 shares f32's exponent, so only the mantissa rounds:
        // values with <= 8 significant bits round-trip bit-exactly...
        for v in [0.0f32, 1.0, -2.5, 0.03125, -127.0, 256.0] {
            let back = bf16_bits_to_f32(f32_to_bf16_bits(v));
            assert_eq!(v.to_bits(), back.to_bits(), "bf16 round-trip must be exact for {v}");
        }
        // ...and everything else (any full 24-bit mantissa, like 1e-30 or PI)
        // is bounded by bf16's 8-bit mantissa.
        for v in [
            1.0f32 / 3.0,
            0.123_456_79,
            std::f32::consts::PI,
            1e-17,
            1e20,
            1e-30,
            1e30,
        ] {
            let back = bf16_bits_to_f32(f32_to_bf16_bits(v));
            let rel = (v - back).abs() / v.abs();
            assert!(
                rel <= 2.0f32.powi(-8) + 1e-6,
                "bf16 rel error {rel} exceeds 2^-8 for {v}"
            );
        }
    }
}

impl QwenWeights {
    /// Resident bytes of all weights (matrices at their storage dtype,
    /// norms/biases as f32) — the registry's budget accounting input.
    pub fn weight_bytes(&self) -> usize {
        let mut b = self.embed.bytes() + 4 * self.norm.len();
        for l in &self.layers {
            b += l.q.bytes()
                + l.k.bytes()
                + l.v.bytes()
                + l.o.bytes()
                + l.gate.bytes()
                + l.up.bytes()
                + l.down.bytes()
                + 4 * (l.qb.len() + l.kb.len() + l.vb.len() + l.norm_in.len() + l.norm_post.len());
        }
        b
    }
}

impl QwenWeights {
    /// Load a Qwen2.5 model directory (config.json + model.safetensors).
    /// Peak RAM: f32 weights + mapped fp16 file (~3x model size in fp16
    /// terms); the mapping is dropped before returning.
    pub fn load(dir: &Path) -> Result<(ModelCfg, Self), String> {
        let cfg_path = dir.join("config.json");
        let cfg_bytes =
            std::fs::read(&cfg_path).map_err(|e| format!("reading {cfg_path:?}: {e}"))?;
        let cfg_json: serde_json::Value = serde_json::from_slice(&cfg_bytes)
            .map_err(|e| format!("parsing config.json: {e}"))?;
        let cfg = ModelCfg::from_json(&cfg_json)?;

        let st_path = dir.join("model.safetensors");
        let file = File::open(&st_path).map_err(|e| format!("opening {st_path:?}: {e}"))?;
        // SAFETY: read-only mapping of a file we only interpret.
        let mmap = unsafe { Mmap::map(&file) }.map_err(|e| format!("mmap {st_path:?}: {e}"))?;
        let st = SafeTensors::deserialize(&mmap)
            .map_err(|e| format!("parsing safetensors: {e}"))?;

        let h = cfg.hidden;
        let qd = cfg.num_q_heads * cfg.head_dim;
        let kvd = cfg.num_kv_heads * cfg.head_dim;
        let inter = cfg.intermediate;

        let embed = tensor_to_mat(&st, "model.embed_tokens.weight", &[cfg.vocab, h])?;
        if embed.elem_len() != cfg.vocab * h {
            return Err(format!(
                "embed size {} != vocab*hidden {}",
                embed.elem_len(),
                cfg.vocab * h
            ));
        }
        let norm = tensor_to_f32(&st, "model.norm.weight", &[h])?;

        let mut layers = Vec::with_capacity(cfg.num_layers);
        for i in 0..cfg.num_layers {
            // NOTE: closure borrows `st` (no move) — moving would consume the
            // deserialized index on the first iteration.
            // Big matrices: dtype-preserving storage (bf16 stays bf16 — half
            // the bytes streamed per token, docs/WEIGHTS16.md).
            let pmat = |n: &str, want: &[usize]| {
                tensor_to_mat(&st, &format!("model.layers.{i}.{n}"), want)
            };
            // Small vectors: converted to f32 once at load (biases + norms
            // are read per layer per forward; exactness matters, size doesn't).
            let pvec = |n: &str, want: &[usize]| {
                tensor_to_f32(&st, &format!("model.layers.{i}.{n}"), want).map(Arc::new)
            };
            let w = LayerWeights {
                q: pmat("self_attn.q_proj.weight", &[qd, h])?,
                k: pmat("self_attn.k_proj.weight", &[kvd, h])?,
                v: pmat("self_attn.v_proj.weight", &[kvd, h])?,
                o: pmat("self_attn.o_proj.weight", &[h, qd])?,
                qb: pvec("self_attn.q_proj.bias", &[qd])?,
                kb: pvec("self_attn.k_proj.bias", &[kvd])?,
                vb: pvec("self_attn.v_proj.bias", &[kvd])?,
                gate: pmat("mlp.gate_proj.weight", &[inter, h])?,
                up: pmat("mlp.up_proj.weight", &[inter, h])?,
                down: pmat("mlp.down_proj.weight", &[h, inter])?,
                norm_in: pvec("input_layernorm.weight", &[h])?,
                norm_post: pvec("post_attention_layernorm.weight", &[h])?,
            };
            let shape_ok = w.q.len() == qd * h
                && w.k.len() == kvd * h
                && w.v.len() == kvd * h
                && w.o.len() == h * qd
                && w.gate.len() == inter * h
                && w.up.len() == inter * h
                && w.down.len() == h * inter
                && w.norm_in.len() == h
                && w.norm_post.len() == h;
            if !shape_ok {
                return Err(format!("layer {i}: unexpected weight shape"));
            }
            layers.push(w);
        }

        // Untied lm_head, if present.
        let tied_head = match st.tensor("lm_head.weight") {
            Ok(_) => {
                if cfg.tie_word_embeddings {
                    return Err("lm_head present but config says tied — refusing to guess".into());
                }
                // Stored separately in that case; not supported in v1 (0.5B is tied).
                return Err("untied lm_head loading is not implemented in v1".into());
            }
            Err(_) => true, // no lm_head tensor: use embedding
        };

        drop(st);
        drop(mmap); // release the fp16 mapping before returning

        Ok((
            cfg,
            Self {
                embed,
                layers,
                norm: Arc::new(norm),
                tied_head,
            },
        ))
    }
}
