//! safetensors loading for Qwen2 checkpoints: fp16 -> f32 conversion,
//! native HF weight layout, no transposed copies.

use memmap2::Mmap;
use safetensors::{Dtype, SafeTensors};
use std::fs::File;
use std::path::Path;

use crate::config::ModelCfg;

pub struct LayerWeights {
    /// self_attn projections, native [out, in] row-major layout.
    pub q: Vec<f32>,
    pub k: Vec<f32>,
    pub v: Vec<f32>,
    pub o: Vec<f32>,
    /// QKV bias — Qwen2.5 adds bias to q/k/v projections (72 bias tensors
    /// confirmed in the checkpoint header). o_proj and the MLP have none.
    pub qb: Vec<f32>,
    pub kb: Vec<f32>,
    pub vb: Vec<f32>,
    /// MLP (SwiGLU): down(silu(gate(x)) * up(x))
    pub gate: Vec<f32>,
    pub up: Vec<f32>,
    pub down: Vec<f32>,
    /// RMSNorm gains.
    pub norm_in: Vec<f32>,
    pub norm_post: Vec<f32>,
}

pub struct QwenWeights {
    /// Token embedding [vocab, hidden]; doubles as lm_head when weights are
    /// tied (Qwen2.5-0.5B: tie_word_embeddings = true).
    pub embed: Vec<f32>,
    pub layers: Vec<LayerWeights>,
    /// Final RMSNorm gain [hidden].
    pub norm: Vec<f32>,
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

/// Read one tensor as f32, converting bf16/fp16 on the fly, validating the
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

        let embed = tensor_to_f32(&st, "model.embed_tokens.weight", &[cfg.vocab, h])?;
        if embed.len() != cfg.vocab * h {
            return Err(format!(
                "embed size {} != vocab*hidden {}",
                embed.len(),
                cfg.vocab * h
            ));
        }
        let norm = tensor_to_f32(&st, "model.norm.weight", &[h])?;

        let mut layers = Vec::with_capacity(cfg.num_layers);
        for i in 0..cfg.num_layers {
            // NOTE: closure borrows `st` (no move) — moving would consume the
            // deserialized index on the first iteration.
            let p = |n: &str, want: &[usize]| {
                tensor_to_f32(&st, &format!("model.layers.{i}.{n}"), want)
            };
            let w = LayerWeights {
                q: p("self_attn.q_proj.weight", &[qd, h])?,
                k: p("self_attn.k_proj.weight", &[kvd, h])?,
                v: p("self_attn.v_proj.weight", &[kvd, h])?,
                o: p("self_attn.o_proj.weight", &[h, qd])?,
                qb: p("self_attn.q_proj.bias", &[qd])?,
                kb: p("self_attn.k_proj.bias", &[kvd])?,
                vb: p("self_attn.v_proj.bias", &[kvd])?,
                gate: p("mlp.gate_proj.weight", &[inter, h])?,
                up: p("mlp.up_proj.weight", &[inter, h])?,
                down: p("mlp.down_proj.weight", &[h, inter])?,
                norm_in: p("input_layernorm.weight", &[h])?,
                norm_post: p("post_attention_layernorm.weight", &[h])?,
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
                norm,
                tied_head,
            },
        ))
    }
}
