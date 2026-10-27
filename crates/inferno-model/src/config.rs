//! config.json parsing (Qwen2 family). Value-based extraction — no serde
//! derive dependency; unknown fields are ignored on purpose.

use serde_json::Value;

#[derive(Clone, Debug)]
pub struct ModelCfg {
    pub num_layers: usize,
    pub num_q_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub hidden: usize,
    pub intermediate: usize,
    pub vocab: usize,
    pub rms_eps: f32,
    pub rope_theta: f32,
    pub tie_word_embeddings: bool,
    pub eos_token: u32,
    pub max_ctx: usize,
}

fn field<'a>(v: &'a Value, name: &str) -> Result<&'a Value, String> {
    v.get(name).ok_or_else(|| format!("config.json missing field: {name}"))
}

fn as_usize(v: &Value, name: &str) -> Result<usize, String> {
    field(v, name)?
        .as_u64()
        .map(|x| x as usize)
        .ok_or_else(|| format!("config field {name} is not an integer"))
}

impl ModelCfg {
    pub fn from_json(v: &Value) -> Result<Self, String> {
        let q_heads = as_usize(v, "num_attention_heads")?;
        let kv_heads = as_usize(v, "num_key_value_heads")?;
        let head_dim = as_usize(v, "head_dim").unwrap_or_else(|_| {
            // Older configs omit head_dim: derive from hidden/heads.
            let hidden = as_usize(v, "hidden_size").unwrap_or(0);
            if q_heads > 0 { hidden / q_heads } else { 0 }
        });
        let eos = field(v, "eos_token_id")?
            .as_u64()
            .ok_or("config field eos_token_id is not an integer")? as u32;
        let max_ctx = as_usize(v, "max_position_embeddings").unwrap_or(32_768);
        Ok(Self {
            num_layers: as_usize(v, "num_hidden_layers")?,
            num_q_heads: q_heads,
            num_kv_heads: kv_heads,
            head_dim,
            hidden: as_usize(v, "hidden_size")?,
            intermediate: as_usize(v, "intermediate_size")?,
            vocab: as_usize(v, "vocab_size")?,
            rms_eps: field(v, "rms_norm_eps")?
                .as_f64()
                .ok_or("config field rms_norm_eps is not a float")? as f32,
            rope_theta: field(v, "rope_theta")?
                .as_f64()
                .unwrap_or(10_000.0) as f32,
            tie_word_embeddings: field(v, "tie_word_embeddings")?
                .as_bool()
                .unwrap_or(false),
            eos_token: eos,
            max_ctx,
        })
    }

    /// Qwen2.5-0.5B-Instruct geometry, for tests that must not hit the disk.
    pub fn qwen05b() -> Self {
        Self {
            num_layers: 24,
            num_q_heads: 14,
            num_kv_heads: 2,
            head_dim: 64,
            hidden: 896,
            intermediate: 4864,
            vocab: 151_936,
            rms_eps: 1e-6,
            rope_theta: 1_000_000.0,
            tie_word_embeddings: true,
            eos_token: 151_645,
            max_ctx: 32_768,
        }
    }
}
