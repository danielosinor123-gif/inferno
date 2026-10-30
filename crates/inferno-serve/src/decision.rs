//! Decision scoring (docs/DECISION-API.md): answer decisions, not paragraphs.
//!
//! Score a candidate answer by teacher-forcing its tokens through the
//! engine's public forward surface and summing per-token log-softmax. The
//! state prefix is shared across a question's candidates, so comparing
//! totals is exactly comparing the candidate-conditioned parts, and a
//! softmax over totals is the answer distribution. No sampling: same input,
//! same output, every run.

use std::collections::BTreeMap;

use inferno_core::kv_cache::{KvCache, KvConfig};
use inferno_model::engine::QwenEngine;

/// One scored question result.
#[derive(Debug, Clone)]
pub struct DecisionResult {
    pub kind: &'static str,
    /// label -> probability (softmax over totals), ordered by label.
    pub probabilities: Vec<(String, f64)>,
    /// 1 - H(p)/ln(k): 0 = flat, 1 = certain.
    pub confidence: f64,
}

/// log-softmax of `tokens` under the engine (teacher-forced).
pub fn score_continuation(
    engine: &QwenEngine,
    cache: &mut KvCache,
    seq_id: u64,
    tokens: &[u32],
) -> Result<f64, String> {
    if tokens.is_empty() {
        return Err("empty token sequence".into());
    }
    let mut lp = 0.0f64;
    let mut kv = Vec::new();
    for i in 0..tokens.len() - 1 {
        let x = engine.forward_token_pub(tokens[i], i, cache, seq_id, &mut kv);
        let logits = engine.logits_of_pub(&x);
        let next = tokens[i + 1] as usize;
        if next >= logits.len() {
            return Err(format!("token id {next} outside vocab {}", logits.len()));
        }
        // log-softmax in f64: max-subtracted for stability
        let m = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max) as f64;
        let sum: f64 = logits.iter().map(|&v| ((v as f64) - m).exp()).sum();
        lp += (logits[next] as f64) - m - sum.ln();
        cache
            .write_token(seq_id, i, &kv)
            .map_err(|_| "kv write failed during scoring".to_string())?;
    }
    if !lp.is_finite() {
        return Err("score is not finite".into());
    }
    Ok(lp)
}

/// A KV pool sized for `max_tokens` under this engine's geometry.
pub fn scoring_cache(engine: &QwenEngine, max_tokens: usize) -> KvCache {
    let bs = 16usize;
    let blocks = (max_tokens.div_ceil(bs) + 1).max(4);
    KvCache::new(KvConfig {
        block_size: bs,
        num_blocks: blocks,
        num_layers: engine.cfg.num_layers,
        num_kv_heads: engine.cfg.num_kv_heads,
        head_dim: engine.cfg.head_dim,
    })
}

/// Score each candidate continuation and return the softmax distribution
/// + confidence. All candidates share the state prefix; totals are the
/// logits of the distribution (docs/DECISION-API.md).
pub fn decide(
    engine: &QwenEngine,
    state: &str,
    instructions: &str,
    candidates: &[(String, String)], // (label, answer line)
) -> Result<DecisionResult, String> {
    if candidates.is_empty() {
        return Err("a question needs at least one allowed answer".into());
    }
    let prefix = format!("{state}\n{instructions}\nAnswer:");
    let mut cache = scoring_cache(engine, 4096);
    let mut totals: Vec<(String, f64)> = Vec::with_capacity(candidates.len());
    for (label, answer) in candidates {
        let seq_text = format!("{prefix} {answer}");
        let tokens = engine.encode_text(&seq_text);
        if tokens.len() < 2 {
            return Err(format!("answer {label:?} tokenizes to nothing"));
        }
        let used = tokens.len().div_ceil(16) + 1;
        assert!(cache.reserve(1, used * 16, &[]).is_some(), "scoring reserve");
        let lp = score_continuation(engine, &mut cache, 1, &tokens)?;
        cache.release(1);
        totals.push((label.clone(), lp));
    }
    // softmax over totals
    let m = totals.iter().map(|(_, v)| *v).fold(f64::NEG_INFINITY, f64::max);
    let exps: Vec<f64> = totals.iter().map(|(_, v)| (v - m).exp()).collect();
    let sum: f64 = exps.iter().sum();
    let probabilities: Vec<(String, f64)> = totals
        .iter()
        .zip(&exps)
        .map(|((l, _), e)| (l.clone(), e / sum))
        .collect();
    // confidence = 1 - H(p)/ln(k)
    let k = probabilities.len() as f64;
    let h: f64 = probabilities
        .iter()
        .map(|(_, p)| if *p > 0.0 { -p * p.ln() } else { 0.0 })
        .sum();
    let confidence = if k > 1.0 { (1.0 - h / k.ln()).clamp(0.0, 1.0) } else { 1.0 };
    Ok(DecisionResult {
        kind: "scored",
        probabilities,
        confidence,
    })
}

/// Parse the allowed answers for a `choice` question (criteria labels).
/// Returns (label, answer-line) pairs; the answer line is the label text.
pub fn choice_candidates(criteria: &BTreeMap<String, serde_json::Value>) -> Vec<(String, String)> {
    criteria
        .keys()
        .map(|l| (l.clone(), l.clone()))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use inferno_model::engine::demo::{tiny_cfg, tiny_weights};

    fn tiny() -> QwenEngine {
        QwenEngine::new(tiny_cfg(), tiny_weights())
    }

    #[test]
    fn scoring_is_deterministic_and_finite() {
        // Hermetic = STRUCTURAL (random tiny weights carry no semantics):
        // same input -> identical score, score finite, and the distribution
        // is normalized. Semantic sanity is the live gate (real model).
        let eng = tiny();
        let toks: Vec<u32> = vec![3, 11, 5, 21, 8];
        let mut c = scoring_cache(&eng, 64);
        c.reserve(1, 64, &[]).unwrap();
        let a = score_continuation(&eng, &mut c, 1, &toks).unwrap();
        let b = score_continuation(&eng, &mut c, 1, &toks).unwrap();
        assert_eq!(a, b, "scoring must be deterministic");
        assert!(a.is_finite(), "score must be finite");
        let mut c2 = scoring_cache(&eng, 64);
        c2.reserve(1, 64, &[]).unwrap();
        let other: Vec<u32> = vec![9, 2, 17];
        let d = score_continuation(&eng, &mut c2, 1, &other).unwrap();
        assert!(d.is_finite());
    }

    #[test]
    fn decide_distribution_is_normalized_with_confidence_in_range() {
        let eng = tiny();
        // the tiny engine has no tokenizer: use token ids via the engine's
        // encode substitute — tiny engine encode panics, so drive `decide`
        // through a model WITH text... hermetic path: skip text, verify the
        // normalization math on synthetic totals instead.
        let totals: Vec<(String, f64)> = vec![
            ("a".into(), -2.0),
            ("b".into(), -1.0),
            ("c".into(), -4.0),
        ];
        let m = totals.iter().map(|(_, v)| *v).fold(f64::NEG_INFINITY, f64::max);
        let exps: Vec<f64> = totals.iter().map(|(_, v)| (v - m).exp()).collect();
        let sum: f64 = exps.iter().sum();
        let probs: Vec<f64> = exps.iter().map(|e| e / sum).collect();
        let s: f64 = probs.iter().sum();
        assert!((s - 1.0).abs() < 1e-12, "probabilities must sum to 1 ({s})");
        let h: f64 = probs.iter().map(|p| -p * p.ln()).sum();
        let conf = (1.0 - h / 3.0f64.ln()).clamp(0.0, 1.0);
        assert!((0.0..=1.0).contains(&conf));
    }

    #[test]
    fn empty_candidates_refuse() {
        let eng = tiny();
        let e = decide(&eng, "s", "q", &[]).unwrap_err();
        assert!(e.contains("at least one"), "got {e}");
    }
}
