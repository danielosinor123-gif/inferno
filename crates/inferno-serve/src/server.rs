//! Engine service state: the Rust side of the ADR-001 boundary — routes,
//! per-model batchers over the ModelRegistry, OpenAI-shaped responses.
//!
//! Concurrency model (honest v1): one `Mutex<ServeState>` serializes
//! request handling while the gemv pool inside the engine parallelizes the
//! math. Connection fan-out is the Go plane's job (ADR-001). Per-model
//! batchers give each model its own KV cache — no cross-model leakage.

use std::collections::HashMap;
use std::path::PathBuf;

use inferno_core::batching::{Batcher, BatcherConfig, Priority, Submit};
use inferno_core::kv_cache::KvConfig;
use inferno_core::sched::NullAdmission;
use inferno_model::engine::QwenEngine;
use inferno_model::registry::{Loader, ModelRegistry};

use crate::http::{Request, Response};

const BLOCK_SIZE: usize = 16;
const MAX_NUM_SEQS: usize = 8;
const PREFILL_CHUNK: usize = 64;
const DEFAULT_MAX_TOKENS: usize = 16;

pub struct ServeState {
    reg: ModelRegistry,
    batchers: HashMap<String, Batcher>,
    blocks_per_model: usize,
    next_id: u64,
}

impl ServeState {
    pub fn new(budget_bytes: usize, blocks_per_model: usize) -> Self {
        Self {
            reg: ModelRegistry::new(budget_bytes),
            batchers: HashMap::new(),
            blocks_per_model,
            next_id: 1,
        }
    }

    /// Model names the service knows (registered or not).
    pub fn known_models(&self) -> Vec<String> {
        self.reg.known()
    }

    /// Register a model: `name` served from the directory `path`.
    pub fn register(&mut self, name: &str, path: PathBuf) {
        let reg_name = name.to_string();
        let loader: Loader = Box::new(move || {
            QwenEngine::load(&path)
                .map_err(|e| format!("loading {reg_name}: {e}"))
        });
        // KV config is built per model after the engine reveals its geometry
        // (see `batcher_for`), so register with a placeholder-free loader.
        self.reg.register(name, KvConfig::default(), loader);
    }

    /// Route one request. Every failure is a typed JSON response — the house
    /// rule at the boundary (docs/ENGINE-API.md gate 5).
    pub fn handle(&mut self, req: &Request) -> Response {
        match (req.method.as_str(), req.path.as_str()) {
            ("GET", "/healthz") => Response::json(
                200,
                &serde_json::json!({"status": "ok", "service": "inferno-serve"}),
            ),
            ("GET", "/v1/models") => self.list_models(),
            ("POST", p) if p.starts_with("/v1/models/") && p.ends_with("/load") => {
                let name = &p["/v1/models/".len()..p.len() - "/load".len()];
                self.load_model(name)
            }
            ("POST", "/v1/chat/completions") => self.chat_completions(req),
            _ => Response::json(
                404,
                &serde_json::json!({"error": {"type": "not_found",
                    "message": format!("no route {} {}", req.method, req.path)}}),
            ),
        }
    }

    fn list_models(&self) -> Response {
        let data: Vec<serde_json::Value> = self
            .reg
            .known()
            .into_iter()
            .map(|name| {
                let resident = self.reg.resident_names().contains(&name);
                serde_json::json!({
                    "id": name,
                    "object": "model",
                    "owned_by": "inferno",
                    "resident": resident,
                })
            })
            .collect();
        Response::json(
            200,
            &serde_json::json!({"object": "list", "data": data}),
        )
    }

    fn load_model(&mut self, name: &str) -> Response {
        if name.is_empty() {
            return bad_request("model name required");
        }
        match self.reg.begin_request(name) {
            Ok(()) => {
                // build the batcher now so the first chat request doesn't
                // pay the geometry plumbing
                if let Err(e) = self.ensure_batcher(name) {
                    self.reg.end_request(name);
                    return bad_request(&e.to_string());
                }
                self.reg.end_request(name);
                Response::json(
                    200,
                    &serde_json::json!({"id": name, "object": "model", "status": "loaded"}),
                )
            }
            Err(e) => match e {
                inferno_model::registry::RegistryError::UnknownModel(_) => Response::json(
                    404,
                    &serde_json::json!({"error": {"type": "model_not_found",
                        "message": e.to_string()}}),
                ),
                _ => Response::json(
                    503,
                    &serde_json::json!({"error": {"type": "load_failed",
                        "message": e.to_string()}}),
                ),
            },
        }
    }

    fn chat_completions(&mut self, req: &Request) -> Response {
        let body: serde_json::Value = match serde_json::from_slice(&req.body) {
            Ok(v) => v,
            Err(e) => return bad_request(&format!("body is not valid JSON: {e}")),
        };
        let Some(model) = body.get("model").and_then(|v| v.as_str()) else {
            return bad_request("field 'model' is required");
        };
        let max_tokens = body
            .get("max_tokens")
            .and_then(|v| v.as_u64())
            .map(|v| v as usize)
            .unwrap_or(DEFAULT_MAX_TOKENS);

        // Token source: `prompt_ids` (debug/hermetic clients) or text via the
        // engine's tokenizer. A text request to a tokenizer-less engine is a
        // typed 400, never a panic (the engine's encode_text requires one).
        let prompt_ids: Option<Vec<u32>> = body
            .get("prompt_ids")
            .and_then(|v| v.as_array())
            .map(|ids| ids.iter().filter_map(|v| v.as_u64()).map(|v| v as u32).collect());
        if prompt_ids.as_ref().map(|p| p.is_empty()).unwrap_or(false) {
            return bad_request("field 'prompt_ids' must be non-empty when present");
        }
        let prompt_text = if prompt_ids.is_some() {
            String::new()
        } else {
            match extract_prompt(&body) {
                Ok(p) => p,
                Err(msg) => return bad_request(&msg),
            }
        };

        // Pin the model for the whole request — eviction cannot touch it
        // (the registry's refcount rule, now at the HTTP layer).
        if let Err(e) = self.reg.begin_request(model) {
            return match e {
                inferno_model::registry::RegistryError::UnknownModel(_) => Response::json(
                    404,
                    &serde_json::json!({"error": {"type": "model_not_found",
                        "message": format!("unknown model {model:?}")}}),
                ),
                other => Response::json(
                    503,
                    &serde_json::json!({"error": {"type": "model_unavailable",
                        "message": other.to_string()}}),
                ),
            };
        }
        // From here on, every exit path must release the pin.
        let tokens = match prompt_ids {
            Some(ids) => Ok(ids),
            None => match self.reg.get_mut(model) {
                Ok(engine) if engine.has_tokenizer() => Ok(engine.encode_text(&prompt_text)),
                Ok(_) => Err(format!(
                    "model {model:?} has no tokenizer; send 'prompt_ids' (token ids) instead"
                )),
                Err(e) => Err(e.to_string()),
            },
        };
        let result = match tokens {
            Ok(toks) => self.generate(model, toks, max_tokens),
            Err(msg) => Err(msg),
        };
        self.reg.end_request(model);
        match result {
            Ok(gen) => {
                let content = gen.text.unwrap_or_default();
                let id = format!("cmpl-{}", self.next_id);
                self.next_id += 1;
                Response::json(
                    200,
                    &serde_json::json!({
                        "id": id,
                        "object": "chat.completion",
                        "model": model,
                        "choices": [{
                            "index": 0,
                            "message": {"role": "assistant", "content": content},
                            "finish_reason": if gen.hit_max {"length"} else {"stop"},
                        }],
                        "token_ids": gen.tokens,
                        "usage": {
                            "prompt_tokens": gen.prompt_tokens,
                            "completion_tokens": gen.tokens.len(),
                            "total_tokens": gen.prompt_tokens + gen.tokens.len(),
                        },
                    }),
                )
            }
            Err(msg) => bad_request(&msg),
        }
    }

    /// Run one request to completion through the model's batcher.
    /// `prompt_tokens` is the tokenized prompt (the HTTP layer tokenized it
    /// via the engine, or took `prompt_ids` directly).
    fn generate(
        &mut self,
        model: &str,
        prompt_tokens: Vec<u32>,
        max_tokens: usize,
    ) -> Result<Generation, String> {
        // Batcher first (it reads the engine's geometry), then the engine
        // borrow for the drain — the two borrows must not overlap.
        self.ensure_batcher(model).map_err(|e| e.to_string())?;
        let batcher = self.batchers.get_mut(model).expect("just ensured");
        let id = batcher.submit(Submit {
            prompt: prompt_tokens.clone(),
            max_gen: max_tokens,
            priority: Priority::Normal,
            ttft_slo_ms: 60_000.0,
        });
        // Drain this model's queue (NullAdmission: the core service doesn't
        // consult the kernel scheduler; the plane/fleet layer does that).
        let engine = self.reg.get_mut(model).map_err(|e| e.to_string())?;
        batcher.drain(engine, &NullAdmission);
        let Some((_, toks)) = batcher
            .finished_tokens
            .iter()
            .find(|(i, _)| *i == id)
            .cloned()
        else {
            return Err("request did not finish".into());
        };
        let text = if engine.has_tokenizer() {
            Some(engine.decode_text(&toks))
        } else {
            None
        };
        Ok(Generation {
            tokens: toks.clone(),
            text,
            hit_max: toks.len() >= max_tokens && !toks.is_empty(),
            prompt_tokens: prompt_tokens.len(),
        })
    }

    fn ensure_batcher(&mut self, model: &str) -> Result<(), String> {
        if self.batchers.contains_key(model) {
            return Ok(());
        }
        let (layers, kv_heads, head_dim) = {
            let e = self.reg.get(model).map_err(|e| e.to_string())?;
            (e.cfg.num_layers, e.cfg.num_kv_heads, e.cfg.head_dim)
        };
        let b = Batcher::new(
            KvConfig {
                block_size: BLOCK_SIZE,
                num_blocks: self.blocks_per_model,
                num_layers: layers,
                num_kv_heads: kv_heads,
                head_dim,
            },
            BatcherConfig {
                prefill_chunk_tokens: PREFILL_CHUNK,
                max_num_seqs: MAX_NUM_SEQS,
            },
        );
        self.batchers.insert(model.to_string(), b);
        Ok(())
    }
}

struct Generation {
    tokens: Vec<u32>,
    text: Option<String>,
    hit_max: bool,
    prompt_tokens: usize,
}

/// OpenAI 'messages' (concatenated contents) or a plain 'prompt'.
fn extract_prompt(body: &serde_json::Value) -> Result<String, String> {
    if let Some(p) = body.get("prompt").and_then(|v| v.as_str()) {
        if !p.is_empty() {
            return Ok(p.to_string());
        }
    }
    let Some(msgs) = body.get("messages").and_then(|v| v.as_array()) else {
        return Err("one of 'messages' or 'prompt' is required".into());
    };
    let mut parts = Vec::new();
    for m in msgs {
        if let Some(c) = m.get("content").and_then(|v| v.as_str()) {
            let role = m.get("role").and_then(|v| v.as_str()).unwrap_or("user");
            parts.push(format!("{role}: {c}"));
        }
    }
    if parts.is_empty() {
        return Err("'messages' carries no content".into());
    }
    Ok(parts.join("\n"))
}

fn bad_request(msg: &str) -> Response {
    Response::json(
        400,
        &serde_json::json!({"error": {"type": "invalid_request", "message": msg}}),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use inferno_model::engine::demo::{tiny_cfg, tiny_weights};
    use inferno_model::registry::Loader;

    fn tiny_state() -> ServeState {
        let mut s = ServeState::new(64 * 1024 * 1024, 8);
        let loader: Loader =
            Box::new(|| Ok(QwenEngine::new(tiny_cfg(), tiny_weights())));
        s.reg.register("tiny", KvConfig::default(), loader);
        s
    }

    #[test]
    fn healthz() {
        let mut s = tiny_state();
        let r = s.handle(&Request {
            method: "GET".into(),
            path: "/healthz".into(),
            query: vec![],
            headers: vec![],
            body: vec![],
        });
        assert_eq!(r.status, 200);
    }

    #[test]
    fn models_list_and_load() {
        let mut s = tiny_state();
        let r = s.handle(&Request {
            method: "GET".into(),
            path: "/v1/models".into(),
            query: vec![],
            headers: vec![],
            body: vec![],
        });
        assert_eq!(r.status, 200);
        let v: serde_json::Value = serde_json::from_slice(&r.body).unwrap();
        assert_eq!(v["data"][0]["id"], "tiny");
        let r = s.handle(&Request {
            method: "POST".into(),
            path: "/v1/models/tiny/load".into(),
            query: vec![],
            headers: vec![],
            body: vec![],
        });
        assert_eq!(r.status, 200);
        assert!(s.batchers.contains_key("tiny"), "load builds the batcher");
    }

    #[test]
    fn chat_completion_small_engine() {
        let mut s = tiny_state();
        // Hermetic engine has no tokenizer: the prompt arrives as token ids
        // (the documented escape hatch), proving the full routing path.
        let body = serde_json::json!({"model": "tiny", "prompt_ids": [3, 11, 5], "max_tokens": 2});
        let r = s.handle(&Request {
            method: "POST".into(),
            path: "/v1/chat/completions".into(),
            query: vec![],
            headers: vec![],
            body: serde_json::to_vec(&body).unwrap(),
        });
        assert_eq!(r.status, 200);
        let v: serde_json::Value = serde_json::from_slice(&r.body).unwrap();
        assert_eq!(v["object"], "chat.completion");
        assert_eq!(v["model"], "tiny");
        // greedy argmax over the tiny vocab: 2 generated tokens (or eos early)
        let toks = v["token_ids"].as_array().unwrap();
        assert!(toks.len() <= 2, "max_tokens respected");
        assert!(v["usage"]["prompt_tokens"].as_u64().unwrap() >= 1);
        // no tokenizer in the hermetic tiny engine: content is empty, ids present
        assert_eq!(v["choices"][0]["message"]["content"], "");
    }

    #[test]
    fn unknown_model_is_typed_404() {
        let mut s = tiny_state();
        let body = serde_json::json!({"model": "nope", "prompt": "hi"});
        let r = s.handle(&Request {
            method: "POST".into(),
            path: "/v1/chat/completions".into(),
            query: vec![],
            headers: vec![],
            body: serde_json::to_vec(&body).unwrap(),
        });
        assert_eq!(r.status, 404);
        let v: serde_json::Value = serde_json::from_slice(&r.body).unwrap();
        assert_eq!(v["error"]["type"], "model_not_found");
    }

    #[test]
    fn malformed_body_is_400() {
        let mut s = tiny_state();
        let r = s.handle(&Request {
            method: "POST".into(),
            path: "/v1/chat/completions".into(),
            query: vec![],
            headers: vec![],
            body: b"{".to_vec(),
        });
        assert_eq!(r.status, 400);
        let v: serde_json::Value = serde_json::from_slice(&r.body).unwrap();
        assert_eq!(v["error"]["type"], "invalid_request");
    }
}
