//! Model registry — the lifecycle layer of the company computer
//! (docs/VISION.md phase 2, docs/LIFECYCLE.md).
//!
//! A company computer serves MANY models on ONE machine: a chat model, an
//! embedding model, a small routing model. That needs load-on-demand with a
//! memory budget, least-recently-used eviction, and a hard rule: **a model in
//! active use is never evicted** (the registry refuses loudly instead — the
//! house rule: never silently corrupt a running request).
//!
//! Loaders are factories (`Fn() -> Result<QwenEngine>`), so the registry knows
//! nothing about paths/files — the admin plane above it plugs in whatever
//! provenance it likes (local dirs, model hub, cache)...

use std::collections::HashMap;
use std::time::Instant;

use inferno_core::kv_cache::{KvCache, KvConfig};

use crate::engine::QwenEngine;

/// Builds an engine on demand. Load-on-miss + reload-after-evict.
pub type Loader = Box<dyn Fn() -> Result<QwenEngine, String> + Send + Sync>;

struct Resident {
    engine: QwenEngine,
    bytes: usize,
    /// Active requests pinning this model — nonzero blocks eviction.
    refs: usize,
    last_used: Instant,
}

#[derive(Debug)]
pub enum RegistryError {
    UnknownModel(String),
    LoadFailed(String),
    /// Evicting this model would break the budget, but every evictable
    /// (referenced == 0) model is pinned; or this model alone exceeds the
    /// whole budget. Loud refusal — never evict an in-use model.
    BudgetExhausted {
        needed: usize,
        budget: usize,
        pinned: Vec<String>,
    },
    NotResident(String),
}

impl std::fmt::Display for RegistryError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RegistryError::UnknownModel(n) => write!(f, "unknown model {n:?}"),
            RegistryError::LoadFailed(e) => write!(f, "model load failed: {e}"),
            RegistryError::BudgetExhausted { needed, budget, pinned } => write!(
                f,
                "budget {budget} B cannot fit {needed} B; pinned (in use): {pinned:?}"
            ),
            RegistryError::NotResident(n) => write!(f, "model {n:?} is not resident"),
        }
    }
}

pub struct ModelRegistry {
    loaders: HashMap<String, Loader>,
    kv_cfgs: HashMap<String, KvConfig>,
    resident: HashMap<String, Resident>,
    budget_bytes: usize,
    /// Diagnostics: load events per name (evictions are visible here — a
    /// reload after evict increments the count).
    load_events: HashMap<String, usize>,
    evictions: usize,
}

impl ModelRegistry {
    pub fn new(budget_bytes: usize) -> Self {
        Self {
            loaders: HashMap::new(),
            kv_cfgs: HashMap::new(),
            resident: HashMap::new(),
            budget_bytes,
            load_events: HashMap::new(),
            evictions: 0,
        }
    }

    /// Register a model factory under `name` with the KV pool it wants when
    /// resident. Registration does not load anything.
    pub fn register(&mut self, name: &str, kv: KvConfig, loader: Loader) {
        self.loaders.insert(name.to_string(), loader);
        self.kv_cfgs.insert(name.to_string(), kv);
    }

    /// Model names the registry knows (loaded or not).
    pub fn known(&self) -> Vec<String> {
        let mut v: Vec<String> = self.loaders.keys().cloned().collect();
        v.sort();
        v
    }

    /// Names currently resident in memory.
    pub fn resident_names(&self) -> Vec<String> {
        let mut v: Vec<String> = self.resident.keys().cloned().collect();
        v.sort();
        v
    }

    pub fn resident_bytes(&self) -> usize {
        self.resident.values().map(|r| r.bytes).sum()
    }

    pub fn budget(&self) -> usize {
        self.budget_bytes
    }

    pub fn evictions(&self) -> usize {
        self.evictions
    }

    /// Load events for `name` (eviction + reload shows up as > 1).
    pub fn load_events(&self, name: &str) -> usize {
        *self.load_events.get(name).unwrap_or(&0)
    }

    /// Begin serving a request with `name`: load (evicting LRU if the budget
    /// demands) and pin the model so it cannot be evicted mid-request.
    pub fn begin_request(&mut self, name: &str) -> Result<(), RegistryError> {
        self.ensure_resident(name)?;
        let r = self
            .resident
            .get_mut(name)
            .ok_or_else(|| RegistryError::UnknownModel(name.to_string()))?;
        r.refs += 1;
        r.last_used = Instant::now();
        Ok(())
    }

    /// Release a request pin.
    pub fn end_request(&mut self, name: &str) {
        if let Some(r) = self.resident.get_mut(name) {
            r.refs = r.refs.saturating_sub(1);
        }
    }

    /// Borrow a resident engine (read-only diagnostics). Touches LRU.
    pub fn get(&mut self, name: &str) -> Result<&QwenEngine, RegistryError> {
        self.ensure_resident(name)?;
        self.resident
            .get(name)
            .map(|r| &r.engine)
            .ok_or_else(|| RegistryError::NotResident(name.to_string()))
    }

    /// Borrow a resident engine mutably. Touches LRU. Eviction protection is
    /// the caller's job via begin/end_request (a mut borrow cannot be held
    /// across an eviction call — Rust enforces the invariant the doc demands).
    pub fn get_mut(&mut self, name: &str) -> Result<&mut QwenEngine, RegistryError> {
        self.ensure_resident(name)?;
        self.resident
            .get_mut(name)
            .map(|r| &mut r.engine)
            .ok_or_else(|| RegistryError::NotResident(name.to_string()))
    }

    /// Drop a resident model's memory. Refuses while pinned — the load
    /// events counter and RSS tell the truth about what happened.
    pub fn evict(&mut self, name: &str) -> Result<(), RegistryError> {
        let r = self
            .resident
            .get(name)
            .ok_or_else(|| RegistryError::NotResident(name.to_string()))?;
        if r.refs > 0 {
            return Err(RegistryError::BudgetExhausted {
                needed: 0,
                budget: 0,
                pinned: vec![name.to_string()],
            });
        }
        self.resident.remove(name);
        Ok(())
    }

    /// Evict least-recently-used unpinned models until the budget can fit
    /// `needed` more bytes. Returns the names evicted.
    fn make_room(&mut self, needed: usize) -> Result<Vec<String>, RegistryError> {
        let mut evicted = Vec::new();
        loop {
            if self.resident_bytes() + needed <= self.budget_bytes {
                return Ok(evicted);
            }
            // LRU among unpinned
            let victim = self
                .resident
                .iter()
                .filter(|(_, r)| r.refs == 0)
                .min_by_key(|(_, r)| r.last_used)
                .map(|(n, _)| n.clone());
            let Some(victim) = victim else {
                // Everything resident is pinned — refuse loudly, name them.
                let pinned: Vec<String> = self
                    .resident
                    .iter()
                    .filter(|(_, r)| r.refs > 0)
                    .map(|(n, _)| n.clone())
                    .collect();
                return Err(RegistryError::BudgetExhausted {
                    needed,
                    budget: self.budget_bytes,
                    pinned,
                });
            };
            self.resident.remove(&victim);
            self.evictions += 1;
            evicted.push(victim);
        }
    }

    /// Load `name` if not resident, evicting LRU as the budget requires.
    /// If the model alone (weights + KV pool) exceeds the budget, the freshly
    /// loaded engine is dropped and the original error surfaces.
    fn ensure_resident(&mut self, name: &str) -> Result<(), RegistryError> {
        if self.resident.contains_key(name) {
            let r = self.resident.get_mut(name).unwrap();
            r.last_used = Instant::now();
            return Ok(());
        }
        let loader = self
            .loaders
            .get(name)
            .ok_or_else(|| RegistryError::UnknownModel(name.to_string()))?
            .as_ref();
        let kv = self
            .kv_cfgs
            .get(name)
            .cloned()
            .ok_or_else(|| RegistryError::UnknownModel(name.to_string()))?;
        let engine = loader().map_err(RegistryError::LoadFailed)?;
        let bytes = engine.weight_bytes() + KvCache::pool_bytes(&kv);
        if bytes > self.budget_bytes {
            return Err(RegistryError::BudgetExhausted {
                needed: bytes,
                budget: self.budget_bytes,
                pinned: Vec::new(),
            });
        }
        // Make room before inserting (a no-op when the budget already fits).
        self.make_room(bytes)?;
        self.resident.insert(
            name.to_string(),
            Resident {
                engine,
                bytes,
                refs: 0,
                last_used: Instant::now(),
            },
        );
        *self.load_events.entry(name.to_string()).or_insert(0) += 1;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::{demo, QwenEngine};
    use inferno_core::kv_cache::{KvCache, KvConfig};

    fn kv() -> KvConfig {
        KvConfig {
            block_size: 4,
            num_blocks: 8,
            num_layers: 2,
            num_kv_heads: 1,
            head_dim: 4,
        }
    }

    fn tiny_loader() -> Loader {
        Box::new(|| Ok(QwenEngine::new(demo::tiny_cfg(), demo::tiny_weights())))
    }

    fn reg_with(names: &[&str], budget: usize) -> ModelRegistry {
        let mut r = ModelRegistry::new(budget);
        for n in names {
            r.register(n, kv(), tiny_loader());
        }
        r
    }

    fn one_model_bytes() -> usize {
        let eng = QwenEngine::new(demo::tiny_cfg(), demo::tiny_weights());
        eng.weight_bytes() + KvCache::pool_bytes(&kv())
    }

    #[test]
    fn lru_eviction_order() {
        let b = one_model_bytes();
        let mut reg = reg_with(&["a", "b", "c"], 2 * b + 16);
        reg.get("a").unwrap();
        reg.get("b").unwrap();
        // a is LRU now; loading c must evict a (budget fits 2)
        reg.get("c").unwrap();
        assert_eq!(reg.resident_names(), vec!["b".to_string(), "c".to_string()]);
        assert_eq!(reg.evictions(), 1);
        // touching b refreshes LRU: the next load evicts c, not b
        reg.get("b").unwrap();
        reg.get("a").unwrap();
        assert_eq!(reg.resident_names(), vec!["a".to_string(), "b".to_string()]);
        assert_eq!(reg.evictions(), 2);
    }

    #[test]
    fn pinned_models_are_never_evicted() {
        let b = one_model_bytes();
        let mut reg = reg_with(&["a", "b", "c"], 2 * b + 16);
        reg.begin_request("a").unwrap();
        reg.begin_request("b").unwrap();
        // Budget full of pinned models: loading c must REFUSE loudly, naming
        // what it refuses to evict — never evict an in-use model.
        let e = match reg.get("c") {
            Err(e) => e,
            Ok(_) => panic!("loading c must refuse while a+b are pinned"),
        };
        match e {
            RegistryError::BudgetExhausted { pinned, .. } => {
                assert!(pinned.contains(&"a".to_string()));
                assert!(pinned.contains(&"b".to_string()));
            }
            other => panic!("expected BudgetExhausted, got {other:?}"),
        }
        assert_eq!(reg.resident_bytes(), 2 * b, "nothing was evicted");
        // releasing one pin unblocks: a (now LRU unpinned) is the victim
        reg.end_request("a");
        reg.get("c").unwrap();
        assert!(!reg.resident_names().contains(&"a".to_string()));
        assert!(reg.resident_names().contains(&"c".to_string()));
        assert!(reg.resident_names().contains(&"b".to_string()));
    }

    #[test]
    fn evict_reload_is_bit_identical() {
        let b = one_model_bytes();
        let mut reg = reg_with(&["a", "b"], b + 16);
        let want = {
            let eng = reg.get("a").unwrap();
            eng.logits_digest_pub(3, 0)
        };
        reg.evict("a").unwrap();
        assert!(reg.resident_names().is_empty());
        let eng = reg.get("a").unwrap();
        assert_eq!(
            eng.logits_digest_pub(3, 0),
            want,
            "reload after eviction must be bit-identical"
        );
        assert_eq!(reg.load_events("a"), 2, "evict+reload must be visible");
        drop(reg);
    }

    #[test]
    fn accounting_matches_storage_dtype() {
        let eng = QwenEngine::new(demo::tiny_cfg(), demo::tiny_weights());
        let wb = eng.weight_bytes();
        // tiny model math: embed 32*8 bf16 = 512 B; per layer: matrices
        // (q 64 + k 32 + v 32 + o 64 + gate 48 + up 48 + down 48 = 336 elems
        // * 2 B = 672 B) + f32 vectors (qb 8 + kb 4 + vb 4 + norm 8 + post 8
        // = 32 elems * 4 B = 128 B) = 800 B; 2 layers = 1600 B; final norm
        // 8 * 4 = 32 B; total 512 + 1600 + 32 = 2144 B.
        assert!(wb > 2000 && wb < 2300, "weight bytes {wb} outside expected band");
    }
}
