//! MoE routing + expert streaming (docs/MOE-STREAMING.md).
//!
//! A mixture-of-experts layer activates a few experts per token, so experts
//! can live in a slow tier and be fetched on demand. THE INVARIANT:
//! placement decides speed, never output — the streamed result is
//! bit-identical to the dense reference on the same weights (gates 1-4 of
//! the pre-registration).
//!
//! This is the hermetic engine: the slow tier is an in-memory `Vec<Expert>`
//! standing in for disk (mmap/pread is the real-model step), and weights
//! are f32 so bit-identity is achievable.

use std::collections::{HashMap, HashSet, VecDeque};

/// Geometry of one MoE layer.
#[derive(Clone, Copy, Debug)]
pub struct MoeCfg {
    pub hidden: usize,
    pub n_experts: usize,
    pub top_k: usize,
}

/// One expert = a SwiGLU MLP (native HF [out, in] row-major layout).
#[derive(Clone, Debug)]
pub struct Expert {
    pub gate: Vec<f32>, // [inter, hidden]
    pub up: Vec<f32>,   // [inter, hidden]
    pub down: Vec<f32>, // [hidden, inter]
    pub inter: usize,
}

impl Expert {
    pub fn nbytes(&self) -> usize {
        4 * (self.gate.len() + self.up.len() + self.down.len())
    }

    /// down(silu(gate(x)) * up(x)) — the same formula as the dense engine.
    pub fn forward(&self, x: &[f32], out: &mut [f32]) {
        debug_assert_eq!(x.len(), self.gate.len() / self.inter);
        let h = x.len();
        let mut gate = vec![0.0f32; self.inter];
        let mut up = vec![0.0f32; self.inter];
        gemv_rows(&self.gate, x, self.inter, h, &mut gate);
        gemv_rows(&self.up, x, self.inter, h, &mut up);
        let mut act = vec![0.0f32; self.inter];
        for i in 0..self.inter {
            let g = gate[i] / (1.0 + (-gate[i]).exp());
            act[i] = g * up[i];
        }
        gemv_rows(&self.down, &act, h, self.inter, out);
    }
}

/// Row-major gemv: `out[j] = dot(w[j*k..], x)`.
fn gemv_rows(w: &[f32], x: &[f32], rows: usize, k: usize, out: &mut [f32]) {
    for j in 0..rows {
        let row = &w[j * k..j * k + k];
        let mut s = 0.0f32;
        for i in 0..k {
            s += row[i] * x[i];
        }
        out[j] = s;
    }
}

/// Routing: `logits = router_w @ x` ([n_experts, hidden]), top-k by logit,
/// softmax over the selected logits. Returns (expert_id, weight) sorted by
/// expert id — the order BOTH the dense and streamed paths accumulate in,
/// which is what makes their outputs bit-identical.
pub fn route(
    router_w: &[f32],
    x: &[f32],
    cfg: &MoeCfg,
) -> Vec<(usize, f32)> {
    let h = cfg.hidden;
    let mut logits = vec![0.0f32; cfg.n_experts];
    gemv_rows(router_w, x, cfg.n_experts, h, &mut logits);
    let mut idx: Vec<usize> = (0..cfg.n_experts).collect();
    idx.sort_by(|&a, &b| logits[b].partial_cmp(&logits[a]).unwrap());
    idx.truncate(cfg.top_k);
    idx.sort(); // accumulate in expert-id order (both paths)
    let m = idx.iter().map(|&i| logits[i]).fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = idx.iter().map(|&i| (logits[i] - m).exp()).collect();
    let sum: f32 = exps.iter().sum();
    idx.iter().zip(&exps).map(|(&i, e)| (i, e / sum)).collect()
}

/// The slow tier: all experts, "on disk".
pub struct ExpertStore {
    pub layers: Vec<Vec<Expert>>,
}

impl ExpertStore {
    pub fn expert_bytes(&self) -> usize {
        self.layers.iter().flatten().map(|e| e.nbytes()).sum()
    }
}

/// Per-(layer, expert) key.
type Key = (usize, usize);

/// The resident tier: per-layer LRU with session-learned pins and honest
/// accounting. Eviction takes the least-recently-used UNPINNED entry.
pub struct ExpertCache {
    cap: usize,
    lru: VecDeque<Key>,
    resident: HashMap<Key, Expert>,
    pins: HashSet<Key>,
    /// routing observations per key (the learning signal)
    obs: HashMap<Key, u32>,
    pin_threshold: u32,
    // accounting
    hits: usize,
    misses: usize,
    fetched_bytes: usize,
}

impl ExpertCache {
    pub fn new(cap: usize) -> Self {
        Self {
            cap,
            lru: VecDeque::new(),
            resident: HashMap::new(),
            pins: HashSet::new(),
            obs: HashMap::new(),
            pin_threshold: u32::MAX, // off until a threshold is set
            hits: 0,
            misses: 0,
            fetched_bytes: 0,
        }
    }

    /// Learn pins from routing: any expert observed >= threshold times is
    /// pinned (survives eviction). Set threshold to u32::MAX to disable.
    pub fn set_pin_threshold(&mut self, n: u32) {
        self.pin_threshold = n;
    }

    pub fn observe(&mut self, layer: usize, expert: usize) {
        let c = self.obs.entry((layer, expert)).or_insert(0);
        *c += 1;
        if self.pin_threshold != u32::MAX && *c >= self.pin_threshold {
            self.pins.insert((layer, expert));
        }
    }

    pub fn hits(&self) -> usize {
        self.hits
    }

    pub fn misses(&self) -> usize {
        self.misses
    }

    pub fn fetched_bytes(&self) -> usize {
        self.fetched_bytes
    }

    pub fn resident_bytes(&self) -> usize {
        self.resident.values().map(|e| e.nbytes()).sum()
    }

    pub fn resident_len(&self) -> usize {
        self.resident.len()
    }

    /// Fetch (or serve from cache) one expert, touching the LRU order.
    pub fn get(&mut self, layer: usize, expert: usize, store: &ExpertStore) -> &Expert {
        let key = (layer, expert);
        if self.resident.contains_key(&key) {
            self.hits += 1;
            self.touch(key);
        } else {
            self.misses += 1;
            while self.resident.len() >= self.cap {
                // evict LRU among UNPINNED
                let victim = self.lru.iter().find(|k| !self.pins.contains(k)).copied();
                match victim {
                    Some(v) => {
                        self.lru.retain(|k| *k != v);
                        self.resident.remove(&v);
                    }
                    None => break, // everything pinned: exceed cap rather than break the invariant
                }
            }
            let e = store.layers[layer][expert].clone();
            self.fetched_bytes += e.nbytes();
            self.lru.push_back(key);
            self.resident.insert(key, e);
        }
        // one borrow, after all mutation
        self.resident.get(&key).expect("resident after get")
    }

    fn touch(&mut self, key: Key) {
        if let Some(pos) = self.lru.iter().position(|k| *k == key) {
            self.lru.remove(pos);
        }
        self.lru.push_back(key);
    }
}

/// The DENSE reference: no cache, every routed expert read from the store.
/// This is what the streaming path must match bit-for-bit.
pub fn moe_layer_dense(
    cfg: &MoeCfg,
    layer: usize,
    router_w: &[f32],
    x: &[f32],
    store: &ExpertStore,
    out: &mut [f32],
) -> Vec<(usize, f32)> {
    let sel = route(router_w, x, cfg);
    let mut acc = vec![0.0f32; cfg.hidden];
    let mut tmp = vec![0.0f32; cfg.hidden];
    for &(e, w) in &sel {
        store.layers[layer][e].forward(x, &mut tmp);
        for i in 0..cfg.hidden {
            acc[i] += w * tmp[i];
        }
    }
    out.copy_from_slice(&acc);
    sel
}

/// The STREAMED engine: same routing, experts through the cache.
pub fn moe_layer_streamed(
    cfg: &MoeCfg,
    layer: usize,
    router_w: &[f32],
    x: &[f32],
    store: &ExpertStore,
    cache: &mut ExpertCache,
    out: &mut [f32],
) -> Vec<(usize, f32)> {
    let sel = route(router_w, x, cfg);
    let mut acc = vec![0.0f32; cfg.hidden];
    let mut tmp = vec![0.0f32; cfg.hidden];
    for &(e, w) in &sel {
        for (l, ex) in [(layer, e)] {
            cache.observe(l, ex);
        }
        let expert = cache.get(layer, e, store);
        expert.forward(x, &mut tmp);
        for i in 0..cfg.hidden {
            acc[i] += w * tmp[i];
        }
    }
    out.copy_from_slice(&acc);
    sel
}

#[cfg(test)]
mod tests {
    use super::*;

    fn prng(state: &mut u64) -> f32 {
        let mut x = *state | 1;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        *state = x;
        ((x >> 40) as f32 / (1u32 << 24) as f32) - 0.5
    }

    fn build(layers: usize, n_experts: usize, hidden: usize, inter: usize) -> (MoeCfg, ExpertStore, Vec<Vec<f32>>) {
        let cfg = MoeCfg { hidden, n_experts, top_k: 2 };
        let mut st = 0x5eed_1234u64;
        let mut store_layers = Vec::new();
        for _ in 0..layers {
            let mut experts = Vec::with_capacity(n_experts);
            for _ in 0..n_experts {
                experts.push(Expert {
                    gate: (0..inter * hidden).map(|_| prng(&mut st) * 0.3).collect(),
                    up: (0..inter * hidden).map(|_| prng(&mut st) * 0.3).collect(),
                    down: (0..inter * hidden).map(|_| prng(&mut st) * 0.3).collect(),
                    inter,
                });
            }
            store_layers.push(experts);
        }
        // per-layer router weights
        let routers: Vec<Vec<f32>> = (0..layers)
            .map(|l| (0..n_experts * hidden).map(|_i| prng(&mut st) * 0.2 + l as f32 * 0.01).collect())
            .collect();
        (cfg, ExpertStore { layers: store_layers }, routers)
    }

    fn xvec(hidden: usize, seed: u64) -> Vec<f32> {
        let mut st = seed;
        (0..hidden).map(|_| prng(&mut st)).collect()
    }

    /// zero router: uniform logits -> top-k is deterministic by sort order
    /// (stable sort picks the lowest expert ids) — a fixed routing pattern
    /// for the cache tests.
    fn zero_router(hidden: usize, n_experts: usize) -> Vec<f32> {
        vec![0.0f32; n_experts * hidden]
    }

    #[test]
    fn gate1_placement_never_changes_output() {
        let (cfg, store, routers) = build(2, 8, 8, 16);
        let mut cache = ExpertCache::new(64);
        for (l, rw) in routers.iter().enumerate() {
            for seed in 0..6u64 {
                let x = xvec(cfg.hidden, 100 + seed * 7 + l as u64);
                let mut dense = vec![0.0f32; cfg.hidden];
                let mut streamed = vec![0.0f32; cfg.hidden];
                let sel_d = moe_layer_dense(&cfg, l, rw, &x, &store, &mut dense);
                let sel_s = moe_layer_streamed(&cfg, l, rw, &x, &store, &mut cache, &mut streamed);
                assert_eq!(sel_d, sel_s, "routing must not depend on placement");
                for i in 0..cfg.hidden {
                    assert_eq!(
                        dense[i].to_bits(),
                        streamed[i].to_bits(),
                        "BIT-IDENTITY broken at layer {l} seed {seed} dim {i}"
                    );
                }
            }
        }
    }

    #[test]
    fn gate2_lru_matches_reference_order() {
        let (cfg, store, _) = build(1, 6, 8, 16);
        let mut cache = ExpertCache::new(2);
        // drive a known access sequence directly through the cache and
        // check the resident set against a reference LRU
        let seq = [3usize, 1, 3, 0, 1];
        let mut ref_lru: VecDeque<usize> = VecDeque::new();
        for &e in &seq {
            let _ = cache.get(0, e, &store); // fetch (or hit) expert e
            if let Some(p) = ref_lru.iter().position(|k| *k == e) {
                ref_lru.remove(p);
            }
            ref_lru.push_back(e);
            while ref_lru.len() > 2 {
                ref_lru.pop_front();
            }
        }
        let resident: HashSet<usize> = (0..cfg.n_experts)
            .filter(|e| cache.resident.contains_key(&(0, *e)))
            .collect();
        let ref_set: HashSet<usize> = ref_lru.iter().copied().collect();
        assert_eq!(resident, ref_set, "resident set must match reference LRU");
        assert!(cache.hits() > 0, "repeats must hit");
        assert!(cache.misses() >= 2, "first touches must miss");
    }

    #[test]
    fn gate3_pins_survive_eviction() {
        let (cfg, store, _) = build(1, 8, 8, 16);
        let mut cache = ExpertCache::new(2);
        // pin expert 1 by observation count, then crowd the cache
        cache.set_pin_threshold(2);
        cache.observe(0, 1);
        cache.observe(0, 1);
        assert!(cache.pins.contains(&(0, 1)));
        let x = xvec(cfg.hidden, 2);
        let rw = zero_router(cfg.hidden, cfg.n_experts);
        // touch expert 1 (pinned), then 3, 4, 5 — the unpinned must evict
        // each other, never the pin
        for &e in &[1usize, 3, 4, 5, 1] {
            let _ = cache.get(0, e, &store);
        }
        assert!(
            cache.resident.contains_key(&(0, 1)),
            "pinned expert must survive eviction pressure"
        );
        assert!(cache.resident.len() <= 3, "cap 2 + the unpinned victim space");
        let _ = moe_layer_streamed(&cfg, 0, &rw, &x, &store, &mut cache, &mut vec![0.0; cfg.hidden]);
    }

    #[test]
    fn gate4_accounting_is_honest() {
        let (cfg, store, _) = build(1, 4, 8, 16);
        let mut cache = ExpertCache::new(4);
        let one = store.layers[0][0].nbytes();
        assert_eq!(cache.fetched_bytes(), 0, "no fetch before any get");
        let _ = cache.get(0, 0, &store);
        let after_first = cache.fetched_bytes();
        assert_eq!(after_first, one, "one miss fetches exactly one expert");
        let _ = cache.get(0, 0, &store); // hit
        assert_eq!(cache.fetched_bytes(), after_first, "hits must not refetch");
        assert_eq!(cache.resident_bytes(), one, "resident bytes = the two cached experts");
        assert_eq!(cache.hits(), 1);
        assert_eq!(cache.misses(), 1);
    }
}
