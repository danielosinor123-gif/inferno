//! Paged KV cache with prefix sharing — reference implementation of the
//! vLLM design (Kwon et al., "Efficient Memory Management for Large Language
//! Model Serving with PagedAttention", arXiv:2309.06180).
//!
//! Semantics implemented here, mapped to the paper:
//! * Fixed-size blocks (default 16 tokens) drawn from a preallocated pool
//!   (§4.2 KV Cache Manager). No fragmentation: every block is fully usable.
//! * Per-sequence block tables (§4.2) — a logical sequence maps to an
//!   arbitrary set of physical blocks.
//! * Copy-on-write on shared blocks (§4.2): a block with refcount > 1 is
//!   never written in place; the writer gets a private copy.
//! * Retained-KV prefix caching (RadixAttention semantics, arXiv:2312.07104):
//!   completed blocks are registered under a
//!   content+position hash (hash covers tokens[0..=(k+1)*block_size), so it
//!   is position-dependent — RoPE makes absolute position matter, so sharing
//!   a block whose prefix differs would be wrong).
//!
//! CPU reference storage is `Vec<f32>`; the GPU port keeps the exact same
//! block/table/refcount logic and swaps storage for device pages.

use std::collections::HashMap;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Which {
    K = 0,
    V = 1,
}

#[derive(Clone, Copy, Debug)]
pub struct KvConfig {
    /// Tokens per physical block (vLLM default 16).
    pub block_size: usize,
    pub num_blocks: usize,
    pub num_layers: usize,
    /// KV heads (GQA/MQA aware — decode may use fewer heads than Q).
    pub num_kv_heads: usize,
    pub head_dim: usize,
}

impl Default for KvConfig {
    fn default() -> Self {
        // Small reference geometry so unit tests + scheduler benchmarks run
        // anywhere; real deployments override via config.
        Self {
            block_size: 16,
            num_blocks: 1024,
            num_layers: 4,
            num_kv_heads: 2,
            head_dim: 32,
        }
    }
}

#[derive(Debug)]
pub struct ReserveInfo {
    /// Total blocks held by the sequence after the call.
    pub total: usize,
    /// Blocks satisfied from the prefix cache (shared, refcount += 1).
    pub reused: usize,
}

#[derive(Debug)]
pub enum CacheError {
    /// Pool exhausted — caller must preempt (release + requeue), the vLLM
    /// response to allocation failure at decode time.
    Full,
}

#[derive(Default)]
struct SeqTable {
    blocks: Vec<u32>,
}

pub struct KvCache {
    cfg: KvConfig,
    data: Vec<f32>,
    free: Vec<u32>,
    tables: HashMap<u64, SeqTable>,
    /// hash -> canonical block
    prefix: HashMap<u64, u32>,
    /// block -> hash it canonically represents (for cleanup on free)
    canonical_of: HashMap<u32, u64>,
    /// block -> refcount (>= 1 while allocated)
    refs: HashMap<u32, u32>,
    /// Cap on prefix map size (do not insert new canonical entries past this)
    max_prefix_entries: usize,
}

impl Default for KvCache {
    fn default() -> Self {
        Self::new(KvConfig::default())
    }
}

impl KvCache {
    pub fn new(cfg: KvConfig) -> Self {
        assert!(cfg.block_size >= 1 && cfg.num_blocks >= 1);
        let stride = Self::stride_of(&cfg);
        Self {
            data: vec![0.0; cfg.num_blocks * stride],
            free: (0..cfg.num_blocks as u32).rev().collect(),
            cfg,
            tables: HashMap::new(),
            prefix: HashMap::new(),
            canonical_of: HashMap::new(),
            refs: HashMap::new(),
            // generous default; halved so the map never dominates memory
            max_prefix_entries: cfg.num_blocks / 2,
        }
    }

    pub fn cfg(&self) -> &KvConfig {
        &self.cfg
    }

    pub fn blocks_free(&self) -> usize {
        self.free.len()
    }

    pub fn prefix_entries(&self) -> usize {
        self.prefix.len()
    }

    pub fn seq_blocks(&self, seq_id: u64) -> Option<&[u32]> {
        self.tables.get(&seq_id).map(|t| t.blocks.as_slice())
    }

    pub fn seq_len_blocks(&self, seq_id: u64) -> usize {
        self.tables.get(&seq_id).map(|t| t.blocks.len()).unwrap_or(0)
    }

    pub fn blocks_for_tokens(tokens: usize, block_size: usize) -> usize {
        tokens.div_ceil(block_size)
    }

    fn stride_of(cfg: &KvConfig) -> usize {
        cfg.num_layers * 2 * cfg.block_size * cfg.num_kv_heads * cfg.head_dim
    }

    /// Tokens of a seq table's capacity (blocks * block_size).
    pub fn seq_capacity(&self, seq_id: u64) -> usize {
        self.seq_len_blocks(seq_id) * self.cfg.block_size
    }

    /// Ensure `seq_id` holds >= `tokens` worth of blocks. Prefix-cached blocks
    /// are used for every block whose hash is already registered; the rest are
    /// allocated fresh. Returns None on pool exhaustion (nothing allocated).
    ///
    /// `prefix_hashes[k]` must be the incremental hash of tokens[0..(k+1)*bs].
    pub fn reserve(
        &mut self,
        seq_id: u64,
        tokens: usize,
        prefix_hashes: &[u64],
    ) -> Option<ReserveInfo> {
        let bs = self.cfg.block_size;
        let n_needed = Self::blocks_for_tokens(tokens, bs);
        let mut reused = 0usize;

        // Fast path: table already big enough.
        {
            let have = self.seq_len_blocks(seq_id);
            if have >= n_needed {
                return Some(ReserveInfo { total: have, reused: 0 });
            }
        }

        // Count shareable prefix blocks first, to fail before mutating state
        // (all-or-nothing: an admission decision must never half-allocate).
        let have = self.seq_len_blocks(seq_id);
        let mut shareable = 0usize;
        for i in have..n_needed {
            if i < prefix_hashes.len() && self.prefix.contains_key(&prefix_hashes[i]) {
                shareable += 1;
            } else {
                break;
            }
        }
        let need_fresh = n_needed - have - shareable;
        if need_fresh > self.free.len() {
            return None;
        }
        // Seed the table entry (all-or-nothing check passed; an empty
        // table left behind on later failure is harmless).
        self.tables.entry(seq_id).or_default();

        for i in have..n_needed {
            if i < shareable + have {
                let h = prefix_hashes[i];
                let block = *self.prefix.get(&h).expect("checked above");
                match self.refs.get_mut(&block) {
                    Some(r) => *r += 1, // live sharer: refcount up
                    None => {
                        // free-but-cached block: revive (remove from the
                        // free list, restore refcount)
                        if let Some(pos) = self.free.iter().position(|&b| b == block) {
                            self.free.remove(pos);
                        }
                        self.refs.insert(block, 1);
                    }
                }
                self.tables.get_mut(&seq_id).unwrap().blocks.push(block);
                reused += 1;
            } else {
                let block = self.free.pop().unwrap();
                // Fresh allocation evicts whatever content this block used
                // to canonically represent (its data is now garbage for the
                // new owner, who writes before reading).
                if let Some(old_hash) = self.canonical_of.remove(&block) {
                    self.prefix.remove(&old_hash);
                }
                self.refs.insert(block, 1);
                self.tables.get_mut(&seq_id).unwrap().blocks.push(block);
            }
        }
        Some(ReserveInfo {
            total: n_needed,
            reused,
        })
    }

    /// Grow a sequence by `n` fresh blocks (decode-time growth). Returns
    /// false on exhaustion with no partial state (caller preempts).
    pub fn grow(&mut self, seq_id: u64, n: usize) -> bool {
        if n > self.free.len() {
            return false;
        }
        let table = self.tables.get_mut(&seq_id).expect("grow on unknown seq");
        for _ in 0..n {
            let block = self.free.pop().unwrap();
            self.refs.insert(block, 1);
            table.blocks.push(block);
        }
        true
    }

    /// Register a completed block under `hash` so later sequences can share
    /// it. Only blocks fully covered by processed tokens may be registered
    /// (the caller enforces this). Returns true when this call created a new
    /// canonical entry.
    pub fn register(&mut self, seq_id: u64, block_idx: usize, hash: u64) -> bool {
        let block = match self.tables.get(&seq_id).and_then(|t| t.blocks.get(block_idx)) {
            Some(&b) => b,
            None => return false,
        };
        if self.prefix.len() >= self.max_prefix_entries {
            return false; // cache table full; keep correctness, lose sharing
        }
        if self.prefix.contains_key(&hash) {
            return false; // someone else already canonical for this content
        }
        self.prefix.insert(hash, block);
        self.canonical_of.insert(block, hash);
        true
    }

    /// Release every block held by `seq_id` (refcount-aware; shared blocks
    /// survive until the last holder drops).
    pub fn release(&mut self, seq_id: u64) {
        let table = match self.tables.remove(&seq_id) {
            Some(t) => t,
            None => return,
        };
        for block in table.blocks {
            self.dec_ref(block);
        }
    }

    fn dec_ref(&mut self, block: u32) {
        let r = self.refs.get_mut(&block).expect("refcount for live block");
        *r -= 1;
        if *r == 0 {
            self.refs.remove(&block);
            // RadixAttention semantics (arXiv:2312.07104 — deep-read verified:
            // the vLLM *paper* does NOT prescribe this; SGLang's LRU KV cache
            // does): freed blocks REMAIN content-addressable —
            // they go to the free list still holding valid K/V for their
            // hash, and stay in prefix/canonical_of until a fresh allocation
            // evicts them. This is what makes duplicate prompts after the
            // first request finished still hit the cache.
            self.free.push(block);
        }
    }

    /// Make block `idx` of `seq` private (vLLM §4.2 CoW). Returns the block id
    /// to write into, or None when the pool is full.
    fn ensure_mutable(&mut self, seq_id: u64, idx: usize) -> Option<u32> {
        let block = *self
            .tables
            .get(&seq_id)
            .and_then(|t| t.blocks.get(idx))
            .expect("ensure_mutable on allocated block");
        if *self.refs.get(&block).unwrap() == 1 {
            return Some(block);
        }
        let new_block = self.free.pop()?;
        // copy the full block region
        let stride = self.stride();
        let src = block as usize * stride;
        let dst = new_block as usize * stride;
        self.data.copy_within(src..src + stride, dst);
        // old block loses this sharer. ensure_mutable only runs when
        // refs >= 2, so the old block still has >= 1 holder after this
        // decrement and can never be freed here (freed-on-zero would leave
        // the prefix map pointing at a free-list block — the bug CoW must
        // never introduce). Assert that invariant.
        let r = self.refs.get_mut(&block).unwrap();
        *r -= 1;
        debug_assert!(*r >= 1, "CoW old block reached 0 refs; impossible by construction");
        self.refs.insert(new_block, 1);
        let table = self.tables.get_mut(&seq_id).unwrap();
        table.blocks[idx] = new_block;
        Some(new_block)
    }

    fn stride(&self) -> usize {
        Self::stride_of(&self.cfg)
    }

    fn offset(&self, block: u32, layer: usize, which: Which, slot: usize) -> usize {
        let d = self.cfg.num_kv_heads * self.cfg.head_dim;
        let per_which = self.cfg.block_size * d;
        let per_layer = 2 * per_which;
        let base = block as usize * self.stride();
        base + layer * per_layer + (which as usize) * per_which + slot * d
    }

    /// Write one token's K/V across all layers.
    /// `kv` layout: [layer][which][heads*head_dim], total layers*2*heads*hd.
    /// Returns Err(Full) when CoW cannot get a private block.
    pub fn write_token(&mut self, seq_id: u64, abs_pos: usize, kv: &[f32]) -> Result<(), CacheError> {
        let bs = self.cfg.block_size;
        let idx = abs_pos / bs;
        let slot = abs_pos % bs;
        let d = self.cfg.num_kv_heads * self.cfg.head_dim;
        let block = self
            .ensure_mutable(seq_id, idx)
            .ok_or(CacheError::Full)?;
        for l in 0..self.cfg.num_layers {
            let k_off = l * 2 * d;
            let v_off = k_off + d;
            let base = self.offset(block, l, Which::K, slot);
            self.data[base..base + d].copy_from_slice(&kv[k_off..k_off + d]);
            let base_v = self.offset(block, l, Which::V, slot);
            self.data[base_v..base_v + d].copy_from_slice(&kv[v_off..v_off + d]);
        }
        Ok(())
    }

    /// Read a contiguous `block_size * d` region: one block, one layer, one of K/V.
    /// Used by the attention kernel over the block table.
    pub fn read_block_layer(
        &self,
        seq_id: u64,
        block_idx: usize,
        layer: usize,
        which: Which,
    ) -> Option<&[f32]> {
        let block = *self.tables.get(&seq_id)?.blocks.get(block_idx)?;
        let base = self.offset(block, layer, which, 0);
        let d = self.cfg.block_size * self.cfg.num_kv_heads * self.cfg.head_dim;
        Some(&self.data[base..base + d])
    }

    /// Stats for /metrics export.
    pub fn stats(&self) -> (usize, usize, usize) {
        (self.free.len(), self.tables.len(), self.prefix.len())
    }
}

/// FNV-1a over a token slice — content hash for prefix caching.
pub fn fnv_tokens(h: u64, tokens: &[u32]) -> u64 {
    let mut h = h;
    for &t in tokens {
        h ^= t as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}

/// Incremental block-prefix hashes: hashes[k] covers tokens[0..(k+1)*bs].
/// Identity of an all-zero initial hash is avoided by seeding with 0xcbf29ce484222325.
pub fn prefix_hashes(tokens: &[u32], block_size: usize) -> Vec<u64> {
    let mut out = Vec::new();
    let mut h = 0xcbf29ce484222325u64;
    let mut i = 0usize;
    while i < tokens.len() {
        let end = (i + block_size).min(tokens.len());
        // Position-dependence: fold the block INDEX into the hash as well,
        // so a repeated token run at a different absolute position cannot
        // alias (RoPE correctness). Only COMPLETE blocks are registrable,
        // enforced by the caller; partial final blocks never share.
        if end - i == block_size {
            h = fnv_tokens(h, &tokens[i..end]);
            out.push(h);
        }
        i = end;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg() -> KvConfig {
        KvConfig {
            block_size: 4,
            num_blocks: 16,
            num_layers: 2,
            num_kv_heads: 2,
            head_dim: 8,
        }
    }

    fn kv_of(tok: u32, layers: usize, d: usize) -> Vec<f32> {
        let mut v = Vec::with_capacity(layers * 2 * d);
        for l in 0..layers {
            for w in 0..2 {
                for i in 0..d {
                    v.push((tok as f32) * 0.125 + l as f32 + (w as f32) * 0.5 + i as f32 * 0.001);
                }
            }
        }
        v
    }

    #[test]
    fn reserve_allocates_expected_blocks() {
        let mut c = KvCache::new(cfg());
        let info = c.reserve(1, 9, &[]).unwrap(); // ceil(9/4) = 3
        assert_eq!(info.total, 3);
        assert_eq!(info.reused, 0);
        assert_eq!(c.seq_blocks(1).unwrap().len(), 3);
        assert_eq!(c.blocks_free(), 13);
    }

    #[test]
    fn prefix_share_and_release_refcounts() {
        let mut c = KvCache::new(cfg());
        let prompt: Vec<u32> = (1..=8).collect(); // 2 complete blocks
        let hashes = prefix_hashes(&prompt, 4);

        let info = c.reserve(10, 8, &hashes).unwrap();
        assert_eq!(info.reused, 0);
        // process + register both blocks
        assert!(c.register(10, 0, hashes[0]));
        assert!(c.register(10, 1, hashes[1]));

        let info = c.reserve(11, 8, &hashes).unwrap();
        assert_eq!(info.reused, 2); // fully shared
        assert_eq!(c.blocks_free(), 14); // 16 - 2 canonical blocks

        // B releases: shared blocks survive (A still holds refs=1 each)
        c.release(11);
        assert_eq!(c.seq_blocks(11), None);
        assert_eq!(c.blocks_free(), 14);
        // A releases: blocks drop refcount to 0 and go to the free list —
        // but stay content-addressable (vLLM cached-free semantics)
        c.release(10);
        assert_eq!(c.blocks_free(), 16);
        assert_eq!(c.prefix_entries(), 2);
    }

    #[test]
    fn duplicate_prompt_is_zero_cost() {
        let mut c = KvCache::new(cfg());
        let prompt: Vec<u32> = (1..=8).collect();
        let hashes = prefix_hashes(&prompt, 4);
        c.reserve(1, 8, &hashes).unwrap();
        c.register(1, 0, hashes[0]);
        c.register(1, 1, hashes[1]);
        let before = c.blocks_free();
        let info = c.reserve(2, 8, &hashes).unwrap();
        assert_eq!(info.reused, 2);
        assert_eq!(c.blocks_free(), before); // zero fresh allocation
    }

    #[test]
    fn exhaustion_returns_none_all_or_nothing() {
        let mut c = KvCache::new(KvConfig { num_blocks: 4, ..cfg() });
        let big: Vec<u32> = (0..20).collect(); // needs 5 blocks > 4
        assert!(c.reserve(1, 20, &[]).is_none());
        assert_eq!(c.blocks_free(), 4); // nothing half-allocated
        assert_eq!(c.seq_len_blocks(1), 0);
        let _ = big;
    }

    #[test]
    fn write_read_roundtrip() {
        let mut c = KvCache::new(cfg());
        let hashes = prefix_hashes(&[7, 7, 7, 7], 4);
        c.reserve(5, 4, &hashes).unwrap();
        let kv = kv_of(7, 2, 16);
        c.write_token(5, 3, &kv).unwrap(); // last slot of block 0
        let got = c.read_block_layer(5, 0, 0, Which::K).unwrap();
        let d = 16;
        let expect = &kv[0..d]; // layer 0 K of this token only in slot 3
        let slot_base = 3 * d;
        assert_eq!(&got[slot_base..slot_base + d], expect);
        // V layer 1 slot 3
        let got_v = c.read_block_layer(5, 0, 1, Which::V).unwrap();
        let v_off = 3 * d; // layer1 K then V: layout [layer][which], so layer1 V = (1*2 + 1)*d
        let expect_v = &kv[v_off..v_off + d];
        assert_eq!(&got_v[slot_base..slot_base + d], expect_v);
    }

    #[test]
    fn cow_gives_private_copy_not_in_place_write() {
        let mut c = KvCache::new(cfg());
        let prompt: Vec<u32> = (1..=4).collect();
        let hashes = prefix_hashes(&prompt, 4);
        c.reserve(1, 4, &hashes).unwrap();
        c.write_token(1, 0, &kv_of(1, 2, 16)).unwrap();
        c.register(1, 0, hashes[0]);

        c.reserve(2, 4, &hashes).unwrap(); // shares block 0, refs=2
        // B writes the shared slot -> must CoW, not clobber A's data
        let kv_b = kv_of(99, 2, 16);
        c.write_token(2, 0, &kv_b).unwrap();
        // A still sees original
        let a = c.read_block_layer(1, 0, 0, Which::K).unwrap();
        assert!((a[0] - (1f32 * 0.125)).abs() < 1e-6);
        // B sees its own
        let b = c.read_block_layer(2, 0, 0, Which::K).unwrap();
        assert!((b[0] - (99f32 * 0.125)).abs() < 1e-6);
        // refs split
        let bid_a = c.seq_blocks(1).unwrap()[0];
        let bid_b = c.seq_blocks(2).unwrap()[0];
        assert_ne!(bid_a, bid_b);
    }

    #[test]
    fn partial_block_never_registered() {
        let mut c = KvCache::new(cfg());
        let tokens: Vec<u32> = (1..=7).collect(); // 1 complete + 1 partial block
        let hashes = prefix_hashes(&tokens, 4);
        assert_eq!(hashes.len(), 1); // only the complete block
        c.reserve(1, 7, &hashes).unwrap();
        assert!(c.register(1, 0, hashes[0]));
    }
}
