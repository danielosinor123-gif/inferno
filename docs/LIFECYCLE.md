# Model lifecycle — pre-registered before implementation

## Why (from VISION.md)

The company computer serves many models on one machine (a chat model, an
embedding model, a small model for routing). That needs a **model registry**:
load on demand, keep hot models resident, evict cold ones under a memory
budget, and never evict a model mid-request. This is the prerequisite for the
admin plane (per-team quotas) and the multi-model router.

## Claim under test

`ModelRegistry` can hold N model engines behind per-name loaders with a total
memory budget: `get(name)` loads-or-returns; loading beyond the budget evicts
the least-recently-used *unreferenced* model; eviction frees the weight memory
(RSS measureable); a model in active use is never evicted (loud refusal
instead); and evict-then-reload reproduces bit-identical outputs.

## Gates (fixed before measurement)

1. **Correctness**: tiny engine — capture a logits digest before eviction,
   reload, same digest (bit-identical reload identity).
2. **Safety**: `begin_request` (refcount) blocks eviction; an over-budget
   load that would need an in-use model REFUSES loudly instead of evicting.
3. **LRU**: with budget for 2, touching A→B→A→C evicts B (LRU) and keeps A.
4. **Memory honesty**: `memory_bytes()` accounting per dtype (BF16 = 2 B/elem,
   F32 = 4 B/elem, KV pool = num_blocks × stride × 4) AND cross-checked
   against real RSS from `/proc/self/statm` in the demo (labeled estimate vs
   measured).
5. **Demo**: bench `models` subcommand loads three names (one file, replicated
   — the registry is file-agnostic; documented as such) with a budget that
   forces rotation; prints load/evict/reload events + RSS deltas.

## Status

- [x] Pre-registered written
- [x] Implement (ModelRegistry: loaders, budget accounting, LRU eviction,
      begin/end_request refcount pins, loud BudgetExhausted refusal)
- [x] Tests (gates 1–4: evict/reload bit-identical, pinned never evicted,
      LRU order, dtype-accurate byte accounting) — 37/37 total
- [x] `models` bench demo (load/evict/reload + in-flight refusal + RSS
      cross-check via /proc/self/statm)
- [ ] Demo executed in-VM with RSS numbers (pending run; gates 1–4 already
      hold in tests)

## Notes

- The registry is file-agnostic (loaders are `Fn() -> QwenEngine`), so the
  same mechanism serves local dirs, a model hub, or a cache later.
- Eviction protection is structural: a mut borrow of a resident engine and an
  `evict` call cannot coexist in Rust — the type system enforces the
  invariant the doc demands; refcounts cover the indirect path
  (begin/end_request).
