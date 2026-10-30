# G18: MoE expert streaming — pre-registered before implementation

## Why (the colibri study's central lesson)

A mixture-of-experts model activates only a few experts per token, so the
model does not have to fit in fast memory — it has to be *placed*. Dense
weights stay resident; routed experts are fetched on demand from a slower
tier through a cache that learns. This is the model tier the product needs
(35B-A3B class: 23 GB on disk, ~10-20 GB resident, 6-10 tok/s on consumer
silicon per colibri's published numbers) and it is the difference between
"our 0.5B toy" and the company computer's actual payload.

## The invariant that makes it a product (borrowed from colibri, same as our
## other gates in spirit)

**Placement decides speed, never output.** The router's decisions and the
model's outputs must be identical whether an expert was served from cache,
RAM, or the (simulated) disk tier. Verified, not assumed.

## Scope of the hermetic slice

A synthetic MoE in-memory: deterministic router weights + experts, with a
**dense reference** (compute every expert, pick the routed ones) versus the
**streaming engine** (fetch routed experts from a slow tier through a
per-layer LRU cache with session-learned pins). The "slow tier" is a
`Vec<u8>` standing in for disk; mmap/pread is the real-model step.

## Gates (fixed before measurement)

1. **Identity**: same tokens + weights -> streaming output BIT-IDENTICAL to
   the dense reference (the invariant). f32 everywhere in this slice makes
   bit-identity achievable; the real quantized path gets its own gate.
2. **Cache behavior**: with a cache smaller than the expert set, a repeating
   routing pattern shows hits (hit rate > 0 on repeats), and the LRU order
   matches a reference LRU implementation in the test.
3. **Pins**: pinned experts (learned from observed routing) survive
   eviction; unpinned evict LRU-first.
4. **Budget accounting**: per-tier bytes reported honestly (resident cache
   bytes + fetched bytes), no hidden state.

## Deliberately not in this slice

- Real MoE weights (needs ~10 GB resident — hardware-blocked; the hermetic
  engine is the design proof, the real model follows on the reference box).
- Quantization of experts (colibri's int4 tier; its own pre-registration).
- Router training/adaptation — routing weights come with the model; the
  *pins* are our learning mechanism and what gate 3 covers.

## Status — HERMETIC SLICE COMPLETE (all 4 gates pass)

- [x] Implemented `crates/inferno-model/src/moe.rs`: top-k routing with
  softmax over the selected (accumulated in expert-id order — the order
  both paths share, which is what makes the outputs bit-identical), the
  `ExpertStore` slow tier, the `ExpertCache` resident tier (per-key LRU,
  observation-count pins, honest counters), dense + streamed layer
  forwards.
- [x] **Gate 1** — placement never changes output: streaming output is
  BIT-IDENTICAL (`to_bits` equality) to the dense reference over 2 layers x
  6 seeds, and the routing selection matches (`sel_d == sel_s`).
- [x] **Gate 2** — LRU matches a reference `VecDeque` LRU on a fixed access
  sequence; repeats hit, first touches miss.
- [x] **Gate 3** — observation-threshold pins survive eviction pressure.
- [x] **Gate 4** — accounting honest: one miss fetches exactly one expert's
  bytes, hits fetch nothing, resident bytes == cached experts.

## The real-model path (next, hardware-gated)

Wire the store to mmap'd expert shards (pread per expert, colibri's "never
wait for the disk twice" pattern), a real MoE family's geometry (Qwen3.6-35B
-A3B: 23 GB download, ~10-20 GB resident — needs the reference box, not the
3.9 GB VM), then the placement-vs-speed measurement with the same gate 1
invariant. Expert quantization (colibri's int4 tier) is its own
pre-registration.
