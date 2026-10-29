# int8 weight storage — pre-registered before implementation

## Claim under test

The remaining per-token cost is the weight stream: ~1.0 GB/token at bf16
(24 layers + the tied lm_head, each read once per token by construction of
dense gemv — 4 workers share one DRAM bus, so the stream doesn't parallelize).
Storing the big matrices as **int8 with per-row f32 scales** quarters the
stream again (~0.5 GB/token bf16 → ~0.25 GB + scales). The dot becomes
`sum(q_i * s * x_i) = s * sum(q_i * x_i)` — same summation order per row,
one extra multiply per row, bit-shape identical results *up to the
quantization of the weights themselves*.

Unlike bf16 (exact round-trip), int8 LOSES information. So the gate is not
bit-identity — it is *generation equivalence*: the quantized engine must
produce the same greedy token stream as the bf16 engine on our probe
prompts, with logit deltas bounded and documented.

## Gates (fixed before measurement)

1. **Generation equivalence**: int8 engine vs bf16 engine, same prompts —
   (a) identical greedy ids on the standard probes ("The capital of France
   is" → " Paris" family; the long cafes prompt), (b) top-5 logit delta
   bounded (report the actual numbers; bound set by measurement, not
   aspiration), (c) the tiny model's reference test still passes (the
   reference uses the same quantized storage — catches wiring bugs).
2. **Speed**: paired same-session A/B (W8int8 vs W7fusedr workload) —
   TTFT + TPOT, honest either direction. Paired because this host's
   inter-session variance is ±30% (documented in WEIGHTS16.md).
3. **Memory**: per-model `bytes()` accounting quatered vs bf16, RSS
   cross-check in the models demo.
4. **Fallback stays**: WeightMat gains an INT8 variant; bf16 remains the
   default for anything where exactness beats speed. Norms/biases stay
   f32.

## Scope

- Quantize ONLY the big projection matrices (q/k/v/o/gate/up/down + embed);
  per-row absmax scale, symmetric, round-to-nearest.
- Per-row dequant inside the row dot (dispatch once per row, as bf16 —
  the per-element dispatch lesson from WEIGHTS16.md).
- int8 checkpoints (if ever encountered) load natively; bf16 checkpoints
  quantize ONCE at load (conversion cost measured, reported).

## Status — MEASURED, AND IT LOSES

- [x] Pre-registered written
- [x] Implement (WeightMat::INT8 + per-row absmax scales + row-streamed
      loader conversion, zero-transient; `--weights int8|bf16` flag)
- [x] Gate 1 — **generation equivalence: PASSES**. int8 engine on the real
      model: identical greedy ids to bf16 (`12095, 13, 1084, 374, 279, 7772,
      3283, 304` = " Paris. It is the largest city in"), probe logit deltas
      ≤ 0.2 (" Paris" 17.235 vs 17.217). Tiny-model reference gate passes on
      INT8 storage too (39/39 tests).
- [x] Gate 2 — **paired same-session A/B: int8 is ~35% SLOWER**:

  | | bf16-fused (W9bf16p) | int8 (W9int8p) |
  |---|---|---|
  | TTFT | 27.7 / 27.1 s | 47.1 / 31.9 s |
  | TPOT | 1154 / 1283 ms | 1731 / 1571 ms |

- [x] Gate 3 — memory: int8 quarters the stream as designed (~0.5 GB/model
  resident vs ~1.0 GB bf16) — and it still loses on speed.
- [x] Reported, including the loss.

## Conclusion (the negative result is the finding)

Halving-the-stream worked at bf16 (−46% bytes → 2.05× speed) because the
bf16 dot stays a clean shift+bitcast that vectorizes. Quartering at int8
(−50% more bytes) lost, because the dot becomes a per-element sign-extend
cast + a per-row rescale that apparently costs more than the saved DRAM
traffic at 4 workers on one bus. **At this machine's geometry the dot is
arithmetic-bound, not purely bandwidth-bound.**

That redirects the next optimization honestly: stop shrinking the weights,
start vectorizing the dot (explicit SIMD or a structure LLVM autovectorizes
well) — that's the only lever that helps both bf16 and int8. Pre-register it
before implementing. The INT8 code stays (gated, tested, flag-selectable):
on the reference machine with wider cores the arithmetic/bandwidth ratio
may differ, and the negative result here is documented so nobody re-tries it
blind. **bf16 remains the default.**
