# Vectorized dot (AVX2+FMA) — pre-registered before implementation

## Why

The int8 experiment (docs/WEIGHTS8.md) proved the per-token cost on this
machine is **arithmetic-bound in the row dot**, not (only) bandwidth-bound:
quartering the bytes lost 35% because the per-element dequant cast cost more
than the saved stream. So the next lever is the dot's *arithmetic*: the
current inner loops are scalar reductions — `s += convert(w[i]) * x[i]` —
which LLVM may or may not vectorize, and almost certainly does not fuse.

## Claim under test

An explicit AVX2+FMA row dot (8 f32 lanes, runtime-detected, scalar
fallback on non-x86/no-AVX2) speeds up TPOT materially vs the current
loops, on both the f32 and bf16 paths.

## The honest methodology change this forces

A vectorized dot **changes the summation order** (8 lane accumulators +
horizontal sum instead of one sequential accumulator). Results therefore
differ from the scalar path by rounding — the last ULPs. That breaks the
*literal* bit-identity the earlier gates used, and we say so plainly:

- **Gate 1 (wiring)**: the tiny reference test compares the vectorized
  engine vs the scalar reference with the EXISTING tolerances (1e-4 KV /
  1e-3 logits, relative) — those tolerances absorb lane-order rounding by
  construction; a real wiring bug is still orders of magnitude beyond them.
- **Gate 2 (generation equivalence)**: real-model probes must produce the
  SAME greedy ids as the bf16-fused engine (`" Paris"` family; the cafes
  prompt stream), with top-5 logit deltas reported. ULP-level logit noise
  can flip a near-tie in principle — if it does, it gets documented, not
  hidden.
- **Gate 3 (speed)**: paired same-session A/B (W10avx2 vs W9bf16p
  workload), TTFT + TPOT, honest either direction.
- The scalar path stays as the fallback and as the reference; the engine
  selects the vector path at runtime via `is_x86_feature_detected!`.

## Scope

- x86_64 AVX2 + FMA only (std::arch, runtime-detected); everything else
  keeps today's scalar loops.
- Both storage paths (f32 elements; bf16 elements widened per 8-lane
  batch with _mm256_slli_epi32 + cast).
- No new deps. The dot stays per-row-parallel across the existing pool.

## Status — MEASURED, AND IT LOSES (with a bigger finding attached)

- [x] Pre-registered written
- [x] AVX2+FMA present in the VM (i5-8265U: avx2 + fma flags confirmed)
- [x] Implemented: `dot_row_avx2` (f32/bf16/int8 8-lane FMA + hsum + scalar
      tail), runtime-detected, scalar fallback; `INFERNO_SIMD=1` opt-in
- [x] Gate 1 — reference tests pass both ways (39/39)
- [x] Gate 2 — **real-model equivalence PASSES**: AVX2 probe logits identical
      to scalar at 3 decimals (`17.217/16.320/15.695/15.571/15.387`), greedy
      ids identical (`12095, 13, 1084, 374, 279, 7772, 3283, 304`)
- [x] Gate 3 — **paired same-session A/B: AVX2 is ~2x SLOWER than scalar**:

  | | scalar (INFERNO_NO_SIMD=1) | AVX2+FMA |
  |---|---|---|
  | TPOT | **328.2 / 321.4 ms** | 730.0 / 622.8 ms |

## Conclusion (negative result + the real finding)

LLVM was **already auto-vectorizing the scalar dot**; the hand-written
AVX2 path is worse code: `#[target_feature]` functions cannot inline into
the caller (a real call per row, ~200k rows/token), the 128→256 widening
per bf16 batch, and a horizontal sum per row — none of which the
compiler-generated version pays. Scalar is the default again; the AVX2
path stays behind `INFERNO_SIMD=1`, tested, as an experiment.

**The bigger finding is the scalar number itself**: TPOT **~325 ms on a
quiet VM** — versus the 695–1283 ms we had been reporting as "the" engine
speed in W7/W9. The engine got ~2x "faster" by the host going quiet. Our
measurement discipline flagged the variance (WEIGHTS16.md: ±30%) but this
is worse than ±30% across sessions: **the honest headline for this
engine's single-stream speed is ~325 ms/token, and earlier figures were
host-noise floors, not engine properties.** Every future speed claim gets
made with the paired-run protocol and the machine state quoted alongside.

Net position after three optimization experiments (threading +9%, bf16
2.05x, fusion 6.5%, int8 -35%, SIMD 2x-loss): the compiler + bf16 + fused
dispatch, on a quiet machine, at **~325 ms TPOT** — down from the 1650 ms
scalar-f32 v1 baseline on the same comparable-quiet basis (that number was
also noise-flattered; treat the ratio directionally, not precisely).
