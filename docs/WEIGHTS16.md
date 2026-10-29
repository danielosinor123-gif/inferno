# bf16 weight storage — pre-registered before implementation

## Claim under test

Per token, the forward streams ~1.9 GB of weights (24 layers × ~60 MB +
the 544 MB tied lm_head) — the dominant memory traffic, and 4 cores share
one DRAM bus, so it does not parallelize. Storing weight matrices as **bf16**
(halves the bytes) with per-row conversion at use time cuts the stream to
~1.0 GB/token. Bonus: the load peak drops from ~2.9 GB (f32 arrays + mmap
transients) to <1 GB — the WSL VM (4 GB, host at 1.1 GB free) stops dying.

bf16→f32 conversion (`bits << 16`) is EXACT and costs one shift per element
against a MAC — negligible next to the DRAM bytes saved. For a bf16
checkpoint (ours: Qwen2.5-0.5B, sha256-verified) the loader becomes a pure
memcpy of the mmap region — zero conversion, zero transient.

## Gates (fixed before measurement)

1. **Bit-identical**: bf16-round-trip(f32) values are exact, so a bf16-stored
   engine must produce bit-identical logits/generations vs the f32 path on
   identical inputs — verified by the reference test (which now runs the
   parallel bf16 engine vs a scalar bf16 reference) + the real-text
   determinism gate (staggered repeats, identical ids) + loader precision
   tests (bf16 round-trip of known values incl. negatives/subnormals).
2. **Honest speedup**: W6 = the SAME workload as W5threaded (prompt 32,
   max-gen 16, conc 1, reps 2, same seed) — compare against W5threaded
   (TTFT 33.85 s / TPOT 1507.8 ms) and the W1b baseline (37.36 / 1649.7).
3. **fp32/f16 checkpoints stay exact**: storage keeps the input dtype
   (f32 stays f32, f16 stays f16); only the bf16 path avoids conversion.

## Implementation constraints

- `WeightMat` enum { F32(Arc<Vec<f32>>), BF16(Arc<Vec<u16>>), F16(Arc<Vec<u16>>) }
  with a per-row dot accessor; workers dispatch on the tag once per row.
- f32→bf16 rounding: round-to-nearest-even on the truncated mantissa.
- No new deps; inner-loop order unchanged (bit-identity by construction).

## Status

- [x] Pre-registered written
- [x] Implement (WeightMat {F32/BF16/F16}, loader keeps native dtype, engine
      dispatches per ROW, tests + reference gate run bf16)
- [x] Gates: bit-identical — reference test (parallel bf16 engine vs scalar
      bf16 reference) ✓; bf16 round-trip precision ✓; real-text probe logits
      bit-identical to the f32 era (" Paris" 17.217 both) ✓; staggered
      determinism gate: ids [576,3283,374,1083,3881,369,1181,9080] identical
      across cold+warm runs AND identical to the f32-era runs ✓; reuse 1/1 ✓
- [x] Measured, reported honestly (below)

## Results

| | W1b baseline (scalar f32) | W5threaded (4w f32) | W6bf16 (per-element dispatch) | **W6bf16b (per-row dispatch)** |
|---|---|---|---|---|
| TTFT | 37.36 s | 33.85 s | 50.3 / 40.3 s | **29.9 / 18.4 s** |
| TPOT | 1649.7 ms | 1507.8 ms | 1827 / 2283 ms | **739.8 / 733.4 ms** |

**TPOT: 2.05× vs W5threaded, 2.25× vs the scalar baseline.** Both W6 rows ran
with host loadavg rising 2.3→3.0 (contention was AGAINST us), so the win is
not a noise artifact; rep-to-rep TPOT variance is 0.9%.

**The first implementation FAILED and it's instructive**: dispatching the
dtype match per ELEMENT (inside the dot) measured 12–50% WORSE than f32 —
the per-element branch blocks auto-vectorization of the inner loop. Moving
the dispatch to once-per-row (type-uniform inner loops) delivered the win.
Claim tested → failed → fixed → re-measured, all recorded.

**Unplanned bonus**: with bf16 storage, VM `mem_avail` during runs rose from
~1.57 GB to ~2.5 GB — the load peak dropped from ~2.9 GB to <1 GB (the loader
is now a bit-copy for bf16 checkpoints). The WSL VM fragility that killed
three benchmark legs is materially reduced.

## Dispatch fusion (W7) — the follow-up candidate, measured

Fusing the per-layer gemv dispatches (`q/k/v` share the attention norm;
`gate/up` share the MLP norm — one `map_chunks` each, 7→4 dispatches per
layer, 169→97 per token) is implemented and gated bit-identical (37/37 tests,
reference gate rerun). **Measured via back-to-back interleaved A/B in one VM
session** (the only trustworthy comparison on this host — see note):

| | W6bf16r (per-matrix dispatch) | W7fusedr (fused) |
|---|---|---|
| TTFT | 18.82 / 18.76 s | 16.68 / 17.90 s |
| TPOT | 788.5 / 787.3 ms | 774.0 / 695.2 ms |

Fusion: **TTFT −8%, TPOT −6.5%** — kept.

**Methodology finding worth its own line**: the first W7fused measurement
(944/1018 ms) looked WORSE than W6bf16b (733/740 ms), but those two rows came
from different sessions with different host load — this host's run-to-run
variance is ±30%. *Interleaved same-session A/B is the only comparison worth
trusting here*; the session-ab-ac package above is the honest one. The
bench harness's machine-state logging flags the variance; the fix for it is
paired runs, not better averaging.

**What's left of the bottleneck**: TPOT ~735 ms is now mostly the remaining
weight stream (~1.0 GB/token, 4 workers on one DRAM bus — arithmetic says the
bf16 halving already did what it can for a shared bus). Further gains need
either a wider machine (the reference-hardware question) or a fundamentally
different memory story (weight caching across tokens is already in place;
the stream is the 24 layers + lm_head, read once per token by construction
of dense gemv — int8 weights or sparse skipping are the next candidates,
both to be pre-registered first).
