# Threading step — pre-registered before implementation

## Claim under test

The v1 engine is single-threaded by design (correctness gate). The gemv row
loop is the hot spot: every projection is a contiguous row-loop over the
native HF `[out, in]` layout, each row independent. Parallelizing ROWS across
`inferno_core::threads::ThreadPool` should scale near-linearly on the large
MLP matrices (4864-column dots), less on the small QKV ones.

## Gates (fixed before any measurement)

1. **Correctness first**: outputs must be BIT-IDENTICAL to the single-threaded
   engine on the same weights/inputs (row-parallel gemv with unchanged per-row
   summation order is exact by construction — this gate verifies that claim,
   including through the determinism/retention benches).
2. **Honest speedup**: run the SAME W1b workload (prompt 32, max-gen 16,
   reps 2, conc 1, tag W5threaded) against the recorded single-threaded
   baseline (W1b: TTFT 37.36 s / TPOT 1649.7 ms). Report the measured
   speedup even if it is ~free on memory-bound small matrices.
3. **No silent concurrency**: threading is the engine's only concurrency; the
   batcher/scheduler single-threaded loop stays.

## Implementation constraints

- Row-parallel `gemv`: chunk rows across pool workers, each row fully computed
  by one worker, unchanged inner-loop summation order → bit-identical results.
- Pool sized to available parallelism, created once at engine construction.
- No rayon, no external deps: inferno-core's own ThreadPool only.

## Status

- [x] Pre-registered written
- [x] Implement (row-parallel gemv over inferno-core's ThreadPool; weights Arc'd
      for 'static captures; x copied per call — negligible vs the gemv)
- [x] Verify bit-identical — 32/32 tests, reference test runs the PARALLEL
      engine vs the scalar reference (nproc=4, so the pool path is live)
- [x] Measure W5threaded vs W1b baseline, report honestly

## Result (same workload, same seed, healthy machine state)

| | W1b baseline (scalar) | W5threaded (4 workers) | delta |
|---|---|---|---|
| TTFT | 37.36 s | 33.85 s | **−9.4%** |
| TPOT | 1649.7 ms | 1507.8 ms | **−8.6%** |

(TTFT rep-0 comparators: 38.9 s vs 37.4 s baseline — within noise; the
healthy-rep pair above is the honest comparison.)

**~9%, not the 2–3× the row-parallel form should allow.** Where the time
actually goes (measured, not speculated): per token the forward streams
~1.9 GB of weights (24 layers × ~60 MB + the 544 MB tied lm_head) through
169 gemv calls. The scalar inner loop runs far below peak (bounds-checked
f32 dot, cold cache lines on a 2 GB working set), so each call is part
compute, part memory streaming — and 4 workers share one DRAM bus, so the
memory half doesn't parallelize at all. Per-call dispatch (channel +
condvar + result copy) on 169 small calls adds its own tax.

**Kept** because it is free, verified bit-identical, and never regresses —
but the next optimization is not "more threads". It is: (a) fuse the seven
per-layer gemvs into one worker dispatch per layer (fewer, larger tasks —
channel/tax per call drops 7×), and (b) keep the weights resident across
tokens (they already are — but the lm_head at [151936,896] is re-streamed
per token; bf16 storage or an fp16 lm_head would halve that stream). Both
will be pre-registered before implementation.
