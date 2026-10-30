# G1: GPU compute â€” pre-registered before implementation

## The decision (and why the obvious answer was rejected)

The thesis needs AI on a machine whose primary job is something else. Three
architectures were considered:

- **(c) hand-written CUDA kernels**: full control, months of work, NVIDIA-only.
- **(b) fork llama.cpp and rebuild its scheduling**: inherits multi-vendor
  kernels AND quantization, but forfeits our scheduling contract or forces a
  fork-maintenance treadmill.
- **(a) wrap llama.cpp as the engine**: fastest, but it replaces our
  *paged-KV engine* â€” the layer that carries the 28x prefix-retention win,
  per-model batchers, and the Engine-trait gates. Borrowing it wholesale
  surrenders the moat.

**Decision: (d) borrow the kernels, keep the architecture.** Use llama.cpp's
compute backend (ggml) behind our EXISTING `Engine` trait:

```
batcher  â”€â”€â–º  inferno Engine trait (unchanged)  â”€â”€â–º  ggml backend (new)
registry â”€â”€â–º    prefill_chunk / decode_kv / decode_step       â–²
fore     â”€â”€â–º    our paged KV, our CoW, our retention          â”‚
sight           our reference test still the scalar gate  â”€â”€â”€â”˜
```

The expensive part (hand-tuned GPU kernels, multiple vendor backends,
quantization) is borrowed. The moat layer (scheduling, paged KV, lifecycle,
honest-failure gates) stays ours and stays tested. This is the same bet as
ADR-001: the moat lives in the layer we own.

## Gates (fixed before measurement)

1. **Correctness / conformance**: the GPU engine and the CPU engine produce
   IDENTICAL greedy token streams on the probe set ("The capital of France
   is" -> " Paris" family; the cafes prompt). Not approximate â€” identical
   ids, same as every earlier gate. The scalar CPU engine remains the
   reference; the GPU path must match it exactly.
2. **No regression**: all 51 existing tests + the live HTTP gates stay green
   with the GPU engine untangled (feature-gated so the CPU path is the
   default build).
3. **Speed, honest, paired**: same-session A/B (CPU vs GPU engine) on the
   standard workload â€” TTFT + TPOT, with machine state. Expected GPU win is
   large (10-100x on TPOT); if it is not, that is the finding.
4. **VRAM accounting honest**: ggml buffer sizes + model + KV pool reported
   per model; cross-checked against nvidia-smi (the VM has an idle MX130 to
   test against, and the reference tier is 12 GB cards).
5. **Failure is typed**: every ggml call wrapped; a GPU that cannot fit the
   model returns the registry's BudgetExhausted-style refusal, never a
   panic, never a silent fallback.

## Scope of the first slice

- ggml backend behind `Engine` on one model family (Qwen2.5 geometry we
  already parse), quantized weights where ggml supports them, our paged KV
  replaced by ggml's context KV for THIS engine only (documented: the GPU
  engine trades prefix retention for compute; the retention win remains a
  CPU-engine + registry feature until a later pre-registered step ports it).
- CPU path untouched and default.

## Status

- [x] Decision recorded (d)
- [x] Pre-registered written
- [ ] Build ggml into the workspace (static, vendored â€” the network lesson)
- [ ] Engine impl over ggml + gates 1,2,5 (hermetic first: tiny model, CPU
      reference comparison)
- [ ] Real-model conformance (gate 1)
- [ ] Paired A/B + VRAM accounting (gates 3,4)