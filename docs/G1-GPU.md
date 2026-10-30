# G1: GPU compute - pre-registered before implementation

## The decision (and why the obvious answer was rejected)

The thesis needs AI on a machine whose primary job is something else. Three
architectures were considered:

- **(c) hand-written CUDA kernels**: full control, months of work, NVIDIA-only.
- **(b) fork llama.cpp and rebuild its scheduling**: inherits multi-vendor
  kernels AND quantization, but forfeits our scheduling contract or forces a
  fork-maintenance treadmill.
- **(a) wrap llama.cpp as the engine**: fastest, but it replaces our
  *paged-KV engine* - the layer that carries the 28x prefix-retention win,
  per-model batchers, and the Engine-trait gates. Borrowing it wholesale
  surrenders the moat.

**Decision: (d) borrow the kernels, keep the architecture.** Use llama.cpp's
compute backend (ggml) behind our EXISTING `Engine` trait:

```
batcher   -->  inferno Engine trait (unchanged)  -->  ggml backend (new)
registry  -->    prefill_chunk / decode_kv / decode_step      ^
foresight -->    our paged KV, our CoW, our retention         |
               our reference test still the scalar gate -----+
```

The expensive part (hand-tuned GPU kernels, multiple vendor backends,
quantization) is borrowed. The moat layer (scheduling, paged KV, lifecycle,
honest-failure gates) stays ours and stays tested. This is the same bet as
ADR-001: the moat lives in the layer we own.

## Gates (fixed before measurement)

1. **Correctness / conformance**: the GPU engine and the CPU engine produce
   IDENTICAL greedy token streams on the probe set ("The capital of France
   is" -> " Paris" family; the cafes prompt). Not approximate - identical
   ids, same as every earlier gate. The scalar CPU engine remains the
   reference; the GPU path must match it exactly.
2. **No regression**: all 51 existing tests + the live HTTP gates stay green
   with the GPU engine untangled (feature-gated so the CPU path is the
   default build).
3. **Speed, honest, paired**: same-session A/B (CPU vs GPU engine) on the
   standard workload - TTFT + TPOT, with machine state. Expected GPU win is
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
- [x] **ggml vendored and building**: pinned v0.8.0
      (8642541b255bfb5dc39cd351a178be09013a55d6, MIT, third_party/VENDORED.md),
      static-built by `crates/inferno-gpu/build.rs` (glob file selection,
      gnu11, _GNU_SOURCE, no llamafile - a later option -, stdc++ linked).
      The build fights worth recording: CMake-generated ggml-version.h (now
      generated in build.rs), strict-c11 hiding POSIX declarations,
      SCHED_BATCH needing _GNU_SOURCE, the unconditional tiled-matmul
      dependency (ggml-cpu/tiled/), and C++ symbols needing the C++ runtime
      at link time.
- [x] **Bridge gates: FOUR ops match the scalar kernels** (rms_norm, rope
      NEOX vs `kernels::rope`, swiglu/silu-mul, mul_mat) — all within the
      reference tolerance, tests in crates/inferno-gpu/src/lib.rs.
- **SIGSEGV caught by the gates, worth the record**: my first ggml_rope_ext
  call used an 11-arg ABI (scalar position) — the pinned header takes 13
  args with b = an i32 positions TENSOR. Lesson: FFI signatures get verified
  against the pinned header, not against memory of the API. The classic
  ggml_rope would have been wrong anyway: it hardcodes freq_base=10000
  (Qwen2.5 needs theta=1e6), so the ext form with our theta is correct.
- [ ] Engine-level hermetic conformance (full forward through ggml ops vs
      the scalar reference - the next step)
- [ ] Real-model conformance (gate 1)
- [ ] Paired A/B + VRAM accounting (gates 3,4)
- [ ] No-regression + typed failures (gates 2,5; CPU stays default)

## Naming note

The bridge currently runs ggml's CPU backend (the VM has no CUDA toolkit -
`nvcc` absent, MX130 idle). The CUDA path is a documented follow-on; the
CPU-backend bridge is the proof that the borrowed-kernel architecture
works before any GPU toolchain enters the picture.
