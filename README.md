# inferno — the runtime for the company computer

`inferno` is the **request/runtime layer** of the foresight arc: a from-no-deps
Rust runtime implementing the scheduling, memory, and concurrency machinery a
high-speed LLM inference server needs, **built on top of** the
[foresight-sched](https://github.com/danielosinor123-gif/foresight-sched)
sched_ext CPU scheduler rather than replacing it.

**The product vision** ([docs/VISION.md](docs/VISION.md)): a *multi-purpose
company computer that is also the company's AI machine* — one fleet carrying
enterprise workloads and private AI together, instead of a dedicated AI
appliance next to the machines you already own. The moat is the two-level
co-design this repo implements: the batcher schedules tokens, the kernel
schedules the CPU — so AI yields instantly to the other work on the machine.

```
HTTP plane (OpenAI-compat API)                     [next PR]
┌────────────────────────────────────────────────┐
│ inferno-model (real weights: Qwen2.5-0.5B,     │
│   safetensors + HF tokenizer, gemv engine)     │
│ inferno-core                                   │
│   threads    — worker pool ("easy threads")    │
│   memory     — arena + recycled buffers        │
│   kv_cache   — paged KV, prefix sharing, CoW   │
│   batching   — iteration-level continuous      │
│                batching + chunked prefill      │
│   kernels    — CPU reference (online-softmax   │
│                paged attention, RMSNorm, RoPE) │
│   sched      — CPU admission onto              │
└──────────────────────┬─────────────────────────┘   foresight-sched
                       ▼
   /sys/fs/bpf/foresight/{task_class,knobs,stats,demotions,pid_probe}
                       ▼
   foresight_sched BPF scheduler (kernel 6.18.33.2, SCX lanes)
```

**Status: real-weights phase in progress** — `inferno-model` (Qwen2.5-0.5B-Instruct
via safetensors, fp16→f32, one forward pass per generated token, greedy
sampling) is wired into the same `Engine`/paged-KV path the scheduler layer
was benchmarked on. First real TTFT/TPOT numbers land with the next commit.
Design + limitation notes live in `crates/inferno-model/src/engine.rs` and
`docs/DEEP-READ.md`.

## The four pillars

| Pillar | Mechanism | Grounded in |
|---|---|---|
| **Speed** | online-softmax attention (single pass, no score materialization); blocked matmul; per-step budgeted scheduling | FlashAttention (2205.14135), FlashDecoding++ (2311.01282) |
| **Concurrency** | iteration-level scheduling: sequences join/leave the batch every step; chunked prefill keeps long prompts from stalling decodes; SLO-aware priority admission | Orca (OSDI'22), Sarathi-Serve (2403.02310), DistServe (2401.09670) |
| **Memory efficiency** | paged KV cache (zero fragmentation by construction), refcounted block sharing, copy-on-write, freed blocks stay content-addressable until evicted | vLLM/PagedAttention (2309.06180), RadixAttention (2312.07104) |
| **Easy accessible threads** | `ThreadPool::new(n)` + `spawn(f) -> JoinHandle` + `map_chunks(data, f)`; ~100 verified lines, shutdown-correct under park | our design (see threads.rs header for why not work-stealing) |

## Status

- [x] Worker pool, arena + byte pool, paged KV cache (reserve/grow/register/
      share/CoW/exhaustion/retention), continuous batcher (chunked prefill,
      priority, preemption, reuse), reference kernels — all with tests
- [x] foresight-sched BPF client (Linux): knobs/stats/demotions reads,
      task_class tagging, pid_probe offset discovery
- [x] **Real-weights phase**: Qwen2.5-0.5B-Instruct via safetensors + HF
      tokenizer behind the same `Engine`/paged-KV path. sha256-verified
      weights, QKV-bias loader, tied head, self-extension attention. Greedy
      generation verified coherent ("The capital of France is" → " Paris").
- [x] **Correctness gates**: engine-vs-independent-reference test,
      determinism across cold/warm paths and engine versions, retention
      reuse bit-identical, 33/33 tests
- [x] **Benchmarks v1** ([docs/BENCHMARK.md](docs/BENCHMARK.md)): pre-registered
      protocol with machine-state outlier detection; raw data published
      including suspect reps. Single-stream TPOT 733 ms, 28× TTFT on
      repeated prefixes, 2-way batching at ~zero marginal cost.
- [x] **Threading step** ([docs/THREADING.md](docs/THREADING.md)):
      row-parallel gemv, bit-identical, +9% (honest: below hoped scaling)
- [x] **bf16 weight storage** ([docs/WEIGHTS16.md](docs/WEIGHTS16.md)):
      dtype-preserving matrices, per-row dispatch — 2.05× over threaded,
      load peak <1 GB, VM memory floor 1.6 → 2.5 GB
- [ ] Model lifecycle (load/swap/evict, multi-model per machine) — the #1
      platform blocker for the company computer
- [ ] Admin plane API (models, quotas, health — operable by a company)
- [ ] OpenAI-compatible HTTP plane
- [ ] GPU kernel phase (the CPU reference kernels define the algorithm; PTX
      comes after)

## Honest measurement notes

* All benchmark claims live in `docs/BENCHMARK.md` with the pre-registered
  protocol, every rep raw, suspect reps flagged by machine state, and failed
  attempts documented (the first bf16 dispatch measured 12–50% *worse* —
  the fix is in the same doc).
* Raw kernel throughput is intentionally not the product claim: this
  hardware measures ~25–70× slower per token than cloud GPU fleets. The
  category bet is private, local, shared-machine serving (see VISION.md).

## Quick start

```bash
# Windows (toolchain note below) or any Linux:
cargo test --workspace
cargo run --release -p inferno-bench -- threads
cargo run --release -p inferno-bench -- load
# WSL2 with foresight-sched loaded: same commands report foresight-live
```

Toolchain note (this specific Windows box): the user home directory contains
a space, which breaks mingw's `ld.exe` (`cannot find .../default-manifest.o`).
Workaround used here: copy the mingw64 toolchain to a spaceless directory
(`C:\m64tool\bin`) and prepend it to PATH for cargo invocations. MSVC is
unavailable (no VS Build Tools). This is documented because it will bite
every future Rust build on this machine.
