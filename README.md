# inferno — a concurrent inference runtime on top of foresight

`inferno` is the **request/runtime layer** of the foresight arc: a from-no-deps
Rust runtime implementing the scheduling, memory, and concurrency machinery a
high-speed LLM inference server needs, **built on top of** the
[foresight-sched](https://github.com/danielosinor123-gif/foresight-sched)
sched_ext CPU scheduler rather than replacing it.

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

## Status (v0.1 — scheduler layer)

- [x] Worker pool: 6/6 tests, shutdown-correct (deadlock regression test included)
- [x] Arena + byte pool: 2/2 tests
- [x] Paged KV cache: reserve/grow/register/share/CoW/exhaustion/retention — 8/8 tests
- [x] Continuous batcher: drain/chunked-prefill/priority/preemption/reuse — 7/7 tests
- [x] Reference kernels: naive-vs-optimized agreement (attention/RMSNorm/RoPE/matmul) — 4/4 tests
- [x] foresight-sched BPF client (Linux): raw-syscall reads of knobs/stats/demotions, task_class tagging of own threads, pid_probe offset discovery
- [x] **27/27 tests pass**
- [ ] Real-weights backend (safetensors + HF tokenizer) behind the same `Engine` trait
- [ ] OpenAI-compatible HTTP plane
- [ ] GPU kernel phase (the CPU reference kernels define the algorithm; PTX comes after)

## Honest measurement notes (v0.1)

* The load benchmark runs a **pseudo-model**: it reproduces the cost shape
  (prefill ~ linear in tokens, decode ~ constant) and performs REAL KV block
  writes, but it is not a language model. Numbers below are scheduler-layer
  plumbing capacity, not model throughput.
* The reuse metric caught a load-generator bug (all clients shared one seed
  → duplicate streams inflating "prefix reuse" to 47-73%). After the fix,
  random prompts show 0.0% reuse — as they should.
* This Windows host currently gives spawned threads ≈1.5 effective cores
  (63% total load; raw `std::thread` shows the same 5.3x slowdown as our
  pool — it is the environment, not the pool). The meaningful A/B — the same
  benchmark under `foresight-live` inside WSL2 with batch/interactive lanes —
  is the next experiment and the reason this repo exists.

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
