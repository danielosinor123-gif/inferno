# Research Grounding — papers → design decisions

Every non-obvious design choice in inferno traces to a specific paper (or is
flagged as our own). This file is the map; module doc-comments carry the
citations inline too.

| Paper | What we take from it | Where it lives |
|---|---|---|
| **vLLM / PagedAttention** — Kwon et al., *Efficient Memory Management for LLM Serving with PagedAttention*, arXiv:2309.06180 (SOSP'23) | Paged KV storage: fixed-size blocks + per-sequence block tables + refcounted sharing + copy-on-write + free-list retention of content-addressable blocks; recompute-preemption as the response to pool exhaustion | `kv_cache.rs` (blocks/tables/CoW/retention), `batching.rs` (`preempt`) |
| **Orca** — Yu et al., *Orca: A Distributed Serving System for Transformer-Based Generative Models*, OSDI 2022 | Iteration-level scheduling: requests join/leave the running batch EVERY decode step, not per-request-lifetime. This is the single biggest throughput primitive in modern serving | `batching.rs::step` (admit → prefill → decode → retire loop) |
| **Sarathi-Serve** — Agrawal et al., arXiv:2403.02310 (OSDI'24) | Chunked prefill: cap prefill tokens per step so long prompts never stall running decodes; token-budget mixing of prefill+decode | `BatcherConfig::prefill_chunk_tokens`, prefill phase in `step` |
| **FlashAttention** — Dao et al., arXiv:2205.14135; **FlashAttention-2** — arXiv:2307.08691 | Online softmax recursion (running max m, running sum l) — single pass, no materialized score vector | `kernels.rs::paged_attention` (CPU reference; GPU port replaces the inner loop with MMA tiles) |
| **FlashAttention-3** — Shah et al., arXiv:2407.08608 | Hopper-specific warp specialization + TMA — deferred to the GPU phase; the recursion we implement is FA-compatible | noted in `kernels.rs` header |
| **FlashDecoding++** — He et al., arXiv:2310.11453 | Split-K attention for long-context decode (partial softmax per chunk, combine at the end) — the SAME combine structure as our online-softmax recursion, at block granularity | `kernels.rs::paged_attention` |
| **SGLang / RadixAttention** — Zheng et al., arXiv:2312.07104 | Content-addressed prefix reuse (their radix tree over token prefixes; we use incremental block hashes — same share-by-prefix idea at page granularity) | `kv_cache.rs::prefix_hashes`/`register` |
| **DistServe** — Zhong et al., arXiv:2401.09670; **Splitwise** — Patel et al., arXiv:2311.18677 | Prefill and decode have opposite SLO profiles → schedule them as separate domains; informs our `Priority` + per-phase class mapping onto foresight's lanes | `batching.rs` (prefill/decode phases), `sched.rs` (class mapping) |
| **SpecInfer** — Miao et al., arXiv:2310.10545; **Medusa** — Cai et al., arXiv:2401.10774 | Speculative decoding: draft-then-verify amortizes decode steps — GPU-phase roadmap, not v1 | roadmap |
| **Mooncake** — Qin et al., arXiv:2407.00079 | KV-cache-centric disaggregation: the KV cache is the resource, compute flows to it — informs the future multi-node/offload design of the block pool | roadmap |
| **Attention Is All You Need** — Vaswani et al., arXiv:1706.03762 | Transformer shapes (layers/heads/head_dim, RoPE θ) that parameterize the reference kernels | `KvConfig`, `kernels.rs::rope` |

## What is OURS (no paper)

* **foresight-sched integration**: request-class → sched_ext task_class mapping
  over the pinned BPF maps, reusing foresight's `pid_probe` offset translator
  (foresight-sched is the user's own prior work — the layering is the novel
  part here, documented in docs/FORESIGHT-INTEGRATION.md).
* The single-lock worker pool with shutdown-under-park correctness (deliberately
  NOT a work-stealing deque — justified in `threads.rs` header).
* The adversarial protocol itself: pre-registered benchmarks, tests as the
  final arbiter, reuse-metric-catching-a-load-generator-bug (it did, PR #1).

## Known simplifications (v1 honesty)

* `pseudo.rs` is a cost-shape simulator, not a language model. Scheduler-level
  benchmarks (threads/KV/batching/admission) are meaningful; MODEL quality
  claims must wait for the real-weights backend.
* CPU reference storage (`Vec<f32>`); GPU pages change the storage, not the
  algorithm.
* Attention here is single-token-per-call (decode shape). Chunked-Q (prefill
  shape, the real FlashAttention win) lands with the GPU kernel phase.
