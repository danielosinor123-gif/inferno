# Deep Read — actual paper reading pass (2026-10-27)

This is the correction to an earlier failure mode: the first version of
`RESEARCH.md` cited papers **from memory**. On audit, 3 of 13 arXiv IDs were
wrong and 1 mechanism was attributed to the wrong paper. This document was
produced by fetching and reading the actual texts (ar5iv full-text HTML for the
core five; abstract-level for the roadmap papers) and comparing what the papers
say against what inferno v0.1 implements.

**Scope honesty:**

| Paper | What was read here | Level |
|---|---|---|
| vLLM 2309.06180 | full text (ar5iv), targeted extraction around block table / CoW / eviction / block size | full |
| Sarathi-Serve 2403.02310 | full text (ar5iv), scheduling design + Algorithm 3 + budgets | full |
| FlashAttention 2205.14135 | full text (ar5iv), Algorithm 1 + IO analysis | full |
| SGLang 2312.07104 | full text (ar5iv), RadixAttention + LRU radix tree | full |
| FlashDecoding++ 2311.01282 | full text (ar5iv), contributions/abstract section | mechanism pass |
| SpecInfer 2305.09781 | full text (ar5iv), title/venue block + abstract region | abstract |
| Orca (OSDI'22) | **not on arXiv.** USENIX page verified; PDF not parseable in this environment. Mechanism cross-read through the passages of vLLM/Sarathi that describe Orca | secondary |
| FA-2, FA-3, DistServe, Splitwise, Medusa, Mooncake, Vaswani | abstract/metadata verification only | ID check |

---

## Corrections log (what deep reading caught)

1. **Orca was never on arXiv.** `2206.02331` (previously cited) is a
   remote-sensing change-detection paper — nothing to do with serving. Orca
   exists as OSDI'22 proceedings only:
   <https://www.usenix.org/conference/osdi22/presentation/yu>.
2. **FlashDecoding++ is arXiv:2311.01282**, not 2310.11453 (that ID is BitNet,
   the 1-bit LLM paper — a different real paper my memory conflated).
3. **SpecInfer is arXiv:2305.09781**, not 2310.10545 (that ID is an unrelated
   statistics paper).
4. **The freed-block-retention semantics in `kv_cache.rs` were attributed to
   "vLLM §4.3" — the paper does not contain them.** Targeted search of the full
   vLLM text finds no "free queue", no content-addressable retention of
   completed requests' blocks. What the vLLM paper *actually* prescribes for
   memory exhaustion is **all-or-nothing eviction per sequence** ("either evict
   all or none of the blocks of a sequence"), recovered by **swap-to-CPU or
   recomputation**. The retain-content-addressable-blocks-until-evicted
   mechanism — which our tests found and we kept, correctly, as the better
   semantic — is **SGLang/RadixAttention's** contribution: "our system
   maintains an LRU cache of the KV cache for all requests within a radix
   tree … uses a radix tree for efficient matching, insertion, and eviction".
   Reattributed in RESEARCH.md.
5. **Terminology**: the FlashAttention paper does not use the phrase "online
   softmax"; it says "incrementally performing the softmax reduction (also
   known as tiling)". The term "online softmax" comes from
   Milakov & Gimelshein, arXiv:1805.02867. Our comments use it because it is
   the standard community term — noted for precision.

---

## vLLM (2309.06180) — PagedAttention

**What the paper prescribes, verified in text:**

- KV storage as fixed-size physical blocks + per-sequence logical block
  tables; "Storing multiple tokens within a KV block (block size > 1) enables
  the PagedAttention kernel to process the KV cache across more positions in
  parallel … a larger block size also increases memory fragmentation" —
  our `KvConfig::block_tokens` knob exists for exactly this trade-off.
- **Per-layer/per-head separate block tables**: "The two designs have no
  performance difference and we choose the second one for easy
  implementation." inferno does the same (per-layer tables in `kv_cache.rs`).
- **Refcount + copy-on-write at block granularity**, triggered exactly when
  the refcount > 1 and a sharing sequence must write into a shared block;
  only the *last* logical block of the shared prefix needs CoW. Matches
  `kv_cache.rs` refcounting/`cow_split` design point-for-point.
- **Waste bound**: "a new physical block is only allocated when all previous
  blocks are full, vLLM limits all the memory wastes for a request within one
  block" — this is the near-zero-waste claim (≤ 1 block tail per sequence,
  vs. tens of % fragmentation with contiguous preallocation).
- **Exhaustion policy**: all-or-nothing eviction per sequence group, then
  swap-to-CPU or recomputation; "recomputation latency can be significantly
  lower than the original latency, as the tokens generated at decoding can be
  concatenated with the original user prompt" — our `preempt()` recompute path
  matches the recompute branch; we do NOT yet implement the CPU swap branch
  (no CPU pager) — listed as a delta below.
- Headline numbers: 2–4× throughput vs FasterTransformer-era systems, "without
  affecting the model accuracy at all"; improvements grow with longer
  sequences / more complex decoding.

**Delta in inferno vs. paper:** we `reserve()` all KV blocks up-front at
admission (all-or-nothing admission, simpler SLO reasoning); vLLM allocates
block-by-block lazily as tokens appear and handles exhaustion by preempting
mid-flight. For chunked prefill workloads, up-front reservation over-renders
memory safe but can reject longer prompts earlier than necessary. Candidate
follow-up: lazy per-chunk reservation + mid-flight preemption re-using our
existing `preempt`.

## Orca (OSDI'22) — iteration-level scheduling

Not on arXiv; cross-read through the vLLM/Sarathi texts (Sarathi §3: "Orca
supports hybrid batches composed of both prefill and decode requests";
vLLM §5 cites Yu et al. 2022 as the iteration-level comparison point) and the
USENIX abstract page. The core primitive — requests join/leave the running
batch **every iteration** (per-token step), not per-request — is the basis of
our `Batcher::step` admit → prefill → decode → retire loop, and of Orca-style
hybrid prefill+decode batches that Sarathi later refines with chunking.
Caveat: until we read the OSDI PDF directly, Orca-specific claims in
inferno docs rest on secondary-source descriptions (labeled accordingly).

## Sarathi-Serve (2403.02310) — chunked prefill, stall-free batching

**Verified in text:**

- The failure mode is named **generation stalls**: full prefills scheduled
  between decode iterations inflate time-between-tokens (TBT) for every
  running stream. Orca cannot fix long prompts (one giant batch step);
  vLLM's prefill-before-decode policy stalls decodes.
- Fix = **chunked prefill with a token budget**: "it throttles the number of
  prefill tokens in each iteration … bounds the latency of each iteration,
  and makes it nearly independent of the total length of input prompts."
- **Algorithm 3 order**: fill batch with ongoing decode tokens → add one
  prefill chunk → admit new requests, all within `token_budget` derived from
  the TBT SLO (`compute_token_budget(T_max, TBT SLO)`). inferno's `step` uses
  the same order (decodes first, prefill chunk, then new admissions), with
  `prefill_chunk_tokens` as the budget knob.
- Their budget evaluations: **256** (Mistral-7B, 1×A100) and **512**
  (LLaMA2-70B, 4×A100). Our default chunk = 512 sits inside the evaluated
  range — chosen from the paper, not tuned on hardware we don't have.

## FlashAttention (2205.14135)

**Verified in text:**

- Algorithm 1: block sizes `B_c = ⌈M/4d⌉, B_r = min(⌈M/4d⌉, d)`; running
  state `O` (output), `ℓ` (sum), `m` (max) initialized to 0/0/−∞; per-tile
  rescale of previous partial output. inferno's `paged_attention` runs the
  same recursion **per token** rather than per tile — mathematically
  equivalent (softmax is invariant to the granularity of the running-max
  normalization), just more rescale steps. On CPU this is fine; for the GPU
  port the tile-wise version is the one to implement (fewer renorms, matches
  the paper's IO account).
- The win is IO, not FLOPs: "computing exact attention with far fewer memory
  accesses … avoid reading and writing the attention matrix to and from HBM";
  storing `ℓ, m` and recomputing attention in the backward pass instead of
  materializing the N×N score matrix. Our CPU reference stores no score
  matrix either (scores live in registers within a block pass) — same
  invariant, weaker motivation (no HBM), honestly noted in `kernels.rs`.
- "online softmax" — see terminology correction above (paper says "softmax
  reduction"/tiling; the term originates in arXiv:1805.02867).

## SGLang / RadixAttention (2312.07104)

**Verified in text:** KV cache kept after request completion as a **cache**,
not garbage: "our system maintains an LRU cache of the KV cache for all
requests within a radix tree … efficient matching, insertion, and eviction"
with a **cache-aware scheduling policy** that routes requests with shared
prefixes into batch positions that maximize reuse. The anti-pattern they call
out: "the KV cache of a request is discarded after processing is completed."

inferno implements the same semantic with a different data structure:
incremental per-block content hashes + a chain per sequence (equivalent to a
radix tree flattened at block boundaries) + retention of freed blocks until
evicted. This is the corrected home for the semantics previously mislabeled
"vLLM §4.3". Their `--schedule` cache-aware ordering is the roadmap item we
don't have yet (our scheduler is priority-based, not cache-aware).

## FlashDecoding++ (2311.01282) — corrected ID

Contributions verified in text: (1) **asynchronized softmax with unified max
value** — pick one shared max across partial softmax computations so partial
results never need a synchronized update; enables fine-grained pipelining;
(2) flat GEMM double buffering (pad to 8, not 64); (3) heuristic dataflow
per hardware. Numbers: up to 4.86× (NVIDIA) / 3.93× (AMD) vs HF, 1.37× vs
SOTA engines. For inferno: the split-partial-softmax + combine structure
matches our block-granularity combine; the *unified max* trick is a GPU-era
synchronization-avoidance move (on CPU our per-token rescale has no
synchronization problem). Roadmap note for the GPU kernel phase.

## SpecInfer (2305.09781) — corrected ID

Tree-based speculative inference + verification (ASPLOS'24): draft models
produce a token tree, the target model verifies in fewer passes. GPU-phase
roadmap alongside Medusa (2401.10774, multiple decoding heads — verified).

---

## Net effect on inferno v0.1

- Implementation verified against papers on: paged storage, per-layer tables,
  refcount+CoW, block-tail waste bound, iteration-level scheduling, chunked
  prefill ordering + budget, online-softmax recursion, retained KV as a cache.
- Citation layer corrected: 3 IDs fixed, 1 mechanism reattributed to SGLang,
  Orca honestly marked as secondary-source reading.
- New deltas surfaced by the reading (queued, not yet built):
  1. **Lazy block reservation** (vLLM-style) + mid-flight preemption as the
     exhaustion response — replaces/augments all-or-nothing `reserve()`.
  2. **CPU swap branch** of vLLM's exhaustion policy (we only recompute).
  3. **Cache-aware batch ordering** (SGLang's scheduling policy) on top of our
     retention cache — route shared-prefix requests to maximize reuse.
  4. GPU phase: tile-wise FA rescale (not token-wise), FlashDecoding++-style
     unified max to kill cross-block sync.
