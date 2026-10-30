# Study: JustVugg/colibri (2026-10-30)

45k stars, Apache-2.0, 3,738 commits, 13 engines, one C file per model family,
no deps, no GPU required. Tagline: **"Run frontier MoE models on hardware you
already own."** Same thesis as ours, proven at a scale we have not reached.

## What it actually is

- **Pure C engines** (one per model family) over shared headers, no BLAS, no
  Python at runtime; CPU (OpenMP), Vulkan (any GPU), CUDA, Metal backends.
- **MoE streaming as the core trick**: a 744B GLM-5.2 model activates ~40B per
  token and only ~11 GB of that changes token to token (the routed experts).
  So the dense part stays in RAM and the experts stream from NVMe through a
  per-layer LRU cache plus pins learned from your own sessions. A 2.8T Kimi
  K3 runs on a 32 GB machine this way.
- **Placement tiers** (VRAM / RAM / NVMe) decided by measured routing heat,
  with the invariant: *placement decides speed, never which weights or which
  router decisions fire*.
- **OpenAI + Anthropic compatible API**, tool calling, multi-conversation KV
  slots, image in/out, and `POST /v1/systemone` — a *decision with a
  probability* (choice/noul/score; nothing generated, so answers stay inside
  your allowed set).
- **Install is the front door**: one-step setup detects the machine, recommends
  a model from a RAM/disk table, builds or fetches the engine, downloads the
  model with resume, starts a dashboard. AI-assistant setup doc + an MCP
  server (`coli mcp`).
- **Honest numbers**: Qwen3.6-35B-A3B = 6.0 tok/s CPU, 9.9 tok/s Vulkan on an
  integrated Radeon 780M, 30 tok/s CUDA (3070); GLM-5.2 744B = 1.83 tok/s on a
  128 GB Ryzen AI Max+ 395, 9.0-9.2 on 6x RTX 5090, 0.05-0.1 on a 25 GB
  laptop. Every number carries its machine and settings.
- **Measurement culture at or above ours**: CI checks every engine against its
  model's reference implementation; quality measured alongside speed; "nothing
  is adopted because a microbenchmark looks fast"; the open-research table
  publishes hypotheses, evidence so far, and the experiments still needed —
  including an explicit invitation to publish negative results.

## What it proves for our thesis

1. **The market is proven.** 45k stars and an active issue/PR cadence on a
   thesis identical to the VISION (local models on owned hardware) removes the
   "does anyone want this" question. Our consumer wedge is not speculative.
2. **MoE is the tier that matters.** The useful local model is a 35B-A3B MoE
   (23 GB download, 10-20 GB resident, 6-10 tok/s on consumer silicon), not a
  0.5B dense toy. Our engine currently serves the toy.
3. **Placement is a first-class discipline** and their invariant ("placement
   never changes output") is the same spirit as our bit-identity gates — a
   different axis, converging answer.
4. **The install experience IS the product** for consumers. Their one-step
   setup is the single biggest gap between our working runtime and a thing a
   household can run.
5. **Most business value is decisions, not chat.** System One (score against
   allowed answers, return probabilities) is a natural fit for a greedy engine
   whose logits already contain the distribution — a cheap, high-value API we
   do not have.

## Where we are differentiated (or should be)

- **They do not schedule for shared machines.** colibri takes the machine and
  runs. Our two-level co-design (batcher + foresight kernel) is exactly the
  coexistence they do not attempt — the "gaming PC that also does AI" layer.
- **Lifecycle + retention**: registry LRU/eviction/pins (theirs is per-layer
  expert cache only), all-or-nothing admission, loud failures. Ours is more
  serving-shaped; theirs is more engine-shaped.
- **Honest measurement**: comparable discipline. Their CI reference-check per
  engine is a practice worth adopting outright.

## The honest scoreboard

Their 35B MoE decodes 6-9.9 tok/s on consumer silicon; our 0.5B dense decodes
~1.36 tok/s on a laptop VM. Throughput headroom is theirs by 4-20x *on larger
models*. The gap is not shame; it is the roadmap.

## Adopted into GAPS.md (new items)

- **G18 MoE expert streaming** — the model tier the product actually needs;
  their per-layer LRU + learned pins is the reference design, and our
  placement invariant becomes a gate ("routing decisions identical across
  tiers", which is their rule and ours).
- **G19 Placement tiers** — NVMe/RAM/VRAM residency with byte accounting per
  model (extends the registry we have).
- **G20 Decision API** (`/v1/systemone`-style scoring) — native fit for our
  argmax/logits path.
- **G21 CI reference checks** — every engine verified against its reference
  implementation in CI.
- **G22 One-step consumer setup** — hardware detect, model recommendation,
  download+resume, dashboard; AI-assistant setup doc and an MCP server are
  ecosystem plays for the fleet layer.
- **G23 Anthropic-compatible surface** — Claude Code and the Anthropic SDKs
  are the highest-value client population for a local endpoint.

## Strategic read

colibri validates the thesis and has solved the parts we have not (MoE
streaming, install-as-product). We have the part it has not (coexistence
scheduling, lifecycle, retention, gate discipline). The union is the company
computer: their engine class on our serving substrate. Short term, their repo
is the most useful reference implementation in the space — and the benchmark
our 0.5B-v1 numbers should eventually be quoted beside.
