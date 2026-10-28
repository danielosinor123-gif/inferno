# VISION — the company computer

*North star for every decision in this repository. Written 2026-10-28.*

## The product, in one sentence

**A multi-purpose company computer that is also the company's AI machine** — a
general-purpose machine that runs the workloads companies already run, and
*natively* runs the company's AI privately, without a second infrastructure.

## The category

NVIDIA's answer to "we need AI" is: *buy a DGX appliance, keep your normal
computers, and shuttle your data to the box.* Our answer is: **your computers
should be the AI machine.** One fleet, both jobs.

We are not selling inference software and we are not selling a faster
inference box. We are selling the machine class that makes private AI an
*infrastructure* decision instead of a data-center project.

## Why nobody has built this (and the moat)

AI software assumes it owns the machine. Put it on a computer that also does
real work and either the AI ruins the machine or the machine ruins the AI —
which is why every company that wants AI ends up with a dedicated room, and
why the current category is "AI appliances you buy in addition to everything
you already own."

The unlock is **coexistence, engineered at two levels**:

- **Token level** (userspace batcher): schedules tokens, KV blocks, prefill and
  decode, admission — knows what AI work costs.
- **CPU level** (kernel co-design via foresight-sched): the batch load declares
  itself as batch work, gets scheduled in the spare slices, and yields the
  instant interactive work needs the machine.

One machine, both loads, no tug-of-war. Anyone can copy a kernel; a *machine
class* that demonstrably carries enterprise workload and AI on one box is a
category, not a feature.

### Evidence we already have (measured, not claimed)

- 0 preemptions and correct state under real host pressure — the host itself
  was being OOM-killed by Windows mid-benchmark and the serving loop held.
- 28× first-token speedup on repeated-prefix workloads (chat, templates,
  multi-turn documents) — the dominant pattern of real business use.
- 2-way concurrent requests at ~zero marginal cost (aggregate throughput
  unchanged).
- Bit-identical deterministic output across engine changes (threading, dtype
  storage) and across cold vs cached paths.
- Loud failure under impossible conditions instead of silent corruption —
  the property regulated buyers pay for.
- A real model + paged KV in ~1 GB of memory.

## What we do NOT claim

Raw per-token compute against dedicated GPU fleets: we measured ourselves
25–70× slower than cloud-class hardware per token and that race stays
intentionally unwinnable. The product thesis is *where the dedicated box
structurally cannot go*: inside the building, on owned hardware, shared with
everything else the machine already does.

## Product layers

```
 ┌─────────────────────────────────────────────┐
 │  Vertical solutions   (regulated industries)│  ← later: compliance-certified
 ├─────────────────────────────────────────────┤
 │  Fleet platform       (management, telemetry)│  ← recurring revenue
 ├─────────────────────────────────────────────┤
 │  Admin plane          (one screen: models,   │  ← the IT-person surface
 │                        quotas, health)       │
 ├─────────────────────────────────────────────┤
 │  Runtime              (inferno: this repo)   │  ← the irreplaceable core
 ├─────────────────────────────────────────────┤
 │  Reference machine    (hardware integration) │  ← commodity Linux box
 └─────────────────────────────────────────────┘
```

## Roadmap (phased)

**Phase 1 — Reference machine.** Runtime (done enough to prove the category)
on commodity hardware. Demonstration: one box carrying a synthetic enterprise
load + AI serving, zero user-visible interference, under pressure. First
vertical: buyers who *cannot* use cloud AI — data residency is their purchase
trigger (healthcare, legal, finance, industrial, government).

**Phase 2 — Admin plane + model lifecycle.** Multi-model load/swap/eviction on
a machine, per-team quotas, a provisioning screen that a company IT person can
operate without a data-center team. This converts a demo into a product.

**Phase 3 — Fleet platform.** Every machine after the first needs
management, telemetry, updates. The recurring-revenue layer.

**Phase 4 — Vertical solutions.** Compliance-certified product lines built on
the fleet platform.

## Engineering priorities that follow from this vision

1. **Model lifecycle** (load/swap/evict, multi-model per machine) — the #1
   platform blocker, and the prerequisite for the admin plane.
2. **Admin plane API** — the machine must be operable by a company, not by us.
3. **Observability** — operators don't adopt what they can't see.
4. **Kernel performance on the reference hardware** — including GPU kernels
   when the reference design includes GPUs (the pre-registered threading
   path is the same work).
5. Everything above is gated by the standing rule: **loud failure, never
   silent corruption; bit-identical determinism; honest measurement with the
   failures published.**

## How this document is used

Any feature, benchmark, hire, or partnership gets checked against it: does it
make the company computer a better company computer? Kernel work that only
wins a benchmark nobody buys does not. The measurement discipline in
`docs/BENCHMARK.md` (pre-register, report all reps, publish failures) is part
of the product, not an engineering preference — regulators and procurement
teams are the audience for it.
