# VISION — the multi-purpose machine

*North star for every decision in this repository. Rewritten 2026-10-30 around
the consumer thesis (the original enterprise framing survives below as phase 3).*

## The product, in one sentence

**The gaming PC, the living-room box, the family computer — stays exactly what
it was bought for, and runs local, private AI alongside it.** The software
schedules the second job around the first so neither degrades: the game is
never starved, and the machine does meaningful AI work whenever the game
isn't using it — no manual juggling, no second computer.

## Why now: the opening

NVIDIA is actively decommissioning consumer cards for AI (inference nerfs,
supply steered to datacenter). The installed base of gaming GPUs is the
world's largest AI-capable fleet, and the vendors' answer to "I want AI on my
own machine" is *buy a different computer* or *rent it in the cloud*. Our
answer: **the one you already own, truer to its original purpose than before** —
the AI yields to the game, hands the VRAM back on demand, and comes back when
the machine is free.

## The enabling mechanism (already built and measured)

Two-level contract — the kernel level and the token level:

- **Kernel level** (foresight-sched, this repo's client): the game is
  classified interactive and protected; local AI is batch and takes the
  leftover slices. The human's experience is the OS's job, not the game's.
- **Token level** (batcher + registry): AI work is bursty at human
  timescales (a question, an answer, ten idle minutes), so it can yield
  instantly and re-plan around the machine's real state.

Plus the feature that makes it a product rather than a demo: **VRAM
handoff.** When a AAA game launches, the AI model swaps off the GPU to
system RAM (registry LRU eviction, extended to VRAM pressure signals); the
game gets the full card; on exit the model returns. The user does nothing.

## What the machine is for

- Local private AI: chat, coding help, RAG over the family's own documents,
  transcription, image work — no subscription, no data leaving the house.
- The games and the media, undegraded, always first.
- Model tier follows the card: 8 GB → 3–4B class, 12 GB → 7–8B class
  (the sweet spot is the 3060-12GB / 4070 tier NVIDIA is leaving behind).

## The honest limits (stated so nobody over-claims)

- **Continuous AAA gaming + heavy continuous AI on one mid-range GPU will
  compromise something.** We claim clean *alternation* (game never
  degraded; AI gets the full machine when idle), not free parallelism at
  full tilt.
- **Capacity is the card's VRAM.** That is a tiering fact, not a blocker.
- **The GPU layer is the critical path for this vision.** Everything
  measured so far is CPU inference (which proves the software thesis); the
  GPU kernels + VRAM handoff are what make it real on the target machine.

## Hardware position (decided — do not re-litigate per sprint)

**We are a software company; we do not build the computer.**

- Building boxes makes us a PC vendor — thin margins, inventory, a support
  organization — and drags attention off the moat (scheduling + VRAM
  handoff + honest failure).
- Instead: **reference BOM + certification.** Publish known-good builds
  ("this card tier + this RAM + our software = this model tier under your
  games"), let OEMs/assemblers fulfill them, and sell the software layer.
  (System76/pattern: the brand and the software carry the margin.)
- **The one component worth owning: the model store.** An NVMe cache tier
  (in-box or a small NAS) that makes model swap/eviction instant. It only
  makes sense inside our product, extends the registry's job, and is cheap
  to build. If we ever ship an appliance, this is it.
- **Guidance for the reference tier**: 12 GB VRAM cards first.

## Product layers (consumer first, enterprise second)

```
 ┌────────────────────────────────────────────┐
 │ Fleet layer      (households, then teams)  │  ← later: model sync, family sharing
 ├────────────────────────────────────────────┤
 │ Model store + certification                │  ← the owned component + BOM guidance
 ├────────────────────────────────────────────┤
 │ VRAM handoff + registry (lifecycle under   │  ← the headline feature
 │   VRAM pressure; LRU; pins)                │
 ├────────────────────────────────────────────┤
 │ coherent serving runtime (inferno: this repo)│
 ├────────────────────────────────────────────┤
 │ GPU kernels (CUDA) + foresight co-design   │  ← the critical path
 └────────────────────────────────────────────┘
```

## Roadmap

1. **GPU kernel phase** (CUDA inference on consumer cards) — the bet the
   whole thesis rests on. Pre-registered before implementation.
2. **VRAM handoff**: VRAM-pressure signals wired into ModelRegistry
   eviction (extend today's RAM-budget registry); model store design.
3. **Game-mode UX**: one command ("leave the machine to the game" / "AI may
   use the idle machine") — the foresight knobs exposed honestly.
4. **Certification + reference BOM**: the 12 GB tier first.
5. **Fleet layer**: households first (many machines, one model store), then
   teams — the enterprise "company computer" thesis, unchanged, second.

## How this document is used

Every feature, benchmark, hire, and partnership gets checked against it: does
it make an ordinary machine do both jobs better? Kernel work that only wins a
benchmark nobody runs on a gaming PC does not. The measurement discipline in
docs/ (pre-register, paired runs, publish failures) is part of the product:
consumers forgive bugs, not betrayal.
