# GAPS — what is missing, in dependency order

*Standing list. Each item: what / why it blocks the vision (docs/VISION.md) /
size / blocked-by. Written 2026-10-30. Sizes: S = days, M = weeks, L = months.*

## Tier 1 — the consumer thesis does not exist until these do

**G1. GPU inference kernels (CUDA) on consumer cards.**
Per-layer Qwen-family kernels over safetensors on 8–12 GB cards; same Engine
trait, dispatch by device. Everything measured so far is CPU — this is the bet
the product stands on, not an optimization. *L. Blocks G2, G4, and the demo.*

**G2. VRAM handoff.**
ModelRegistry extended with VRAM-pressure signals (cuda OOM / explicit
game-mode event / nvidia-smi poll): model swaps to system RAM, returns when
the card is free. The headline feature of the vision doc; today the registry
budgets RAM only. *M. Blocked by G1.*

**G3. Game-mode control (user-facing).**
"Leave the machine to the game" / "AI may use the idle machine" — the
foresight knobs exposed as one honest switch, plus a tray indicator.
*S/M. Blocked by nothing (kernel client exists).*

**G4. The coexistence proof (demo-grade measurement).**
Frame-time histogram of a real game under: baseline / CPU-AI load / GPU-AI
load / mid-game model swap. Pre-registered protocol, published raw. Without
this the claim is marketing; with it, this is the whole funnel. *M. Blocked
by G1.*

## Tier 2 — a household can install and use it

**G5. Installer + service packaging.**
One command (curl | bash or single binary), systemd unit, driver detection,
first-run wizard picks a model by VRAM tier. No manual cargo. *S.*

**G6. A UI beyond curl.** The Go plane serves one page: chat window, model
manager, the G3 toggle, VRAM/status readout. Desktop-shell optional later.
*M. Blocked by nothing.*

**G7. Model catalog + one-click download.** HF hub integration, 3–4B and
7–8B presets with VRAM guidance, checksum-verified (the loader's rule).
*M.*

**G8. Privacy made structural.** No telemetry by default, no network calls
except explicit downloads, a visible "nothing leaves this machine" state.
*S, but designed in, not bolted on.*

## Tier 3 — it becomes a product you can sell

**G9. Licensing + entitlement.** One-time key, offline verification, N-machine
household terms. *S/M.*

**G10. Update channel.** Signed releases, engine + models updated separately.
*S.*

**G11. Opt-in crash/usage telemetry.** Off by default (G8), structured when on.
*S.*

## Tier 4 — the platform layer

**G12. The model store.** NVMe cache tier (in-box or small appliance) making
swap/eviction instant; integrates with the registry as a storage tier below
RAM. *M (hardware + registry).*

**G13. Serving concurrency.** The engine's state mutex serializes requests
(honest v1, documented); real concurrent serving needs the batcher driving
multiple sequences across models. *M.*

**G14. Batched decode gemm.** One matrix-multiply serves N concurrent decodes
(pre-registered concept in WEIGHTS16.md) — where concurrency becomes
profitable. *M. Blocked by G13.*

**G15. Other model families.** The loader hardcodes Qwen2 geometry; add
Llama/Mistral/etc. as geometry profiles + family kernels. *M each.*

**G16. Fleet layer.** Household sync (many machines, one model store), then
teams. *L.*

**G17. OEM certification tooling.** Reference-BOM validation harness that
prebuilts can run to earn "runs inferno" marks. *M.*

## Tier 1.5 - adopted from the colibri study (docs/STUDY-colibri.md)

**G18. MoE expert streaming.** The useful local model is a 35B-A3B class MoE
(23 GB download, 10-20 GB resident, 6-10 tok/s on consumer silicon per
colibri's published numbers) - not our 0.5B dense. Stream routed experts from
NVMe through a per-layer LRU + session-learned pins. Their invariant becomes
our gate: *placement decides speed, never which weights or router decisions
fire* (verified by identical outputs across tiers). *L. Blocked by G1.*

**G19. Placement tiers** - NVMe/RAM/VRAM residency with byte accounting per
model (extends the ModelRegistry). *M. Blocked by G18.*

**G20. Decision API** (`/v1/systemone`-style: score a state against allowed
answers, return probabilities + confidence, generate nothing). Native fit for
our greedy engine - the logits ARE the distribution. *S/M.*

**G21. CI reference checks** - every engine verified against its model's
reference implementation in CI (colibri does this; we do it by hand today). *S.*

**G22. One-step consumer setup** - hardware detect, model recommendation from
the RAM/disk tables, download with resume, dashboard; AI-assistant setup doc
and an MCP server are ecosystem plays for the fleet layer. *M. Includes G5.*

**G23. Anthropic-compatible surface** - Claude Code and the Anthropic SDKs are
the highest-value client population for a local endpoint. *S.*## Known debts (small, in-repo now)
- Real two-different-dirs multi-model test (same-weights covered the routing).
- `out.prefill_tokens` metric never incremented (reporting only).
- Mid-decode stash staleness panic on preemption (v1 loud failure; queued
  re-reserve fix).
- README quick-start still says the load bench without models registered.

## Order of attack (what unblocks what)

G1 → G2 → G4 → G3 — the thesis and its proof. Then G5–G8 (the household
sale), G9–G11 (charging), and Tier 4 in whatever order the first customers
pull. G13/G14 are the speed story; G15 is breadth; G16/G17 are the company.
