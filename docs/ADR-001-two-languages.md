# ADR-001: two languages — Rust core, Go plane

Status: accepted (2026-10-29)

## Context

The company computer (docs/VISION.md) has two halves with opposite
requirements: a latency-deterministic serving core, and a churn-heavy
control/API plane. This session's experiments settled the language
questions with data:

- The hand-written AVX2 dot (docs/DOTFMA.md) LOST 2x to LLVM's
  auto-vectorized scalar loop — the core's speed lives in Rust's codegen,
  in loops a GC'd language with a weaker float pipeline would lose.
- The 2.1GB-flat loader and byte-exact registry accounting
  (docs/WEIGHTS16.md, docs/LIFECYCLE.md) are load-bearing on a host that
  OOM-kills the VM; Go's allocator obscures exactly those numbers.
- "Polite coexistence" is the product thesis — a runtime with stop-the-world
  pauses cannot credibly sell predictable latency on a shared machine.
- The BPF/foresight client is kernel-adjacent (no_std-capable Rust).

Meanwhile the plane is I/O-bound, connection-heavy, requirement-churny work:
HTTP servers, streaming, fan-out, admin endpoints — Go's stdlib and
goroutine scheduler are the mature answer, and Go's compile speed removes
the iteration tax that cost this project hours (40+ build cycles at
1.5-4 min each in Rust/WSL).

## Decision

**Two languages, process boundary:**

- **Rust** (`crates/inferno-core`, `inferno-model`): kernels, paged KV,
  batcher, registry, the engine HTTP service. Everything in the hot path.
  std-only stays.
- **Go** (`plane/`): the control plane — OpenAI-compatible API, SSE
  streaming, model lifecycle admin endpoints, metrics, and later the fleet
  agent. New Go module, no cgo, no FFI.

## Boundary

The plane is a separate daemon speaking HTTP to the Rust engine service.
No FFI, no shared memory. Rationale: crash isolation (a plane bug cannot
take down serving), no unsafe cross-language coupling, and it composes with
the fleet-agent future. The hot path never crosses the boundary.

## Consequences

- Two toolchains in the repo (documented; `cargo` for the core, `go` for
  the plane — Go pinned by `plane/go.mod`).
- The plane's contract (engine HTTP API) is versioned like any interface;
  it lives in `docs/ENGINE-API.md` when written.
- Performance work stays in Rust; the plane's success metric is I/O
  throughput and developer velocity, not kernel latency.
- If the plane ever needs kernel-level work (unlikely — it's I/O), that
  piece moves INTO the Rust core rather than reaching across the boundary.

## Rejected alternatives

- **All-Go**: loses the determinism/codegen/memory-exactness the core is
  sold on.
- **All-Rust with an HTTP framework**: reintroduces the dependency weight
  and iteration tax for the layer that changes most; the std-only
  constraint also fights it.
- **cgo/FFI or a shared cdylib**: unsafe coupling across the exact boundary
  where the safety story matters, for latency the boundary's traffic
  pattern doesn't need.
