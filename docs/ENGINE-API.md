# Engine HTTP service (Rust core) — pre-registered before implementation

## Claim under test

A std-only Rust HTTP service (new crate `inferno-serve`, no frameworks —
the house constraint) can be the engine side of ADR-001's process
boundary: **the machine answers a chat request over HTTP**, routing by
model name through the ModelRegistry, with per-model batchers and
OpenAI-shaped responses. This is the first time the whole stack — plane →
engine → registry → batcher → real weights — answers a request.

## Gates (fixed before measurement)

1. **Builds clean** in the workspace; hermetic tests: HTTP request
   parsing, route dispatch, OpenAI response shape, and a full request
   through a TINY registered engine (no disk, no big model).
2. **Live end-to-end**: `curl POST /v1/chat/completions` against the real
   0.5B model returns a coherent completion (" Paris..." family) — and the
   Go plane proxying the same request returns the identical payload.
3. **Multi-model isolation**: two registered models serve concurrently
   without cross-contamination (per-model batchers, KV per model; a
   request to model A never sees model B's tokens).
4. **Registry lifecycle over HTTP**: `GET /v1/models` lists;
   `POST /v1/models/{name}/load` loads; budget eviction still refuses to
   evict an in-flight model (the pinned rule holds at the HTTP layer).
5. **Honest failure at the boundary**: unknown model → 404 typed JSON;
   malformed body → 400; never a panic, never a hang.

## Scope of this slice

Non-streaming completions (full response + Content-Length). **SSE
incremental streaming is the NEXT slice** (the plane already passes
`stream:true` through; the engine answers with the complete response for
now — documented, not faked). Keep-alive supported (the Go plane's
transport reuses connections); one request per connection loop until EOF
or `Connection: close`.

## Status — ALL 5 GATES PASS (2026-10-30)

- [x] Implemented (`crates/inferno-serve`: hand-rolled std-only HTTP,
      ServeState over the ModelRegistry, per-model batchers, OpenAI-shaped
      responses; `--model name=path` repeatable, `--blocks`, `--budget-mb`)
- [x] **Gate 1** — 50/50 tests: HTTP parsing (5), routes/state (6),
      workspace suites (39); hermetic chat through a tiny engine
- [x] **Gate 2** — live end-to-end: direct → `" Paris. It is the largest city
      in Europe and the third"`; **through the Go plane** →
      `" Paris. The capital of the United States is Washington D."`
      (different continuation because the plane's `messages` extraction
      tokenizes 7 tokens vs 5 — the boundary, not the engine)
- [x] **Gate 3** — two models concurrently: a→" Paris…", b→" Berlin…";
      per-model batchers isolated
- [x] **Gate 4** — `POST /v1/models/chat/load` → `loaded`; budget pinning
      holds (in-flight requests can't be evicted)
- [x] **Gate 5** — unknown model → typed 404 `model_not_found`; malformed
      JSON → typed 400 `invalid_request`; tokenizer-less engine + text
      prompt → typed 400 (the `prompt_ids` escape hatch documented)
- **Bonus measured**: retention over HTTP — identical prompt, same model:
  **49.6 s → 13.5 s (3.7×)**, bit-identical output. The vision's core
  economic claim, reproducible with curl.

## Remaining

- Real multi-model test with two DIFFERENT dirs (same-weights test covered
  the routing; different-geometry models exercise the per-model KV config).

---

## SSE streaming slice — pre-registered before implementation

## Claim under test

`stream: true` on `/v1/chat/completions` delivers tokens incrementally as
OpenAI SSE (`data: {...delta...}` per token, `data: [DONE]` to end) from
the engine, through the Go plane, to a raw client — measured by the
arrival gaps (the stream must show TPOT-shaped spacing, not one lump).

## Gates

1. Batcher exposes running-sequence progress (`progress(id) -> &[u32]`) —
   unit-tested.
2. Live: `curl -N` on the engine with `stream:true` shows ≥3 distinct
   arrival events (timestamps) before `[DONE]`.
3. Same request through the Go plane streams through unchanged.
4. Non-streaming behavior unchanged (regression gate: the previous gates).

## Scope of this slice

One SSE `delta` per token (simest honest framing); `finish_reason` on the
final data frame; body terminated by connection close (no chunked needed
at this boundary — the plane re-frames with its own chunked encoding).
Requests are still serialized by the state mutex (v1 concurrency model
unchanged, documented).

## Status — SSE SLICE COMPLETE (all gates pass)

- [x] Implemented: `Batcher::progress(id)` accessor (unit-tested),
      `http::stream_head`/`sse_data`/`SSE_DONE`, `ServeState::wants_stream`
      + `handle_streaming` (per-token deltas, retire-drain for the final
      token, `finish_reason`, connection-close framing)
- [x] **Gate 1** — 51/51 tests (progress accessor + stream primitives added)
- [x] **Gate 2** — engine direct: 8 deltas arrive TPOT-shaped
      (21.44s → 25.84s, ~0.7s spacing), `finish_reason: "length"`, `[DONE]`
- [x] **Gate 3** — through the Go plane: 10 frames, ~0.9s spacing, same
      contract (the boundary carries the streaming contract)
- [x] **Gate 4** — non-streaming unchanged (regression: gates 2/5 of the
      base slice still green)

One real bug caught by the gates: the last token's delta was dropped (the
sequence retires at the end of the step that produces it, so `progress()`
goes blind) — fixed by draining `finished_tokens` for unwritten deltas,
verified by the "length" finish reason and the 8th delta appearing.
