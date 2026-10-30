# G20: decision API — pre-registered before implementation

## Claim under test

An endpoint that answers *decisions, not paragraphs*: given a state and typed
questions, return for each question the chosen label, a probability per
allowed answer, and a confidence — nothing generated, so no answer can fall
outside the allowed set. This is the colibri study's clearest product insight
(most business value from small/local models is classification and routing),
and it is a native fit for our greedy engine: the logits already ARE the
distribution.

## Gates (fixed before measurement)

1. **Hermetic correctness** (tiny engine): scoring a continuation that
   follows from a state must score HIGHER than a random continuation;
   per-question probabilities softmax-normalize to 1; confidence in [0, 1];
   deterministic (same input -> identical output, every run).
2. **Live**: the real 0.5B answers a real decision (e.g. "review this diff:
   merge / request changes / close") with a sane distribution, over HTTP,
   through the Go plane.
3. **No regression**: the OpenAI-compatible paths and all existing tests
   stay green.
4. **Typed failures**: unknown question type / empty criteria / unknown
   model -> typed JSON errors, never a panic, never a partial answer.

## API shape (colibri-compatible — that is the point)

`POST /v1/systemone`:

```json
{
  "state": "340 lines, 8 files, no tests. CI is green but nothing covers that path.",
  "questions": {
    "review": {"type": "choice", "instructions": "What should the reviewer do?",
               "criteria": {"merge": null, "request changes": null, "close": null}},
    "risky":  {"type": "noul", "instructions": "Is this change risky?"}
  }
}
```

Reply per question: `choice` -> `{"choice": label, "probabilities": {...},
"confidence": 0..1}`; `noul` -> `{"probability": p_yes, "confidence": ...}`.

## Scope of this slice

- Scoring = sum of per-token log-softmax of the option's tokens given the
  state (teacher forcing through `forward_token_pub`/`logits_of_pub`, the
  same public surface the gates already use). Simple, honest, ours.
- `choice` and `noul` types; `score` type deferred (documented, not faked).
- Per-question KV isolation: each (state + option) scores in its own sequence
  region; the state is re-encoded per option (documented: prompt-prefix
  caching across options is the next optimization, not this slice).
- Confidence = 1 - H(p)/log(k) (0 = flat, 1 = certain).

## Status — ALL 4 GATES PASS (2026-10-30)

- [x] Pre-registered written
- [x] Implemented: `crates/inferno-serve/src/decision.rs` (teacher-forced
  log-softmax scoring through the engine's public forward surface, per-
  candidate KV isolation, softmax + 1-H/ln(k) confidence), the
  `/v1/systemone` handler (choice + noul; typed validation; tokenizer-less
  models refused with a typed 400), and the Go plane's route.
- [x] **Gate 1** — hermetic: 16/16 serve tests (determinism, finiteness,
  normalization to 1, confidence in [0,1], empty-candidate refusal, and
  every validation path typed).
- [x] **Gate 2** — live on the real 0.5B, direct and through the plane:
  "What should the reviewer do?" -> merge 0.837 / close 0.158 /
  request changes 0.005 (confidence 0.574); "Is this change risky?" ->
  yes 0.734 (confidence 0.164). Determinism: identical modulo id.
- [x] **Gate 3** — no regression: 60/60 Rust workspace + Go suite green.
- [x] **Gate 4** — typed failures: every malformed input path returns a
  typed 400/404, including the tokenizer-less-engine case (never a panic).

## Honest observations

- **Judgment quality is a model-capability question, not an API-correctness
  one.** The 0.5B reads "CI is green" and answers merge 0.837 for an
  untested 340-line PR — a human reviewer disagrees, but the distribution
  is well-formed and the API is doing exactly what it claims: reporting
  what the model actually believes. Chat templates + a bigger model are the
  fix, and both are roadmap items.
- **Latency**: ~74s for 5 candidate scorings on the CPU engine, because
  the state is re-encoded and re-forwarded per candidate. Shared-prefix
  scoring (encode once, branch at the answer) is the documented next
  optimization — with it, an N-option question costs ~1 state + N short
  continuations instead of N full sequences.
- The scoring prompt shape (`state\ninstructions\nAnswer: X`) is part of
  the contract; changing it changes the numbers (documented so nobody
  "improves" it silently).
