# Control plane (Go) — pre-registered before implementation

## Claim under test

A standalone Go daemon (`plane/`, stdlib only, no cgo — ADR-001) can be the
company computer's control surface: OpenAI-compatible chat completions
(fanned out to the Rust engine service over HTTP), model lifecycle admin
endpoints, and clean failure at the HTTP boundary.

## Gates (fixed before measurement)

1. **Builds clean**: `go build ./...` + `go vet ./...` + `go test ./...`
   under go1.22, stdlib only (no external modules).
2. **/healthz**: 200 + JSON status.
3. **Loud failure at the boundary** (house rule): engine unreachable →
   proxied endpoints return a clean **503 with a JSON error** (typed
   `engine_unreachable`), not a panic, not a hang.
4. **OpenAI shape**: `POST /v1/chat/completions` parses the standard
   request (model/messages/max_tokens/stream) and routes by model name;
   malformed bodies → 400, not 500.
5. **Graceful shutdown**: SIGINT/SIGTERM drains in-flight requests
   (http.Server.Shutdown) — no dropped connections on restart.

## Scope of this slice

The plane's skeleton: health, models proxy, chat-completions proxy +
streaming passthrough, model load admin endpoint, shutdown. The **Rust
engine HTTP service it talks to is the next pre-registered step**
(`docs/ENGINE-API.md`); until it exists, the plane's error path is the
verified behavior (gate 3 proves it against a closed port).

## Status

- [x] Pre-registered written
- [x] ADR-001 (two languages, process boundary)
- [x] go1.22 installed in the VM
- [x] Plane skeleton implemented (`plane/`, stdlib only: healthz,
      /v1/models proxy, /v1/models/{name}/load, /v1/chat/completions with
      OpenAI-shape validation + SSE passthrough, graceful shutdown)
- [x] Gates 1–5 all pass: `go build`/`go vet`/`go test` clean; healthz
      200 JSON; dead engine → typed 503 `engine_unreachable`; malformed
      request → 400 `invalid_request`; stub passthrough + [DONE] stream
      verified; SIGTERM drain wired (signal.NotifyContext + Shutdown)
- [ ] Engine HTTP service (Rust core) — next pre-registered step
      (`docs/ENGINE-API.md`): the plane has nothing real to talk to yet.
