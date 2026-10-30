// inferno-plane — the company computer's control surface (ADR-001:
// Go plane over the Rust engine core, process boundary, stdlib only).
//
// Endpoints:
//   GET  /healthz                    plane health + build info
//   GET  /v1/models                  models the engine knows (registry)
//   POST /v1/models/{name}/load      load a model into the registry
//   POST /v1/chat/completions        OpenAI-shaped chat, routed by model;
//                                    stream:true proxies SSE incrementally
//
// House rule enforced here: every boundary failure is loud, typed, and
// machine-readable (docs/PLANE.md gate 3) — a dead engine yields 503
// engine_unreachable, never a panic or a hang.
package main

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"io"
	"log"
	"net/http"
	"os"
	"os/signal"
	"syscall"
	"time"
)

// Config from the environment (12-factor: the plane is a daemon).
type Config struct {
	Addr   string // listen address
	Engine string // engine base URL
}

func loadConfig() Config {
	addr := os.Getenv("INFERNO_PLANE_ADDR")
	if addr == "" {
		addr = "127.0.0.1:8080"
	}
	engine := os.Getenv("INFERNO_ENGINE_URL")
	if engine == "" {
		engine = "http://127.0.0.1:8081"
	}
	return Config{Addr: addr, Engine: engine}
}

// Plane is the control-plane server.
type Plane struct {
	cfg    Config
	engine *EngineClient
}

func (p *Plane) routes() *http.ServeMux {
	mux := http.NewServeMux()
	mux.HandleFunc("GET /healthz", p.healthz)
	mux.HandleFunc("GET /v1/models", p.proxyModels)
	mux.HandleFunc("POST /v1/models/{name}/load", p.loadModel)
	mux.HandleFunc("POST /v1/chat/completions", p.chatCompletions)
	mux.HandleFunc("POST /v1/systemone", p.systemOne)
	return mux
}

func (p *Plane) healthz(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(map[string]any{
		"status":  "ok",
		"service": "inferno-plane",
		"engine":  p.engine.base.String(),
	})
}

// proxyModels passes the engine's registry listing through unchanged.
func (p *Plane) proxyModels(w http.ResponseWriter, r *http.Request) {
	resp, err := p.engine.send(http.MethodGet, "/v1/models", nil, "")
	if err != nil {
		engineUnreachable(w, p.engine, err)
		return
	}
	defer resp.Body.Close()
	copyResponse(w, resp, false)
}

// loadModel asks the engine to load a registered model (lifecycle admin).
func (p *Plane) loadModel(w http.ResponseWriter, r *http.Request) {
	name := r.PathValue("name")
	if name == "" {
		writeJSONError(w, http.StatusBadRequest, "invalid_request", "model name required")
		return
	}
	resp, err := p.engine.send(http.MethodPost, "/v1/models/"+name+"/load", nil, "")
	if err != nil {
		engineUnreachable(w, p.engine, err)
		return
	}
	defer resp.Body.Close()
	copyResponse(w, resp, false)
}

// chatRequest is the OpenAI-compatible subset the plane routes by.
type chatRequest struct {
	Model     string            `json:"model"`
	Messages  []json.RawMessage `json:"messages,omitempty"`
	Prompt    string            `json:"prompt,omitempty"`
	MaxTokens int               `json:"max_tokens,omitempty"`
	Stream    bool              `json:"stream,omitempty"`
}

// chatCompletions validates the OpenAI shape, then proxies. Validation is
// the plane's job (gate 4: malformed -> 400, not 500); generation is the
// engine's.
func (p *Plane) chatCompletions(w http.ResponseWriter, r *http.Request) {
	body, err := io.ReadAll(io.LimitReader(r.Body, 1<<20))
	if err != nil {
		writeJSONError(w, http.StatusBadRequest, "invalid_request", "unreadable body")
		return
	}
	var req chatRequest
	if err := json.Unmarshal(body, &req); err != nil {
		writeJSONError(w, http.StatusBadRequest, "invalid_request", "body is not valid JSON: "+err.Error())
		return
	}
	if req.Model == "" {
		writeJSONError(w, http.StatusBadRequest, "invalid_request", "field 'model' is required")
		return
	}
	if len(req.Messages) == 0 && req.Prompt == "" {
		writeJSONError(w, http.StatusBadRequest, "invalid_request", "one of 'messages' or 'prompt' is required")
		return
	}
	if req.MaxTokens < 0 {
		writeJSONError(w, http.StatusBadRequest, "invalid_request", "field 'max_tokens' must be >= 0")
		return
	}
	resp, err := p.engine.send(http.MethodPost, "/v1/chat/completions", bytes.NewReader(body), "application/json")
	if err != nil {
		engineUnreachable(w, p.engine, err)
		return
	}
	defer resp.Body.Close()
	// Streaming: flush each chunk as it arrives (OpenAI SSE contract).
	copyResponse(w, resp, req.Stream)
}

// engineUnreachable maps a transport failure to the typed 503.
func engineUnreachable(w http.ResponseWriter, c *EngineClient, err error) {
	if c.Unreachable(err) {
		writeJSONError(w, http.StatusServiceUnavailable, "engine_unreachable",
			"inferno engine service is unreachable at "+c.base.String())
		return
	}
	writeJSONError(w, http.StatusBadGateway, "engine_transport", err.Error())
}

// copyResponse proxies an engine response; when streaming, flush per chunk.
func copyResponse(w http.ResponseWriter, resp *http.Response, flushPerWrite bool) {
	for k, vs := range resp.Header {
		for _, v := range vs {
			w.Header().Add(k, v)
		}
	}
	if w.Header().Get("Content-Type") == "" {
		w.Header().Set("Content-Type", "application/json")
	}
	w.WriteHeader(resp.StatusCode)
	if !flushPerWrite {
		_, _ = io.Copy(w, resp.Body)
		return
	}
	flusher, ok := w.(http.Flusher)
	buf := make([]byte, 4096)
	for {
		n, err := resp.Body.Read(buf)
		if n > 0 {
			_, _ = w.Write(buf[:n])
			if ok {
				flusher.Flush()
			}
		}
		if err != nil {
			if !errors.Is(err, io.EOF) {
				log.Printf("stream from engine ended with: %v", err)
			}
			return
		}
	}
}

// systemOneRequest is the decision API's shape (docs/DECISION-API.md).
type systemOneRequest struct {
	State     string            `json:"state"`
	Questions map[string]json.RawMessage `json:"questions"`
}

// systemOne validates the decision shape (state + questions), then proxies.
// The engine does the typed per-question validation; the plane only guards
// the boundary (gate: malformed -> 400, never 500).
func (p *Plane) systemOne(w http.ResponseWriter, r *http.Request) {
	body, err := io.ReadAll(io.LimitReader(r.Body, 8<<20))
	if err != nil {
		writeJSONError(w, http.StatusBadRequest, "invalid_request", "unreadable body")
		return
	}
	var req systemOneRequest
	if err := json.Unmarshal(body, &req); err != nil {
		writeJSONError(w, http.StatusBadRequest, "invalid_request", "body is not valid JSON: "+err.Error())
		return
	}
	if req.State == "" {
		writeJSONError(w, http.StatusBadRequest, "invalid_request", "field 'state' is required")
		return
	}
	if len(req.Questions) == 0 {
		writeJSONError(w, http.StatusBadRequest, "invalid_request", "field 'questions' is required and non-empty")
		return
	}
	resp, err := p.engine.send(http.MethodPost, "/v1/systemone", bytes.NewReader(body), "application/json")
	if err != nil {
		engineUnreachable(w, p.engine, err)
		return
	}
	defer resp.Body.Close()
	copyResponse(w, resp, false)
}

func main() {
	cfg := loadConfig()
	engine, err := NewEngineClient(cfg.Engine)
	if err != nil {
		log.Fatalf("plane config: %v", err)
	}
	p := &Plane{cfg: cfg, engine: engine}

	srv := &http.Server{
		Addr:              cfg.Addr,
		Handler:           p.routes(),
		ReadHeaderTimeout: 10 * time.Second,
	}

	// Graceful shutdown (gate 5): drain in-flight requests, then exit.
	ctx, stop := signal.NotifyContext(context.Background(), syscall.SIGINT, syscall.SIGTERM)
	defer stop()

	go func() {
		log.Printf("inferno-plane listening on %s, engine %s", cfg.Addr, cfg.Engine)
		if err := srv.ListenAndServe(); err != nil && !errors.Is(err, http.ErrServerClosed) {
			log.Fatalf("listen: %v", err)
		}
	}()

	<-ctx.Done()
	log.Printf("shutting down; draining in-flight requests")
	shutdownCtx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()
	if err := srv.Shutdown(shutdownCtx); err != nil {
		log.Printf("shutdown: %v", err)
		os.Exit(1)
	}
	log.Printf("bye")
}
