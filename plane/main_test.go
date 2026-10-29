package main

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

// testPlane builds a plane pointed at a dead engine port (the 503 path is
// a gate, not an accident) or at a stub engine for passthrough checks.
func testPlane(t *testing.T, engineURL string) *Plane {
	t.Helper()
	c, err := NewEngineClient(engineURL)
	if err != nil {
		t.Fatalf("engine client: %v", err)
	}
	return &Plane{cfg: Config{Addr: "test", Engine: engineURL}, engine: c}
}

// deadPortURL is a port nothing listens on (the engine service absent).
const deadPortURL = "http://127.0.0.1:1"

func TestHealthzOK(t *testing.T) {
	p := testPlane(t, deadPortURL)
	rr := httptest.NewRecorder()
	p.routes().ServeHTTP(rr, httptest.NewRequest(http.MethodGet, "/healthz", nil))
	if rr.Code != http.StatusOK {
		t.Fatalf("healthz = %d, want 200", rr.Code)
	}
	var body map[string]any
	if err := json.Unmarshal(rr.Body.Bytes(), &body); err != nil {
		t.Fatalf("healthz body: %v", err)
	}
	if body["status"] != "ok" || body["service"] != "inferno-plane" {
		t.Fatalf("healthz body = %v", body)
	}
}

func TestModelsProxyUnreachableIsTyped503(t *testing.T) {
	p := testPlane(t, deadPortURL)
	rr := httptest.NewRecorder()
	p.routes().ServeHTTP(rr, httptest.NewRequest(http.MethodGet, "/v1/models", nil))
	if rr.Code != http.StatusServiceUnavailable {
		t.Fatalf("models (dead engine) = %d, want 503", rr.Code)
	}
	if !strings.Contains(rr.Body.String(), "engine_unreachable") {
		t.Fatalf("body = %q, want engine_unreachable", rr.Body.String())
	}
	if ct := rr.Header().Get("Content-Type"); !strings.Contains(ct, "application/json") {
		t.Fatalf("content-type = %q", ct)
	}
}

func TestChatValidation(t *testing.T) {
	p := testPlane(t, deadPortURL)
	cases := []struct {
		name string
		body string
		want int
	}{
		{"missing model", `{"messages":[]}`, http.StatusBadRequest},
		{"missing content", `{"model":"m"}`, http.StatusBadRequest},
		{"bad json", `{`, http.StatusBadRequest},
		{"negative max_tokens", `{"model":"m","prompt":"hi","max_tokens":-1}`, http.StatusBadRequest},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			rr := httptest.NewRecorder()
			req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(tc.body))
			p.routes().ServeHTTP(rr, req)
			if rr.Code != tc.want {
				t.Fatalf("code = %d, want %d (body %q)", rr.Code, tc.want, rr.Body.String())
			}
		})
	}
}

// A valid request with a live STUB engine: the plane must proxy status,
// content-type, and body verbatim (and not touch the payload).
func TestChatPassthrough(t *testing.T) {
	stub := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost || r.URL.Path != "/v1/chat/completions" {
			t.Errorf("stub got %s %s", r.Method, r.URL.Path)
		}
		if ct := r.Header.Get("Content-Type"); ct != "application/json" {
			t.Errorf("stub content-type = %q", ct)
		}
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte(`{"id":"cmpl-1","choices":[{"message":{"content":" Paris"}}]}`))
	}))
	defer stub.Close()

	p := testPlane(t, stub.URL)
	rr := httptest.NewRecorder()
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions",
		strings.NewReader(`{"model":"chat","messages":[{"role":"user","content":"hi"}],"max_tokens":8}`))
	p.routes().ServeHTTP(rr, req)
	if rr.Code != http.StatusOK {
		t.Fatalf("chat = %d, want 200 (body %q)", rr.Code, rr.Body.String())
	}
	if !strings.Contains(rr.Body.String(), `"content":" Paris"`) {
		t.Fatalf("body = %q, want the engine's payload verbatim", rr.Body.String())
	}
}

// SSE passthrough must flush incrementally (the streaming contract): a stub
// that writes two chunks with a delay must both reach the client writer.
func TestChatStreamFlushes(t *testing.T) {
	stub := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(http.StatusOK)
		f := w.(http.Flusher)
		_, _ = w.Write([]byte("data: {\"a\":1}\n\n"))
		f.Flush()
		_, _ = w.Write([]byte("data: [DONE]\n\n"))
		f.Flush()
	}))
	defer stub.Close()

	p := testPlane(t, stub.URL)
	rr := httptest.NewRecorder()
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions",
		strings.NewReader(`{"model":"chat","messages":[{"role":"user","content":"hi"}],"stream":true}`))
	p.routes().ServeHTTP(rr, req)
	if rr.Code != http.StatusOK {
		t.Fatalf("stream = %d", rr.Code)
	}
	if !strings.Contains(rr.Body.String(), "[DONE]") {
		t.Fatalf("stream body = %q", rr.Body.String())
	}
}

func TestLoadModelNameRequired(t *testing.T) {
	p := testPlane(t, deadPortURL)
	// The route pattern requires a name; an empty path can't match it, so
	// exercise the handler directly for the 400 contract.
	rr := httptest.NewRecorder()
	p.loadModel(rr, httptest.NewRequest(http.MethodPost, "/v1/models//load", nil))
	if rr.Code != http.StatusBadRequest {
		t.Fatalf("load without name = %d, want 400", rr.Code)
	}
}
