package main

import (
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"strings"
	"syscall"
	"time"
)

// EngineClient proxies to the Rust engine service (ADR-001: process
// boundary, HTTP, no FFI).
type EngineClient struct {
	base *url.URL
	http *http.Client
}

// NewEngineClient parses the engine base URL. Defaults to the local engine
// service port.
func NewEngineClient(base string) (*EngineClient, error) {
	if base == "" {
		base = "http://127.0.0.1:8081"
	}
	u, err := url.Parse(base)
	if err != nil {
		return nil, fmt.Errorf("engine base URL %q: %w", base, err)
	}
	if u.Scheme == "" || u.Host == "" {
		return nil, fmt.Errorf("engine base URL %q needs scheme and host", base)
	}
	return &EngineClient{
		base: u,
		http: &http.Client{
			// Generous for prefill (a 512-token prompt on a slow machine is
			// seconds); streaming responses are NOT bounded by this because
			// the body is proxied incrementally.
			Timeout: 10 * time.Minute,
		},
	}, nil
}

// Unreachable reports whether the engine service is down (dial-level
// failure) rather than answering with an HTTP error. Dial failures surface
// as ECONNREFUSED through the url/os error chains.
func (c *EngineClient) Unreachable(err error) bool {
	if err == nil {
		return false
	}
	var ue *url.Error
	if errors.As(err, &ue) {
		err = ue.Err
	}
	if errors.Is(err, syscall.ECONNREFUSED) {
		return true
	}
	msg := err.Error()
	return strings.Contains(msg, "connection refused") ||
		strings.Contains(msg, "no route to host") ||
		strings.Contains(msg, "network is unreachable")
}

// send performs a request to the engine and returns its response. The
// caller must close the body.
func (c *EngineClient) send(method, path string, body io.Reader, contentType string) (*http.Response, error) {
	u := *c.base
	u.Path = strings.TrimSuffix(c.base.Path, "/") + path
	req, err := http.NewRequest(method, u.String(), body)
	if err != nil {
		return nil, err
	}
	if contentType != "" {
		req.Header.Set("Content-Type", contentType)
	}
	return c.http.Do(req)
}

// APIError is the OpenAI-shaped error the plane emits on its own failures
// (the house rule: loud, typed, machine-readable at every boundary).
type APIError struct {
	Status  int    `json:"-"`
	Type    string `json:"type"`
	Message string `json:"message"`
}

func (e *APIError) Error() string { return e.Type + ": " + e.Message }

func writeJSONError(w http.ResponseWriter, status int, errType, msg string) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(map[string]any{
		"error": map[string]string{"type": errType, "message": msg},
	})
}
