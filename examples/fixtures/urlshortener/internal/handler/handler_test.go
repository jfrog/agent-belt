// (c) JFrog Ltd. (2026)

package handler

import (
	"bytes"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/example/urlshortener/internal/shortener"
	"github.com/example/urlshortener/internal/store"
)

func newTestHandler() *Handler {
	s := store.NewMemoryStore()
	sh := shortener.New(s)
	return New(sh)
}

func TestHealthEndpoint(t *testing.T) {
	h := newTestHandler()
	req := httptest.NewRequest("GET", "/health", nil)
	w := httptest.NewRecorder()
	h.ServeHTTP(w, req)

	if w.Code != http.StatusOK {
		t.Errorf("expected 200, got %d", w.Code)
	}
}

func TestShortenEndpoint(t *testing.T) {
	h := newTestHandler()
	body, _ := json.Marshal(map[string]string{"url": "https://example.com"})
	req := httptest.NewRequest("POST", "/shorten", bytes.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()
	h.ServeHTTP(w, req)

	if w.Code != http.StatusCreated {
		t.Errorf("expected 201, got %d", w.Code)
	}

	var resp shortenResponse
	json.NewDecoder(w.Body).Decode(&resp)
	if resp.Code == "" {
		t.Error("expected non-empty code")
	}
}

func TestShortenEmptyURL(t *testing.T) {
	h := newTestHandler()
	body, _ := json.Marshal(map[string]string{"url": ""})
	req := httptest.NewRequest("POST", "/shorten", bytes.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()
	h.ServeHTTP(w, req)

	if w.Code != http.StatusBadRequest {
		t.Errorf("expected 400, got %d", w.Code)
	}
}

func TestRedirectEndpoint(t *testing.T) {
	h := newTestHandler()

	// First create a short URL
	body, _ := json.Marshal(map[string]string{"url": "https://example.com"})
	req := httptest.NewRequest("POST", "/shorten", bytes.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()
	h.ServeHTTP(w, req)

	var resp shortenResponse
	json.NewDecoder(w.Body).Decode(&resp)

	// Now follow the redirect
	req2 := httptest.NewRequest("GET", "/"+resp.Code, nil)
	w2 := httptest.NewRecorder()
	h.ServeHTTP(w2, req2)

	if w2.Code != http.StatusMovedPermanently {
		t.Errorf("expected 301, got %d", w2.Code)
	}
}
