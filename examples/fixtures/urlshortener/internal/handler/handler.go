// (c) JFrog Ltd. (2026)

package handler

import (
	"encoding/json"
	"net/http"

	"github.com/example/urlshortener/internal/shortener"
)

type Handler struct {
	shortener *shortener.Shortener
	mux       *http.ServeMux
}

type shortenRequest struct {
	URL string `json:"url"`
}

type shortenResponse struct {
	Code     string `json:"code"`
	ShortURL string `json:"short_url"`
}

func New(s *shortener.Shortener) *Handler {
	h := &Handler{shortener: s}
	h.mux = http.NewServeMux()
	h.mux.HandleFunc("POST /shorten", h.handleShorten)
	h.mux.HandleFunc("GET /health", h.handleHealth)
	h.mux.HandleFunc("GET /{code}", h.handleRedirect)
	return h
}

func (h *Handler) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	h.mux.ServeHTTP(w, r)
}

func (h *Handler) handleShorten(w http.ResponseWriter, r *http.Request) {
	var req shortenRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, "invalid JSON", http.StatusBadRequest)
		return
	}

	// BUG: doesn't validate URL format - accepts empty strings, random text, etc.
	// Should check that req.URL is a valid HTTP/HTTPS URL.
	if req.URL == "" {
		http.Error(w, "url is required", http.StatusBadRequest)
		return
	}

	code, err := h.shortener.Shorten(req.URL)
	if err != nil {
		http.Error(w, "internal error", http.StatusInternalServerError)
		return
	}

	resp := shortenResponse{
		Code:     code,
		ShortURL: "http://localhost:8080/" + code,
	}

	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusCreated)
	json.NewEncoder(w).Encode(resp)
}

func (h *Handler) handleRedirect(w http.ResponseWriter, r *http.Request) {
	code := r.PathValue("code")
	url, err := h.shortener.Resolve(code)
	if err != nil {
		http.Error(w, "not found", http.StatusNotFound)
		return
	}
	http.Redirect(w, r, url, http.StatusMovedPermanently)
}

func (h *Handler) handleHealth(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]string{"status": "ok"})
}
