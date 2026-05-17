// (c) JFrog Ltd. (2026)

package shortener

import (
	"testing"

	"github.com/example/urlshortener/internal/store"
)

func TestShorten(t *testing.T) {
	s := New(store.NewMemoryStore())
	code, err := s.Shorten("https://example.com")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(code) != CodeLength {
		t.Errorf("expected code length %d, got %d", CodeLength, len(code))
	}
}

func TestResolve(t *testing.T) {
	s := New(store.NewMemoryStore())
	code, _ := s.Shorten("https://example.com")
	url, err := s.Resolve(code)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if url != "https://example.com" {
		t.Errorf("expected https://example.com, got %s", url)
	}
}

func TestResolveNotFound(t *testing.T) {
	s := New(store.NewMemoryStore())
	_, err := s.Resolve("nonexistent")
	if err == nil {
		t.Error("expected error for nonexistent code")
	}
}

func TestSameURLSameCode(t *testing.T) {
	// This test documents the bug: same URL always produces same code
	s := New(store.NewMemoryStore())
	code1, _ := s.Shorten("https://example.com")
	code2, _ := s.Shorten("https://example.com")
	if code1 != code2 {
		t.Errorf("expected same code for same URL, got %s and %s", code1, code2)
	}
}
