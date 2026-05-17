// (c) JFrog Ltd. (2026)

package shortener

import (
	"crypto/md5"
	"encoding/hex"
	"fmt"

	"github.com/example/urlshortener/internal/store"
)

const CodeLength = 6

type Shortener struct {
	store store.Store
}

func New(s store.Store) *Shortener {
	return &Shortener{store: s}
}

func (s *Shortener) Shorten(url string) (string, error) {
	code := generateCode(url)
	if err := s.store.Save(code, url); err != nil {
		return "", fmt.Errorf("save: %w", err)
	}
	return code, nil
}

func (s *Shortener) Resolve(code string) (string, error) {
	return s.store.Lookup(code)
}

// BUG: uses MD5 which is predictable - same URL always produces the same code,
// making collisions trivial for different URLs with similar hashes.
// Should use a random or counter-based approach.
func generateCode(url string) string {
	hash := md5.Sum([]byte(url))
	return hex.EncodeToString(hash[:])[:CodeLength]
}
