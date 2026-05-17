// (c) JFrog Ltd. (2026)

package store

import "fmt"

// MemoryStore stores URLs in a plain map.
// BUG: not goroutine-safe - concurrent reads/writes will race.
// Needs a sync.RWMutex to protect the map.
type MemoryStore struct {
	data map[string]string
}

func NewMemoryStore() *MemoryStore {
	return &MemoryStore{data: make(map[string]string)}
}

func (m *MemoryStore) Save(code, url string) error {
	m.data[code] = url
	return nil
}

func (m *MemoryStore) Lookup(code string) (string, error) {
	url, ok := m.data[code]
	if !ok {
		return "", fmt.Errorf("code %q not found", code)
	}
	return url, nil
}

func (m *MemoryStore) Exists(code string) (bool, error) {
	_, ok := m.data[code]
	return ok, nil
}
