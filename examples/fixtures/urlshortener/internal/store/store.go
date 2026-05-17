// (c) JFrog Ltd. (2026)

package store

// Store is the interface for URL persistence.
type Store interface {
	Save(code, url string) error
	Lookup(code string) (string, error)
	Exists(code string) (bool, error)
}
