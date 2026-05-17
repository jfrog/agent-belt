// (c) JFrog Ltd. (2026)

package store

import (
	"database/sql"
	"fmt"

	_ "github.com/lib/pq"
)

type PostgresStore struct {
	db *sql.DB
}

func NewPostgresStore(connStr string) (*PostgresStore, error) {
	db, err := sql.Open("postgres", connStr)
	if err != nil {
		return nil, fmt.Errorf("open db: %w", err)
	}
	if err := db.Ping(); err != nil {
		return nil, fmt.Errorf("ping db: %w", err)
	}
	return &PostgresStore{db: db}, nil
}

func (p *PostgresStore) Save(code, url string) error {
	_, err := p.db.Exec("INSERT INTO urls (code, url) VALUES ($1, $2)", code, url)
	return err
}

func (p *PostgresStore) Lookup(code string) (string, error) {
	var url string
	err := p.db.QueryRow("SELECT url FROM urls WHERE code = $1", code).Scan(&url)
	if err == sql.ErrNoRows {
		return "", fmt.Errorf("code %q not found", code)
	}
	return url, err
}

func (p *PostgresStore) Exists(code string) (bool, error) {
	var count int
	err := p.db.QueryRow("SELECT COUNT(*) FROM urls WHERE code = $1", code).Scan(&count)
	return count > 0, err
}
