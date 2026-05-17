-- (c) JFrog Ltd. (2026)

CREATE TABLE IF NOT EXISTS urls (
    id SERIAL PRIMARY KEY,
    code VARCHAR(10) UNIQUE NOT NULL,
    url TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX idx_urls_code ON urls(code);
