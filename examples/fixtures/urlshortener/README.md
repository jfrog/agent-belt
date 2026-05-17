# URL Shortener

A simple URL shortener microservice in Go.

## Endpoints

- `POST /shorten` - Create a short URL (`{"url": "https://example.com"}`)
- `GET /:code` - Redirect to original URL
- `GET /health` - Health check

## Architecture

- `cmd/server/main.go` - Entry point, HTTP server
- `internal/handler/` - HTTP request handlers
- `internal/store/` - Storage interface + memory/postgres implementations
- `internal/shortener/` - URL shortening logic

## Evaluation Scenarios

This fixture has 7 scenarios at 4 difficulty levels in
`examples/scenarios/experience/urlshortener-claude/`:

```bash
# Read-only (L1) - agent reviews the codebase
belt eval examples/scenarios/experience/urlshortener-claude --tags L1 --modes rules

# Editing (L2) - agent fixes a single bug
belt eval examples/scenarios/experience/urlshortener-claude --tags L2 --modes rules

# All levels
belt eval examples/scenarios/experience/urlshortener-claude --modes rules
```

See [examples/README.md](../../README.md) for prerequisites and full usage.

## Known Issues

This project has intentional bugs for evaluation:

- `internal/store/memory.go` - not goroutine-safe (no mutex)
- `internal/handler/handler.go` - no URL validation on input
- `internal/shortener/shortener.go` - predictable hash (MD5 prefix)
- `cmd/server/main.go` - no graceful shutdown
- `Dockerfile` - runs as root
