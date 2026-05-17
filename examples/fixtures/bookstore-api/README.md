# Bookstore API

A REST API for managing books and authors, built with Express and TypeScript.

## Endpoints

- `GET /books` - List books (supports `?page=1&limit=10`)
- `GET /books/:id` - Get a book by ID
- `POST /books` - Create a book
- `GET /authors` - List authors
- `GET /authors/:id` - Get an author by ID
- `POST /authors` - Create an author

## Architecture

- `src/routes/` - Express route handlers
- `src/middleware/` - Auth and validation middleware
- `src/models/` - Type definitions
- `src/services/` - Business logic and DB access
- `src/utils/` - Shared utilities (pagination)

## Evaluation Scenarios

This fixture has 7 scenarios at 4 difficulty levels in
`examples/scenarios/experience/bookstore-api-claude/`:

```bash
# Read-only (L1) - agent reviews the codebase
belt eval examples/scenarios/experience/bookstore-api-claude --tags L1 --modes rules

# Editing (L2) - agent fixes a single bug
belt eval examples/scenarios/experience/bookstore-api-claude --tags L2 --modes rules

# All levels
belt eval examples/scenarios/experience/bookstore-api-claude --modes rules
```

See [examples/README.md](../../README.md) for prerequisites and full usage.

## Known Issues

This project has intentional bugs for evaluation:

- `src/services/bookService.ts` - SQL injection via string concatenation
- `src/middleware/auth.ts` - no token expiry validation
- `src/utils/pagination.ts` - off-by-one in page calculation
- `src/routes/books.ts` - missing input validation on POST
- `src/services/authorService.ts` - N+1 query pattern
