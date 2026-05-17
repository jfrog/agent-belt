# (c) JFrog Ltd. (2026)

.PHONY: build test run

build:
	go build -o bin/urlshortener ./cmd/server

test:
	go test ./...

run: build
	./bin/urlshortener
