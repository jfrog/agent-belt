# (c) JFrog Ltd. (2026)

# BUG: runs as root - should use a non-root user for security
FROM golang:1.22-alpine AS builder

WORKDIR /app
COPY go.mod go.sum ./
RUN go mod download
COPY . .
RUN go build -o /urlshortener ./cmd/server

FROM alpine:3.19
COPY --from=builder /urlshortener /urlshortener
EXPOSE 8080
CMD ["/urlshortener"]
