// (c) JFrog Ltd. (2026)

package main

import (
	"fmt"
	"log"
	"net/http"
	"os"

	"github.com/example/urlshortener/internal/handler"
	"github.com/example/urlshortener/internal/shortener"
	"github.com/example/urlshortener/internal/store"
)

func main() {
	port := os.Getenv("PORT")
	if port == "" {
		port = "8080"
	}

	s := store.NewMemoryStore()
	sh := shortener.New(s)
	h := handler.New(sh)

	// BUG: no graceful shutdown - on SIGTERM/SIGINT the server just dies,
	// dropping in-flight requests. Should use http.Server with context
	// and signal.NotifyContext for clean shutdown.

	addr := fmt.Sprintf(":%s", port)
	log.Printf("Starting URL shortener on %s", addr)
	if err := http.ListenAndServe(addr, h); err != nil {
		log.Fatalf("server error: %v", err)
	}
}
