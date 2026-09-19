// Command fakes serves stand-ins for the services noergler talks to, so the
// smoke script and a local `serve` can boot without the real ones.
//
// Phase 3 covers only what startup touches: the Bitbucket and Jira probes, plus
// the riptide ping. The gateway and the review endpoints join in Phase 8, when
// there is a pipeline to drive end to end.
//
// Usage: fakes [-addr :18099]
package main

import (
	"encoding/json"
	"flag"
	"log"
	"net/http"
	"strings"
)

func main() {
	addr := flag.String("addr", ":18099", "listen address")
	flag.Parse()

	mux := http.NewServeMux()

	// Bitbucket: the startup connectivity probe.
	mux.HandleFunc("GET /rest/api/1.0/application-properties", func(w http.ResponseWriter, r *http.Request) {
		writeJSON(w, r, map[string]any{"version": "8.19.2", "displayName": "Bitbucket"})
	})

	// Jira: the startup identity probe.
	mux.HandleFunc("GET /rest/api/2/myself", func(w http.ResponseWriter, r *http.Request) {
		writeJSON(w, r, map[string]any{"name": "noergler", "displayName": "Noergler Bot"})
	})

	// Riptide: the per-team startup ping. A request without a bearer token gets
	// the 401 that disables a team, so that path can be exercised too.
	mux.HandleFunc("GET /auth/ping", func(w http.ResponseWriter, r *http.Request) {
		if !strings.HasPrefix(r.Header.Get("Authorization"), "Bearer ") {
			w.WriteHeader(http.StatusUnauthorized)
			writeJSON(w, r, map[string]any{"detail": "missing token"})
			return
		}
		writeJSON(w, r, map[string]any{"status": "ok", "team": "smoke"})
	})

	// Riptide: swallow rollups so a local run does not error on them.
	mux.HandleFunc("POST /webhooks/noergler", func(w http.ResponseWriter, r *http.Request) {
		log.Printf("riptide rollup received")
		w.WriteHeader(http.StatusAccepted)
	})

	mux.HandleFunc("/", func(w http.ResponseWriter, r *http.Request) {
		log.Printf("unhandled: %s %s", r.Method, r.URL.Path)
		w.WriteHeader(http.StatusNotFound)
		writeJSON(w, r, map[string]any{"errors": []any{map[string]any{"message": "not implemented by hack/fakes"}}})
	})

	log.Printf("fakes listening on %s (bitbucket, jira, riptide probes)", *addr)
	srv := &http.Server{Addr: *addr, Handler: mux}
	if err := srv.ListenAndServe(); err != nil {
		log.Fatal(err)
	}
}

func writeJSON(w http.ResponseWriter, r *http.Request, body any) {
	w.Header().Set("Content-Type", "application/json")
	if err := json.NewEncoder(w).Encode(body); err != nil {
		log.Printf("encode %s %s: %v", r.Method, r.URL.Path, err)
	}
}
