// Command fakes serves stand-ins for the services noergler talks to, so the
// smoke script and a local `serve` can boot without the real ones.
//
// Covers what startup and one replayed webhook touch: the Bitbucket and Jira
// probes, the riptide ping, and the gateway (model listing plus completions).
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

	// Bitbucket: the PR diff the review path fetches.
	mux.HandleFunc("GET /rest/api/1.0/projects/{project}/repos/{repo}/pull-requests/{id}/diff",
		func(w http.ResponseWriter, r *http.Request) {
			w.Header().Set("Content-Type", "text/plain")
			_, _ = w.Write([]byte(sampleDiff))
		})

	// Bitbucket: raw file bodies. AGENTS.md is the repo instructions the
	// reviewer requires; anything else is a file it wants to expand context
	// from. A 404 here is a real answer, not a gap in the fake.
	mux.HandleFunc("GET /rest/api/1.0/projects/{project}/repos/{repo}/raw/{path...}",
		func(w http.ResponseWriter, r *http.Request) {
			if r.PathValue("path") == "AGENTS.md" {
				w.Header().Set("Content-Type", "text/plain")
				_, _ = w.Write([]byte("# Rules\n\nBe terse.\n"))
				return
			}
			w.WriteHeader(http.StatusNotFound)
			writeJSON(w, r, map[string]any{"errors": []any{map[string]any{"message": "no such file"}}})
		})

	// Bitbucket: the comments the review posts. Both the inline comments and
	// the summary land here; the id lets the store record them.
	commentID := 1000
	mux.HandleFunc("POST /rest/api/1.0/projects/{project}/repos/{repo}/pull-requests/{id}/comments",
		func(w http.ResponseWriter, r *http.Request) {
			commentID++
			log.Printf("comment posted id=%d", commentID)
			w.WriteHeader(http.StatusCreated)
			writeJSON(w, r, map[string]any{"id": commentID, "version": 0})
		})

	// Gateway: the model listing per-team startup reads the context window
	// from. The id is the gateway alias, not the llmwire profile id, and
	// max_input_tokens must clear the 1M floor or the team is disabled.
	mux.HandleFunc("GET /models", func(w http.ResponseWriter, r *http.Request) {
		writeJSON(w, r, map[string]any{"object": "list", "data": []any{
			map[string]any{
				"id":                gatewayAlias,
				"object":            "model",
				"max_input_tokens":  1_000_000,
				"max_output_tokens": 128_000,
			},
		}})
	})

	// Gateway: completions. Two callers share this endpoint. The startup ping
	// wants any non-empty reply; a review wants a schema-valid object. They
	// are told apart by response_format, which only the review sends.
	mux.HandleFunc("POST /chat/completions", func(w http.ResponseWriter, r *http.Request) {
		var req struct {
			ResponseFormat json.RawMessage `json:"response_format"`
		}
		_ = json.NewDecoder(r.Body).Decode(&req)

		content := "ok"
		if len(req.ResponseFormat) > 0 {
			content = cannedReview
		}
		// Priced, so the run stores a cost instead of falling open to NULL.
		w.Header().Set("x-litellm-response-cost", "0.0123")
		writeJSON(w, r, map[string]any{
			"id":     "chatcmpl-fake",
			"object": "chat.completion",
			"model":  gatewayAlias,
			"choices": []any{map[string]any{"index": 0, "finish_reason": "stop",
				"message": map[string]any{"role": "assistant", "content": content}}},
			"usage": map[string]any{"prompt_tokens": 1200, "completion_tokens": 300, "total_tokens": 1500},
		})
	})

	mux.HandleFunc("/", func(w http.ResponseWriter, r *http.Request) {
		log.Printf("unhandled: %s %s", r.Method, r.URL.Path)
		w.WriteHeader(http.StatusNotFound)
		writeJSON(w, r, map[string]any{"errors": []any{map[string]any{"message": "not implemented by hack/fakes"}}})
	})

	log.Printf("fakes listening on %s (bitbucket, jira, riptide, gateway)", *addr)
	srv := &http.Server{Addr: *addr, Handler: mux}
	if err := srv.ListenAndServe(); err != nil {
		log.Fatal(err)
	}
}

// sampleDiff is one reviewable file, enough for the pipeline to produce a
// finding and a summary.
const sampleDiff = `diff --git a/a.go b/a.go
index 1111111..2222222 100644
--- a/a.go
+++ b/a.go
@@ -1,3 +1,4 @@
 package a

-func old() {}
+func newThing() int { return 42 }
+func other() {}
`

// gatewayAlias is what LLMWIRE_LITELLM_MODELS maps the profile id onto in
// hack/smoke.sh. The listing is keyed by the alias, not the profile.
const gatewayAlias = "ai-gateway-gpt-5.5"

// cannedReview is one schema-valid review, so a replayed webhook produces an
// inline comment and a summary rather than an unparseable-response notice.
const cannedReview = `{
  "overview": "A small change to the smoke fixture.",
  "strengths": ["Focused diff"],
  "security_performance": "No security or performance impact.",
  "test_coverage": "Covered by the smoke run.",
  "verdict": {"decision": "approve", "rationale": "Nothing blocking."},
  "findings": [{"file": "a.go", "line": 1, "severity": "suggestion",
    "confidence": 90, "headline": "Name the constant",
    "comment": "A literal here would read better as a named constant.",
    "suggestion": null}],
  "compliance_requirements": null
}`

func writeJSON(w http.ResponseWriter, r *http.Request, body any) {
	w.Header().Set("Content-Type", "application/json")
	if err := json.NewEncoder(w).Encode(body); err != nil {
		log.Printf("encode %s %s: %v", r.Method, r.URL.Path, err)
	}
}
