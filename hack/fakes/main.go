// Command fakes serves stand-ins for the services noergler talks to, so the
// smoke script and a local `serve` can boot without the real ones.
//
// Covers what startup and one replayed webhook touch: the Bitbucket and Jira
// probes, the riptide ping, and the gateway (model listing plus completions).
//
// With -record, every body that carries review output is written to a file so
// two runs can be diffed against each other: the comments posted to Bitbucket,
// the completion requests sent to the gateway, and the riptide rollup. The
// name is the payload kind plus the order it arrived in, because the order is
// part of what the diff has to compare.
//
// Usage: fakes [-addr :18099] [-record dir] [-review file]
package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"log"
	"net/http"
	"os"
	"path/filepath"
	"slices"
	"strconv"
	"strings"
	"sync"
)

// recorder writes request bodies to a directory, numbered per kind in arrival
// order. A zero recorder (no -record) discards everything, so the default
// behaviour of the fakes is unchanged.
type recorder struct {
	dir string

	mu sync.Mutex
	n  map[string]int
}

func newRecorder(dir string) *recorder {
	return &recorder{dir: dir, n: make(map[string]int)}
}

// save writes body as <kind>-<seq>.json and returns the bytes it consumed, so
// a handler can still decode what it recorded.
func (rec *recorder) save(kind string, body []byte) {
	if rec.dir == "" {
		return
	}
	rec.mu.Lock()
	rec.n[kind]++
	seq := rec.n[kind]
	rec.mu.Unlock()

	name := filepath.Join(rec.dir, fmt.Sprintf("%s-%d.json", kind, seq))
	if err := os.WriteFile(name, body, 0o644); err != nil {
		log.Printf("record %s: %v", name, err)
		return
	}
	log.Printf("recorded %s (%d bytes)", filepath.Base(name), len(body))
}

// readAndRecord drains r.Body, saves it and hands it back for decoding.
func (rec *recorder) readAndRecord(kind string, r *http.Request) []byte {
	body, err := io.ReadAll(r.Body)
	if err != nil {
		log.Printf("read %s body: %v", kind, err)
		return nil
	}
	rec.save(kind, body)
	return body
}

// posted holds the comment bodies this process accepted, keyed by the id it
// handed out, so a GET of one can answer with the text instead of a 404.
var (
	postedMu sync.Mutex
	posted   = map[string]string{}
)

// Onboarding state. The webhooks have to survive between requests: onboarding
// is a write followed by a read, and a stateless fake would answer the status
// call with "missing" no matter what the onboard call did, so the smoke check
// would pass without proving anything.
//
// Keyed by the target path ("PROJ" or "PROJ/repo"), which is what the client
// addresses; the values are the bodies as Bitbucket would hand them back.
var (
	hooksMu  sync.Mutex
	hooks    = map[string][]map[string]any{}
	hookSeq  = 5000
	projRepo = map[string][]string{}
)

// adminToken is the token hack/smoke.sh sends for the calls that are supposed
// to succeed. Onboarding proves admin rights by listing webhooks, so the
// listing has to refuse a non-admin token too, not just the writes: were only
// the writes gated, a non-admin `status` would read "ok" and the negative case
// in the smoke run would prove nothing.
const adminToken = "admin-token"

// isAdmin reports whether this request carries the admin token.
func isAdmin(r *http.Request) bool {
	return strings.TrimSpace(r.Header.Get("Authorization")) == "Bearer "+adminToken
}

// denyNonAdmin answers 401 the way Bitbucket does when a token lacks the
// rights, and reports whether it handled the request.
func denyNonAdmin(w http.ResponseWriter, r *http.Request) bool {
	if isAdmin(r) {
		return false
	}
	log.Printf("onboarding: refused %s %s (non-admin token)", r.Method, r.URL.Path)
	w.WriteHeader(http.StatusUnauthorized)
	writeJSON(w, r, map[string]any{"errors": []any{map[string]any{
		"message": "You are not permitted to access this resource"}}})
	return true
}

// storedConfiguration is what a listing reports for a hook's configuration.
// Bitbucket never hands the stored secret back, but it does return the object,
// and onboarding's diff only asks whether it is non-empty: an empty map reads
// as "secret unset" and every status call would then report the hook stale.
func storedConfiguration() map[string]any {
	return map[string]any{"secret": "***"}
}

// hookIDOf reads a stored hook's id. It went in as an int and comes back out
// as one, but a decoded body would carry a float64, so both are accepted.
func hookIDOf(h map[string]any) int {
	switch v := h["id"].(type) {
	case int:
		return v
	case float64:
		return int(v)
	}
	return 0
}

// target is the key both webhook paths share: a project, or a repo under it.
func target(r *http.Request) string {
	if repo := r.PathValue("repo"); repo != "" {
		return r.PathValue("project") + "/" + repo
	}
	return r.PathValue("project")
}

// writePage answers with one full page. The client's paged() stops on
// isLastPage, so a single page is all any of these listings needs.
func writePage(w http.ResponseWriter, r *http.Request, values []map[string]any) {
	if values == nil {
		values = []map[string]any{}
	}
	writeJSON(w, r, map[string]any{
		"values": values, "size": len(values), "start": 0,
		"limit": len(values), "isLastPage": true,
	})
}

func main() {
	addr := flag.String("addr", ":18099", "listen address")
	recordDir := flag.String("record", "", "directory to record posted comments, completion requests and riptide rollups into")
	reviewFile := flag.String("review", "", "file holding the canned review JSON (default: the built-in one)")
	flag.Parse()

	rec := newRecorder(*recordDir)
	if *recordDir != "" {
		if err := os.MkdirAll(*recordDir, 0o755); err != nil {
			log.Fatalf("record dir: %v", err)
		}
	}

	// Two runs being diffed have to get the same review bytes, so the body
	// can come from a shared file instead of the canned default.
	review := cannedReview
	if *reviewFile != "" {
		blob, err := os.ReadFile(*reviewFile)
		if err != nil {
			log.Fatalf("review file: %v", err)
		}
		review = string(blob)
	}

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
		rec.readAndRecord("rollup", r)
		log.Printf("riptide rollup received")
		w.WriteHeader(http.StatusAccepted)
	})

	// Bitbucket: the PR diff the review path fetches.
	mux.HandleFunc("GET /rest/api/1.0/projects/{project}/repos/{repo}/pull-requests/{id}/diff",
		func(w http.ResponseWriter, r *http.Request) {
			w.Header().Set("Content-Type", "text/plain")
			_, _ = w.Write([]byte(sampleDiff))
		})

	// Bitbucket: the PR's file list. Only the too-large log path asks for it,
	// to name the files a diff too big to fetch would have touched. Paths
	// match sampleDiff, plus one the diff does not carry: past the byte cap is
	// exactly where this endpoint earns its keep.
	mux.HandleFunc("GET /rest/api/1.0/projects/{project}/repos/{repo}/pull-requests/{id}/changes",
		func(w http.ResponseWriter, r *http.Request) {
			writeJSON(w, r, map[string]any{
				"values": []any{
					map[string]any{"nodeType": "FILE", "type": "MODIFY", "path": map[string]any{"toString": "a.go"}},
					map[string]any{"nodeType": "FILE", "type": "MODIFY", "path": map[string]any{"toString": "util.py"}},
					map[string]any{"nodeType": "FILE", "type": "ADD", "path": map[string]any{"toString": "fixtures/big.json"}},
				},
				"isLastPage": true,
			})
		})

	// Bitbucket: raw file bodies. AGENTS.md is the repo instructions the
	// reviewer requires; anything else is a file it wants to expand context
	// from. A 404 here is a real answer, not a gap in the fake.
	mux.HandleFunc("GET /rest/api/1.0/projects/{project}/repos/{repo}/raw/{path...}",
		func(w http.ResponseWriter, r *http.Request) {
			path := r.PathValue("path")
			if path == "AGENTS.md" {
				w.Header().Set("Content-Type", "text/plain")
				_, _ = w.Write([]byte("# Rules\n\nBe terse.\n"))
				return
			}
			if body, ok := sampleFiles[path]; ok {
				w.Header().Set("Content-Type", "text/plain")
				_, _ = w.Write([]byte(body))
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
			body := rec.readAndRecord("comment", r)
			// Keep the text so a later GET of this id can return it rather
			// than 404, which both implementations read as "deleted".
			//
			// The id is handed out under the same lock that stores the text:
			// two concurrent posts reading the counter would otherwise get the
			// same id, and the second would overwrite the first, so a GET
			// would answer with the wrong comment. The posts are interleaved,
			// so this is reachable.
			var c struct{ Text string }
			_ = json.Unmarshal(body, &c)
			postedMu.Lock()
			commentID++
			id := commentID
			posted[strconv.Itoa(id)] = c.Text
			postedMu.Unlock()
			log.Printf("comment posted id=%d", id)
			w.WriteHeader(http.StatusCreated)
			writeJSON(w, r, map[string]any{"id": id, "version": 0})
		})

	// Bitbucket: reading one comment back. Both implementations fetch the
	// summary they recorded to see whether a human deleted it, and a 404 here
	// means "deleted", which makes the PR ignored from then on. Serving the
	// bodies posted in this process keeps a second replay on the review path
	// instead of that branch.
	mux.HandleFunc("GET /rest/api/1.0/projects/{project}/repos/{repo}/pull-requests/{id}/comments/{commentID}",
		func(w http.ResponseWriter, r *http.Request) {
			id := r.PathValue("commentID")
			postedMu.Lock()
			text, ok := posted[id]
			postedMu.Unlock()
			if !ok {
				w.WriteHeader(http.StatusNotFound)
				writeJSON(w, r, map[string]any{"errors": []any{map[string]any{"message": "no such comment"}}})
				return
			}
			n, _ := strconv.Atoi(id)
			writeJSON(w, r, map[string]any{"id": n, "text": text, "version": 0})
		})

	// Bitbucket: editing a comment in place. A re-review updates its summary
	// rather than posting a second one, so without this the two sides diverge
	// on comment count for a reason that is the fake's, not theirs.
	mux.HandleFunc("PUT /rest/api/1.0/projects/{project}/repos/{repo}/pull-requests/{id}/comments/{commentID}",
		func(w http.ResponseWriter, r *http.Request) {
			id := r.PathValue("commentID")
			body := rec.readAndRecord("comment-update", r)
			var c struct {
				Text    string `json:"text"`
				Version int    `json:"version"`
			}
			_ = json.Unmarshal(body, &c)
			postedMu.Lock()
			_, known := posted[id]
			if known {
				posted[id] = c.Text
			}
			postedMu.Unlock()
			if !known {
				w.WriteHeader(http.StatusNotFound)
				writeJSON(w, r, map[string]any{"errors": []any{map[string]any{"message": "no such comment"}}})
				return
			}
			log.Printf("comment updated id=%s", id)
			n, _ := strconv.Atoi(id)
			writeJSON(w, r, map[string]any{"id": n, "text": c.Text, "version": c.Version + 1})
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
		body, err := io.ReadAll(r.Body)
		if err != nil {
			log.Printf("read completions body: %v", err)
		}
		var req struct {
			ResponseFormat json.RawMessage `json:"response_format"`
		}
		_ = json.Unmarshal(body, &req)

		content := "ok"
		if len(req.ResponseFormat) > 0 {
			content = review
			// Only the review call is worth diffing. The startup ping is a
			// fixed two-word prompt and every boot makes one.
			rec.save("completion", body)
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

	// -- Bitbucket: the onboarding surface -- //
	//
	// Everything below exists for the /onboard route, which writes webhooks
	// into a customer's Bitbucket with a human's admin token. It is the
	// riskiest thing this service does and the smoke run had no coverage of
	// it at all; these eight handlers are what backend/internal/onboarding's
	// AdminClient and BotClient need to run end to end.

	// The bot proving it can read a project. On the bot's own token, so it is
	// deliberately not gated on the admin one.
	//
	// A project whose key ends in NOREAD is refused, so the smoke run has a
	// target the bot cannot read. Without one the grant-bot path is
	// unreachable: onboarding only grants when the bot's own read fails, so
	// against a fake that answers every read the grant is correctly skipped
	// and the call proves nothing.
	mux.HandleFunc("GET /rest/api/1.0/projects/{project}", func(w http.ResponseWriter, r *http.Request) {
		project := r.PathValue("project")
		if strings.HasSuffix(project, "NOREAD") && !isAdmin(r) {
			w.WriteHeader(http.StatusNotFound)
			writeJSON(w, r, map[string]any{"errors": []any{map[string]any{
				"message": "Project " + project + " does not exist or you do not have permission"}}})
			return
		}
		writeJSON(w, r, map[string]any{"key": project, "id": 1, "name": project, "public": false})
	})

	// The bot proving it can read a repo.
	mux.HandleFunc("GET /rest/api/1.0/projects/{project}/repos/{repo}", func(w http.ResponseWriter, r *http.Request) {
		repo := r.PathValue("repo")
		writeJSON(w, r, map[string]any{
			"slug": repo, "id": 1, "name": repo,
			"project": map[string]any{"key": r.PathValue("project")},
		})
	})

	// Every repo in a project, which the stray-hook sweep walks.
	mux.HandleFunc("GET /rest/api/1.0/projects/{project}/repos", func(w http.ResponseWriter, r *http.Request) {
		if denyNonAdmin(w, r) {
			return
		}
		project := r.PathValue("project")
		hooksMu.Lock()
		slugs := append([]string(nil), projRepo[project]...)
		hooksMu.Unlock()
		values := make([]map[string]any, 0, len(slugs))
		for _, slug := range slugs {
			values = append(values, map[string]any{
				"slug": slug, "name": slug,
				"project": map[string]any{"key": project},
			})
		}
		writePage(w, r, values)
	})

	// Listing the hooks on a project or a repo. This doubles as onboarding's
	// admin-rights proof, hence the gate.
	listHooks := func(w http.ResponseWriter, r *http.Request) {
		if denyNonAdmin(w, r) {
			return
		}
		hooksMu.Lock()
		got := append([]map[string]any(nil), hooks[target(r)]...)
		hooksMu.Unlock()
		writePage(w, r, got)
	}
	mux.HandleFunc("GET /rest/api/1.0/projects/{project}/webhooks", listHooks)
	mux.HandleFunc("GET /rest/api/1.0/projects/{project}/repos/{repo}/webhooks", listHooks)

	// Creating a hook. The stored body is what a later listing returns, minus
	// the secret: Bitbucket never hands that back, and onboarding's diff only
	// checks that `configuration` is non-empty.
	createHook := func(w http.ResponseWriter, r *http.Request) {
		if denyNonAdmin(w, r) {
			return
		}
		var body map[string]any
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
			w.WriteHeader(http.StatusBadRequest)
			writeJSON(w, r, map[string]any{"errors": []any{map[string]any{"message": "bad body"}}})
			return
		}
		key := target(r)
		hooksMu.Lock()
		hookSeq++
		id := hookSeq
		body["id"] = id
		body["configuration"] = storedConfiguration()
		hooks[key] = append(hooks[key], body)
		if repo := r.PathValue("repo"); repo != "" {
			project := r.PathValue("project")
			if !slices.Contains(projRepo[project], repo) {
				projRepo[project] = append(projRepo[project], repo)
			}
		}
		hooksMu.Unlock()
		log.Printf("onboarding: webhook created target=%s id=%d", key, id)
		w.WriteHeader(http.StatusCreated)
		writeJSON(w, r, body)
	}
	mux.HandleFunc("POST /rest/api/1.0/projects/{project}/webhooks", createHook)
	mux.HandleFunc("POST /rest/api/1.0/projects/{project}/repos/{repo}/webhooks", createHook)

	// Replacing a hook's settings.
	updateHook := func(w http.ResponseWriter, r *http.Request) {
		if denyNonAdmin(w, r) {
			return
		}
		var body map[string]any
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
			w.WriteHeader(http.StatusBadRequest)
			writeJSON(w, r, map[string]any{"errors": []any{map[string]any{"message": "bad body"}}})
			return
		}
		id, _ := strconv.Atoi(r.PathValue("webhookID"))
		key := target(r)
		hooksMu.Lock()
		found := false
		for i, h := range hooks[key] {
			if hookIDOf(h) == id {
				body["id"] = id
				body["configuration"] = storedConfiguration()
				hooks[key][i] = body
				found = true
				break
			}
		}
		hooksMu.Unlock()
		if !found {
			w.WriteHeader(http.StatusNotFound)
			writeJSON(w, r, map[string]any{"errors": []any{map[string]any{"message": "no such webhook"}}})
			return
		}
		log.Printf("onboarding: webhook updated target=%s id=%d", key, id)
		writeJSON(w, r, body)
	}
	mux.HandleFunc("PUT /rest/api/1.0/projects/{project}/webhooks/{webhookID}", updateHook)
	mux.HandleFunc("PUT /rest/api/1.0/projects/{project}/repos/{repo}/webhooks/{webhookID}", updateHook)

	// Removing a hook, which is what `remove` does once its checks pass.
	deleteHook := func(w http.ResponseWriter, r *http.Request) {
		if denyNonAdmin(w, r) {
			return
		}
		id, _ := strconv.Atoi(r.PathValue("webhookID"))
		key := target(r)
		hooksMu.Lock()
		kept := hooks[key][:0]
		for _, h := range hooks[key] {
			if hookIDOf(h) != id {
				kept = append(kept, h)
			}
		}
		hooks[key] = kept
		hooksMu.Unlock()
		log.Printf("onboarding: webhook deleted target=%s id=%d", key, id)
		w.WriteHeader(http.StatusNoContent)
	}
	mux.HandleFunc("DELETE /rest/api/1.0/projects/{project}/webhooks/{webhookID}", deleteHook)
	mux.HandleFunc("DELETE /rest/api/1.0/projects/{project}/repos/{repo}/webhooks/{webhookID}", deleteHook)

	// Granting the bot write access. Bitbucket takes this as query parameters,
	// not a body, so the log line records them for the smoke run to assert on.
	//
	// The grant is also REMEMBERED, because the status check reads it back:
	// answering a canned permission would let the onboarder claim the bot can
	// comment on a target nothing ever granted.
	var permMu sync.Mutex
	granted := map[string]string{}
	grantPerm := func(w http.ResponseWriter, r *http.Request) {
		if denyNonAdmin(w, r) {
			return
		}
		name, perm := r.URL.Query().Get("name"), r.URL.Query().Get("permission")
		permMu.Lock()
		granted[target(r)+"\x00"+name] = perm
		permMu.Unlock()
		log.Printf("onboarding: permission granted target=%s name=%s permission=%s",
			target(r), name, perm)
		w.WriteHeader(http.StatusNoContent)
	}
	mux.HandleFunc("PUT /rest/api/1.0/projects/{project}/permissions/users", grantPerm)
	mux.HandleFunc("PUT /rest/api/1.0/projects/{project}/repos/{repo}/permissions/users", grantPerm)

	// Reading a user's effective permission, which is how the status check
	// proves the bot can post a review comment. Bitbucket's `filter` is a
	// substring match, so this answers with a page of users rather than one.
	readPerm := func(w http.ResponseWriter, r *http.Request) {
		if denyNonAdmin(w, r) {
			return
		}
		filter := r.URL.Query().Get("filter")
		permMu.Lock()
		// Only what was granted on THIS target, as Bitbucket does: a
		// project-level grant does not appear in a repository's listing.
		// The client is the one that falls back to the project, and a fake
		// that answered for both would hide it if that stopped working.
		perm, ok := granted[target(r)+"\x00"+filter]
		permMu.Unlock()
		values := []any{}
		if ok {
			values = append(values, map[string]any{
				"user":       map[string]any{"name": filter},
				"permission": perm,
			})
		}
		writeJSON(w, r, map[string]any{"values": values, "size": len(values), "isLastPage": true})
	}
	mux.HandleFunc("GET /rest/api/1.0/projects/{project}/permissions/users", readPerm)
	mux.HandleFunc("GET /rest/api/1.0/projects/{project}/repos/{repo}/permissions/users", readPerm)

	mux.HandleFunc("/", func(w http.ResponseWriter, r *http.Request) {
		log.Printf("unhandled: %s %s", r.Method, r.URL.Path)
		w.WriteHeader(http.StatusNotFound)
		writeJSON(w, r, map[string]any{"errors": []any{map[string]any{"message": "not implemented by hack/fakes"}}})
	})

	log.Printf("fakes listening on %s (bitbucket, jira, riptide, gateway)", *addr)
	if *recordDir != "" {
		log.Printf("recording comment, completion and rollup bodies to %s", *recordDir)
	}
	srv := &http.Server{Addr: *addr, Handler: mux}
	if err := srv.ListenAndServe(); err != nil {
		log.Fatal(err)
	}
}

// sampleDiff is one reviewable file, enough for the pipeline to produce a
// finding and a summary.
// Two files in different languages, each with a hunk in the middle of a longer
// file. The languages make file ordering (group, language, path) observable,
// and the surrounding lines give context expansion something to expand: with a
// 404 on the file body the reviewer falls back to diff-only and neither path
// runs, which is what the first recorded run silently did.
const sampleDiff = `diff --git a/a.go b/a.go
index 1111111..2222222 100644
--- a/a.go
+++ b/a.go
@@ -6,3 +6,4 @@ func helper() string {
 	return "helper"
 }

-func old() {}
+func newThing() int { return 42 }
+func other() {}
diff --git a/util.py b/util.py
index 3333333..4444444 100644
--- a/util.py
+++ b/util.py
@@ -5,3 +5,3 @@ def existing():
     return 1


-def removed():
+def renamed():
     return 2
`

// sampleFiles are the post-change bodies of the files in sampleDiff, served
// from the raw endpoint so the reviewer can expand context around the hunks.
// The hunk line numbers above index into these.
var sampleFiles = map[string]string{
	"a.go": `package a

import "fmt"

// helper is here to give the hunk some context above it.
func helper() string {
	return "helper"
}

func newThing() int { return 42 }
func other() {}

func trailing() { fmt.Println("after") }
`,
	"util.py": `"""Module docstring, context above the hunk."""


def existing():
    return 1


def renamed():
    return 2


def trailing():
    return 3
`,
}

// gatewayAlias is what LLMWIRE_LITELLM_MODELS maps the profile id onto in
// hack/smoke.sh. The listing is keyed by the alias, not the profile.
const gatewayAlias = "ai-gateway-gpt-5.5"

// cannedReview is the built-in default: one schema-valid review, so a replayed
// webhook produces an inline comment and a summary rather than an
// unparseable-response notice. It is what smoke.sh gets, and one finding is all
// smoke.sh asserts on.
//
// Not the same bytes as hack/testdata/review.json, deliberately: -review
// serves that file to two runs being diffed, and it carries a finding per file
// in sampleDiff so the posting order of several comments is compared too.
// Keeping the default here means the fakes still work with no flags.
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
