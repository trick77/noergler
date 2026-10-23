package api

import (
	"encoding/json"
	"errors"
	"net/http"
	"strings"
	"testing"
	"time"

	"github.com/trick77/noergler/internal/store"
)

func n64(v int64) *int64 { return &v }

// tok is a PR link token, the PR page's and its runs' only key.
const tok = "AAAAAAAAAAAAAAAAAAAAAA"

const runPath = "/api/dashboard/prs/" + tok + "/runs/2"

// An unknown token, a malformed one, a run the PR does not have, a number
// in another spelling and an old /runs/{token} path are the same 404.
func TestRunDetailNotFound(t *testing.T) {
	h := newDashHarness(t)
	h.st.prToken, h.st.runN = "BBBBBBBBBBBBBBBBBBBBBB", 2
	for _, path := range []string{"/api/dashboard/prs/" + tok + "/runs/2", "/api/dashboard/prs/abc/runs/2",
		"/api/dashboard/prs/BBBBBBBBBBBBBBBBBBBBBB/runs/3", "/api/dashboard/prs/BBBBBBBBBBBBBBBBBBBBBB/runs/0",
		"/api/dashboard/prs/BBBBBBBBBBBBBBBBBBBBBB/runs/02", "/api/dashboard/prs/BBBBBBBBBBBBBBBBBBBBBB/runs/+2",
		"/api/dashboard/prs/BBBBBBBBBBBBBBBBBBBBBB/runs/x", "/api/dashboard/prs/BBBBBBBBBBBBBBBBBBBBBB/runs/99999999999",
		"/api/dashboard/runs/BBBBBBBBBBBBBBBBBBBBBB"} {
		w := h.get(t, path)
		if w.Code != http.StatusNotFound {
			t.Errorf("%s: status = %d, want 404", path, w.Code)
		}
	}
}

func TestRunDetailFailsLoudlyWhenTheStoreDoes(t *testing.T) {
	h := newDashHarness(t)
	h.st.err = errors.New("database is down")
	w := h.get(t, runPath)
	if w.Code != http.StatusInternalServerError {
		t.Fatalf("status = %d, want 500", w.Code)
	}
	var body map[string]any
	if err := json.Unmarshal(w.Body.Bytes(), &body); err != nil {
		t.Fatal(err)
	}
	if got := body["error"]; got != "could not read the run" {
		t.Errorf("error = %#v, want the fixed phrase", got)
	}
	if strings.Contains(w.Body.String(), "database is down") {
		t.Errorf("body carries the store error: %s", w.Body.String())
	}
}

func TestRunDetailBody(t *testing.T) {
	h := newDashHarness(t)
	h.st.prToken, h.st.runN = tok, 2
	h.st.run = store.RunDetail{
		ID: 7, N: 2, Key: store.PRKey{Project: "PAY", Repo: "checkout", PRID: 1482}, TeamSlug: "platform",
		Kind: store.RunAuto, Outcome: "ok", ElapsedMS: n64(73100), CreatedAt: time.Unix(0, 0).UTC(),
		PRTitle: "Retry captures",
		Run: &store.RunFacts{ID: 3, Incremental: true, FromCommit: "9f8e7d6", ToCommit: "3c41a0b",
			ModelLabel: "gpt-5.5-medium", PromptTokens: 41210, CachedTokens: 28416, CompletionTokens: 2830,
			CostNanoUSD: n64(47_845_000), FindingsPosted: 1},
		Call: &store.LLMCall{Profile: "gpt-5.5-medium", Model: "gpt-5.5-2026-08-06",
			CacheReadTokens: n64(28416), CostNanoUSD: n64(47_845_000),
			CostCacheReadNanoUSD: n64(3_552_000), KeySpendNanoUSD: n64(812_400_000_000)},
		Findings: []store.RunFinding{{Severity: "issue", FilePath: "retry.go", Line: 88,
			Headline: "Capture wird wiederholt", CommentText: "Die `Capture` in `RetryPolicy.run` ...",
			Suggestion: strp("idempotency key"), DisputeText: strp("ist gewollt"), CommentID: n64(555)}},
		Detail: &store.Pipeline{PRScope: &store.PRScope{Added: 400, Removed: 9, Files: 12}},
	}

	w := h.get(t, runPath)
	if w.Code != http.StatusOK {
		t.Fatalf("status = %d: %s", w.Code, w.Body.String())
	}
	var body map[string]any
	if err := json.Unmarshal(w.Body.Bytes(), &body); err != nil {
		t.Fatal(err)
	}
	if body["tag"] != "PAY/checkout#1482" || body["team_name"] != "Platform Engineering" ||
		body["title"] != "Retry captures" || body["n"] != float64(2) {
		t.Errorf("identity = %v / %v / %v / %v", body["tag"], body["team_name"], body["title"], body["n"])
	}
	// The run's cost at 3 decimals, as on every other dashboard figure.
	if body["cost_usd"] != "0.048" {
		t.Errorf("cost_usd = %v, want 0.048", body["cost_usd"])
	}
	call := body["call"].(map[string]any)
	// Lanes at 6 decimals: $0.004 would hide a cache read worth a tenth of
	// that.
	if call["cost_cache_read_usd"] != "0.003552" {
		t.Errorf("cache read lane = %v, want 0.003552", call["cost_cache_read_usd"])
	}
	// Absent is null, never zero.
	if call["cost_cache_write_usd"] != nil || call["cache_write_tokens"] != nil {
		t.Errorf("absent lanes = %v / %v, want null", call["cost_cache_write_usd"], call["cache_write_tokens"])
	}
	if call["key_spend_usd"] != "812.400" {
		t.Errorf("key spend = %v", call["key_spend_usd"])
	}
	if body["bitbucket_url"] != "https://bitbucket.example.com" {
		t.Errorf("bitbucket_url = %v", body["bitbucket_url"])
	}
	// The PR's history and total are the PR page's.
	if _, ok := body["pr"]; ok {
		t.Errorf("run body carries the PR part: %v", body["pr"])
	}
	// The whole PR's counts are the PR page's figure, not this run's.
	if strings.Contains(w.Body.String(), "pr_scope") {
		t.Errorf("run body carries the PR scope: %s", w.Body.String())
	}
	// The full text, code included: the page is reached only by its token.
	f := body["findings"].([]any)[0].(map[string]any)
	if f["comment"] != "Die `Capture` in `RetryPolicy.run` ..." || f["suggestion"] != "idempotency key" || f["dispute"] != "ist gewollt" {
		t.Errorf("finding = %v", f)
	}
	// The token leaves the server in the summary comment only: never in a
	// body, not even the page's own.
	if strings.Contains(w.Body.String(), tok) {
		t.Errorf("body carries the link token: %s", w.Body.String())
	}
	// The record the store handed over is not altered for the next reader.
	if h.st.run.Detail.PRScope == nil {
		t.Error("the handler cleared the store's record")
	}
}

func strp(s string) *string { return &s }

// Never a secret on the page: a posted finding's text with a secret-like
// token is withheld, the author's dispute included, the rest still shown.
// The body is never cached.
func TestRunDetailWithholdsSecrets(t *testing.T) {
	h := newDashHarness(t)
	h.st.prToken, h.st.runN = tok, 2
	h.st.run = store.RunDetail{ID: 1, Outcome: "ok", Run: &store.RunFacts{ID: 1},
		Findings: []store.RunFinding{{Headline: "Key in config", CommentText: "the key leaks",
			DisputeText: strp("it is AKIAIOSFODNN7EXAMPLE, rotated")}}} // ship:allow-secret: fixture for the secret check
	w := h.get(t, runPath)
	if strings.Contains(w.Body.String(), "AKIA") || !strings.Contains(w.Body.String(), "the key leaks") {
		t.Errorf("body = %s", w.Body.String())
	}
	if w.Header().Get("Cache-Control") != "no-store" {
		t.Errorf("Cache-Control = %q, want no-store", w.Header().Get("Cache-Control"))
	}
}

// No gateway discount or margin (decided), and never a numeric run link.
func TestRunDetailCarriesNoMargin(t *testing.T) {
	h := newDashHarness(t)
	h.st.prToken, h.st.runN = tok, 2
	h.st.run = store.RunDetail{ID: 1, Outcome: "ok", Run: &store.RunFacts{ID: 1},
		Findings: []store.RunFinding{{Headline: "h"}}, Call: &store.LLMCall{Profile: "p"}}
	body := h.get(t, runPath).Body.String()
	for _, banned := range []string{"margin", "discount", "original", "from_attempt"} {
		if strings.Contains(body, banned) {
			t.Errorf("body has %q: %s", banned, body)
		}
	}
}

// The detail record goes out as the review wrote it, under "detail"; none
// is null.
func TestRunDetailPassesTheRecordThrough(t *testing.T) {
	h := newDashHarness(t)
	h.st.prToken, h.st.runN = tok, 2
	h.st.run = store.RunDetail{ID: 3, Outcome: "ok", Detail: &store.Pipeline{Stages: []store.StageTime{{Name: "diff", MS: 3}}}}
	var body map[string]any
	if err := json.Unmarshal(h.get(t, runPath).Body.Bytes(), &body); err != nil {
		t.Fatal(err)
	}
	stages := body["detail"].(map[string]any)["stages"].([]any)
	if len(stages) != 1 || stages[0].(map[string]any)["name"] != "diff" {
		t.Errorf("detail = %v", body["detail"])
	}
	h.st.run.Detail = nil
	if err := json.Unmarshal(h.get(t, runPath).Body.Bytes(), &body); err != nil {
		t.Fatal(err)
	}
	if body["detail"] != nil {
		t.Errorf("no record: detail = %v, want null", body["detail"])
	}
}

// An unparseable answer wrote no run, but was billed: the page shows the
// attempt's cost instead of nothing.
func TestRunDetailBilledWithoutRun(t *testing.T) {
	h := newDashHarness(t)
	h.st.prToken, h.st.runN = tok, 2
	h.st.run = store.RunDetail{ID: 2, Outcome: "unparseable", AttemptCostNanoUSD: n64(400_000_000)}
	var body map[string]any
	if err := json.Unmarshal(h.get(t, runPath).Body.Bytes(), &body); err != nil {
		t.Fatal(err)
	}
	if body["cost_usd"] != "0.400" || body["run"] != nil || body["call"] != nil {
		t.Errorf("cost = %v, run = %v, call = %v", body["cost_usd"], body["run"], body["call"])
	}
}
