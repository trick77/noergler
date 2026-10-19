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

// tok is a link token, the run page's only key.
const tok = "AAAAAAAAAAAAAAAAAAAAAA"

// An unknown token, a malformed one and a numeric attempt id are the same
// 404.
func TestRunDetailNotFound(t *testing.T) {
	h := newDashHarness(t)
	h.st.detailToken = "BBBBBBBBBBBBBBBBBBBBBB"
	for _, path := range []string{"/api/dashboard/runs/" + tok, "/api/dashboard/runs/abc", "/api/dashboard/runs/7",
		"/api/dashboard/runs/BBBBBBBBBBBBBBBBBBBBB", "/api/dashboard/runs/BBBBBBBBBBBBBBBBBBBBBBB"} {
		w := h.get(t, path)
		if w.Code != http.StatusNotFound {
			t.Errorf("%s: status = %d, want 404", path, w.Code)
		}
		if w.Header().Get("Referrer-Policy") != "no-referrer" {
			t.Errorf("%s: Referrer-Policy = %q", path, w.Header().Get("Referrer-Policy"))
		}
	}
}

func TestRunDetailBody(t *testing.T) {
	h := newDashHarness(t)
	h.st.detailToken = tok
	h.st.prCost = n64(109_000_000)
	h.st.detail = store.RunDetail{
		ID: 7, Key: store.PRKey{Project: "PAY", Repo: "checkout", PRID: 1482}, TeamSlug: "platform",
		Kind: store.RunAuto, Outcome: "ok", ElapsedMS: n64(73100), CreatedAt: time.Unix(0, 0).UTC(),
		PRTitle: "Retry captures",
		Run: &store.RunFacts{ID: 3, Incremental: true, FromCommit: "9f8e7d6", ToCommit: "3c41a0b",
			ModelLabel: "gpt-5.5-medium", PromptTokens: 41210, CachedTokens: 28416, CompletionTokens: 2830,
			CostNanoUSD: n64(47_845_000), FindingsPosted: 1},
		Call: &store.LLMCall{Profile: "gpt-5.5-medium", Model: "gpt-5.5-2026-08-06",
			CacheReadTokens: n64(28416), CostNanoUSD: n64(47_845_000),
			CostCacheReadNanoUSD: n64(3_552_000), KeySpendNanoUSD: n64(812_400_000_000)},
		Findings: []store.RunFinding{{Severity: "high", FilePath: "retry.go", Line: 88,
			Headline: "Capture wird wiederholt", CommentText: "Die `Capture` in `RetryPolicy.run` ...",
			Suggestion: strp("idempotency key"), DisputeText: strp("ist gewollt"), CommentID: n64(555)}},
		PRAttempts: []store.PRAttempt{{ID: 6, Kind: store.RunAuto, Outcome: "skipped", Reason: "head_unchanged",
			CreatedAt: time.Unix(0, 0).UTC()}, {ID: 7, Kind: store.RunAuto, Outcome: "ok",
			CreatedAt: time.Unix(1, 0).UTC(), CostNanoUSD: n64(47_845_000)}},
	}

	w := h.get(t, "/api/dashboard/runs/"+tok)
	if w.Code != http.StatusOK {
		t.Fatalf("status = %d: %s", w.Code, w.Body.String())
	}
	var body map[string]any
	if err := json.Unmarshal(w.Body.Bytes(), &body); err != nil {
		t.Fatal(err)
	}
	if body["tag"] != "PAY/checkout#1482" || body["team_name"] != "Platform Engineering" || body["title"] != "Retry captures" {
		t.Errorf("identity = %v / %v / %v", body["tag"], body["team_name"], body["title"])
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
	pr := body["pr"].(map[string]any)
	if pr["cost_usd"] != "0.109" {
		t.Errorf("pr cost = %v, want PRCost's 0.109", pr["cost_usd"])
	}
	if attempts := pr["attempts"].([]any); len(attempts) != 2 {
		t.Errorf("pr attempts = %d, want 2", len(attempts))
	}
	if body["bitbucket_url"] != "https://bitbucket.example.com" {
		t.Errorf("bitbucket_url = %v", body["bitbucket_url"])
	}
	// The full text, code included: the page is reached only by its token.
	f := body["findings"].([]any)[0].(map[string]any)
	if f["comment"] != "Die `Capture` in `RetryPolicy.run` ..." || f["suggestion"] != "idempotency key" || f["dispute"] != "ist gewollt" {
		t.Errorf("finding = %v", f)
	}
	if w.Header().Get("Referrer-Policy") != "no-referrer" || w.Header().Get("X-Robots-Tag") == "" {
		t.Errorf("headers = %v", w.Header())
	}
}

func strp(s string) *string { return &s }

// No gateway discount or margin (decided), and no other run's page: the
// body names no link token.
func TestRunDetailCarriesNoMarginOrOtherRunLink(t *testing.T) {
	h := newDashHarness(t)
	h.st.detailToken = tok
	h.st.detail = store.RunDetail{ID: 1, Outcome: "ok", Run: &store.RunFacts{ID: 1},
		Findings: []store.RunFinding{{Headline: "h"}}, Call: &store.LLMCall{Profile: "p"}}
	body := h.get(t, "/api/dashboard/runs/"+tok).Body.String()
	for _, banned := range []string{"margin", "discount", "original", "token\"", "from_attempt"} {
		if strings.Contains(body, banned) {
			t.Errorf("body has %q: %s", banned, body)
		}
	}
}

// A failed PR total is a failed page, never a null that reads "unpriced"
// and hides a PR at its cap.
func TestRunDetailFailsOnPRCostError(t *testing.T) {
	h := newDashHarness(t)
	h.st.detailToken = tok
	h.st.detail = store.RunDetail{ID: 1, Outcome: "ok"}
	h.st.prCostErr = errors.New("db down")
	if code := h.get(t, "/api/dashboard/runs/"+tok).Code; code != http.StatusInternalServerError {
		t.Errorf("status = %d, want 500", code)
	}
}

// The detail record goes out as the review wrote it, under "detail"; none
// is null.
func TestRunDetailPassesTheRecordThrough(t *testing.T) {
	h := newDashHarness(t)
	h.st.detailToken = tok
	h.st.detail = store.RunDetail{ID: 3, Outcome: "ok", Detail: &store.Pipeline{Stages: []store.StageTime{{Name: "diff", MS: 3}}}}
	var body map[string]any
	if err := json.Unmarshal(h.get(t, "/api/dashboard/runs/"+tok).Body.Bytes(), &body); err != nil {
		t.Fatal(err)
	}
	stages := body["detail"].(map[string]any)["stages"].([]any)
	if len(stages) != 1 || stages[0].(map[string]any)["name"] != "diff" {
		t.Errorf("detail = %v", body["detail"])
	}
	h.st.detail.Detail = nil
	if err := json.Unmarshal(h.get(t, "/api/dashboard/runs/"+tok).Body.Bytes(), &body); err != nil {
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
	h.st.detailToken = tok
	h.st.detail = store.RunDetail{ID: 2, Outcome: "unparseable", AttemptCostNanoUSD: n64(400_000_000)}
	var body map[string]any
	if err := json.Unmarshal(h.get(t, "/api/dashboard/runs/"+tok).Body.Bytes(), &body); err != nil {
		t.Fatal(err)
	}
	if body["cost_usd"] != "0.400" || body["run"] != nil || body["call"] != nil {
		t.Errorf("cost = %v, run = %v, call = %v", body["cost_usd"], body["run"], body["call"])
	}
}

// The PR's reply and question calls go out with the run, each priced at three
// decimals or null when unpriced, and an empty list is [], never null.
func TestRunDetailListsCommentCalls(t *testing.T) {
	h := newDashHarness(t)
	h.st.detailToken = tok
	h.st.detail = store.RunDetail{ID: 4, Outcome: "ok"}
	var body struct {
		PR struct {
			Calls []map[string]any `json:"calls"`
		} `json:"pr"`
	}
	if err := json.Unmarshal(h.get(t, "/api/dashboard/runs/"+tok).Body.Bytes(), &body); err != nil {
		t.Fatal(err)
	}
	if body.PR.Calls == nil {
		t.Error("no calls must be [], not null")
	}
	cost, comment := int64(31_500_000), int64(501)
	h.st.calls = []store.CommentCall{
		{Kind: "reply", CommentID: &comment, Model: "m", CostNanoUSD: &cost, CreatedAt: time.Now()},
		{Kind: "question", Model: "m", CreatedAt: time.Now()},
	}
	if err := json.Unmarshal(h.get(t, "/api/dashboard/runs/"+tok).Body.Bytes(), &body); err != nil {
		t.Fatal(err)
	}
	if len(body.PR.Calls) != 2 || body.PR.Calls[0]["cost_usd"] != "0.032" || body.PR.Calls[1]["cost_usd"] != nil ||
		body.PR.Calls[0]["comment_id"] != float64(501) {
		t.Errorf("calls = %v", body.PR.Calls)
	}
}
