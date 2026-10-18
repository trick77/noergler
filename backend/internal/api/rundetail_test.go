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

func TestRunDetailNotFound(t *testing.T) {
	h := newDashHarness(t)
	h.st.detailID = 7
	for _, path := range []string{"/api/dashboard/runs/8", "/api/dashboard/runs/abc", "/api/dashboard/runs/0", "/api/dashboard/runs/-1"} {
		if code := h.get(t, path).Code; code != http.StatusNotFound {
			t.Errorf("%s: status = %d, want 404", path, code)
		}
	}
}

func TestRunDetailBody(t *testing.T) {
	h := newDashHarness(t)
	h.st.detailID = 7
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
		Findings: []store.FindingHeadline{{Severity: "high", FilePath: "retry.go", Line: 88,
			Headline: "Capture wird wiederholt", CommentID: n64(555), FromAttemptID: n64(7)}},
		PRAttempts: []store.PRAttempt{{ID: 6, Kind: store.RunAuto, Outcome: "skipped", Reason: "head_unchanged",
			CreatedAt: time.Unix(0, 0).UTC()}, {ID: 7, Kind: store.RunAuto, Outcome: "ok",
			CreatedAt: time.Unix(1, 0).UTC(), CostNanoUSD: n64(47_845_000)}},
	}

	w := h.get(t, "/api/dashboard/runs/7")
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
}

// Headlines only (decided): the dashboard is unauthenticated, so the body
// carries no field a finding's text, suggestion or dispute could ride in,
// and no gateway discount or margin.
func TestRunDetailCarriesNoFindingTextOrMargin(t *testing.T) {
	h := newDashHarness(t)
	h.st.detailID = 1
	h.st.detail = store.RunDetail{ID: 1, Outcome: "ok", Run: &store.RunFacts{ID: 1},
		Findings: []store.FindingHeadline{{Headline: "h"}}, Call: &store.LLMCall{Profile: "p"}}
	body := h.get(t, "/api/dashboard/runs/1").Body.String()
	for _, banned := range []string{"comment_text", "suggestion", "dispute", "evidence", "margin", "discount", "original"} {
		if strings.Contains(body, banned) {
			t.Errorf("body has %q: %s", banned, body)
		}
	}
}

// A failed PR total is a failed page, never a null that reads "unpriced"
// and hides a PR at its cap.
func TestRunDetailFailsOnPRCostError(t *testing.T) {
	h := newDashHarness(t)
	h.st.detailID = 1
	h.st.detail = store.RunDetail{ID: 1, Outcome: "ok"}
	h.st.prCostErr = errors.New("db down")
	if code := h.get(t, "/api/dashboard/runs/1").Code; code != http.StatusInternalServerError {
		t.Errorf("status = %d, want 500", code)
	}
}

// An unparseable answer wrote no run, but was billed: the page shows the
// attempt's cost instead of nothing.
// The detail record goes out as the review wrote it, under "detail"; none
// is null.
func TestRunDetailPassesTheRecordThrough(t *testing.T) {
	h := newDashHarness(t)
	h.st.detailID = 3
	h.st.detail = store.RunDetail{ID: 3, Outcome: "ok", Detail: []byte(`{"stages":[{"name":"diff","ms":3}]}`)}
	var body map[string]any
	if err := json.Unmarshal(h.get(t, "/api/dashboard/runs/3").Body.Bytes(), &body); err != nil {
		t.Fatal(err)
	}
	stages := body["detail"].(map[string]any)["stages"].([]any)
	if len(stages) != 1 || stages[0].(map[string]any)["name"] != "diff" {
		t.Errorf("detail = %v", body["detail"])
	}
	h.st.detail.Detail = nil
	if err := json.Unmarshal(h.get(t, "/api/dashboard/runs/3").Body.Bytes(), &body); err != nil {
		t.Fatal(err)
	}
	if body["detail"] != nil {
		t.Errorf("no record: detail = %v, want null", body["detail"])
	}
}

func TestRunDetailBilledWithoutRun(t *testing.T) {
	h := newDashHarness(t)
	h.st.detailID = 2
	h.st.detail = store.RunDetail{ID: 2, Outcome: "unparseable", AttemptCostNanoUSD: n64(400_000_000)}
	var body map[string]any
	if err := json.Unmarshal(h.get(t, "/api/dashboard/runs/2").Body.Bytes(), &body); err != nil {
		t.Fatal(err)
	}
	if body["cost_usd"] != "0.400" || body["run"] != nil || body["call"] != nil {
		t.Errorf("cost = %v, run = %v, call = %v", body["cost_usd"], body["run"], body["call"])
	}
}
