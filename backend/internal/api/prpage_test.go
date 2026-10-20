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

const prPath = "/api/dashboard/prs/" + tok

func TestPRPageNotFound(t *testing.T) {
	h := newDashHarness(t)
	h.st.prToken = "BBBBBBBBBBBBBBBBBBBBBB"
	for _, path := range []string{prPath, "/api/dashboard/prs/abc", "/api/dashboard/prs/7",
		"/api/dashboard/prs/BBBBBBBBBBBBBBBBBBBBB", "/api/dashboard/prs/BBBBBBBBBBBBBBBBBBBBBBB"} {
		if w := h.get(t, path); w.Code != http.StatusNotFound {
			t.Errorf("%s: status = %d, want 404", path, w.Code)
		}
	}
}

// prFixture is a PR with a full run, a skip and an incremental run that
// fixed the first run's finding.
func prFixture() store.PRPage {
	t0 := time.Unix(0, 0).UTC()
	return store.PRPage{
		ID: 9, Key: store.PRKey{Project: "PAY", Repo: "checkout", PRID: 1482}, TeamSlug: "platform", Title: "Retry",
		// The store's sums over the whole history, not the three runs below.
		Totals: store.PRTotals{Runs: 7, ElapsedMS: 223_400, PromptTokens: 188_000, CachedTokens: 24_000, CompletionTokens: 14_000},
		More:   true,
		Runs: []store.PRRun{
			{N: 1, Kind: store.RunAuto, Outcome: "ok", ElapsedMS: n64(72_000), CreatedAt: t0, CostNanoUSD: n64(143_000_000),
				Run: &store.RunFacts{ID: 31, PromptTokens: 47_000, CompletionTokens: 9_000, LinesAdded: 412, FilesChanged: 11},
				Detail: &store.Pipeline{
					Stages:  []store.StageTime{{Name: "diff", MS: 10_000}, {Name: "queued", MS: 12_000}, {Name: "gateway", MS: 30_000}, {Name: "to_post", MS: 1_000}},
					Files:   []store.FileSent{{Path: "b.go", Sent: "full"}, {Path: "a.go", Sent: "full"}},
					PRScope: &store.PRScope{Added: 412, Removed: 88, Files: 11}}},
			{N: 2, Kind: store.RunAuto, Outcome: "skipped", Reason: "head_unchanged", ElapsedMS: n64(400), CreatedAt: t0},
			{N: 3, Kind: store.RunMention, Outcome: "ok", ElapsedMS: n64(51_000), CreatedAt: t0, CostNanoUSD: n64(61_000_000),
				Run:   &store.RunFacts{ID: 32, Incremental: true, PromptTokens: 41_000, CachedTokens: 24_000, CompletionTokens: 5_000, LinesAdded: 40, FilesChanged: 1},
				Fixed: 1,
				Detail: &store.Pipeline{
					Stages: []store.StageTime{{Name: "queued", MS: 2_000}, {Name: "gateway", MS: 25_000}},
					Files:  []store.FileSent{{Path: "a.go", Sent: "diff"}}}},
		},
		Findings: []store.RunFinding{
			{Severity: "issue", FilePath: "a.go", Line: 3, Headline: "old one", CommentText: "SECRET BODY", RunID: 31, FixedRunID: n64(32), Fixed: true},
			{Severity: "suggestion", FilePath: "b.go", Line: 7, Headline: "new one", RunID: 32, Replied: true, CommentID: n64(555)},
			// A finding of a run cut from the history shown.
			{Severity: "issue", FilePath: "c.go", Line: 1, Headline: "cut", RunID: 5},
		},
	}
}

func TestPRPageBody(t *testing.T) {
	h := newDashHarness(t)
	h.st.prToken, h.st.page = tok, prFixture()
	h.st.prCost = n64(207_000_000)
	w := h.get(t, prPath)
	if w.Code != http.StatusOK {
		t.Fatalf("status = %d: %s", w.Code, w.Body.String())
	}
	var body prPageBody
	if err := json.Unmarshal(w.Body.Bytes(), &body); err != nil {
		t.Fatal(err)
	}
	if body.Tag != "PAY/checkout#1482" || body.TeamName != "Platform Engineering" || body.State != "open" || body.Title != "Retry" {
		t.Errorf("identity = %+v", body)
	}
	// PRCost's figure, never a sum of the rows ($0.204 here).
	if body.Cost == nil || *body.Cost != "0.207" {
		t.Errorf("cost = %v, want PRCost's 0.207", body.Cost)
	}
	// The whole PR from the latest run that read it, not a sum of runs.
	if body.Scope == nil || *body.Scope != (store.PRScope{Added: 412, Removed: 88, Files: 11}) {
		t.Errorf("scope = %+v", body.Scope)
	}
	// Tokens, time and the count over the whole history; waits over the
	// runs listed, the only ones with their stages read.
	if body.PromptTokens != 188_000 || body.CachedTokens != 24_000 || body.CompletionTokens != 14_000 {
		t.Errorf("tokens = %d / %d / %d", body.PromptTokens, body.CachedTokens, body.CompletionTokens)
	}
	if body.ElapsedMS != 223_400 || body.WaitMS != 15_000 || body.RunsTotal != 7 || !body.MoreRuns {
		t.Errorf("elapsed = %d, wait = %d, runs = %d, more = %v", body.ElapsedMS, body.WaitMS, body.RunsTotal, body.MoreRuns)
	}
	if len(body.Runs) != 3 || body.Runs[1].Time != nil || body.Runs[1].ReasonLabel == "" {
		t.Fatalf("runs = %+v", body.Runs)
	}
	if tm := body.Runs[0].Time; tm == nil || *tm != (stageSplitBody{WorkMS: 10_000, WaitMS: 13_000, GatewayMS: 30_000}) {
		t.Errorf("run 1 time = %+v", tm)
	}
	if r := body.Runs[2]; r.Fixed != 1 || r.Run == nil || !r.Run.Incremental || r.Kind != "mention" {
		t.Errorf("run 3 = %+v", r)
	}
	// Headline only, the runs by number; a run outside the history is null.
	f := body.Findings
	if len(f) != 3 || f[0].PostedIn == nil || *f[0].PostedIn != 1 || f[0].FixedIn == nil || *f[0].FixedIn != 3 ||
		f[1].PostedIn == nil || *f[1].PostedIn != 3 || f[1].FixedIn != nil || !f[1].Replied || f[2].PostedIn != nil {
		t.Errorf("findings = %+v", f)
	}
	if strings.Contains(w.Body.String(), "SECRET BODY") {
		t.Errorf("finding text on the PR page: %s", w.Body.String())
	}
	// Every file any run listed, sorted, with each run's treatment.
	if len(body.Files) != 2 || body.Files[0].Path != "a.go" || len(body.Files[0].Runs) != 2 ||
		body.Files[0].Runs[1] != (prFileSent{N: 3, Sent: "diff"}) || body.Files[0].Findings != 1 || body.Files[1].Findings != 1 {
		t.Errorf("files = %+v", body.Files)
	}
	if body.Calls == nil || body.Findings == nil {
		t.Error("empty lists must be [], not null")
	}
	if strings.Contains(w.Body.String(), tok) {
		t.Errorf("body carries the link token: %s", w.Body.String())
	}
	if w.Header().Get("Cache-Control") != "no-store" {
		t.Errorf("Cache-Control = %q", w.Header().Get("Cache-Control"))
	}
}

// No run that read the whole diff: the scope is null, never zeros and never
// a run's own increment.
func TestPRPageScopeUnknown(t *testing.T) {
	h := newDashHarness(t)
	pg := prFixture()
	pg.Runs[0].Detail.PRScope = nil
	pg.MergedAt = &time.Time{}
	h.st.prToken, h.st.page = tok, pg
	var body map[string]any
	if err := json.Unmarshal(h.get(t, prPath).Body.Bytes(), &body); err != nil {
		t.Fatal(err)
	}
	if v, ok := body["scope"]; !ok || v != nil {
		t.Errorf("scope = %v, want null", v)
	}
	if body["state"] != "merged" {
		t.Errorf("state = %v", body["state"])
	}
}

// A failed PR total is a failed page, never a null that reads "unpriced"
// and hides a PR at its cap.
func TestPRPageFailsOnPRCostError(t *testing.T) {
	h := newDashHarness(t)
	h.st.prToken, h.st.page = tok, prFixture()
	h.st.prCostErr = errors.New("db down")
	if code := h.get(t, prPath).Code; code != http.StatusInternalServerError {
		t.Errorf("status = %d, want 500", code)
	}
}

// The PR's reply and question calls, each priced at three decimals or null
// when unpriced; an empty list is [], never null.
func TestPRPageListsCommentCalls(t *testing.T) {
	h := newDashHarness(t)
	h.st.prToken = tok
	cost, comment := int64(31_500_000), int64(501)
	h.st.calls = []store.CommentCall{
		{Kind: "reply", CommentID: &comment, Model: "m", CostNanoUSD: &cost, CreatedAt: time.Now()},
		{Kind: "question", Model: "m", CreatedAt: time.Now()},
	}
	var body struct {
		Calls []map[string]any `json:"calls"`
		Runs  []any            `json:"runs"`
	}
	if err := json.Unmarshal(h.get(t, prPath).Body.Bytes(), &body); err != nil {
		t.Fatal(err)
	}
	if body.Runs == nil {
		t.Error("no runs must be [], not null")
	}
	if len(body.Calls) != 2 || body.Calls[0]["cost_usd"] != "0.032" || body.Calls[1]["cost_usd"] != nil ||
		body.Calls[0]["comment_id"] != float64(501) {
		t.Errorf("calls = %v", body.Calls)
	}
}
