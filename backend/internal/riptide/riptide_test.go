package riptide

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"log/slog"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"
)

type fake struct {
	srv    *httptest.Server
	calls  int
	paths  []string
	header http.Header
	body   map[string]any
	raw    string
}

func newFake(t *testing.T, handler func(w http.ResponseWriter, r *http.Request)) (*fake, *Emitter) {
	t.Helper()
	f := &fake{}
	f.srv = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		f.calls++
		f.paths = append(f.paths, r.URL.Path)
		f.header = r.Header.Clone()
		if raw, _ := io.ReadAll(r.Body); len(raw) > 0 {
			f.raw = string(raw)
			_ = json.Unmarshal(raw, &f.body)
		}
		handler(w, r)
	}))
	t.Cleanup(f.srv.Close)
	return f, New(f.srv.URL, "t", slog.New(slog.DiscardHandler))
}

func reply(status int, body string) func(http.ResponseWriter, *http.Request) {
	return func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(status)
		_, _ = io.WriteString(w, body)
	}
}

// sampleRollup mirrors the fixture the Python tests used.
func sampleRollup() Rollup {
	cost := int64(12_300_000) // 0.0123 USD
	return Rollup{
		Outcome:               "merged",
		PRKey:                 "PROJ/repo#1",
		Repo:                  "org/repo",
		SourceCommitSHA:       "abc1234567890abc1234567890abc1234567890a",
		MergeCommitSHA:        "def4567890abc1234567890abc1234567890abcd",
		LinesAdded:            120,
		LinesRemoved:          30,
		FilesChanged:          5,
		TotalRuns:             2,
		TotalPromptTokens:     10,
		TotalCompletionTokens: 20,
		TotalElapsedMS:        1500,
		TotalFindingsCount:    2,
		TotalCostNanoUSD:      &cost,
		ModelsUsed:            []string{"gpt-4o"},
		FirstReviewAt:         time.Date(2026, 4, 29, 11, 0, 0, 0, time.UTC),
		ClosedAt:              time.Date(2026, 4, 29, 12, 0, 0, 0, time.UTC),
	}
}

// --- enabled flag ------------------------------------------------------------

func TestEnabledFlag(t *testing.T) {
	log := slog.New(slog.DiscardHandler)
	tests := []struct {
		name, url, token string
		want             bool
	}{
		{"both set", "http://r", "t", true},
		{"no url", "", "t", false},
		{"no token", "http://r", "", false},
		{"neither", "", "", false},
		{"url is only a slash", "/", "t", false},
		{"whitespace only", "   ", "t", false},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			if got := New(tc.url, tc.token, log).Enabled(); got != tc.want {
				t.Errorf("Enabled() = %v, want %v", got, tc.want)
			}
		})
	}
}

// A team without riptide configured must make no calls at all.
func TestDisabledIsANoOp(t *testing.T) {
	e := New("", "", slog.New(slog.DiscardHandler))
	e.EmitPRCompleted(context.Background(), sampleRollup()) // must not panic
	team, err := e.VerifyAtStartup(context.Background())
	if team != "" || err != nil {
		t.Fatalf("VerifyAtStartup = (%q, %v), want empty", team, err)
	}
}

// --- startup verification ----------------------------------------------------

func TestVerifyAtStartupReturnsTeam(t *testing.T) {
	f, e := newFake(t, reply(200, `{"status":"ok","team":"checkout"}`))
	team, err := e.VerifyAtStartup(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	if team != "checkout" {
		t.Errorf("team = %q, want checkout", team)
	}
	if f.paths[0] != pingPath {
		t.Errorf("path = %q, want %q", f.paths[0], pingPath)
	}
	if got := f.header.Get("Authorization"); got != "Bearer t" {
		t.Errorf("Authorization = %q", got)
	}
}

// A dead token must stop the team rather than silently lose weeks of data.
func TestVerifyAtStartup401IsFatal(t *testing.T) {
	_, e := newFake(t, reply(401, `{"detail":"bad token"}`))
	_, err := e.VerifyAtStartup(context.Background())
	if !errors.Is(err, ErrAuth) {
		t.Fatalf("err = %v, want ErrAuth", err)
	}
}

// Anything else is survivable: riptide being unhappy is not our problem.
func TestVerifyAtStartupOtherErrorsAreNotFatal(t *testing.T) {
	for _, status := range []int{403, 404, 500, 503} {
		_, e := newFake(t, reply(status, `nope`))
		team, err := e.VerifyAtStartup(context.Background())
		if err != nil {
			t.Errorf("status %d: err = %v, want nil", status, err)
		}
		if team != "" {
			t.Errorf("status %d: team = %q", status, team)
		}
	}
}

func TestVerifyAtStartupUnreachableIsNotFatal(t *testing.T) {
	e := New("http://127.0.0.1:1", "t", slog.New(slog.DiscardHandler))
	team, err := e.VerifyAtStartup(context.Background())
	if err != nil || team != "" {
		t.Fatalf("got (%q, %v), want an empty team and no error", team, err)
	}
}

// A non-string team, or an unreadable body, leaves the name unknown but boots.
func TestVerifyAtStartupOddBodyIsNotFatal(t *testing.T) {
	for _, body := range []string{`{"team":null}`, `{}`, `not json`, ``} {
		_, e := newFake(t, reply(200, body))
		team, err := e.VerifyAtStartup(context.Background())
		if err != nil {
			t.Errorf("body %q: err = %v", body, err)
		}
		if team != "" {
			t.Errorf("body %q: team = %q", body, team)
		}
	}
}

// --- emitting ----------------------------------------------------------------

func TestEmitPostsTheExpectedBody(t *testing.T) {
	// 202 on purpose: any status under 400 is success, not only 200.
	f, e := newFake(t, reply(202, ``))
	e.EmitPRCompleted(context.Background(), sampleRollup())

	if f.paths[0] != emitPath {
		t.Fatalf("path = %q, want %q", f.paths[0], emitPath)
	}
	if got := f.header.Get("Content-Type"); got != "application/json" {
		t.Errorf("Content-Type = %q", got)
	}
	want := map[string]any{
		"event_type":              "pr_completed",
		"outcome":                 "merged",
		"pr_key":                  "PROJ/repo#1",
		"repo":                    "org/repo",
		"lines_added":             float64(120),
		"lines_removed":           float64(30),
		"files_changed":           float64(5),
		"total_runs":              float64(2),
		"total_prompt_tokens":     float64(10),
		"total_completion_tokens": float64(20),
		"total_elapsed_ms":        float64(1500),
		"total_findings_count":    float64(2),
		"first_review_at":         "2026-04-29T11:00:00Z",
		"closed_at":               "2026-04-29T12:00:00Z",
	}
	for k, v := range want {
		if f.body[k] != v {
			t.Errorf("%s = %#v, want %#v", k, f.body[k], v)
		}
	}
	// Cost crosses as a string so neither side rounds it through a float.
	if got := f.body["total_cost_usd"]; got != "0.0123" {
		t.Errorf("total_cost_usd = %#v, want the string \"0.0123\"", got)
	}
	if models, ok := f.body["models_used"].([]any); !ok || len(models) != 1 || models[0] != "gpt-4o" {
		t.Errorf("models_used = %#v", f.body["models_used"])
	}
}

// An unpriced rollup still ships: the outcome, diff size and token counts are
// worth having. The cost key is absent rather than zero.
func TestEmitOmitsCostWhenUnpriced(t *testing.T) {
	f, e := newFake(t, reply(202, ``))
	r := sampleRollup()
	r.TotalCostNanoUSD = nil
	e.EmitPRCompleted(context.Background(), r)

	if _, present := f.body["total_cost_usd"]; present {
		t.Errorf("total_cost_usd present with value %#v, want the key absent", f.body["total_cost_usd"])
	}
	if f.body["outcome"] != "merged" || f.body["total_runs"] != float64(2) {
		t.Errorf("the rest of the rollup must still ship: %v", f.body)
	}
}

// A declined or deleted PR has no merge commit, and the key is present as null
// rather than missing.
func TestEmitDeclinedSendsNullMergeCommit(t *testing.T) {
	f, e := newFake(t, reply(202, ``))
	r := sampleRollup()
	r.Outcome = "declined"
	r.MergeCommitSHA = ""
	e.EmitPRCompleted(context.Background(), r)

	if f.body["outcome"] != "declined" {
		t.Errorf("outcome = %v", f.body["outcome"])
	}
	v, present := f.body["merge_commit_sha"]
	if !present {
		t.Error("merge_commit_sha missing, want it present and null")
	}
	if v != nil {
		t.Errorf("merge_commit_sha = %#v, want null", v)
	}
}

// The handle and the account kind travel together.
func TestEmitReviewerHandle(t *testing.T) {
	f, e := newFake(t, reply(202, ``))
	r := sampleRollup()
	r.ReviewerHandle = "noergler-svc"
	e.EmitPRCompleted(context.Background(), r)

	if f.body["reviewer_handle"] != "noergler-svc" || f.body["reviewer_account_kind"] != "bot" {
		t.Errorf("handle = %v, kind = %v", f.body["reviewer_handle"], f.body["reviewer_account_kind"])
	}
}

func TestEmitOmitsAbsentReviewerHandle(t *testing.T) {
	f, e := newFake(t, reply(202, ``))
	e.EmitPRCompleted(context.Background(), sampleRollup())

	if _, present := f.body["reviewer_handle"]; present {
		t.Error("reviewer_handle present, want absent")
	}
	if _, present := f.body["reviewer_account_kind"]; present {
		t.Error("reviewer_account_kind present without a handle")
	}
}

// Telemetry must never fail a webhook, and must never be retried.
func TestEmitSwallowsTransportErrors(_ *testing.T) {
	e := New("http://127.0.0.1:1", "t", slog.New(slog.DiscardHandler))
	e.EmitPRCompleted(context.Background(), sampleRollup()) // must not panic or block
}

func TestEmitSwallowsRejections(t *testing.T) {
	for _, status := range []int{400, 422, 500} {
		f, e := newFake(t, reply(status, `{"detail":"nope"}`))
		e.EmitPRCompleted(context.Background(), sampleRollup())
		if f.calls != 1 {
			t.Errorf("status %d: made %d calls, want exactly 1 (never retried)", status, f.calls)
		}
	}
}

// Redirects are not followed, so a 3xx lands here as a response. It has to count
// as a miss: a redirected endpoint would otherwise drop every rollup silently,
// and the claim taken before the POST means there is no second attempt.
func TestEmitTreatsRedirectAsAMiss(t *testing.T) {
	var logged bool
	f, e := newFake(t, func(w http.ResponseWriter, r *http.Request) {
		http.Redirect(w, r, "https://elsewhere.example.com/webhooks/noergler", http.StatusMovedPermanently)
	})
	e.log = slog.New(slog.NewTextHandler(io.Discard, &slog.HandlerOptions{
		Level: slog.LevelWarn,
		ReplaceAttr: func([]string, slog.Attr) slog.Attr {
			logged = true
			return slog.Attr{}
		},
	}))
	e.EmitPRCompleted(context.Background(), sampleRollup())

	if f.calls != 1 {
		t.Errorf("made %d calls, want 1 (the redirect must not be followed)", f.calls)
	}
	if !logged {
		t.Error("a redirected emit produced no warning; it would vanish silently")
	}
}

// Same for the ping: a redirect is a misconfigured URL, not a riptide problem.
func TestVerifyAtStartupTreatsRedirectAsAnUnexpectedStatus(t *testing.T) {
	_, e := newFake(t, func(w http.ResponseWriter, r *http.Request) {
		http.Redirect(w, r, "https://elsewhere.example.com/auth/ping", http.StatusFound)
	})
	team, err := e.VerifyAtStartup(context.Background())
	if err != nil {
		t.Fatalf("err = %v, want nil: a redirect is survivable", err)
	}
	if team != "" {
		t.Errorf("team = %q, want empty", team)
	}
}

// --- formatting --------------------------------------------------------------

// Money is rendered as a plain decimal: no exponent (Python's Decimal produced
// "1E-9" for a single nano-USD, which a strict parser may reject) and no
// trailing zeros.
func TestFormatNanoUSD(t *testing.T) {
	tests := []struct {
		nano int64
		want string
	}{
		{12_300_000, "0.0123"}, // the Python fixture value
		{0, "0"},
		{1, "0.000000001"},
		{1_000_000_000, "1"},
		{999_999_999, "0.999999999"},
		{50_000_000, "0.05"},
		{1_234_567_890_123, "1234.567890123"},
		{2_500_000_000, "2.5"},
		{-12_300_000, "-0.0123"},
	}
	for _, tc := range tests {
		if got := FormatNanoUSD(tc.nano); got != tc.want {
			t.Errorf("FormatNanoUSD(%d) = %q, want %q", tc.nano, got, tc.want)
		}
	}
}

func TestFormatTime(t *testing.T) {
	tests := []struct {
		name string
		in   time.Time
		want string
	}{
		{
			"whole second",
			time.Date(2026, 4, 29, 12, 0, 0, 0, time.UTC),
			"2026-04-29T12:00:00Z",
		},
		{
			"microseconds are six digits",
			time.Date(2026, 4, 29, 12, 0, 0, 123456000, time.UTC),
			"2026-04-29T12:00:00.123456Z",
		},
		{
			"a non-UTC zone is converted",
			time.Date(2026, 4, 29, 14, 0, 0, 0, time.FixedZone("CEST", 2*60*60)),
			"2026-04-29T12:00:00Z",
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			if got := formatTime(tc.in); got != tc.want {
				t.Errorf("formatTime() = %q, want %q", got, tc.want)
			}
		})
	}
}

// models_used must stay an array even when empty: riptide expects a list.
func TestEmitSendsEmptyModelsAsAnArray(t *testing.T) {
	f, e := newFake(t, reply(202, ``))
	r := sampleRollup()
	r.ModelsUsed = nil
	e.EmitPRCompleted(context.Background(), r)

	if models, ok := f.body["models_used"].([]any); !ok || len(models) != 0 {
		t.Errorf("models_used = %#v, want an empty array", f.body["models_used"])
	}
}

func TestNewTrimsTrailingSlash(t *testing.T) {
	e := New("http://riptide.example.com/", "t", slog.New(slog.DiscardHandler))
	if e.url != "http://riptide.example.com" {
		t.Errorf("url = %q", e.url)
	}
}
