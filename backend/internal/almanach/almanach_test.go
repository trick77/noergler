package almanach

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"net/http/httptest"
	"os"
	"reflect"
	"testing"
	"time"

	"github.com/trick77/noergler/internal/store"
)

type fake struct {
	srv    *httptest.Server
	calls  int
	paths  []string
	header http.Header
	body   map[string]any
}

func newFake(t *testing.T, handler func(w http.ResponseWriter, r *http.Request)) (*fake, *Emitter) {
	t.Helper()
	f := &fake{}
	f.srv = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		f.calls++
		f.paths = append(f.paths, r.URL.Path)
		f.header = r.Header.Clone()
		if raw, _ := io.ReadAll(r.Body); len(raw) > 0 {
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

func nano(n int64) *int64 { return &n }

func sampleNote() Note {
	return Note{
		ProjectKey: "PROJ", RepoSlug: "repo", PRID: 1, PRKey: "PROJ/repo#1", PRTitle: "Fix the thing",
		TicketKey: "PROJ-9", JiraIssueType: "Bug", ChangeKind: "bugfix",
		ReviewedCommit: "abc1234", SourceCommitSHA: "abc1234", MergeCommitSHA: "",
		MergedAt: time.Date(2026, 10, 28, 12, 0, 0, 0, time.UTC), RunID: 77, ModelLabel: "gpt-5.5-high",
		IssuesPosted: 2, SuggestionsPosted: 1,
		Note:   json.RawMessage(`{"summary":"s","change_kind":"bugfix"}`),
		Claims: []Claim{{ProjectKey: "PROJ"}, {ProjectKey: "OTHER", RepoSlug: "x"}},
		NoteCall: &Call{Outcome: "ok", NoteStatus: "written", LLMCall: store.LLMCall{Model: "m", CostNanoUSD: nano(12_300_000),
			CacheReadTokens: nano(90_000), Warnings: nil}},
		Rounds: Rounds{Count: 3, InputTokens: 300_000, OutputTokens: 1_500, CostNanoUSD: nano(40_000_000)},
		Review: ReviewTotals{Runs: 3, PromptTokens: 10, CompletionTokens: 20, ElapsedMS: 1500, CostNanoUSD: nil, Models: []string{"gpt-5.5"}},
	}
}

func TestEnabledFlag(t *testing.T) {
	log := slog.New(slog.DiscardHandler)
	for _, tc := range []struct {
		name, url, token string
		want             bool
	}{
		{"both set", "http://a", "t", true},
		{"no url", "", "t", false},
		{"no token", "http://a", "", false},
		{"url is only a slash", "/", "t", false},
	} {
		if got := New(tc.url, tc.token, log).Enabled(); got != tc.want {
			t.Errorf("%s: Enabled() = %v, want %v", tc.name, got, tc.want)
		}
	}
	var none *Emitter
	if none.Enabled() || none.Status() != StatusOff {
		t.Error("a nil emitter is off")
	}
}

func TestDisabledIsANoOp(t *testing.T) {
	e := New("", "", slog.New(slog.DiscardHandler))
	if e.EmitNote(context.Background(), sampleNote()) {
		t.Error("a disabled emitter reported a note taken")
	}
	if team, err := e.VerifyAtStartup(context.Background()); team != "" || err != nil {
		t.Fatalf("VerifyAtStartup = (%q, %v), want empty", team, err)
	}
}

func TestVerifyAtStartup(t *testing.T) {
	f, e := newFake(t, reply(200, `{"status":"ok","team":"checkout"}`))
	team, err := e.VerifyAtStartup(context.Background())
	if err != nil || team != "checkout" || f.paths[0] != pingPath || f.header.Get("Authorization") != "Bearer t" {
		t.Errorf("team %q err %v path %q auth %q", team, err, f.paths[0], f.header.Get("Authorization"))
	}
	if e.Status() != StatusOK {
		t.Errorf("status = %q, want ok", e.Status())
	}

	_, e = newFake(t, reply(401, `{"detail":"bad token"}`))
	if _, err := e.VerifyAtStartup(context.Background()); !errors.Is(err, ErrAuth) {
		t.Fatalf("401: err = %v, want ErrAuth", err)
	}
	for _, status := range []int{403, 404, 500, 503} {
		_, e := newFake(t, reply(status, `nope`))
		if team, err := e.VerifyAtStartup(context.Background()); err != nil || team != "" {
			t.Errorf("status %d: (%q, %v), want survivable", status, team, err)
		}
	}
	e = New("http://127.0.0.1:1", "t", slog.New(slog.DiscardHandler))
	if team, err := e.VerifyAtStartup(context.Background()); err != nil || team != "" || e.Status() != StatusFailing {
		t.Errorf("unreachable: (%q, %v, %s)", team, err, e.Status())
	}
	for _, body := range []string{`{"team":null}`, `{}`, `not json`, ``} {
		_, e := newFake(t, reply(200, body))
		if team, err := e.VerifyAtStartup(context.Background()); err != nil || team != "" {
			t.Errorf("body %q: (%q, %v)", body, team, err)
		}
	}
}

// The payload carries the note, the structure, the review totals and the
// gateway's figures, money as exact strings beside the integers; a 2xx is
// "taken", which is what lets the caller mark the row emitted.
func TestEmitNotePayloadAndOutcome(t *testing.T) {
	f, e := newFake(t, reply(202, `{"status":"accepted"}`))
	if !e.EmitNote(context.Background(), sampleNote()) {
		t.Fatal("a 202 must report the note taken")
	}
	if f.paths[0] != emitPath || f.header.Get("Authorization") != "Bearer t" || f.header.Get("Content-Type") != "application/json" {
		t.Errorf("request = %s %v", f.paths[0], f.header)
	}
	// The shape is the golden's (TestPayloadGolden); here only what the
	// golden cannot show: the sparse lanes travel as null, never as zero.
	b := f.body
	if b["event_type"] != EventType || b["merge_commit_sha"] != nil {
		t.Errorf("identity = %v", b)
	}
	call := b["note_call"].(map[string]any)
	if call["input_tokens"] != nil || call["cost_usd"] != "0.0123" {
		t.Errorf("note_call = %v", call)
	}
	if w, ok := call["warnings"].([]any); !ok || len(w) != 0 {
		t.Errorf("warnings = %v, want an empty array", call["warnings"])
	}
	rv := b["review_totals"].(map[string]any)
	if rv["cost_usd"] != nil || rv["cost_nano_usd"] != nil {
		t.Errorf("review_totals = %v: an unpriced review is null, never 0", rv)
	}
	if e.Status() != StatusOK {
		t.Errorf("status = %q", e.Status())
	}

	n := sampleNote()
	n.NoteCall = nil
	n.Note = nil
	f, e = newFake(t, reply(202, ``))
	e.EmitNote(context.Background(), n)
	if f.body["note_call"] != nil || f.body["note"] != nil {
		t.Errorf("absent call and note = %v %v, want null", f.body["note_call"], f.body["note"])
	}
}

// goldenNote is a note with every lane set and every value distinct from
// its neighbours, so the golden shows Almanach the JSON type of each field
// and a parser reading a sibling key by mistake cannot pass.
func goldenNote() Note {
	return Note{
		ProjectKey: "PROJ", RepoSlug: "repo", PRID: 42, PRKey: "PROJ/repo#42", PRTitle: "Fix the upload retry",
		TicketKey: "PROJ-9", JiraIssueType: "Bug", ChangeKind: "bugfix",
		ReviewedCommit: "abc1234abc1234", SourceCommitSHA: "def5678def5678", MergeCommitSHA: "0123456789abcdef",
		MergedAt: time.Date(2026, 10, 28, 12, 0, 0, 0, time.UTC), RunID: 77, ModelLabel: "gpt-5.5-high",
		IssuesPosted: 2, SuggestionsPosted: 1,
		Note: json.RawMessage(`{"summary":"Der Adapter liest den Mandanten aus der Kopfzeile.","change_kind":"bugfix",` +
			`"bugfix":{"symptom":"Upload bricht ab.","cause":"Retry ohne Backoff.","cause_kind":"logic"},` +
			`"components":[{"name":"Uploader","text":"Nimmt Dateien an.","files":["internal/files/upload.go"]}],` +
			`"rules":[],"invariants":[],"entities":[]}`),
		Claims: []Claim{{ProjectKey: "PROJ"}, {ProjectKey: "OTHER", RepoSlug: "x"}},
		NoteCall: &Call{Outcome: "ok", NoteStatus: "written", LLMCall: store.LLMCall{
			Profile: "gpt-5.5-high", Model: "gpt-5.5", Deployment: "azure/gpt-5.5", ReasoningSent: "high", FinishReason: "stop",
			CallID: "call-1", GatewayVer: "1.80.0",
			InputTokens: nano(100_000), InputUncachedTokens: nano(10_000), CacheReadTokens: nano(90_000), CacheWriteTokens: nano(0),
			OutputTokens: nano(800), ReasoningTokens: nano(300),
			CostNanoUSD: nano(12_300_000), CostInputNanoUSD: nano(2_000_000), CostCacheReadNanoUSD: nano(9_000_000),
			CostCacheWriteNanoUSD: nano(0), CostOutputNanoUSD: nano(1_000_000), CostReasoningNanoUSD: nano(300_000),
			CostToolUsageNanoUSD: nano(0), KeySpendNanoUSD: nano(12_500_000_000), HeadersMS: nano(320), TotalMS: nano(2100),
			Warnings: []string{"unaccounted"},
		}},
		Rounds: Rounds{Count: 3, InputTokens: 300_000, OutputTokens: 2_400, CostNanoUSD: nano(40_000_000)},
		Review: ReviewTotals{Runs: 4, PromptTokens: 900_000, CachedTokens: 600_000, CompletionTokens: 5_000, ElapsedMS: 61_500,
			CostNanoUSD: nano(1_250_000_000), Models: []string{"gpt-5.5"}},
	}
}

// The wire shape is a contract with Almanach, pinned as a file: Almanach
// keeps a copy and parses it in its ingest test, and its gate diffs the
// copy against this file on master, so a key renamed or added on one side
// fails a test on the other. Regenerate with ALMANACH_GOLDEN_UPDATE=1.
func TestPayloadGolden(t *testing.T) {
	got, err := json.MarshalIndent(payload(goldenNote()), "", "  ")
	if err != nil {
		t.Fatal(err)
	}
	got = append(got, '\n')
	const path = "testdata/pr_merged_note.json"
	if os.Getenv("ALMANACH_GOLDEN_UPDATE") != "" {
		if err := os.WriteFile(path, got, 0o644); err != nil {
			t.Fatal(err)
		}
	}
	want, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	// Compared as JSON, not bytes: a CRLF checkout must not fail it.
	var gotV, wantV any
	if err := json.Unmarshal(got, &gotV); err != nil {
		t.Fatal(err)
	}
	if err := json.Unmarshal(want, &wantV); err != nil {
		t.Fatalf("%s: %v", path, err)
	}
	if diff := firstDiff("", gotV, wantV); diff != "" {
		t.Errorf("payload differs from %s at %s; regenerate with ALMANACH_GOLDEN_UPDATE=1 go test ./internal/almanach (from backend/) and copy the file to Almanach",
			path, diff)
	}
}

// firstDiff names the first path where two decoded JSON values differ,
// "" when they are equal.
func firstDiff(path string, a, b any) string {
	switch av := a.(type) {
	case map[string]any:
		bv, ok := b.(map[string]any)
		if !ok {
			return path + " (object vs " + fmt.Sprintf("%T", b) + ")"
		}
		keys := map[string]bool{}
		for k := range av {
			keys[k] = true
		}
		for k := range bv {
			keys[k] = true
		}
		for k := range keys {
			if d := firstDiff(path+"."+k, av[k], bv[k]); d != "" {
				return d
			}
		}
		return ""
	case []any:
		bv, ok := b.([]any)
		if !ok || len(av) != len(bv) {
			return path + " (array length)"
		}
		for i := range av {
			if d := firstDiff(fmt.Sprintf("%s[%d]", path, i), av[i], bv[i]); d != "" {
				return d
			}
		}
		return ""
	default:
		if !reflect.DeepEqual(a, b) {
			return fmt.Sprintf("%s (%v vs %v)", path, a, b)
		}
		return ""
	}
}

// Anything but a 2xx is "not taken": the note stays pending, the status
// word says why in the only vocabulary the dashboard has.
func TestEmitNoteFailures(t *testing.T) {
	for _, c := range []struct {
		status int
		want   string
	}{{401, StatusRejected}, {403, StatusRejected}, {302, StatusFailing}, {500, StatusFailing}, {422, StatusFailing}} {
		_, e := newFake(t, reply(c.status, `nope`))
		if e.EmitNote(context.Background(), sampleNote()) {
			t.Errorf("status %d reported taken", c.status)
		}
		if e.Status() != c.want {
			t.Errorf("status %d: word %q, want %q", c.status, e.Status(), c.want)
		}
	}
	e := New("http://127.0.0.1:1", "t", slog.New(slog.DiscardHandler))
	if e.EmitNote(context.Background(), sampleNote()) || e.Status() != StatusFailing {
		t.Errorf("unreachable: taken or status %q", e.Status())
	}
	// A cancelled context says nothing about Almanach: the word stays.
	_, e = newFake(t, func(_ http.ResponseWriter, r *http.Request) {
		<-r.Context().Done()
	})
	e.c.Set(StatusOK)
	ctx, cancel := context.WithCancel(context.Background())
	go func() { time.Sleep(20 * time.Millisecond); cancel() }()
	if e.EmitNote(ctx, sampleNote()) || e.Status() != StatusOK {
		t.Errorf("cancelled: taken or status %q", e.Status())
	}
}
