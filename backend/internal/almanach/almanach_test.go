package almanach

import (
	"context"
	"encoding/json"
	"errors"
	"flag"
	"io"
	"log/slog"
	"net/http"
	"net/http/httptest"
	"os"
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
	b := f.body
	if b["event_type"] != EventType || b["pr_key"] != "PROJ/repo#1" || b["merge_commit_sha"] != nil || b["source_commit_sha"] != "abc1234" {
		t.Errorf("identity = %v", b)
	}
	if b["merged_at"] != "2026-10-28T12:00:00Z" || b["issues_posted"] != float64(2) || b["change_kind"] != "bugfix" {
		t.Errorf("facts = %v", b)
	}
	if note, ok := b["note"].(map[string]any); !ok || note["summary"] != "s" {
		t.Errorf("note = %v", b["note"])
	}
	claims := b["claims"].([]any)
	if len(claims) != 2 || claims[0].(map[string]any)["repo_slug"] != nil || claims[1].(map[string]any)["repo_slug"] != "x" {
		t.Errorf("claims = %v", claims)
	}
	call := b["note_call"].(map[string]any)
	if call["cost_usd"] != "0.0123" || call["cost_nano_usd"] != float64(12_300_000) || call["input_tokens"] != nil || call["cache_read_tokens"] != float64(90_000) {
		t.Errorf("note_call = %v", call)
	}
	if w, ok := call["warnings"].([]any); !ok || len(w) != 0 {
		t.Errorf("warnings = %v, want an empty array", call["warnings"])
	}
	rounds := b["rounds"].(map[string]any)
	if rounds["count"] != float64(3) || rounds["cost_usd"] != "0.04" {
		t.Errorf("rounds = %v", rounds)
	}
	rv := b["review_totals"].(map[string]any)
	if rv["runs"] != float64(3) || rv["cost_usd"] != nil || rv["cost_nano_usd"] != nil {
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

var update = flag.Bool("update", false, "rewrite testdata/pr_merged_note.json from the current payload")

// The wire shape is a contract with Almanach, pinned as a file: Almanach's
// ingest test parses this same file, so a key renamed on one side fails a
// test on the other. Regenerate with -update and copy it over.
func TestPayloadGolden(t *testing.T) {
	got, err := json.MarshalIndent(payload(sampleNote()), "", "  ")
	if err != nil {
		t.Fatal(err)
	}
	got = append(got, '\n')
	const path = "testdata/pr_merged_note.json"
	if *update {
		if err := os.WriteFile(path, got, 0o644); err != nil {
			t.Fatal(err)
		}
	}
	want, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	if string(got) != string(want) {
		t.Errorf("payload differs from %s; run go test ./internal/almanach -update and copy the file to Almanach\n%s", path, got)
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
