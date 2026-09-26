package main

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/trick77/noergler/internal/evals"
	"github.com/trick77/noergler/internal/inference"
)

// The orchestration is what the patch-coverage gate objects to being
// untested, and it is worth testing on its own terms: the exit code CI reads
// is decided here, and a run that could not happen must not read as a prompt
// regression.

type stubClient struct {
	findings []inference.ReviewFinding
	// owner[i] is a snippet unique to the case findings[i] belongs to. Empty
	// means "answer on every case", which is what the hand-built stubs want.
	owner   []string
	outcome inference.Outcome
}

// Review answers only the findings belonging to the case being reviewed.
// A stub that returned every finding for every case would report eight
// findings on each clean control, which is the invention ErrInvented exists
// to fail - and it would fail on the stub's own sloppiness rather than on
// anything the orchestration did.
//
// Keyed on the case's own diff, not on the file path: unchecked-error and
// wrong-error-wrapped are both on internal/store/claims.go, so a path match
// hands each of them the other's finding as Extra.
func (s stubClient) Review(_ context.Context, req inference.ReviewRequest) inference.ReviewResult {
	var out []inference.ReviewFinding
	for i, f := range s.findings {
		mark := ""
		if i < len(s.owner) {
			mark = s.owner[i]
		}
		if mark == "" || strings.Contains(req.Prompt, mark) {
			out = append(out, f)
		}
	}
	return inference.ReviewResult{
		Outcome: s.outcome,
		Review:  inference.ParsedReview{Findings: out, Summary: inference.NewReviewSummary()},
	}
}

// perfectFindings answers every seeded bug in the corpus exactly: the first
// line of each expected window, with the expectation's first keyword as the
// comment. Derived from the corpus rather than written out, because a
// hardcoded list silently stops being a clean sweep the moment a case is
// added, and the failure reads as a broken orchestration instead of a stale
// fixture.
func perfectFindings(t *testing.T) (stubClient, int) {
	t.Helper()
	cases, err := evals.LoadCorpus()
	if err != nil {
		t.Fatal(err)
	}
	var s stubClient
	seeded := 0
	for _, c := range cases {
		// The case's own diff, which is unique per case and reaches the
		// prompt verbatim. A path would not be: two cases share claims.go.
		mark := ""
		if len(c.Files) > 0 {
			mark = c.Files[0].Diff
		}
		for _, e := range c.Expected {
			seeded++
			s.findings = append(s.findings, inference.ReviewFinding{
				File: e.File, Line: e.Lines[0], Severity: "issue",
				Comment: e.Keywords[0],
			})
			s.owner = append(s.owner, mark)
		}
	}
	return s, seeded
}

func env(extra map[string]string) func(string) string {
	base := map[string]string{
		"EVAL_BASE_URL": "https://example/v1",
		"EVAL_API_KEY":  "k",
		"EVAL_MODEL":    "some-model",
	}
	for k, v := range extra {
		base[k] = v
	}
	return func(k string) string { return base[k] }
}

func baseOptions(t *testing.T, client evals.Reviewer) options {
	t.Helper()
	return options{
		// The real template: an eval that scores a toy prompt says nothing
		// about the one that ships.
		promptPath: "../../../prompts/review.txt",
		effort:     "some-level",
		timeout:    time.Minute,
		getenv:     env(nil),
		stdout:     &strings.Builder{},
		newClient: func(evals.Settings, string) (evals.Reviewer, error) {
			return client, nil
		},
	}
}

// Every seeded bug caught: exit 0, no error, and missed must be false or a
// green run would be reported as a regression.
func TestRun_CleanSweepIsNotAMiss(t *testing.T) {
	// The corpus' own seeded bugs, answered exactly.
	client, _ := perfectFindings(t)
	missed, err := run(context.Background(), baseOptions(t, client))
	if err != nil || missed {
		t.Fatalf("run = (%v, %v), want (false, nil) on a clean sweep", missed, err)
	}
}

// A miss is exit 1: the one signal CI gates on.
func TestRun_MissedBugReportsMissed(t *testing.T) {
	missed, err := run(context.Background(), baseOptions(t, stubClient{}))
	if err == nil {
		t.Fatal("want an error when every seeded bug is missed")
	}
	if !missed {
		t.Error("missed = false on a missed bug; CI would read a regression as exit 2")
	}
	if !strings.Contains(err.Error(), "missed") {
		t.Errorf("error = %q, want it to say what was missed", err)
	}
}

// A case that never completed is exit 2, not 1: it is not a prompt result.
func TestRun_IncompleteRunIsNotAPromptRegression(t *testing.T) {
	client := stubClient{outcome: inference.OutcomeTimedOut}
	missed, err := run(context.Background(), baseOptions(t, client))
	if err == nil {
		t.Fatal("want an error when a case did not complete")
	}
	if missed {
		t.Error("missed = true on a timed-out run; that is exit 2, not a regression")
	}
	if !strings.Contains(err.Error(), "did not complete") {
		t.Errorf("error = %q, want it to name the incomplete run", err)
	}
}

func TestRun_MissingCredentialsNamesThem(t *testing.T) {
	opt := baseOptions(t, stubClient{})
	opt.getenv = func(string) string { return "" }
	missed, err := run(context.Background(), opt)
	if err == nil || missed {
		t.Fatalf("run = (%v, %v), want a plain error naming the variables", missed, err)
	}
	for _, want := range []string{"EVAL_BASE_URL", "EVAL_API_KEY"} {
		if !strings.Contains(err.Error(), want) {
			t.Errorf("error %q does not name %s", err, want)
		}
	}
}

func TestRun_UnreadablePromptIsNotAMiss(t *testing.T) {
	opt := baseOptions(t, stubClient{})
	opt.promptPath = filepath.Join(t.TempDir(), "absent.txt")
	missed, err := run(context.Background(), opt)
	if err == nil || missed {
		t.Fatalf("run = (%v, %v), want a plain error", missed, err)
	}
	if !strings.Contains(err.Error(), "read prompt") {
		t.Errorf("error = %q, want it to name the prompt", err)
	}
}

// A gateway that cannot be reached is exit 2, and the builder's error must
// survive: "not listed by the gateway for this key" is what told us the
// alias was wrong.
func TestRun_ClientBuilderErrorSurvives(t *testing.T) {
	opt := baseOptions(t, stubClient{})
	opt.newClient = func(evals.Settings, string) (evals.Reviewer, error) {
		return nil, errNotServed
	}
	missed, err := run(context.Background(), opt)
	if err == nil || missed {
		t.Fatalf("run = (%v, %v), want the builder's error", missed, err)
	}
	if !strings.Contains(err.Error(), "not listed by the gateway") {
		t.Errorf("error = %q, want the builder's message kept", err)
	}
}

// The -json report is the committed artifact, so it has to be written and
// to carry the score.
func TestRun_WritesTheJSONReport(t *testing.T) {
	client, seeded := perfectFindings(t)
	opt := baseOptions(t, client)
	opt.jsonOut = filepath.Join(t.TempDir(), "run.json")
	if _, err := run(context.Background(), opt); err != nil {
		t.Fatal(err)
	}
	blob, err := os.ReadFile(opt.jsonOut)
	if err != nil {
		t.Fatal(err)
	}
	var score evals.Score
	if err := json.Unmarshal(blob, &score); err != nil {
		t.Fatal(err)
	}
	if score.Seeded != seeded || score.Caught != seeded {
		t.Errorf("report says caught %d of %d, want %d of %d",
			score.Caught, score.Seeded, seeded, seeded)
	}
	// A perfect review invents nothing. This also pins the stub: keyed on
	// the path instead of the case, the two claims.go cases would each
	// receive the other's finding and this would read 2.
	if score.Extra != 0 {
		t.Errorf("extra = %d, want 0 on a perfect review", score.Extra)
	}
	if out := opt.stdout.(*strings.Builder).String(); !strings.Contains(out, "wrote ") {
		t.Errorf("stdout does not mention the report:\n%s", out)
	}
}

// An invented finding on a clean control is exit 1, the same code a miss
// gets: both mean the prompt got worse. Covers the errors.Join wiring, which
// TestRun_MissedBugReportsMissed only reaches through ErrMissed.
func TestRun_InventedFindingOnACleanControlIsExitOne(t *testing.T) {
	client, _ := perfectFindings(t)
	// Answer every case, including the ones that seed nothing.
	client.owner = nil
	missed, err := run(context.Background(), baseOptions(t, client))
	if err == nil {
		t.Fatal("want an error when a clean control gets a finding")
	}
	if !missed {
		t.Error("missed = false, want true: invention shares exit 1 with a miss")
	}
	if !strings.Contains(err.Error(), "invented finding(s)") {
		t.Errorf("error does not name the invention: %v", err)
	}
}

func TestRun_ReportsEveryCaseOnStdout(t *testing.T) {
	_, seeded := perfectFindings(t)
	opt := baseOptions(t, stubClient{})
	if _, err := run(context.Background(), opt); err == nil {
		t.Fatal("want the missed-bug error")
	}
	out := opt.stdout.(*strings.Builder).String()
	for _, want := range []string{"some-model", "effort some-level", "MISS",
		fmt.Sprintf("seeded %d", seeded)} {
		if !strings.Contains(out, want) {
			t.Errorf("stdout does not mention %q:\n%s", want, out)
		}
	}
}

var errNotServed = errorString("gateway unreachable or profile not served: " +
	"model is not listed by the gateway for this key")

type errorString string

func (e errorString) Error() string { return string(e) }

// labelledClient is a reviewer that reports the label a started client has:
// the model plus the level llmwire actually sent.
type labelledClient struct {
	stubClient
	label string
}

func (c labelledClient) Label() string { return c.label }

// The header names the level that was sent, so two runs on one model at
// different resolved levels can be told apart.
func TestRun_HeaderNamesTheLevelSent(t *testing.T) {
	opt := baseOptions(t, labelledClient{label: "some-model-resolved"})
	opt.effort = ""
	_, _ = run(context.Background(), opt)
	out := opt.stdout.(*strings.Builder).String()
	if !strings.Contains(out, "model some-model-resolved via") {
		t.Errorf("stdout does not name the resolved label:\n%s", out)
	}
}
