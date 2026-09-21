package main

import (
	"context"
	"encoding/json"
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
	outcome  inference.Outcome
}

func (s stubClient) Review(context.Context, inference.ReviewRequest) inference.ReviewResult {
	return inference.ReviewResult{
		Outcome: s.outcome,
		Review:  inference.ParsedReview{Findings: s.findings, Summary: inference.NewReviewSummary()},
	}
}

func env(extra map[string]string) func(string) string {
	base := map[string]string{
		"EVAL_BASE_URL": "https://example/v1",
		"EVAL_API_KEY":  "k",
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
		effort:     "high",
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
	client := stubClient{findings: []inference.ReviewFinding{
		{File: "internal/review/summary.go", Line: 19, Severity: "issue",
			Comment: "r.Ticket is nil here, so this dereference will panic"},
		{File: "internal/store/claims.go", Line: 21, Severity: "issue",
			Comment: "the Commit error is ignored, so a failed commit reports success"},
	}}
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
	client := stubClient{findings: []inference.ReviewFinding{
		{File: "internal/review/summary.go", Line: 19, Severity: "issue",
			Comment: "nil dereference panics here"},
		{File: "internal/store/claims.go", Line: 21, Severity: "issue",
			Comment: "the Commit error is ignored"},
	}}
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
	if score.Seeded != 2 || score.Caught != 2 {
		t.Errorf("report says caught %d of %d, want 2 of 2", score.Caught, score.Seeded)
	}
	if out := opt.stdout.(*strings.Builder).String(); !strings.Contains(out, "wrote ") {
		t.Errorf("stdout does not mention the report:\n%s", out)
	}
}

func TestRun_ReportsEveryCaseOnStdout(t *testing.T) {
	opt := baseOptions(t, stubClient{})
	if _, err := run(context.Background(), opt); err == nil {
		t.Fatal("want the missed-bug error")
	}
	out := opt.stdout.(*strings.Builder).String()
	for _, want := range []string{"mimo-v2.5-pro", "effort high", "MISS", "seeded 2"} {
		if !strings.Contains(out, want) {
			t.Errorf("stdout does not mention %q:\n%s", want, out)
		}
	}
}

var errNotServed = errorString("gateway unreachable or profile not served: " +
	"model is not listed by the gateway for this key")

type errorString string

func (e errorString) Error() string { return string(e) }
