package evals

import (
	"context"
	"encoding/json"
	"errors"
	"regexp"
	"slices"
	"strconv"
	"strings"
	"testing"

	"github.com/trick77/noergler/internal/inference"
)

// stubReviewer answers each case with canned findings, so the scoring is
// testable without a gateway. The eval tool has to be pinned like anything
// else: a scorer that quietly counts a miss as a catch would make every
// future prompt change look safe.
type stubReviewer struct {
	findings []inference.ReviewFinding
	outcome  inference.Outcome
}

func (s stubReviewer) Review(context.Context, inference.ReviewRequest) inference.ReviewResult {
	return inference.ReviewResult{
		Outcome: s.outcome,
		Review:  inference.ParsedReview{Findings: s.findings, Summary: inference.NewReviewSummary()},
	}
}

func finding(file string, line int, comment string) inference.ReviewFinding {
	return inference.ReviewFinding{File: file, Line: line, Severity: "issue", Comment: comment}
}

func oneCase(expected ...Expected) []Case {
	return []Case{{Name: "c", Expected: expected}}
}

var seeded = Expected{
	ID: "bug", File: "a/b.go", Lines: [2]int{10, 14},
	Keywords: []string{"nil", "panic"}, Why: "x",
}

func countStub(s string) int { return len(s) / 4 }

func TestRun_CatchesASeededBug(t *testing.T) {
	client := stubReviewer{findings: []inference.ReviewFinding{
		finding("a/b.go", 12, "this can panic when the pointer is nil"),
	}}
	score := Run(context.Background(), client, "{files}", oneCase(seeded), countStub)
	if score.Caught != 1 || score.Seeded != 1 {
		t.Fatalf("caught %d of %d, want 1 of 1", score.Caught, score.Seeded)
	}
	if score.Extra != 0 {
		t.Errorf("extra = %d, want 0: the finding matched a seeded bug", score.Extra)
	}
}

func TestRun_FindingOutsideTheLineWindowIsAMiss(t *testing.T) {
	client := stubReviewer{findings: []inference.ReviewFinding{
		finding("a/b.go", 40, "this can panic when the pointer is nil"),
	}}
	score := Run(context.Background(), client, "{files}", oneCase(seeded), countStub)
	if score.Caught != 0 {
		t.Fatalf("caught %d, want 0: line 40 is outside 10-14", score.Caught)
	}
	if score.Extra != 1 {
		t.Errorf("extra = %d, want 1: the finding pinned nothing", score.Extra)
	}
	if why := score.Results[0].Matches[0].Why; !strings.Contains(why, "outside") {
		t.Errorf("Why = %q, want it to name the line window", why)
	}
}

// The check that stops a comment landing on the right line by luck from
// scoring as a catch.
func TestRun_RightLineWrongSubjectIsAMiss(t *testing.T) {
	client := stubReviewer{findings: []inference.ReviewFinding{
		finding("a/b.go", 12, "consider renaming this variable for clarity"),
	}}
	score := Run(context.Background(), client, "{files}", oneCase(seeded), countStub)
	if score.Caught != 0 {
		t.Fatalf("caught %d, want 0: no keyword mentioned", score.Caught)
	}
	if why := score.Results[0].Matches[0].Why; !strings.Contains(why, "mentions none") {
		t.Errorf("Why = %q, want it to name the keywords", why)
	}
}

func TestRun_KeywordMayBeInTheHeadlineOrSuggestion(t *testing.T) {
	head := "Possible nil dereference"
	f := finding("a/b.go", 12, "see headline")
	f.Headline = &head
	score := Run(context.Background(), client(f), "{files}", oneCase(seeded), countStub)
	if score.Caught != 1 {
		t.Fatalf("caught %d, want 1: the keyword is in the headline", score.Caught)
	}
}

func TestRun_WrongFileIsAMiss(t *testing.T) {
	score := Run(context.Background(),
		client(finding("other/c.go", 12, "nil panic")), "{files}", oneCase(seeded), countStub)
	if score.Caught != 0 {
		t.Fatalf("caught %d, want 0", score.Caught)
	}
}

// A model may answer with the path as the diff header spelled it.
func TestRun_PathSuffixCounts(t *testing.T) {
	score := Run(context.Background(),
		client(finding("b.go", 12, "nil panic")), "{files}", oneCase(seeded), countStub)
	if score.Caught != 1 {
		t.Fatalf("caught %d, want 1: b.go is the suffix of a/b.go", score.Caught)
	}
}

// Two seeded bugs on the same lines need two findings; one finding must not
// satisfy both, or a single vague remark would score a perfect run.
func TestRun_OneFindingSatisfiesAtMostOneSeededBug(t *testing.T) {
	second := seeded
	second.ID = "bug2"
	score := Run(context.Background(),
		client(finding("a/b.go", 12, "nil panic")), "{files}",
		oneCase(seeded, second), countStub)
	if score.Caught != 1 {
		t.Fatalf("caught %d of 2, want 1: one finding cannot catch two bugs", score.Caught)
	}
	// The loser must not be told "no finding on a/b.go": one exists, in the
	// window, with a keyword. It was taken.
	for _, m := range score.Results[0].Matches {
		if m.Found {
			continue
		}
		if strings.Contains(m.Why, "no finding") {
			t.Errorf("Why = %q, but a qualifying finding exists and was claimed "+
				"by the other expectation", m.Why)
		}
		if !strings.Contains(m.Why, "claimed by another expectation") {
			t.Errorf("Why = %q, want it to say the finding was taken", m.Why)
		}
	}
}

// Greedy in-corpus-order assignment undercounts: e1 ("nil") would take the
// finding that also satisfies it, leaving e2 ("error") with one that mentions
// no keyword, and the run reports a miss on a review that caught both.
func TestRun_AssignmentDoesNotLetOneExpectationStarveAnother(t *testing.T) {
	e1 := Expected{ID: "e1", File: "a/b.go", Lines: [2]int{10, 14},
		Keywords: []string{"nil"}, Why: "x"}
	e2 := Expected{ID: "e2", File: "a/b.go", Lines: [2]int{10, 14},
		Keywords: []string{"error"}, Why: "x"}
	client := stubReviewer{findings: []inference.ReviewFinding{
		finding("a/b.go", 11, "nil pointer causes an error"), // fits e1 and e2
		finding("a/b.go", 12, "nil deref"),                   // fits e1 only
	}}
	score := Run(context.Background(), client, "{files}", oneCase(e1, e2), countStub)
	if score.Caught != 2 {
		t.Fatalf("caught %d of 2: the pairing e1->f1, e2->f0 satisfies both", score.Caught)
	}
	if score.Extra != 0 {
		t.Errorf("extra = %d, want 0", score.Extra)
	}
}

// encoding/json renders an error as {}, so the -json report would record
// that a case failed without saying why: a 401 and a timeout would be
// indistinguishable in a committed history.
func TestResult_JSONCarriesTheErrorText(t *testing.T) {
	r := Result{Case: "c", Err: errors.New("gateway 401"), ErrMsg: "gateway 401"}
	blob, err := json.Marshal(r)
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(string(blob), "gateway 401") {
		t.Errorf("JSON = %s, want it to carry the error text", blob)
	}
	if strings.Contains(string(blob), `"Err":`) {
		t.Errorf("JSON = %s, want the bare error field omitted", blob)
	}
}

func envOf(m map[string]string) func(string) string {
	return func(k string) string { return m[k] }
}

func TestResolveSettings_DefaultsTheModelButNotTheCredentials(t *testing.T) {
	s, err := ResolveSettings(envOf(map[string]string{
		"EVAL_BASE_URL": "https://x/v1", "EVAL_API_KEY": "k",
	}), 0)
	if err != nil {
		t.Fatal(err)
	}
	if s.Model != DefaultModel {
		t.Errorf("Model = %q, want the default %q", s.Model, DefaultModel)
	}
}

func TestResolveSettings_NamesEveryMissingVariable(t *testing.T) {
	_, err := ResolveSettings(envOf(nil), 0)
	if err == nil {
		t.Fatal("want an error when nothing is set")
	}
	for _, want := range []string{"EVAL_BASE_URL", "EVAL_API_KEY"} {
		if !strings.Contains(err.Error(), want) {
			t.Errorf("error %q does not name %s", err, want)
		}
	}
}

// resolveWindow's advice is to set OPENAI_CONTEXT_WINDOW; honouring it here
// is what makes that followable rather than a permanent failure.
func TestResolveSettings_ContextWindowFromFlagThenEnv(t *testing.T) {
	base := map[string]string{"EVAL_BASE_URL": "u", "EVAL_API_KEY": "k"}
	s, err := ResolveSettings(envOf(base), 1_500_000)
	if err != nil || s.ContextWindow != 1_500_000 {
		t.Fatalf("flag: window = %d, err = %v", s.ContextWindow, err)
	}

	withEnv := map[string]string{"EVAL_BASE_URL": "u", "EVAL_API_KEY": "k",
		"OPENAI_CONTEXT_WINDOW": "2000000"}
	s, err = ResolveSettings(envOf(withEnv), 0)
	if err != nil || s.ContextWindow != 2_000_000 {
		t.Fatalf("env: window = %d, err = %v", s.ContextWindow, err)
	}

	// The flag wins: an operator who passed it meant it.
	s, err = ResolveSettings(envOf(withEnv), 1_000_000)
	if err != nil || s.ContextWindow != 1_000_000 {
		t.Fatalf("flag over env: window = %d, err = %v", s.ContextWindow, err)
	}

	if _, err = ResolveSettings(envOf(map[string]string{"EVAL_BASE_URL": "u",
		"EVAL_API_KEY": "k", "OPENAI_CONTEXT_WINDOW": "lots"}), 0); err == nil {
		t.Error("want an error for a non-numeric OPENAI_CONTEXT_WINDOW")
	}
}

// The alias must default to the model's own id. An invented one only works
// on a LiteLLM instance configured with that name: a real run against a
// plain OpenAI-compatible host failed Startup with "not listed by the
// gateway for this key" while listing the model under its real name.
func TestGatewayEnv_AliasDefaultsToTheModelName(t *testing.T) {
	env := GatewayEnv(Settings{BaseURL: "https://x/v1", APIKey: "k", Model: "m"})
	if got := env["LLMWIRE_LITELLM_MODELS"]; got != "m=m" {
		t.Errorf("MODELS = %q, want m=m: the endpoint serves the model under its own name", got)
	}
	if env["LLMWIRE_LITELLM_BASE_URL"] != "https://x/v1" || env["LLMWIRE_LITELLM_API_KEY"] != "k" {
		t.Errorf("env = %v, want the endpoint and key passed through", env)
	}

	// A gateway that does rename it is still reachable.
	env = GatewayEnv(Settings{BaseURL: "u", APIKey: "k", Model: "m", Alias: "house-name"})
	if got := env["LLMWIRE_LITELLM_MODELS"]; got != "m=house-name" {
		t.Errorf("MODELS = %q, want the explicit alias honoured", got)
	}
}

// A gateway failure must not read as a prompt result in either direction.
func TestScore_IncompleteCaseIsNotAPromptResult(t *testing.T) {
	// The dangerous direction: a case with nothing seeded fails, and
	// caught == seeded == 0 would otherwise be a pass.
	s := Score{Results: []Result{{Case: "clean", ErrMsg: "gateway 503"}}}
	if s.ErrMissed() != nil {
		t.Error("ErrMissed fires on a case that seeded nothing; that is why ErrIncomplete exists")
	}
	err := s.ErrIncomplete()
	if err == nil || !strings.Contains(err.Error(), "gateway 503") {
		t.Errorf("ErrIncomplete = %v, want it to name the failure", err)
	}

	// A non-OK outcome counts too, even without an error value.
	s = Score{Results: []Result{{Case: "c", Outcome: inference.OutcomeTimedOut.String()}}}
	if s.ErrIncomplete() == nil {
		t.Error("a timed-out case is not a prompt result")
	}

	s = Score{Results: []Result{{Case: "c", Outcome: inference.OutcomeOK.String()}}}
	if err := s.ErrIncomplete(); err != nil {
		t.Errorf("ErrIncomplete = %v on a completed run, want nil", err)
	}
}

func TestScore_ReportNamesMissesAndWhyTheyMatter(t *testing.T) {
	s := Score{Seeded: 1, Results: []Result{{
		Case: "c", Outcome: inference.OutcomeOK.String(),
		Matches: []Match{{Expected: Expected{ID: "bug", Why: "it panics"},
			Finding: -1, Why: "no finding on a/b.go"}},
	}}}
	var buf strings.Builder
	s.Report(&buf)
	for _, want := range []string{"MISS", "bug", "no finding on a/b.go", "it panics"} {
		if !strings.Contains(buf.String(), want) {
			t.Errorf("report does not mention %q:\n%s", want, buf.String())
		}
	}
}

// The control case: findings on a diff with nothing wrong are all noise.
func TestRun_CleanCaseCountsEveryFindingAsExtra(t *testing.T) {
	client := stubReviewer{findings: []inference.ReviewFinding{
		finding("a/b.go", 3, "nit"), finding("a/b.go", 9, "another nit"),
	}}
	score := Run(context.Background(), client, "{files}", oneCase(), countStub)
	if score.Seeded != 0 || score.Caught != 0 {
		t.Fatalf("seeded %d caught %d, want 0 and 0", score.Seeded, score.Caught)
	}
	if score.Extra != 2 {
		t.Errorf("extra = %d, want 2: a clean case is how false positives are counted", score.Extra)
	}
}

func TestRun_RecordsTheOutcomeName(t *testing.T) {
	score := Run(context.Background(),
		stubReviewer{outcome: inference.OutcomeUnparseable}, "{files}", oneCase(), countStub)
	if got := score.Results[0].Outcome; got != inference.OutcomeUnparseable.String() {
		t.Errorf("Outcome = %q, want %q", got, inference.OutcomeUnparseable.String())
	}
}

// The corpus ships with the binary, so a broken case file is a build-time
// asset problem rather than something only a live run would surface.
func TestLoadCorpus_EveryCaseIsUsable(t *testing.T) {
	cases, err := LoadCorpus()
	if err != nil {
		t.Fatal(err)
	}
	if len(cases) < 2 {
		t.Fatalf("corpus has %d case(s); too few to mean anything", len(cases))
	}
	var clean int
	for _, c := range cases {
		if len(c.Files) == 0 {
			t.Errorf("%s has no files", c.Name)
		}
		for _, f := range c.Files {
			if f.Path == "" || f.Diff == "" {
				t.Errorf("%s: a file has no path or no diff", c.Name)
			}
		}
		if len(c.Expected) == 0 {
			clean++
		}
		for _, e := range c.Expected {
			if e.ID == "" || e.File == "" || len(e.Keywords) == 0 || e.Why == "" {
				t.Errorf("%s: expected %q is missing id, file, keywords or why", c.Name, e.ID)
			}
			if e.Lines[0] <= 0 || e.Lines[1] < e.Lines[0] {
				t.Errorf("%s: expected %q has line window %v", c.Name, e.ID, e.Lines)
			}
			var onAFile bool
			for _, f := range c.Files {
				if samePath(e.File, f.Path) {
					onAFile = true
				}
			}
			if !onAFile {
				t.Errorf("%s: expected %q points at %s, which the case does not contain",
					c.Name, e.ID, e.File)
			}
		}
	}
	if clean == 0 {
		t.Error("no case without expected findings: without one, " +
			"'caught every seeded bug' is satisfied by reporting everything")
	}
}

// A fabricated hunk header makes a case unwinnable. prompts/review.txt asks
// for the line in the NEW version and the file content is rendered
// unnumbered, so the model anchors off the header: a header claiming a start
// the content does not have scores a correct finding outside the window, and
// the eval reports a prompt regression that is really a corpus bug. This
// caught exactly that in both original cases.
func TestCorpus_HunkHeadersAgreeWithContent(t *testing.T) {
	cases, err := LoadCorpus()
	if err != nil {
		t.Fatal(err)
	}
	header := regexp.MustCompile(`^@@ -\d+(?:,\d+)? \+(\d+)(?:,(\d+))? @@`)
	for _, c := range cases {
		for _, f := range c.Files {
			lines := strings.Split(strings.TrimRight(f.Diff, "\n"), "\n")
			m := header.FindStringSubmatch(lines[0])
			if m == nil {
				t.Errorf("%s: %s has no hunk header: %q", c.Name, f.Path, lines[0])
				continue
			}
			// One hunk per file. A second @@ would fall into this hunk's body
			// and be compared against content, so the guard would fail on a
			// valid case; splitting per hunk is the fix when one needs two.
			for _, l := range lines[1:] {
				if strings.HasPrefix(l, "@@") {
					t.Errorf("%s: %s has more than one hunk; this check handles one",
						c.Name, f.Path)
				}
			}
			start, _ := strconv.Atoi(m[1])

			// The hunk's whole new side must sit at `start` in the content.
			// Comparing only the first line would pass on a hunk whose body
			// diverges later, and a blank context line matches anywhere.
			var newSide []string
			for _, l := range lines[1:] {
				if strings.HasPrefix(l, "-") {
					continue
				}
				// Strip exactly one marker byte, not "+" then " ": a line
				// added as "+ foo" would otherwise lose its real leading
				// space and never match the content.
				if l == "" {
					newSide = append(newSide, "")
					continue
				}
				newSide = append(newSide, l[1:])
			}
			newCount := len(newSide)
			content := strings.Split(f.Content, "\n")
			if start < 1 || start+newCount-1 > len(content) {
				t.Errorf("%s: %s hunk claims new lines %d-%d, content has %d",
					c.Name, f.Path, start, start+newCount-1, len(content))
				continue
			}
			if got := content[start-1 : start-1+newCount]; !slices.Equal(got, newSide) {
				t.Errorf("%s: %s header says new line %d, but content there is\n%q\nnot the hunk's new side\n%q",
					c.Name, f.Path, start, got, newSide)
			}
			if m[2] != "" {
				if want, _ := strconv.Atoi(m[2]); want != newCount {
					t.Errorf("%s: %s header counts %d new-side lines, hunk body has %d",
						c.Name, f.Path, want, newCount)
				}
			}
		}
	}
}

// A window outside the file, or one that excludes the line the bug is on,
// makes a case unwinnable in the other direction.
func TestCorpus_ExpectedWindowsLieInsideTheFile(t *testing.T) {
	cases, err := LoadCorpus()
	if err != nil {
		t.Fatal(err)
	}
	for _, c := range cases {
		for _, e := range c.Expected {
			for _, f := range c.Files {
				if !samePath(e.File, f.Path) {
					continue
				}
				n := len(strings.Split(strings.TrimRight(f.Content, "\n"), "\n"))
				if e.Lines[1] > n {
					t.Errorf("%s: %q window %v ends past %s, which has %d lines",
						c.Name, e.ID, e.Lines, f.Path, n)
				}
			}
		}
	}
}

func client(f ...inference.ReviewFinding) stubReviewer { return stubReviewer{findings: f} }
