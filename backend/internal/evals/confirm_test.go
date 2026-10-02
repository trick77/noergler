package evals

import (
	"context"
	"errors"
	"strings"
	"testing"

	"github.com/trick77/noergler/internal/inference"
)

// consolidator answers every consolidation call with the same findings and
// keeps the prompts it was sent.
type consolidator struct {
	findings []inference.ReviewFinding
	outcome  inference.Outcome
	prompts  []string
}

func (c *consolidator) Review(_ context.Context, req inference.ReviewRequest) inference.ReviewResult {
	c.prompts = append(c.prompts, req.Prompt)
	out := inference.ReviewResult{
		Outcome: c.outcome,
		Review:  inference.ParsedReview{Findings: c.findings, Summary: inference.NewReviewSummary()},
		Cost:    inference.CallCost{PromptTokens: 200, CachedTokens: 150, CompletionTokens: 20},
	}
	if c.outcome != inference.OutcomeOK {
		out.Err = errors.New("stream idle")
	}
	return out
}

// sampledOf is a complete three-sample run of the given cases.
func sampledOf(cases []Case, script ...[]inference.ReviewFinding) Sampled {
	s := RunSampled(context.Background(), &scriptedReviewer{script: script}, "{files}", cases, countStub, 3, 1, nil)
	s.Model, s.Effort = "finder", "some-level"
	return s
}

// The case the line-based union gets wrong: two reviews report the one bug
// four lines apart. The union posts it twice; the consolidation call is
// shown both and answers with one.
func TestRunConfirm_ConsolidationMergesWhatTheUnionDoubles(t *testing.T) {
	cases := oneCase(seeded)
	sampled := sampledOf(cases, bug(10), bug(14), nil)
	client := &consolidator{findings: bug(10)}

	got, err := RunConfirm(context.Background(), client, "{files}", "\nBRIEF\n", cases, sampled, countStub, 3, nil)
	if err != nil {
		t.Fatal(err)
	}
	if v := findView(got.Views, ViewUnion3); v.Caught != 1 || v.Extra != 1 {
		t.Errorf("union: caught %.2f extra %.2f, want 1 and 1", v.Caught, v.Extra)
	}
	v := findView(got.Views, ViewConfirm3)
	if v.Caught != 1 || v.Extra != 0 || v.Calls != 4 {
		t.Errorf("confirm: caught %.2f extra %.2f in %.2f call(s), want 1, 0 and 4", v.Caught, v.Extra, v.Calls)
	}
	// Three samples form one group of three, whatever was asked for.
	if len(client.prompts) != 1 || len(got.Cases[0].Groups) != 1 {
		t.Fatalf("%d call(s), %d group(s), want 1 and 1", len(client.prompts), len(got.Cases[0].Groups))
	}

	// The review prompt is an untouched prefix, so it is one the endpoint
	// already cached; the brief and every candidate follow it.
	prompt := client.prompts[0]
	review := assemble("{files}", cases[0], countStub).Prompt
	if !strings.HasPrefix(prompt, review) {
		t.Error("the consolidation prompt does not start with the review prompt")
	}
	tail := strings.TrimPrefix(prompt, review)
	for _, want := range []string{"BRIEF", `"id": "r1-1"`, `"id": "r2-1"`, `"line": 14`} {
		if !strings.Contains(tail, want) {
			t.Errorf("after the review prompt there is no %s:\n%s", want, tail)
		}
	}
	if g := got.Cases[0].Groups[0]; g.Candidates != 2 || g.Cost.CachedTokens != 150 {
		t.Errorf("group: %d candidate(s), %d cached token(s)", g.Candidates, g.Cost.CachedTokens)
	}
	if got.Tokens.Prompt != 200 || got.Tokens.Completion != 20 {
		t.Errorf("token means: %+v", got.Tokens)
	}

	var b strings.Builder
	got.Finder, got.Confirmer = "finder", "confirmer"
	got.Report(&b)
	if out := b.String(); !strings.Contains(out, ViewConfirm3) || !strings.Contains(out, "consolidated by confirmer") {
		t.Errorf("report:\n%s", out)
	}
}

// Nothing to consolidate is no call: a clean change costs three reviews.
func TestRunConfirm_NoCandidatesMakesNoCall(t *testing.T) {
	cases := oneCase()
	client := &consolidator{}
	got, err := RunConfirm(context.Background(), client, "{files}", "BRIEF", cases, sampledOf(cases, nil, nil, nil), countStub, 3, nil)
	if err != nil {
		t.Fatal(err)
	}
	if len(client.prompts) != 0 {
		t.Fatalf("%d call(s) with no candidate", len(client.prompts))
	}
	if v := findView(got.Views, ViewConfirm3); v.Calls != 3 || v.ControlsClean != 1 {
		t.Errorf("confirm: %.2f call(s), controls clean %.2f, want 3 and 1", v.Calls, v.ControlsClean)
	}
	if err := got.ErrIncomplete(); err != nil {
		t.Errorf("a skipped group is not an incomplete one: %v", err)
	}
}

// One candidate is nothing to merge: no call, and the lone finding stands.
// A call here could only repeat it or lose it.
func TestRunConfirm_ALoneCandidateIsKeptWithoutACall(t *testing.T) {
	cases := oneCase(seeded)
	client := &consolidator{}
	got, err := RunConfirm(context.Background(), client, "{files}", "BRIEF", cases, sampledOf(cases, nil, bug(12), nil), countStub, 3, nil)
	if err != nil {
		t.Fatal(err)
	}
	if len(client.prompts) != 0 {
		t.Fatalf("%d call(s) for a single candidate", len(client.prompts))
	}
	if v := findView(got.Views, ViewConfirm3); v.Caught != 1 || v.Calls != 3 {
		t.Errorf("confirm: caught %.2f in %.2f call(s), want 1 and 3", v.Caught, v.Calls)
	}
}

// A consolidation call that never completes is redrawn, then reported.
func TestRunConfirm_FailedCallIsIncomplete(t *testing.T) {
	cases := oneCase(seeded)
	client := &consolidator{outcome: inference.OutcomeTimedOut}
	var progress strings.Builder
	got, err := RunConfirm(context.Background(), client, "{files}", "BRIEF", cases, sampledOf(cases, bug(12), bug(12), nil), countStub, 1, &progress)
	if err != nil {
		t.Fatal(err)
	}
	if len(client.prompts) != 1+maxRedraws {
		t.Errorf("%d attempt(s), want %d", len(client.prompts), 1+maxRedraws)
	}
	if err := got.ErrIncomplete(); err == nil || !strings.Contains(err.Error(), "c[1 2 3]") {
		t.Errorf("err = %v, want the group named", err)
	}
	if !strings.Contains(progress.String(), "1 group(s)") {
		t.Errorf("progress: %q", progress.String())
	}
}

func TestRunConfirm_RefusesWhatItCannotScore(t *testing.T) {
	cases := oneCase(seeded)
	good := sampledOf(cases, nil, nil, nil)
	other := []Case{{Name: "elsewhere", Files: fixtureFiles()}}
	if _, err := RunConfirm(context.Background(), &consolidator{}, "{files}", "", other, good, countStub, 1, nil); err == nil {
		t.Error("a case the sampled run never scored must be refused")
	}
	bad := inference.OutcomeTimedOut
	failed := &scriptedReviewer{
		script:  make([][]inference.ReviewFinding, 5),
		outcome: []inference.Outcome{inference.OutcomeOK, bad, bad, bad, inference.OutcomeOK},
	}
	incomplete := RunSampled(context.Background(), failed, "{files}", cases, countStub, 3, 1, nil)
	if _, err := RunConfirm(context.Background(), &consolidator{}, "{files}", "", cases, incomplete, countStub, 1, nil); err == nil {
		t.Error("an incomplete sampled run must be refused")
	}
}

func TestSpread(t *testing.T) {
	groups := subsets(5, 3)
	if got := spread(groups, 3); len(got) != 3 || &got[0][0] != &groups[0][0] || &got[2][0] != &groups[6][0] {
		t.Errorf("spread(10, 3) = %v, want groups 0, 3 and 6", got)
	}
	if got := spread(groups, 99); len(got) != len(groups) {
		t.Errorf("asking for more than there are gives %d, want all %d", len(got), len(groups))
	}
}

// Optional fields render only when present, and no candidate is no JSON.
func TestRenderCandidates(t *testing.T) {
	if s, n := renderCandidates([][]inference.ReviewFinding{nil, nil}); s != "" || n != 0 {
		t.Errorf("empty group rendered %q (%d)", s, n)
	}
	f := finding("a/b.go", 12, "c")
	head, fix := "H", "x := 1"
	f.Headline, f.Suggestion = &head, &fix
	s, n := renderCandidates([][]inference.ReviewFinding{{f}})
	if n != 1 || !strings.Contains(s, `"headline": "H"`) || !strings.Contains(s, `"suggestion": "x := 1"`) {
		t.Errorf("rendered %d: %s", n, s)
	}
}
