package evals

import (
	"context"
	"encoding/json"
	"errors"
	"math"
	"slices"
	"strings"
	"sync"
	"testing"

	"github.com/trick77/noergler/internal/inference"
)

// scriptedReviewer answers call i with script[i], so a test decides what
// each sample of a case reports. Run it with parallel 1: only then is the
// call order the sample order.
type scriptedReviewer struct {
	mu      sync.Mutex
	calls   int
	script  [][]inference.ReviewFinding
	costs   []*int64
	outcome []inference.Outcome
}

func (s *scriptedReviewer) Review(context.Context, inference.ReviewRequest) inference.ReviewResult {
	s.mu.Lock()
	i := s.calls
	s.calls++
	s.mu.Unlock()
	out := inference.ReviewResult{
		Review: inference.ParsedReview{Findings: s.script[i], Summary: inference.NewReviewSummary()},
	}
	if i < len(s.costs) {
		out.Cost = inference.CallCost{NanoUSD: s.costs[i], PromptTokens: 100, CompletionTokens: 10}
		if i > 0 {
			out.Cost.CachedTokens = 90
		}
	}
	if i < len(s.outcome) {
		out.Outcome = s.outcome[i]
		if out.Outcome != inference.OutcomeOK {
			out.Err = errors.New("stream idle")
		}
	}
	return out
}

func bug(line int) []inference.ReviewFinding {
	return []inference.ReviewFinding{finding("a/b.go", line, "nil pointer")}
}

func near(a, b float64) bool { return math.Abs(a-b) < 1e-9 }

func TestMergeFindings(t *testing.T) {
	f := func(file string, line int) inference.ReviewFinding { return finding(file, line, "x") }
	cases := []struct {
		name    string
		samples [][]inference.ReviewFinding
		quorum  int
		want    int
	}{
		{"same line in two samples", [][]inference.ReviewFinding{{f("a/b.go", 10)}, {f("a/b.go", 10)}, nil}, 2, 1},
		{"two lines apart is the same finding", [][]inference.ReviewFinding{{f("a/b.go", 10)}, {f("a/b.go", 12)}, nil}, 2, 1},
		{"three lines apart is not", [][]inference.ReviewFinding{{f("a/b.go", 10)}, {f("a/b.go", 13)}, nil}, 2, 0},
		{"another file is not", [][]inference.ReviewFinding{{f("a/b.go", 10)}, {f("other/c.go", 10)}, nil}, 2, 0},
		{"a path suffix is the same file", [][]inference.ReviewFinding{{f("a/b.go", 10)}, {f("b.go", 10)}, nil}, 2, 1},
		// A sample votes once: a model repeating itself is not a majority.
		{"one sample stuttering is one vote", [][]inference.ReviewFinding{{f("a/b.go", 10), f("a/b.go", 11)}, nil, nil}, 2, 0},
		{"union keeps a lone finding", [][]inference.ReviewFinding{{f("a/b.go", 10)}, nil, nil}, 1, 1},
		{"unanimity needs all three", [][]inference.ReviewFinding{{f("a/b.go", 10)}, {f("a/b.go", 10)}, nil}, 3, 0},
	}
	for _, c := range cases {
		if got := len(mergeFindings(c.samples, c.quorum)); got != c.want {
			t.Errorf("%s: kept %d, want %d", c.name, got, c.want)
		}
	}
}

// The kept finding is the earliest sample's, so what a vote posts is one
// review's own wording, never a blend.
func TestMergeFindings_RepresentativeIsTheEarliestSample(t *testing.T) {
	first, second := finding("a/b.go", 10, "first"), finding("a/b.go", 11, "second")
	got := mergeFindings([][]inference.ReviewFinding{{first}, {second}}, 2)
	if len(got) != 1 || got[0].Comment != "first" || got[0].Line != 10 {
		t.Fatalf("got %+v, want the first sample's finding", got)
	}
}

func TestSubsets(t *testing.T) {
	for _, c := range []struct{ n, k, want int }{{5, 1, 5}, {5, 2, 10}, {5, 3, 10}, {3, 3, 1}, {2, 3, 0}} {
		if got := len(subsets(c.n, c.k)); got != c.want {
			t.Errorf("subsets(%d, %d) = %d groups, want %d", c.n, c.k, got, c.want)
		}
	}
}

func TestAdaptive(t *testing.T) {
	hit, none := bug(12), []inference.ReviewFinding(nil)
	cases := []struct {
		name      string
		a, b, c   []inference.ReviewFinding
		kept, use int
	}{
		{"both report it: two calls", hit, hit, none, 1, 2},
		{"both silent: two calls", none, none, hit, 0, 2},
		{"split, third confirms", hit, none, hit, 1, 3},
		{"split, third does not", hit, none, none, 0, 3},
	}
	for _, c := range cases {
		got, calls := adaptive(c.a, c.b, c.c)
		if len(got) != c.kept || calls != c.use {
			t.Errorf("%s: kept %d in %d call(s), want %d in %d", c.name, len(got), calls, c.kept, c.use)
		}
	}
}

// One seeded case caught by samples 1 and 2 and missed by sample 3: every
// view's number is derivable by hand, which is what pins the arithmetic.
func TestRunSampled_ViewsOnACaseCaughtTwiceInThree(t *testing.T) {
	client := &scriptedReviewer{script: [][]inference.ReviewFinding{bug(12), bug(13), nil}}
	got := RunSampled(context.Background(), client, "{files}", oneCase(seeded), countStub, 3, 1, nil)

	want := map[string]struct{ caught, calls float64 }{
		ViewSingle:  {2.0 / 3, 1},
		ViewAgree2:  {1.0 / 3, 2},
		ViewEither2: {1, 2},
		ViewVote3:   {1, 3},
		ViewUnion3:  {1, 3},
		// Of the six orders, only the two that start with both catching
		// samples stop at two calls.
		ViewAdaptive: {1, (2*2.0 + 4*3.0) / 6},
	}
	for name, w := range want {
		v := got.view(name)
		if !near(v.Caught, w.caught) || !near(v.Calls, w.calls) {
			t.Errorf("%s: caught %.3f in %.3f call(s), want %.3f in %.3f",
				name, v.Caught, v.Calls, w.caught, w.calls)
		}
		if v.Extra != 0 {
			t.Errorf("%s: extra %.3f, want 0", name, v.Extra)
		}
	}
	if v := got.view(ViewSingle); !near(v.AllCaught, 2.0/3) {
		t.Errorf("single all-caught = %.3f, want 2/3", v.AllCaught)
	}
	if err := got.ErrIncomplete(); err != nil {
		t.Errorf("complete run reported incomplete: %v", err)
	}
	if err := got.ErrRegressed(); err != nil {
		t.Errorf("the vote caught the bug in its only triple: %v", err)
	}
}

// A clean control one sample invents on: the vote drops it, the union keeps
// it, and only the union's share of clean groups falls.
func TestRunSampled_VoteDropsALoneInventionOnAControl(t *testing.T) {
	client := &scriptedReviewer{script: [][]inference.ReviewFinding{nil, bug(30), nil}}
	got := RunSampled(context.Background(), client, "{files}", oneCase(), countStub, 3, 1, nil)

	if v := got.view(ViewVote3); v.Extra != 0 || v.ControlsClean != 1 {
		t.Errorf("vote: extra %.2f, controls clean %.2f, want 0 and 1", v.Extra, v.ControlsClean)
	}
	if v := got.view(ViewUnion3); v.Extra != 1 || v.ControlsClean != 0 {
		t.Errorf("union: extra %.2f, controls clean %.2f, want 1 and 0", v.Extra, v.ControlsClean)
	}
	if v := got.view(ViewSingle); !near(v.ControlsClean, 2.0/3) {
		t.Errorf("single controls clean = %.3f, want 2/3", v.ControlsClean)
	}
	if got.ErrRegressed() != nil {
		t.Error("a finding the vote drops must not fail the vote")
	}
}

func TestSampled_ErrRegressedNamesBothHalves(t *testing.T) {
	// Seeded case never caught, then a control every sample invents on.
	client := &scriptedReviewer{script: [][]inference.ReviewFinding{nil, nil, nil, bug(30), bug(30), bug(31)}}
	cases := append(oneCase(seeded), Case{Name: "clean", Files: fixtureFiles()})
	err := RunSampled(context.Background(), client, "{files}", cases, countStub, 3, 1, nil).ErrRegressed()
	if err == nil || !strings.Contains(err.Error(), "misses") || !strings.Contains(err.Error(), "invents") {
		t.Fatalf("err = %v, want both the miss and the invention named", err)
	}
}

// A sample that never completed is not a silent zero: the run is incomplete
// and says which sample.
func TestSampled_FailedSampleIsIncomplete(t *testing.T) {
	// Sample 2 fails on its draw and on both redraws.
	bad := inference.OutcomeTimedOut
	client := &scriptedReviewer{
		script:  [][]inference.ReviewFinding{bug(12), nil, nil, nil, bug(12)},
		outcome: []inference.Outcome{inference.OutcomeOK, bad, bad, bad, inference.OutcomeOK},
	}
	got := RunSampled(context.Background(), client, "{files}", oneCase(seeded), countStub, 3, 1, nil)
	err := got.ErrIncomplete()
	if err == nil || !strings.Contains(err.Error(), "c#2") {
		t.Fatalf("err = %v, want sample 2 of case c named", err)
	}
	var b strings.Builder
	got.Report(&b)
	if !strings.Contains(b.String(), "INCOMPLETE 2/3") {
		t.Errorf("report does not flag the case:\n%s", b.String())
	}
}

// A view costs one cold call plus warm ones for the rest, so the multiple
// reflects the cache and is not just the call count.
func TestSampled_CostMultipleIsColdPlusWarm(t *testing.T) {
	cold, warm := int64(1000), int64(400)
	client := &scriptedReviewer{
		script: [][]inference.ReviewFinding{nil, nil, nil},
		costs:  []*int64{&cold, &warm, &warm},
	}
	got := RunSampled(context.Background(), client, "{files}", oneCase(), countStub, 3, 1, nil)
	for name, want := range map[string]float64{ViewSingle: 1, ViewAgree2: 1.4, ViewVote3: 1.8, ViewAdaptive: 1.4} {
		m := got.view(name).CostMultiple
		if m == nil || !near(*m, want) {
			t.Errorf("%s cost multiple = %v, want %.1f", name, m, want)
		}
	}
	if got.Tokens.ColdCached != 0 || got.Tokens.WarmCached != 90 {
		t.Errorf("cached tokens cold %.0f warm %.0f, want 0 and 90", got.Tokens.ColdCached, got.Tokens.WarmCached)
	}
	var b strings.Builder
	got.Report(&b)
	if !strings.Contains(b.String(), "1.80x") {
		t.Errorf("report does not carry the vote's cost:\n%s", b.String())
	}
}

// One unpriced sample and no view has a cost: null in the JSON and
// "unpriced" in the report, never a zero.
func TestSampled_UnpricedRunHasNoCostVerdict(t *testing.T) {
	cold := int64(1000)
	client := &scriptedReviewer{
		script: [][]inference.ReviewFinding{nil, nil, nil},
		costs:  []*int64{&cold, nil, &cold},
	}
	got := RunSampled(context.Background(), client, "{files}", oneCase(), countStub, 3, 1, nil)
	for _, v := range got.Views {
		if v.CostMultiple != nil {
			t.Errorf("%s has a cost multiple on an unpriced run", v.Name)
		}
	}
	blob, err := json.Marshal(got.Views[0])
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(string(blob), `"CostMultiple":null`) {
		t.Errorf("unpriced cost is not null in JSON: %s", blob)
	}
	var b strings.Builder
	got.Report(&b)
	if !strings.Contains(b.String(), "unpriced") {
		t.Errorf("report does not say unpriced:\n%s", b.String())
	}
}

// Later samples run concurrently; the race detector is the assertion.
func TestRunSampled_ParallelSamplesAllLand(t *testing.T) {
	client := &scriptedReviewer{script: make([][]inference.ReviewFinding, 6)}
	got := RunSampled(context.Background(), client, "{files}", oneCase(seeded), countStub, 6, 4, nil)
	if n := len(got.Cases[0].Samples); n != 6 || client.calls != 6 {
		t.Fatalf("%d sample(s) from %d call(s), want 6 and 6", n, client.calls)
	}
	for i, s := range got.Cases[0].Samples {
		if s.Case != "c" {
			t.Errorf("sample %d was never filled", i+1)
		}
	}
}

func nit(line int) []inference.ReviewFinding {
	f := finding("a/b.go", line, "nil pointer")
	f.Severity = "suggestion"
	return []inference.ReviewFinding{f}
}

// The case the vote gets wrong: a bug one sample in three finds. The vote
// discards it, the tiered rule keeps it because it was rated an issue, and
// a lone suggestion is still dropped.
func TestRunSampled_TieredKeepsALoneIssueTheVoteDiscards(t *testing.T) {
	client := &scriptedReviewer{script: [][]inference.ReviewFinding{bug(12), nil, nil}}
	got := RunSampled(context.Background(), client, "{files}", oneCase(seeded), countStub, 3, 1, nil)
	if v := got.view(ViewVote3); v.Caught != 0 {
		t.Errorf("vote caught %.2f, want 0: one report in three is no majority", v.Caught)
	}
	if v := got.view(ViewTiered3); v.Caught != 1 {
		t.Errorf("tiered caught %.2f, want 1: a lone issue is kept", v.Caught)
	}

	client = &scriptedReviewer{script: [][]inference.ReviewFinding{nit(30), nil, nil}}
	got = RunSampled(context.Background(), client, "{files}", oneCase(), countStub, 3, 1, nil)
	if v := got.view(ViewTiered3); v.Extra != 0 {
		t.Errorf("tiered extra %.2f, want 0: a lone suggestion is dropped", v.Extra)
	}
	if v := got.view(ViewUnion3); v.Extra != 1 {
		t.Errorf("union extra %.2f, want 1", v.Extra)
	}
}

// The agreement table is what decides between a vote and a union: it must
// say how many samples reported each finding and whether it was real.
func TestSampled_AgreementCountsFindingsByVotes(t *testing.T) {
	client := &scriptedReviewer{script: [][]inference.ReviewFinding{
		// seeded case: the bug twice, plus a lone invented suggestion
		append(bug(12), nit(40)...), bug(13), nil,
		// control: one invented issue, reported by all three
		bug(30), bug(30), bug(31),
	}}
	cases := append(oneCase(seeded), Case{Name: "clean", Files: fixtureFiles()})
	got := RunSampled(context.Background(), client, "{files}", cases, countStub, 3, 1, nil)
	want := []Agreement{
		{Votes: 1, Invented: 1},
		{Votes: 2, Real: 1, RealIssue: 1},
		{Votes: 3, Invented: 1, InventedIssue: 1},
	}
	if !slices.Equal(got.Agreement, want) {
		t.Fatalf("agreement = %+v\nwant        %+v", got.Agreement, want)
	}
	var b strings.Builder
	got.Report(&b)
	if !strings.Contains(b.String(), "findings by how many of 3 samples") {
		t.Errorf("report has no agreement table:\n%s", b.String())
	}
}

// Rescore reads only the stored samples, so a report survives a JSON round
// trip and a view added later is scored on the same reviews.
func TestSampled_RescoreFromJSONMatchesTheRun(t *testing.T) {
	client := &scriptedReviewer{script: [][]inference.ReviewFinding{bug(12), nil, bug(13)}}
	ran := RunSampled(context.Background(), client, "{files}", oneCase(seeded), countStub, 3, 1, nil)
	blob, err := json.Marshal(ran)
	if err != nil {
		t.Fatal(err)
	}
	var back Sampled
	if err := json.Unmarshal(blob, &back); err != nil {
		t.Fatal(err)
	}
	back.Views, back.Agreement = nil, nil
	if err := back.Rescore(oneCase(seeded)); err != nil {
		t.Fatal(err)
	}
	for i, v := range ran.Views {
		if b := back.Views[i]; b.Name != v.Name || !near(b.Caught, v.Caught) || !near(b.Calls, v.Calls) {
			t.Errorf("%s: rescored %+v, ran %+v", v.Name, b, v)
		}
	}
	if !slices.Equal(back.Agreement, ran.Agreement) {
		t.Errorf("agreement: rescored %+v, ran %+v", back.Agreement, ran.Agreement)
	}

	if err := back.Rescore([]Case{{Name: "elsewhere"}}); err == nil {
		t.Error("a case the corpus does not have must be refused")
	}
	back.Samples = 4
	if err := back.Rescore(oneCase(seeded)); err == nil {
		t.Error("a case with fewer samples than the run claims must be refused")
	}
}

// A failed sample is drawn again, and the count is recorded: a redrawn
// sample replaces a call that was cut off, which is not the same sample.
func TestRunSampled_RedrawsAFailedSample(t *testing.T) {
	client := &scriptedReviewer{
		script:  [][]inference.ReviewFinding{bug(12), nil, bug(12), bug(12)},
		outcome: []inference.Outcome{inference.OutcomeOK, inference.OutcomeTimedOut, inference.OutcomeOK, inference.OutcomeOK},
	}
	var progress strings.Builder
	got := RunSampled(context.Background(), client, "{files}", oneCase(seeded), countStub, 3, 1, &progress)
	if err := got.ErrIncomplete(); err != nil {
		t.Fatalf("the redraw completed the sample: %v", err)
	}
	if r := got.Cases[0].Samples[1].Redraws; r != 1 {
		t.Errorf("sample 2 redraws = %d, want 1", r)
	}
	if v := got.view(ViewSingle); v.Caught != 1 {
		t.Errorf("single caught %.2f, want 1: the redrawn sample found the bug", v.Caught)
	}
	if !strings.Contains(progress.String(), "3/3 sample(s) ok, 1 redraw(s)") {
		t.Errorf("progress line: %q", progress.String())
	}
}
