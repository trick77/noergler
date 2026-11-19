package evals

import (
	"context"
	"strconv"
	"strings"
	"testing"

	"github.com/trick77/noergler/internal/inference"
)

// fixedReviewer answers with canned findings and fixed claims.
type fixedReviewer struct {
	findings []inference.ReviewFinding
	fixed    []inference.FixedClaim
}

func (s fixedReviewer) Review(context.Context, inference.ReviewRequest) inference.ReviewResult {
	return inference.ReviewResult{Review: inference.ParsedReview{Findings: s.findings, Fixed: s.fixed,
		Summary: inference.NewReviewSummary()}}
}

func corpusCase(t *testing.T, name string) Case {
	t.Helper()
	cases, err := LoadCorpus()
	if err != nil {
		t.Fatal(err)
	}
	for _, c := range cases {
		if c.Name == name {
			return c
		}
	}
	t.Fatalf("no corpus case %q", name)
	return Case{}
}

func runOne(c Case, r Reviewer) Score {
	return Run(context.Background(), r, "{files}\n{cumulative_pr_diff}\n{previously_posted_findings}\n{previous_overview}",
		[]Case{c}, countStub)
}

// A claim production keeps catches the expected fix; nothing else counts.
func TestRun_ExpectedFixCaughtByAKeptClaim(t *testing.T) {
	c := corpusCase(t, "incremental-fixed")
	score := runOne(c, fixedReviewer{fixed: []inference.FixedClaim{{ID: "F101", Evidence: []string{"w.Flush()", "return w.Error()"}}}})
	if score.Seeded != 1 || score.Caught != 1 {
		t.Fatalf("caught %d of %d, want 1 of 1", score.Caught, score.Seeded)
	}
	if score.FalseFixed != 0 || score.ErrFalseFixed() != nil {
		t.Errorf("false fixed = %d, want 0", score.FalseFixed)
	}
}

// A claim quoting only unchanged lines is dropped by production, so the
// fix is missed and the miss says why.
func TestRun_DroppedFixedClaimIsAMissWithItsReason(t *testing.T) {
	c := corpusCase(t, "incremental-fixed")
	score := runOne(c, fixedReviewer{fixed: []inference.FixedClaim{{ID: "F101", Evidence: []string{"w.Flush()"}}}})
	if score.Caught != 0 {
		t.Fatalf("caught %d, want 0: the claim quotes no changed line", score.Caught)
	}
	m := score.Results[0].FixedMatches[0]
	if !strings.HasPrefix(m.Why, "claimed, dropped: ") {
		t.Errorf("why = %q, want the drop reason", m.Why)
	}
	if score.ErrMissed() == nil {
		t.Error("a missed fix must fail the run like a missed bug")
	}
}

// A kept claim no case asked for is a false fix and fails the run: in
// production it resolves the thread of a live bug.
func TestRun_KeptUnexpectedFixedClaimFailsTheRun(t *testing.T) {
	c := corpusCase(t, "incremental-moved-not-fixed")
	// ValidateFixed's documented limit: any changed line of the file passes.
	score := runOne(c, fixedReviewer{fixed: []inference.FixedClaim{{ID: "F102", Evidence: []string{"return (total + size - 1) / size"}}}})
	if score.FalseFixed != 1 {
		t.Fatalf("false fixed = %d, want 1", score.FalseFixed)
	}
	if score.ErrFalseFixed() == nil {
		t.Error("a false fix must fail the run")
	}
}

// A claim on a closed finding names no id the prompt showed: dropped.
func TestRun_ClosedFindingCannotBeClaimedFixed(t *testing.T) {
	c := corpusCase(t, "incremental-disputed-closed")
	score := runOne(c, fixedReviewer{fixed: []inference.FixedClaim{{ID: "F104", Evidence: []string{"return w.balance"}}}})
	r := score.Results[0]
	if score.FalseFixed != 0 || len(r.Fixed) != 0 {
		t.Fatalf("kept %v, want nothing: a closed finding has no id", r.Fixed)
	}
	if len(r.FixedDropped) != 1 || r.FixedDropped[0].Reason != inference.FixedUnknownID {
		t.Errorf("dropped = %+v, want one unknown_id", r.FixedDropped)
	}
}

// A re-raised earlier finding on a clean incremental case is a finding on
// a control: the run fails, as for any invented finding.
func TestRun_ReRaiseOnACleanIncrementalCaseFailsTheRun(t *testing.T) {
	c := corpusCase(t, "incremental-moved-not-fixed")
	score := runOne(c, fixedReviewer{findings: []inference.ReviewFinding{{
		File: "internal/paging/paging.go", Line: 13, Severity: "issue",
		Comment: "page n is treated as 0-based", Evidence: []string{"start := n * size"},
	}}})
	if score.Extra != 1 || score.ErrInvented() == nil {
		t.Fatalf("extra = %d, invented = %v; want 1 and an error", score.Extra, score.ErrInvented())
	}
}

// A repeat of an earlier finding fails the run even on a case that seeds a
// bug, where it would only be one more Extra.
func TestRun_RepeatOnASeededIncrementalCaseFailsTheRun(t *testing.T) {
	c := corpusCase(t, "incremental-moved-not-fixed")
	// The same case seeding a bug elsewhere: the repeat must still count.
	c.Expected = []Expected{{ID: "other", File: "internal/paging/paging.go", Lines: [2]int{4, 8},
		Keywords: []string{"zero"}, Why: "x"}}
	score := runOne(c, fixedReviewer{findings: []inference.ReviewFinding{{
		File: "internal/paging/paging.go", Line: 13, Severity: "issue",
		Comment:  "Page treats n as 0-based although the doc says 1-based.",
		Evidence: []string{"start := n * size"},
	}}})
	if score.Repeated != 1 || score.ErrRepeated() == nil {
		t.Fatalf("repeated = %d, err = %v; want 1 and an error", score.Repeated, score.ErrRepeated())
	}
}

// A dropped claim is matched by its number however the model wrote the id,
// and the first drop names the reason, not a later duplicate.
func TestRun_DroppedClaimReasonReadsTheIdLikeProduction(t *testing.T) {
	c := corpusCase(t, "incremental-fixed")
	score := runOne(c, fixedReviewer{fixed: []inference.FixedClaim{
		{ID: "101", Evidence: []string{"w.Flush()"}},
		{ID: "F101", Evidence: []string{"w.Flush()"}},
	}})
	m := score.Results[0].FixedMatches[0]
	if m.Found || m.Why != "claimed, dropped: "+string(score.Results[0].FixedDropped[0].Reason) || m.Why == "not claimed" {
		t.Errorf("why = %q, drops = %+v", m.Why, score.Results[0].FixedDropped)
	}
}

// Closed earlier findings are listed first, as production lists them.
func TestLoadCorpus_ClosedEarlierFindingsFirst(t *testing.T) {
	c := corpusCase(t, "incremental-disputed-closed")
	if len(c.PreviouslyPosted) != 2 || !c.PreviouslyPosted[0].Closed || c.PreviouslyPosted[1].Closed {
		t.Errorf("order = %+v, want the closed one first", c.PreviouslyPosted)
	}
}

// The prompt carries what production adds on a push: the cumulative diff,
// the posted findings (a closed one without an id) and the overview. A
// full case carries none of it.
func TestAssemble_IncrementalCaseRendersTheProductionBlocks(t *testing.T) {
	tmpl := "{files}\n{cumulative_pr_diff}\n{previously_posted_findings}\n{previous_overview}"
	c := corpusCase(t, "incremental-disputed-closed")
	p := assemble(tmpl, c, countStub).Prompt
	for _, want := range []string{"Cumulative PR diff", "[F103]", "(author replied: ", "(closed on the PR)", "Previous PR overview"} {
		if !strings.Contains(p, want) {
			t.Errorf("incremental prompt lacks %q", want)
		}
	}
	if strings.Contains(p, "[F104]") {
		t.Error("a closed finding must render without its id")
	}
	full := corpusCase(t, "nil-deref")
	if p := assemble(tmpl, full, countStub).Prompt; strings.Contains(p, "Already-posted findings") || strings.Contains(p, "Cumulative PR diff") {
		t.Error("a full case must not carry incremental blocks")
	}
}

// An incremental case is usable: every expected fix names an open earlier
// finding, and every earlier finding sits on a file of the push.
func TestCorpus_IncrementalCasesAreConsistent(t *testing.T) {
	cases, err := LoadCorpus()
	if err != nil {
		t.Fatal(err)
	}
	var n int
	for _, c := range cases {
		if !c.Incremental {
			continue
		}
		n++
		if strings.TrimSpace(c.CumulativeDiff) == "" || c.PreviousOverview == "" || len(c.PreviouslyPosted) == 0 {
			t.Errorf("%s: incremental case without cumulative diff, overview or earlier findings", c.Name)
		}
		open := map[string]bool{}
		for _, p := range c.PreviouslyPosted {
			if p.ID > 0 {
				open[inference.FixedIDPrefix+strconv.FormatInt(p.ID, 10)] = true
			}
			var onAFile bool
			for _, f := range c.Files {
				onAFile = onAFile || samePath(p.FilePath, f.Path)
			}
			if !onAFile {
				t.Errorf("%s: earlier finding on %s, which the push does not touch", c.Name, p.FilePath)
			}
		}
		for _, e := range c.ExpectedFixed {
			if !open[e.ID] || e.Why == "" {
				t.Errorf("%s: expected fix %q names no open earlier finding, or has no why", c.Name, e.ID)
			}
		}
	}
	if n == 0 {
		t.Error("no incremental case: fixed claims and re-raising go unmeasured")
	}
}
