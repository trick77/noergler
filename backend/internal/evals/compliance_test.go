package evals

import (
	"context"
	"strings"
	"testing"

	"github.com/trick77/noergler/internal/inference"
)

// Every case loads, carries its labels in the ticket the prompt shows, and
// has at least one requirement each way across the corpus: a corpus of only
// provable ones could never see a wrong ✅.
func TestComplianceCorpusLoads(t *testing.T) {
	cases, err := LoadComplianceCorpus()
	if err != nil {
		t.Fatal(err)
	}
	if len(cases) == 0 {
		t.Fatal("empty corpus")
	}
	shown, held := 0, 0
	for _, c := range cases {
		for _, e := range c.Expect {
			if e.Label != "" && !strings.Contains(c.TicketContext, e.Label) {
				t.Errorf("%s: label %s is not in the ticket", c.Name, e.Label)
			}
			if e.Shown {
				shown++
			} else {
				held++
			}
		}
		p := c.Assemble("", func(s string) int { return len(s) / 4 })
		if !strings.Contains(p.Prompt, c.TicketContext) || !strings.Contains(p.Prompt, "compliance_requirements") {
			t.Errorf("%s: the prompt lacks the ticket or the compliance instructions", c.Name)
		}
	}
	if shown == 0 || held == 0 {
		t.Errorf("corpus has %d provable and %d unprovable requirements, want both", shown, held)
	}
}

type fakeComplianceReviewer struct {
	reqs []inference.ComplianceRequirement
}

func (f fakeComplianceReviewer) Review(context.Context, inference.ReviewRequest) inference.ReviewResult {
	return inference.ReviewResult{Outcome: inference.OutcomeOK,
		Review: inference.ParsedReview{ComplianceRequirements: f.reqs}}
}

// The verdicts go through ConfirmCompliance as production does: a proven
// requirement is shown, a removed line drops it, an unmet one counts as unmet
// and is never shown, and a provable one the model never listed is
// unlisted. Showing a requirement the case marks unprovable is the failure.
func TestRunComplianceScores(t *testing.T) {
	cases, err := LoadComplianceCorpus()
	if err != nil {
		t.Fatal(err)
	}
	var c ComplianceCase
	for _, x := range cases {
		if x.Name == "met-outside-pr" {
			c = x
		}
	}
	if c.Name == "" {
		t.Fatal("met-outside-pr is missing")
	}
	count := func(s string) int { return len(s) / 4 }

	right := RunCompliance(context.Background(), fakeComplianceReviewer{reqs: []inference.ComplianceRequirement{
		{Requirement: "AK-1 settings file encrypted", Met: false},
		{Requirement: "AK-2 no environment builder", Met: true, File: "app/settings.config", Quote: `<store mode="plain" />`},
	}}, []ComplianceCase{c}, "", count, 2, 1, nil)
	if right.Totals.Shown != 2 || right.Totals.Provable != 2 || right.Totals.FalseShown != 0 || right.Totals.Unmet != 2 {
		t.Errorf("totals = %+v, want AK-2 shown twice, AK-1 never, two unmet", right.Totals)
	}
	if right.ErrWrong() != nil || right.ErrIncomplete() != nil {
		t.Errorf("a right answer failed: %v %v", right.ErrWrong(), right.ErrIncomplete())
	}

	wrong := RunCompliance(context.Background(), fakeComplianceReviewer{reqs: []inference.ComplianceRequirement{
		{Requirement: "AK-1 settings file encrypted", Met: true, File: "app/settings.config", Quote: `<store mode="plain" />`},
		{Requirement: "AK-2 no environment builder", Met: true, File: "app/settings.config", Quote: `<store builder="EnvBuilder" mode="plain" />`},
	}}, []ComplianceCase{c}, "", count, 1, 1, nil)
	if wrong.Totals.FalseShown != 1 || wrong.Totals.Dropped[string(inference.ComplianceQuoteRemoved)] != 1 {
		t.Errorf("totals = %+v, want AK-1 shown wrongly and AK-2 dropped for a removed line", wrong.Totals)
	}
	if wrong.ErrWrong() == nil {
		t.Error("a requirement shown wrongly did not fail the run")
	}

	unlisted := RunCompliance(context.Background(), fakeComplianceReviewer{reqs: []inference.ComplianceRequirement{}},
		[]ComplianceCase{c}, "", count, 1, 1, nil)
	if unlisted.Totals.Unlisted != 1 || unlisted.Totals.Shown != 0 {
		t.Errorf("totals = %+v, want AK-2 unlisted", unlisted.Totals)
	}
}

// A requirement is scored for the criterion whose label it carries: a
// keyword of another criterion never claims it, and AK-1 is not AK-10.
func TestScoreComplianceMatchesByLabelFirst(t *testing.T) {
	expect := []ComplianceExpectation{
		{Label: "AC-1", Keywords: []string{"per minute"}, Shown: true},
		{Label: "AC-2", Keywords: []string{"429"}, Shown: false},
		{Label: "AC-10", Keywords: []string{"audit"}, Shown: false},
	}
	checks := []inference.ComplianceCheck{
		{Requirement: inference.ComplianceRequirement{Requirement: "AC-1 return 429 above 100 requests per minute", Met: true}},
	}
	got := scoreCompliance(expect, checks)
	want := []string{ComplianceShownRight, ComplianceHeldBack, ComplianceHeldBack}
	for i, w := range want {
		if got[i].Verdict != w {
			t.Errorf("%s = %s, want %s", got[i].Label, got[i].Verdict, w)
		}
	}
	if !hasLabel("see ak-1.", "AK-1") || hasLabel("AK-10 audit", "AK-1") || hasLabel("XAK-1", "AK-1") {
		t.Error("hasLabel does not match whole labels only")
	}
}

// An answer with no compliance list at all (a parse failure) is incomplete,
// never scored as "nothing shown".
func TestRunComplianceIncomplete(t *testing.T) {
	cases, err := LoadComplianceCorpus()
	if err != nil {
		t.Fatal(err)
	}
	res := RunCompliance(context.Background(), fakeComplianceReviewer{reqs: nil}, cases[:1], "",
		func(s string) int { return len(s) / 4 }, 1, 1, nil)
	if res.Totals.Incomplete != 1 || res.ErrIncomplete() == nil {
		t.Errorf("totals = %+v, want one incomplete answer", res.Totals)
	}
}
