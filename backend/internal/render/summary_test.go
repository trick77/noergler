package render

import (
	"encoding/json"
	"os"
	"strings"
	"testing"

	"github.com/trick77/noergler/internal/inference"
	"github.com/trick77/noergler/internal/jira"
)

// summaryGolden holds the output of the Python _build_summary for one case,
// captured from the venv. The Go input for each case is built in
// summaryCases below with the same arguments the generator used.
type summaryGolden struct {
	Out string `json:"out"`
}

func loadSummaryGolden(t *testing.T) map[string]summaryGolden {
	t.Helper()
	blob, err := os.ReadFile("testdata/summary_golden.json")
	if err != nil {
		t.Fatalf("read summary golden: %v", err)
	}
	var g map[string]summaryGolden
	if err := json.Unmarshal(blob, &g); err != nil {
		t.Fatalf("decode summary golden: %v", err)
	}
	return g
}

func ptrf(f float64) *float64 { return &f }
func ptrs(s string) *string   { return &s }

func finding(severity, comment string, headline *string) inference.ReviewFinding {
	return inference.ReviewFinding{File: "a.go", Line: 1, Severity: severity, Comment: comment, Headline: headline}
}

func testTicket(key, title, ac string) *jira.Ticket {
	return &jira.Ticket{
		Key: key, Title: title, AcceptanceCriteria: ac,
		URL: "https://jira/browse/" + key, IssueType: "Story", Status: "In Progress",
	}
}

// base mirrors the generator's defaults: no findings, no ticket, Jira off,
// compliance check on, a 4000-token AGENTS.md warn threshold and a $5 cap.
func base() SummaryInput {
	return SummaryInput{
		Summary:               inference.NewReviewSummary(),
		TicketComplianceCheck: true,
		AgentsMDWarnTokens:    4000,
		MaxPRCostUSD:          5.0,
		ModelLabel:            "gpt-5.5-medium",
	}
}

func summaryCases() map[string]SummaryInput {
	cases := map[string]SummaryInput{}

	cases["empty"] = base()

	mixed := base()
	mixed.Findings = []inference.ReviewFinding{
		finding("issue", "c", ptrs("Null deref")),
		finding("suggestion", "c", ptrs("Rename it")),
	}
	mixed.AgentsMDFound = true
	mixed.Summary = inference.ReviewSummary{
		Overview: "Adds a thing.", Strengths: []string{"Tests included"},
		SecurityPerformance: "Nothing notable.", TestCoverage: "Good.",
		VerdictDecision: "approve", VerdictRationale: "Fine.",
	}
	mixed.TokenUsage = TokenUsage{Prompt: 1000, Completion: 200, Present: true}
	mixed.PromptBreakdown = PromptBreakdown{Template: 500, RepoInstructions: 300, Files: 200, Present: true}
	mixed.ElapsedSeconds, mixed.ElapsedPresent = 12.34, true
	mixed.InputBudget, mixed.ContextWindow = 628000, 1000000
	mixed.FilesReviewed, mixed.TotalFiles, mixed.FilesCountsSet = 3, 5, true
	mixed.DiffAdded, mixed.DiffRemoved = 40, 7
	mixed.RunCostUSD, mixed.CumulativeCostUSD, mixed.KeySpendUSD = ptrf(0.1234), ptrf(1.5), 42.0
	cases["mixed"] = mixed

	truncated := base()
	truncated.Findings = []inference.ReviewFinding{
		finding("issue", "c", ptrs("One")), finding("issue", "c", ptrs("Two")),
	}
	truncated.Truncated = true
	cases["truncated"] = truncated

	security := base()
	security.Findings = []inference.ReviewFinding{finding("issue", "possible sql injection here", ptrs("Injection"))}
	cases["security"] = security

	secMulti := base()
	secMulti.Findings = []inference.ReviewFinding{
		finding("issue", "sql injection", ptrs("A")),
		finding("issue", "xss hole", ptrs("B")),
		finding("issue", "fine", ptrs("C")),
	}
	cases["security_multiple"] = secMulti

	hf := base()
	hf.Findings = []inference.ReviewFinding{finding("issue", "First line of comment.\nSecond line.", nil)}
	cases["headline_fallback"] = hf

	he := base()
	he.Findings = []inference.ReviewFinding{finding("issue", "Body here.", ptrs("  "))}
	cases["headline_empty_string"] = he

	agentsFound := base()
	agentsFound.AgentsMDFound = true
	agentsFound.PromptBreakdown = PromptBreakdown{Template: 1, RepoInstructions: 3000, Files: 1, Present: true}
	agentsFound.TokenUsage = TokenUsage{Prompt: 1, Completion: 1, Present: true}
	cases["no_findings_agents_found"] = agentsFound

	agentsOver := agentsFound
	agentsOver.PromptBreakdown = PromptBreakdown{Template: 1, RepoInstructions: 9000, Files: 1, Present: true}
	cases["agents_over_warn"] = agentsOver

	agentsNoBreakdown := base()
	agentsNoBreakdown.AgentsMDFound = true
	cases["agents_no_breakdown"] = agentsNoBreakdown

	agentsWarnOff := agentsOver
	agentsWarnOff.AgentsMDWarnTokens = 0
	cases["agents_warn_disabled"] = agentsWarnOff

	tNoCompliance := base()
	tNoCompliance.Ticket = testTicket("ABC-1", "Do the thing", "")
	tNoCompliance.JiraEnabled = true
	cases["ticket_no_compliance"] = tNoCompliance
	cases["ticket_no_ac"] = tNoCompliance

	tDisabled := base()
	tDisabled.Ticket = testTicket("ABC-1", "Do the thing", "AC-1 do it")
	tDisabled.JiraEnabled = true
	tDisabled.TicketComplianceCheck = false
	cases["ticket_compliance_disabled"] = tDisabled

	tFailed := base()
	tFailed.Ticket = testTicket("ABC-1", "Do the thing", "AC-1 do it")
	tFailed.JiraEnabled = true
	tFailed.ComplianceExtractionFailed = true
	cases["ticket_extraction_failed"] = tFailed

	// A ticket with no AC and a failed extraction: the AC check comes first,
	// so the reader is told there was nothing to check rather than that the
	// check broke. Nothing pinned that precedence.
	tNoACFailed := base()
	tNoACFailed.Ticket = testTicket("ABC-1", "Do the thing", "")
	tNoACFailed.JiraEnabled = true
	tNoACFailed.ComplianceExtractionFailed = true
	cases["ticket_no_ac_outranks_extraction_failed"] = tNoACFailed

	// An AC, compliance on, extraction fine, and still no requirements: none
	// of them is verifiable from the code changes. This is the default branch
	// of the reason switch, which no case reached.
	tNoReqs := base()
	tNoReqs.Ticket = testTicket("ABC-1", "Do the thing", "AC-1 do it")
	tNoReqs.JiraEnabled = true
	tNoReqs.ComplianceRequirements = []inference.ComplianceRequirement{}
	cases["ticket_no_requirements_code_relevant"] = tNoReqs

	tFull := base()
	tFull.Ticket = testTicket("ABC-1", "Do the thing", "AC-1")
	tFull.JiraEnabled = true
	tFull.ComplianceRequirements = []inference.ComplianceRequirement{{Requirement: "Does X", Met: true}}
	cases["ticket_fully_compliant"] = tFull

	tPartial := tFull
	tPartial.ComplianceRequirements = []inference.ComplianceRequirement{
		{Requirement: "Does X", Met: true}, {Requirement: "Does Y", Met: false},
	}
	cases["ticket_partial"] = tPartial

	tNone := tFull
	tNone.ComplianceRequirements = []inference.ComplianceRequirement{{Requirement: "Does X", Met: false}}
	cases["ticket_none_met"] = tNone

	tParent := base()
	tParent.Ticket = testTicket("ABC-2", "Sub", "")
	tParent.ParentTicket = testTicket("ABC-1", "Parent", "")
	tParent.JiraEnabled = true
	cases["ticket_with_parent"] = tParent

	jiraOn := base()
	jiraOn.JiraEnabled = true
	cases["no_ticket_jira_enabled"] = jiraOn
	cases["no_ticket_jira_disabled"] = base()

	for name, decision := range map[string]string{
		"verdict_request_changes": "request_changes",
		"verdict_followups":       "approve_with_followups",
		"verdict_unknown":         "bogus",
	} {
		v := base()
		rationale := map[string]string{
			"request_changes": "No.", "approve_with_followups": "Mostly.", "bogus": "Hm.",
		}[decision]
		v.Summary = inference.ReviewSummary{VerdictDecision: decision, VerdictRationale: rationale}
		cases[name] = v
	}

	inc := base()
	inc.ReviewedCommit = strings.Repeat("b", 40)
	inc.IncrementalFrom = strings.Repeat("a", 40)
	cases["incremental"] = inc

	sk := base()
	sk.SkippedFiles = []string{"src/big.go", "x/y/huge.ts"}
	sk.ContentSkippedFiles = []string{"z/med.py"}
	cases["skipped_files"] = sk

	cf := base()
	cf.CrossFileSymbols = []string{"Foo", "Bar"}
	cases["cross_file_few"] = cf

	cf1 := base()
	cf1.CrossFileSymbols = []string{"Foo"}
	cases["cross_file_one"] = cf1

	cfm := base()
	cfm.CrossFileSymbols = []string{"A", "B", "C", "D", "E", "F", "G"}
	cases["cross_file_many"] = cfm

	unpriced := base()
	unpriced.TokenUsage = TokenUsage{Prompt: 10, Completion: 2, Present: true}
	unpriced.KeySpendUSD = 7.0
	cases["cost_unpriced"] = unpriced

	zeroKey := base()
	zeroKey.TokenUsage = TokenUsage{Prompt: 10, Completion: 2, Present: true}
	zeroKey.RunCostUSD, zeroKey.CumulativeCostUSD = ptrf(0.5), ptrf(0.5)
	cases["cost_zero_keyspend"] = zeroKey

	noCost := base()
	noCost.TokenUsage = TokenUsage{Prompt: 10, Completion: 2, Present: true}
	cases["cost_none_at_all"] = noCost

	runOnly := noCost
	runOnly.RunCostUSD = ptrf(0.5)
	cases["cost_run_no_cumulative"] = runOnly

	allFiles := base()
	allFiles.FilesReviewed, allFiles.TotalFiles, allFiles.FilesCountsSet = 4, 4, true
	allFiles.DiffAdded = 10
	cases["files_all_reviewed"] = allFiles

	diffOnly := base()
	diffOnly.DiffAdded, diffOnly.DiffRemoved = 5, 3
	cases["diff_only"] = diffOnly

	noWindow := base()
	noWindow.TokenUsage = TokenUsage{Prompt: 1000, Completion: 10, Present: true}
	noWindow.InputBudget = 628000
	cases["budget_no_window"] = noWindow

	return cases
}

// TestSummaryMatchesPython drives the Go builder with the same arguments the
// Python generator used and compares the rendered markdown byte for byte.
// This is what the ~35 test_build_summary_* cases in test_reviewer.py cover.
func TestSummaryMatchesPython(t *testing.T) {
	golden := loadSummaryGolden(t)
	cases := summaryCases()

	for name, want := range golden {
		in, ok := cases[name]
		if !ok {
			t.Errorf("golden case %q has no Go input", name)
			continue
		}
		t.Run(name, func(t *testing.T) {
			if got := Summary(in); got != want.Out {
				t.Errorf("\n--- got ---\n%s\n--- want ---\n%s", got, want.Out)
			}
		})
	}
	for name := range cases {
		if _, ok := golden[name]; !ok {
			t.Errorf("Go case %q has no golden value", name)
		}
	}
}

// Section order is load-bearing: readers scan for Recommendation last and the
// footnote is always after a rule.
func TestSummarySectionOrder(t *testing.T) {
	in := base()
	in.Ticket = testTicket("ABC-1", "T", "")
	in.JiraEnabled = true
	got := Summary(in)

	want := []string{
		"### Overview", "### Strengths", "### Issues / Suggestions",
		"### Security / Performance", "### Test Coverage", "### Ticket",
		"### Recommendation", "\n---\n",
	}
	prev := -1
	for _, section := range want {
		i := strings.Index(got, section)
		if i < 0 {
			t.Fatalf("section %q missing from:\n%s", section, got)
		}
		if i < prev {
			t.Errorf("section %q is out of order", section)
		}
		prev = i
	}
}

// The italic fallbacks mean the model omitted the field; the plain sentinels
// mean it looked and found nothing. Conflating them hides a model defect.
func TestSummaryDistinguishesOmittedFromEmpty(t *testing.T) {
	got := Summary(base())
	for _, want := range []string{"_Not provided._", "None.", "None notable.", "_Not assessed._", "_No rationale provided._"} {
		if !strings.Contains(got, want) {
			t.Errorf("missing %q in:\n%s", want, got)
		}
	}
}
