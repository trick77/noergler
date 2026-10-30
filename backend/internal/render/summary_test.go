package render

import (
	"encoding/json"
	"fmt"
	"os"
	"strings"
	"testing"

	"github.com/trick77/noergler/internal/inference"
	"github.com/trick77/noergler/internal/jira"
	"github.com/trick77/noergler/internal/lang"
)

// summaryGolden holds the expected summary for one case. The input for each
// case is built in summaryCases below.
//
// The headings were reworded after the corpus was first captured: "and"
// instead of a slash, and sentence case ("Issues and suggestions", "Security
// and performance", "Test coverage", "Requirement compliance"). The corpus
// carries the rename and is byte-for-byte otherwise.
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
		Lang:                  lang.EN,
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
	mixed.AgentsMDTokens = 300
	mixed.ElapsedSeconds, mixed.ElapsedPresent = 12.34, true
	mixed.ModelLabel = "gpt-5.5-medium"
	mixed.TotalFiles, mixed.FilesCountsSet = 5, true
	mixed.DiffAdded, mixed.DiffRemoved = 40, 7
	mixed.RunCostUSD, mixed.CumulativeCostUSD, mixed.MaxPRCostUSD = ptrf(0.1234), ptrf(1.5), 5.0
	mixed.PRURL = "https://noergler.example.com/prs/AAAAAAAAAAAAAAAAAAAAAA"
	cases["mixed"] = mixed

	truncated := base()
	truncated.Findings = []inference.ReviewFinding{
		finding("issue", "c", ptrs("One")), finding("issue", "c", ptrs("Two")),
	}
	truncated.Cut = 3
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
	agentsFound.AgentsMDTokens = 3000
	cases["no_findings_agents_found"] = agentsFound

	accounting := base()
	accounting.Findings = []inference.ReviewFinding{finding("issue", "c", ptrs("New one"))}
	accounting.EarlierOpen = []inference.ReviewFinding{finding("suggestion", "c", ptrs("Older one"))}
	accounting.FixedEarlier, accounting.Dropped = 2, 2
	cases["accounting"] = accounting

	accountingNone := base()
	accountingNone.FixedEarlier, accountingNone.Dropped = 1, 1
	cases["accounting_no_findings"] = accountingNone

	// A capped run keeps its own bullet; the line beside it never counts the
	// cut a second time.
	accountingCapped := base()
	accountingCapped.Findings = []inference.ReviewFinding{finding("issue", "c", ptrs("Kept"))}
	accountingCapped.Cut = 1
	accountingCapped.EarlierOpen = []inference.ReviewFinding{finding("issue", "c", ptrs("Still there"))}
	cases["accounting_truncated"] = accountingCapped

	// A push that adds nothing still lists the PR's open findings.
	openOnly := base()
	openOnly.EarlierOpen = []inference.ReviewFinding{
		finding("suggestion", "c", ptrs("Older suggestion")), finding("issue", "c", ptrs("Older issue")),
	}
	cases["accounting_open_only"] = openOnly

	// Past the list cap the rest is counted, issues first in the shown part.
	long := base()
	for i := range 31 {
		long.EarlierOpen = append(long.EarlierOpen, finding("suggestion", "c", ptrs(fmt.Sprintf("Old %d", i))))
	}
	long.Findings = []inference.ReviewFinding{finding("issue", "c", ptrs("New issue"))}
	cases["list_capped"] = long

	agentsOver := agentsFound
	agentsOver.AgentsMDTokens = 9000
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

	// The latest push on its own line; the Overview stays the PR's.
	incPush := inc
	incPush.Summary.Overview = "Adds retries with backoff to the payment client."
	incPush.Summary.LatestPush = "Fixes a typo in the\nretry log message."
	cases["incremental_latest_push"] = incPush

	// A long push sentence wraps inside its list item; a sentinel is dropped.
	incLong := inc
	incLong.Summary.LatestPush = "Moves the retry policy into its own type, threads the caller's context through every " +
		"attempt and replaces the fixed sleep with a jittered backoff that resets after a success."
	cases["incremental_latest_push_wrapped"] = incLong
	incNone := inc
	incNone.Summary.LatestPush = "None."
	cases["incremental_latest_push_sentinel"] = incNone

	// A full review has no latest push: whatever the model put there is not shown.
	fullPush := base()
	fullPush.Summary.LatestPush = "Should not render."
	cases["full_ignores_latest_push"] = fullPush

	sk := base()
	sk.SkippedFiles = []string{"src/big.go", "x/y/huge.ts"}
	sk.OverBudgetFiles = []string{"docs/README.md"}
	sk.ContentSkippedFiles = []string{"z/med.py"}
	cases["skipped_files"] = sk

	// An unpriced run says so, and shows no PR total.
	unpriced := base()
	unpriced.ElapsedSeconds, unpriced.ElapsedPresent = 73.1, true
	cases["cost_unpriced"] = unpriced

	withPR := base()
	withPR.ElapsedSeconds, withPR.ElapsedPresent = 73.1, true
	withPR.RunCostUSD, withPR.CumulativeCostUSD, withPR.MaxPRCostUSD = ptrf(0.5), ptrf(0.5), 5.0
	cases["cost_with_pr_total"] = withPR

	noCost := base()
	cases["cost_none_at_all"] = noCost

	runOnly := noCost
	runOnly.RunCostUSD = ptrf(0.5)
	cases["cost_run_no_cumulative"] = runOnly

	// No public URL: the same line, no link to a host that does not exist.
	noLink := base()
	noLink.ModelLabel = "gpt-5.5-medium"
	noLink.ElapsedSeconds, noLink.ElapsedPresent = 0.85, true
	noLink.RunCostUSD = ptrf(0.01)
	cases["no_pr_url"] = noLink

	link := noLink
	link.PRURL = "https://noergler.example.com/prs/BBBBBBBBBBBBBBBBBBBBBB"
	cases["pr_url"] = link

	linkDE := link
	linkDE.Lang = lang.DE
	linkDE.TotalFiles, linkDE.FilesCountsSet = 9, true
	cases["pr_url_de"] = linkDE

	// Files not reviewed (size gate, token budget) are counted on the line
	// itself, so the count never reads as "all of them reviewed".
	overBudget := base()
	overBudget.TotalFiles, overBudget.FilesCountsSet = 40, true
	overBudget.SkippedFiles = []string{"bundle.properties"}
	overBudget.OverBudgetFiles = []string{"a.go", "b.go"}
	overBudget.DiffAdded = 10
	cases["files_over_budget"] = overBudget

	allFiles := base()
	allFiles.TotalFiles, allFiles.FilesCountsSet = 4, true
	allFiles.DiffAdded = 10
	cases["files_all_reviewed"] = allFiles

	diffOnly := base()
	diffOnly.DiffAdded, diffOnly.DiffRemoved = 5, 3
	cases["diff_only"] = diffOnly

	return cases
}

// Elapsed reads like the dashboard's duration, so the footer and the run
// page agree; seconds round before splitting, never "1m 60s".
func TestElapsed(t *testing.T) {
	for in, want := range map[float64]string{
		0.85: "850ms", 12.34: "12.3s", 59.94: "59.9s", 59.96: "1m 0s", 73.1: "1m 13s", 119.5: "2m 0s",
	} {
		if got := Elapsed(in); got != want {
			t.Errorf("Elapsed(%v) = %q, want %q", in, got, want)
		}
	}
}

// TestSummaryIsPinned drives the builder over the whole case corpus and
// compares the rendered markdown byte for byte.
func TestSummaryIsPinned(t *testing.T) {
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
	in.Visual = []VisualLine{{Path: "a-snapshots/x.png", Text: "Moved."}}
	got := Summary(in)

	want := []string{
		"### PR review summary", "### Overview", "### Strengths", "### Issues and suggestions",
		"### Security and performance", "### Test coverage", "### Visual changes", "### Ticket",
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

// One line per screenshot: the model's words, or why it was not compared.
// No screenshot, no section.
func TestSummaryVisualChanges(t *testing.T) {
	if got := Summary(base()); strings.Contains(got, "Visual changes") {
		t.Errorf("a PR without screenshots has no section:\n%s", got)
	}
	in := base()
	in.Visual = []VisualLine{
		{Path: "apps/e2e/tests/__screenshots__/login.spec.ts/login-chromium.png", Text: "Submit button moved\nbelow the form; explained by `login.css`.", Variants: 2},
		{Path: "a/__screenshots__/header.png", Status: VisualNoVision},
		{Path: "b/__screenshots__/header.png", Status: VisualOverCap},
	}
	got := Summary(in)
	for _, want := range []string{
		// The file name, not the path, then the line flattened to one
		// paragraph and wrapped as the list item it is.
		"### Visual changes\n- `login-chromium.png` — Submit button moved below the form; explained by `login.css`. _(2 more variants)_",
		// A name two screenshots share keeps its path.
		"- `a/__screenshots__/header.png` — _not compared: the model takes no images._",
		"- `b/__screenshots__/header.png` — _not compared yet",
	} {
		if !strings.Contains(got, want) {
			t.Errorf("missing %q in:\n%s", want, got)
		}
	}

	in.Lang = lang.DE
	in.Visual = []VisualLine{{Path: "s-snapshots/a.png", Text: "Schrift grösser, ausserhalb des Diffs nicht erklärt: Straße."}}
	got = Summary(in)
	if !strings.Contains(got, "### Visuelle Änderungen") || !strings.Contains(got, "Strasse") {
		t.Errorf("German section, SwissSS on the model's line:\n%s", got)
	}
}

func TestSummaryVisualChangesAreCapped(t *testing.T) {
	in := base()
	// Each line its own sentence: identical ones would fold into one.
	for i := range maxVisualLines + 3 {
		in.Visual = append(in.Visual, VisualLine{Path: fmt.Sprintf("s-snapshots/%d.png", i), Text: fmt.Sprintf("change %d", i)})
	}
	got := Summary(in)
	if !strings.Contains(got, "_… and 3 more._") || strings.Contains(got, fmt.Sprintf("`s-snapshots/%d.png`", maxVisualLines)) {
		t.Errorf("want %d lines and a count:\n%s", maxVisualLines, got)
	}
}

// Screenshots with the byte-identical sentence are one line (user,
// 2026-10-30: the same "uses the new font" ten times). Exact text only; a
// status line never folds; a folded line names three and counts the rest,
// variants included; the cap counts folded lines.
func TestSummaryVisualChangesFoldIdenticalLines(t *testing.T) {
	in := base()
	same := "The edit field uses the new font;  explained by `theme.css`.\n"
	in.Visual = []VisualLine{
		{Path: "s/a.png", Text: same, Variants: 1},
		{Path: "s/b.png", Status: VisualTooLarge},
		{Path: "s/c.png", Text: "The edit field uses the old font; explained by `theme.css`."},
		{Path: "s/d.png", Text: same},
		{Path: "s/e.png", Text: same, Variants: 2},
		{Path: "s/f.png", Text: same},
		{Path: "s/g.png", Status: VisualTooLarge},
	}
	got := Summary(in)
	for _, want := range []string{
		"- `a.png`, `d.png`, `e.png` and 4 more — The edit field uses the new font; explained by `theme.css`.",
		"- `b.png` — _not compared: image too large._",
		"- `c.png` — The edit field uses the old font",
		"- `g.png` — _not compared: image too large._",
	} {
		if !strings.Contains(got, want) {
			t.Errorf("missing %q in:\n%s", want, got)
		}
	}
	if n := strings.Count(got, "uses the new font"); n != 1 {
		t.Errorf("the shared sentence appears %d times, want once:\n%s", n, got)
	}
	if strings.Contains(got, "more variant") {
		t.Errorf("a folded line counts variants in its total, not as a suffix:\n%s", got)
	}

	// Two of them fold, the rest stay: the cap counts lines, not screenshots.
	in = base()
	for i := range maxVisualLines + 2 {
		in.Visual = append(in.Visual, VisualLine{Path: fmt.Sprintf("s/%d.png", i), Text: "same"})
	}
	in.Visual = append(in.Visual, VisualLine{Path: "s/z.png", Text: "other"})
	got = Summary(in)
	if strings.Contains(got, "more._") || !strings.Contains(got, "`z.png` — other") {
		t.Errorf("folded lines fit under the cap:\n%s", got)
	}

	in.Lang = lang.DE
	in.Visual = []VisualLine{{Path: "s/a.png", Text: "gleich"}, {Path: "s/b.png", Text: "gleich"},
		{Path: "s/c.png", Text: "gleich"}, {Path: "s/d.png", Text: "gleich"}}
	if got = Summary(in); !strings.Contains(got, "- `a.png`, `b.png`, `c.png` und 1 weitere — gleich") {
		t.Errorf("German fold:\n%s", got)
	}
}
