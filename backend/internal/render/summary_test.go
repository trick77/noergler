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

	focus := mixed
	focus.Focus = []inference.FocusItem{
		{File: "app/users.go", Line: 41, Why: "Validates the id list; check the length limit that bounds the loop."},
		{File: "app/store.go", Line: 12, Why: "Batches the lookups; check the empty list."},
	}
	cases["focus"] = focus

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

	// An empty AGENTS.md: the footer says the review had no project rules,
	// never the "add an AGENTS.md" tip.
	agentsEmpty := base()
	agentsEmpty.AgentsMDFound = true
	agentsEmpty.AgentsMDEmpty = true
	cases["agents_empty"] = agentsEmpty

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
	cases["ticket_no_requirements"] = tNoReqs

	tFull := base()
	tFull.Ticket = testTicket("ABC-1", "Do the thing", "AC-1")
	tFull.JiraEnabled = true
	tFull.ComplianceRequirements = []inference.ComplianceRequirement{{Requirement: "Does X", Met: true}}
	cases["ticket_shown"] = tFull

	tPartial := tFull
	tPartial.ComplianceRequirements = []inference.ComplianceRequirement{
		{Requirement: "Does X", Met: true}, {Requirement: "Does Y", Met: false},
	}
	cases["ticket_unmet_left_out"] = tPartial

	// The model copies the ticket's labels into the requirement however it
	// likes; bare, Bitbucket links them to Jira issues that do not exist.
	tLabels := tFull
	tLabels.Ticket = testTicket("ABC-1", "Do the thing", "AK-1 does X\nAK-2 does Y")
	tLabels.ComplianceRequirements = []inference.ComplianceRequirement{
		{Requirement: "AK-1: Does X", Met: true}, {Requirement: "Does Y (AK-2)", Met: false},
	}
	tLabels.Summary.Overview = "Implements AK-1; AK-2 is still open, see ABC-1."
	tLabels.Findings = []inference.ReviewFinding{finding("issue", "AK-2 is not enforced.", ptrs("AK-2 open"))}
	cases["ticket_requirement_labels"] = tLabels

	tNone := tFull
	tNone.ComplianceRequirements = []inference.ComplianceRequirement{{Requirement: "Does X", Met: false}}
	cases["ticket_none_shown"] = tNone

	tParent := base()
	tParent.Ticket = testTicket("ABC-2", "Sub", "")
	tParent.ParentTicket = testTicket("ABC-1", "Parent", "")
	tParent.JiraEnabled = true
	cases["ticket_with_parent"] = tParent

	jiraOn := base()
	jiraOn.JiraEnabled = true
	cases["no_ticket_jira_enabled"] = jiraOn
	jiraDown := base()
	jiraDown.JiraEnabled = true
	jiraDown.JiraUnavailable = true
	cases["jira_unavailable"] = jiraDown
	parentDown := base()
	parentDown.Ticket = testTicket("ABC-2", "Sub", "")
	parentDown.JiraEnabled = true
	parentDown.JiraUnavailable = true
	cases["jira_unavailable_parent"] = parentDown
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

// TestSummaryGoldenUpdate rewrites testdata/summary_golden.json from the
// current renderer when SUMMARY_GOLDEN_UPDATE=1; otherwise it does nothing.
// The golden's diff is what a reviewer of a layout change reads.
func TestSummaryGoldenUpdate(t *testing.T) {
	if os.Getenv("SUMMARY_GOLDEN_UPDATE") != "1" {
		t.Skip("set SUMMARY_GOLDEN_UPDATE=1 to rewrite the golden")
	}
	out := map[string]summaryGolden{}
	for name, in := range summaryCases() {
		out[name] = summaryGolden{Out: Summary(in)}
	}
	blob, err := json.MarshalIndent(out, "", " ")
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile("testdata/summary_golden.json", append(blob, '\n'), 0o644); err != nil {
		t.Fatal(err)
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

// The ticket section never says a requirement is unmet and never rates the
// ticket, in either language, whatever list reaches it: a PR is a delta, so
// "not met" is never proven. Once a requirement met in deployment scripts
// the PR did not touch read "not compliant" on the PR.
func TestTicketNeverSaysUnmet(t *testing.T) {
	lists := [][]inference.ComplianceRequirement{
		{{Requirement: "Does X", Met: false}},
		{{Requirement: "Does X", Met: true}, {Requirement: "Does Y", Met: false}},
		{{Requirement: "Does X", Met: true}},
		{},
	}
	banned := []string{"❌", "⚠️", "Nicht erfüllt", "Teilweise erfüllt", "Vollständig erfüllt",
		"Not compliant", "Partially compliant", "Fully compliant", "Does Y"}
	for _, l := range []lang.Lang{lang.DE, lang.EN} {
		for i, reqs := range lists {
			in := base()
			in.Lang = l
			in.Ticket = testTicket("ABC-1", "Do the thing", "AC-1 do it")
			in.ParentTicket = testTicket("ABC-0", "Parent", "")
			in.JiraEnabled = true
			in.TicketComplianceCheck = true
			in.ComplianceRequirements = reqs
			got := renderTicket(in)
			for _, b := range banned {
				if strings.Contains(got, b) {
					t.Errorf("%s list %d: ticket section carries %q:\n%s", l, i, b, got)
				}
			}
		}
	}
}

// The PR's whole diff unreadable: the latest push's proofs still show, with
// a line saying they are all that was checked; none at all reads "not
// checked", never "none shown".
func TestTicketHistoryUnknown(t *testing.T) {
	in := base()
	in.Ticket = testTicket("ABC-1", "Do the thing", "AC-1 do it")
	in.JiraEnabled = true
	in.TicketComplianceCheck = true
	in.ComplianceHistoryUnknown = true
	in.ComplianceRequirements = []inference.ComplianceRequirement{{Requirement: "Does X", Met: true}}
	if got := renderTicket(in); !strings.Contains(got, "Does X ✅") || !strings.Contains(got, "latest push only") {
		t.Errorf("proofs without the note:\n%s", got)
	}
	in.ComplianceRequirements = []inference.ComplianceRequirement{}
	if got := renderTicket(in); !strings.Contains(got, "Requirements not checked") || strings.Contains(got, "No requirement shown") {
		t.Errorf("an unchecked history reads as none shown:\n%s", got)
	}
}

// Section order is load-bearing: the summary often goes unread past its top,
// so what the reader acts on (the recommendation, the open findings) comes
// first; the footnote is always after a rule.
func TestSummarySectionOrder(t *testing.T) {
	in := base()
	in.Ticket = testTicket("ABC-1", "T", "")
	in.JiraEnabled = true
	in.Visual = []VisualLine{{Path: "a-snapshots/x.png", Text: "Moved."}}
	in.Focus = []inference.FocusItem{{File: "a.go", Line: 1, Why: "Read this."}}
	in.Summary.SecurityPerformance, in.Summary.TestCoverage = "One query per row.", "Handler tested."
	in.Summary.Strengths = []string{"Small"}
	got := Summary(in)

	want := []string{
		"### PR review summary", "### Recommendation", "### Issues and suggestions", "### Overview",
		"### Review focus", "### Security and performance", "### Test coverage", "### Strengths",
		"### Visual changes", "### Ticket", "\n---\n",
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

// The overview, the issue list and the recommendation always render, an
// omitted overview or rationale as its italic fallback (a model defect worth
// seeing). Every other section with nothing to say, omitted or a sentinel,
// is left out: a heading over "None." is a line the reader skips past.
func TestSummaryLeavesOutEmptySections(t *testing.T) {
	got := Summary(base())
	for _, want := range []string{"_Not provided._", "### Issues and suggestions\nNone.", "_No rationale provided._"} {
		if !strings.Contains(got, want) {
			t.Errorf("missing %q in:\n%s", want, got)
		}
	}
	for _, gone := range []string{"### Strengths", "### Security and performance", "### Test coverage", "### Review focus"} {
		if strings.Contains(got, gone) {
			t.Errorf("empty section %q rendered:\n%s", gone, got)
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
		// "more" counts screenshots; the variants keep their own suffix, summed.
		"- `a.png`, `d.png`, `e.png` and 1 more — The edit field uses the new font; explained by `theme.css`. _(3 more",
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

	// A shared base name forces paths; a folded line then names one and
	// counts the rest, or the subject is wider than the line again.
	in = base()
	in.Visual = []VisualLine{
		{Path: "apps/e2e/__screenshots__/login.spec.ts/header.png", Text: "same"},
		{Path: "apps/web/__screenshots__/home.spec.ts/header.png", Text: "same"},
		{Path: "apps/web/__screenshots__/home.spec.ts/footer.png", Text: "same"},
	}
	got = Summary(in)
	if !strings.Contains(got, "- `apps/e2e/__screenshots__/login.spec.ts/header.png` and 2 more — same") {
		t.Errorf("one path and a count:\n%s", got)
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
	if got = Summary(in); !strings.Contains(got, "- `a.png`, `b.png`, `c.png` und 1 weiterer — gleich") {
		t.Errorf("German fold:\n%s", got)
	}
}

// A nested AGENTS.md left out is named in full (every one has the same
// base name), on one line per reason.
func TestSummaryNamesAgentsMDLeftOut(t *testing.T) {
	in := base()
	in.AgentsMDFound = true
	in.AgentsMDLeftOut = []AgentsMDLeft{
		{Path: "services/pay/AGENTS.md", Reason: LeftTokenLimit},
		{Path: "web/AGENTS.md", Reason: LeftOversize},
		{Path: "tools/AGENTS.md", Reason: LeftTokenLimit},
	}
	got := Summary(in)
	for _, want := range []string{
		"Not used (over the AGENTS.md token limit): `services/pay/AGENTS.md`, `tools/AGENTS.md` ⚠️",
		"Not used (file too large): `web/AGENTS.md` ⚠️",
	} {
		if !strings.Contains(got, want) {
			t.Errorf("footer must carry %q:\n%s", want, got)
		}
	}
	in.AgentsMDLeftOut = nil
	if got := Summary(in); strings.Contains(got, "Not used (") {
		t.Errorf("nothing left out, no line:\n%s", got)
	}
}

// The nested files the prompt carried are named; with none, no line (the
// root file alone is every PR's).
func TestSummaryNamesAgentsMDApplied(t *testing.T) {
	in := base()
	in.AgentsMDFound = true
	in.AgentsMDNested = []AgentsMDFile{{Path: "services/AGENTS.md", Tokens: 10}, {Path: "services/pay/AGENTS.md", Tokens: 20}}
	if got := Summary(in); !strings.Contains(got, "Also applied: `services/AGENTS.md`, `services/pay/AGENTS.md` ℹ️") {
		t.Errorf("footer must name the applied files:\n%s", got)
	}
	in.Lang = lang.DE
	if got := Summary(in); !strings.Contains(got, "Zusätzlich angewendet: `services/AGENTS.md`") {
		t.Errorf("German:\n%s", got)
	}
	in.AgentsMDNested = nil
	if got := Summary(in); strings.Contains(got, "angewendet") {
		t.Errorf("root only, no line:\n%s", got)
	}
}

// The bloat warning measures the root and the nested files together and
// names the largest; the root alone keeps its own wording.
func TestSummaryBloatCountsNested(t *testing.T) {
	in := base()
	in.AgentsMDFound = true
	in.AgentsMDWarnTokens = 4000
	in.AgentsMDTokens = 1000
	in.AgentsMDNested = []AgentsMDFile{
		{Path: "a/AGENTS.md", Tokens: 500}, {Path: "b/AGENTS.md", Tokens: 2000},
		{Path: "c/AGENTS.md", Tokens: 600}, {Path: "d/AGENTS.md", Tokens: 100},
	}
	want := "`AGENTS.md` files take ~4200 of 4000 tokens together (105%), largest: `b/AGENTS.md` ~2000, `AGENTS.md` ~1000, `c/AGENTS.md` ~600"
	if got := Summary(in); !strings.Contains(got, want) {
		t.Errorf("want %q:\n%s", want, got)
	}
	in.AgentsMDNested = in.AgentsMDNested[:1]
	if got := Summary(in); strings.Contains(got, "context bloat") {
		t.Errorf("1500 of 4000, no warning:\n%s", got)
	}
	in.AgentsMDNested = nil
	in.AgentsMDTokens = 5000
	if got := Summary(in); !strings.Contains(got, "`AGENTS.md` takes ~5000 of 4000 tokens (125%)") {
		t.Errorf("root only keeps its wording:\n%s", got)
	}
	// A root that was not read (0 tokens) is never named as one to trim.
	in.AgentsMDTokens = 0
	in.AgentsMDNested = []AgentsMDFile{{Path: "a/AGENTS.md", Tokens: 3000}, {Path: "b/AGENTS.md", Tokens: 1500}}
	if got := Summary(in); strings.Contains(got, "`AGENTS.md` ~0") || !strings.Contains(got, "largest: `a/AGENTS.md` ~3000, `b/AGENTS.md` ~1500 —") {
		t.Errorf("unread root named:\n%s", got)
	}
}

// A nested lookup cut short is one line, never a list of paths.
func TestSummarySaysNestedLookupIncomplete(t *testing.T) {
	in := base()
	in.AgentsMDFound = true
	for code, want := range map[string]string{
		LookupUnreadable: "Lookup of AGENTS.md below the root stopped (Bitbucket error) ⚠️",
		LookupProbeCap:   "Not every directory checked for an AGENTS.md (too many in this PR) ⚠️",
	} {
		in.AgentsMDIncomplete = code
		if got := Summary(in); !strings.Contains(got, want) {
			t.Errorf("%s: want %q:\n%s", code, want, got)
		}
	}
	in.AgentsMDIncomplete = ""
	if got := Summary(in); strings.Contains(got, "Lookup of") || strings.Contains(got, "Not every directory") {
		t.Errorf("complete lookup, no line:\n%s", got)
	}
}
