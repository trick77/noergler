package render

import (
	"fmt"
	"path"
	"strings"

	"github.com/trick77/noergler-go/internal/inference"
	"github.com/trick77/noergler-go/internal/jira"
)

// verdictLabels maps the model's verdict enum to its rendered label. An
// unrecognised value falls back to approve, as Python's dict.get does.
var verdictLabels = map[string]string{
	"approve":                "Approve ✅",
	"approve_with_followups": "Approve with follow-ups ⚠️",
	"request_changes":        "Request changes 🛑",
}

// PromptBreakdown is the per-section input token count the footnote renders.
// RepoInstructions doubles as the AGENTS.md figure on the scope line.
type PromptBreakdown struct {
	Template         int
	RepoInstructions int
	Files            int
	Present          bool
}

// TokenUsage is the model's reported prompt and completion token counts.
type TokenUsage struct {
	Prompt     int
	Completion int
	Present    bool
}

// SummaryInput is everything the summary comment renders.
//
// Pointer fields are the ones where absent and zero differ: a nil cost means
// the run was not priced (and the line is omitted entirely), while 0.0 would
// print "$0.00" and read as a broken integration.
type SummaryInput struct {
	Findings  []inference.ReviewFinding
	Truncated bool
	Summary   inference.ReviewSummary

	AgentsMDFound bool
	// SkippedFiles were too large for the model; ContentSkippedFiles were
	// reviewed from their diff without full file context.
	SkippedFiles        []string
	ContentSkippedFiles []string

	TokenUsage      TokenUsage
	PromptBreakdown PromptBreakdown

	Ticket       *jira.Ticket
	ParentTicket *jira.Ticket

	ComplianceRequirements     []inference.ComplianceRequirement
	ComplianceExtractionFailed bool
	TicketComplianceCheck      bool
	JiraEnabled                bool

	ElapsedSeconds   float64
	ElapsedPresent   bool
	ReviewedCommit   string
	IncrementalFrom  string
	FilesReviewed    int
	TotalFiles       int
	FilesCountsSet   bool
	DiffAdded        int
	DiffRemoved      int
	CrossFileSymbols []string

	InputBudget   int
	ContextWindow int
	ModelLabel    string

	// RunCostUSD is nil when the gateway did not price the run.
	RunCostUSD *float64
	// CumulativeCostUSD is the PR total including this run; only read when
	// RunCostUSD is set, matching Python's nesting.
	CumulativeCostUSD *float64
	// KeySpendUSD is the gateway's own gauge for the whole API key. Shown,
	// never summed, suppressed when zero.
	KeySpendUSD  float64
	MaxPRCostUSD float64

	// AgentsMDWarnTokens is the soft threshold the scope line reports
	// against; 0 or less disables the stats entirely.
	AgentsMDWarnTokens int
}

// Summary renders the PR summary comment.
//
// Section order is fixed (reviewer.py:2037): optional incremental header,
// Overview, Strengths, Issues / Suggestions, Security / Performance, Test
// Coverage, the ticket block when a ticket is linked, Recommendation, and the
// footnote after a horizontal rule.
func Summary(in SummaryInput) string {
	var sections []string

	if in.IncrementalFrom != "" && in.ReviewedCommit != "" {
		sections = append(sections, fmt.Sprintf(
			"### Review summary (incremental update)\n- Changes reviewed: `%s` .. `%s`",
			shortSHA(in.IncrementalFrom, 10), shortSHA(in.ReviewedCommit, 10)))
	}

	sections = append(sections, renderOverview(in))
	sections = append(sections, renderStrengths(in))
	sections = append(sections, renderIssues(in))

	// The italic fallbacks mean the model omitted the field, which is a
	// defect worth seeing; the plain sentinels mean it looked and found
	// nothing. Two different states, both rendered.
	sections = append(sections, "### Security / Performance\n"+
		orDefault(WrapProse(strings.TrimSpace(in.Summary.SecurityPerformance)), "None notable."))
	sections = append(sections, "### Test Coverage\n"+
		orDefault(WrapProse(strings.TrimSpace(in.Summary.TestCoverage)), "_Not assessed._"))

	if in.Ticket != nil {
		sections = append(sections, renderTicket(in))
	}

	sections = append(sections, renderRecommendation(in))

	if footnote := renderFootnote(in); footnote != "" {
		sections = append(sections, footnote)
	}

	return strings.Join(sections, "\n\n")
}

func renderOverview(in SummaryInput) string {
	body := "_Not provided._"
	if in.Summary.Overview != "" {
		body = WrapProse(strings.TrimSpace(in.Summary.Overview))
	}
	return "### Overview\n" + body
}

func renderStrengths(in SummaryInput) string {
	if len(in.Summary.Strengths) == 0 {
		return "### Strengths\nNone."
	}
	var b strings.Builder
	b.WriteString("### Strengths")
	for _, s := range in.Summary.Strengths {
		b.WriteString("\n- " + s)
	}
	return b.String()
}

// renderIssues lists numbered headlines only; the detail lives on the inline
// comments.
func renderIssues(in SummaryInput) string {
	lines := []string{"### Issues / Suggestions"}
	if len(in.Findings) == 0 {
		return lines[0] + "\nNone."
	}

	securityCount := 0
	for _, f := range in.Findings {
		if IsSecurityFinding(f.Comment) {
			securityCount++
		}
	}
	if securityCount > 0 {
		lines = append(lines, fmt.Sprintf("- %s 🔒 — review carefully",
			Plural(securityCount, "potential security issue")))
	}
	if in.Truncated {
		lines = append(lines, fmt.Sprintf(
			"- Showing top %d findings by severity. Additional findings were omitted.",
			len(in.Findings)))
	}
	if securityCount > 0 || in.Truncated {
		lines = append(lines, "")
	}
	for i, f := range in.Findings {
		headline := ""
		if f.Headline != nil {
			headline = strings.TrimSpace(*f.Headline)
		}
		if headline == "" {
			headline = "(no description)"
			if f.Comment != "" {
				headline = strings.TrimSpace(firstLine(f.Comment))
			}
		}
		lines = append(lines, fmt.Sprintf("%d. %s", i+1, headline))
	}
	return strings.Join(lines, "\n")
}

func renderTicket(in SummaryInput) string {
	var reqs []inference.ComplianceRequirement
	if in.TicketComplianceCheck {
		reqs = in.ComplianceRequirements
	}
	hasCompliance := len(reqs) > 0

	heading := "### Ticket"
	verdictSuffix := ""
	if hasCompliance {
		met := 0
		for _, r := range reqs {
			if r.Met {
				met++
			}
		}
		label, emoji := "Not compliant", "❌"
		switch {
		case met == len(reqs):
			label, emoji = "Fully compliant", "✅"
		case met > 0:
			label, emoji = "Partially compliant", "⚠️"
		}
		verdictSuffix = fmt.Sprintf(" · **%s** %s", label, emoji)
		heading = "### Requirement Compliance"
	}

	lines := []string{heading}
	if in.ParentTicket != nil {
		lines = append(lines, fmt.Sprintf("**[%s](%s)** — %s", in.ParentTicket.Key, in.ParentTicket.URL, in.ParentTicket.Title))
		lines = append(lines, fmt.Sprintf("**↳ [%s](%s)** — %s%s", in.Ticket.Key, in.Ticket.URL, in.Ticket.Title, verdictSuffix))
	} else {
		lines = append(lines, fmt.Sprintf("**[%s](%s)** — %s%s", in.Ticket.Key, in.Ticket.URL, in.Ticket.Title, verdictSuffix))
	}

	if hasCompliance {
		for _, r := range reqs {
			mark := "❌"
			if r.Met {
				mark = "✅"
			}
			requirement := r.Requirement
			if requirement == "" {
				requirement = "???"
			}
			lines = append(lines, fmt.Sprintf("- %s %s", requirement, mark))
		}
		return strings.Join(lines, "\n")
	}

	// Order matters: a missing AC means there was nothing to extract, so
	// "extraction failed" would be misleading. Ticket-side conditions are
	// checked before LLM-side ones (reviewer.py:2164).
	var reason string
	switch {
	case !in.TicketComplianceCheck:
		reason = "_Compliance check disabled in config_"
	case in.Ticket.AcceptanceCriteria == "":
		reason = "_No acceptance criteria found in ticket_"
	case in.ComplianceExtractionFailed:
		reason = "_Compliance extraction failed during LLM review — check logs_"
	default:
		reason = "_No acceptance criteria are verifiable from code changes_"
	}
	return strings.Join(append(lines, reason), "\n")
}

func renderRecommendation(in SummaryInput) string {
	label, ok := verdictLabels[in.Summary.VerdictDecision]
	if !ok {
		label = verdictLabels[inference.DefaultVerdictDecision]
	}
	rationale := orDefault(WrapProse(strings.TrimSpace(in.Summary.VerdictRationale)), "_No rationale provided._")
	return fmt.Sprintf("### Recommendation\n**%s** — %s", label, rationale)
}

// renderFootnote builds the scope lines then the telemetry lines, each as an
// italic bullet under a horizontal rule.
func renderFootnote(in SummaryInput) string {
	var scope []string

	if in.AgentsMDFound {
		scope = append(scope, agentsMDScopeLine(in))
	} else {
		scope = append(scope, "Tip: Add an `AGENTS.md` to your repository root with project-specific "+
			"review guidelines for more targeted feedback. 💡")
	}

	if in.Ticket == nil {
		if in.JiraEnabled {
			scope = append(scope, "No ticket found in branch name or PR title ℹ️")
		} else {
			scope = append(scope, "Jira is not enabled ℹ️")
		}
	}

	if in.FilesCountsSet {
		filesStr := fmt.Sprintf("Reviewed %d files", in.FilesReviewed)
		if in.FilesReviewed != in.TotalFiles {
			filesStr = fmt.Sprintf("Reviewed %d of %d files (%d skipped: lock files, binaries, config)",
				in.FilesReviewed, in.TotalFiles, in.TotalFiles-in.FilesReviewed)
		}
		if diff := diffParts(in.DiffAdded, in.DiffRemoved); diff != "" {
			filesStr += ", " + diff + " lines"
		}
		scope = append(scope, filesStr)
	} else if in.DiffAdded != 0 || in.DiffRemoved != 0 {
		if diff := diffParts(in.DiffAdded, in.DiffRemoved); diff != "" {
			scope = append(scope, "Diff: "+diff+" lines")
		}
	}

	if n := len(in.CrossFileSymbols); n > 0 {
		shown := in.CrossFileSymbols
		suffix := ""
		if n > 5 {
			shown = shown[:5]
			suffix = fmt.Sprintf(" and %d more", n-5)
		}
		quoted := make([]string, len(shown))
		for i, s := range shown {
			quoted[i] = "`" + s + "`"
		}
		depWord := "dependencies"
		if n == 1 {
			depWord = "dependency"
		}
		scope = append(scope, fmt.Sprintf("%d cross-file %s analyzed (%s%s)",
			n, depWord, strings.Join(quoted, ", "), suffix))
	}

	if len(in.SkippedFiles) > 0 {
		scope = append(scope, "Not reviewed (too large): "+baseNames(in.SkippedFiles)+" ⚠️")
	}
	if len(in.ContentSkippedFiles) > 0 {
		scope = append(scope, "Reviewed without full file context (too large): "+baseNames(in.ContentSkippedFiles)+" ⚠️")
	}

	var telemetry []string
	if in.InputBudget > 0 && in.TokenUsage.Present {
		windowSuffix := ""
		if in.ContextWindow > 0 {
			windowSuffix = fmt.Sprintf(", model max %s", FmtK(in.ContextWindow))
		}
		telemetry = append(telemetry, fmt.Sprintf("Tokens used: %s of %s available (%d%% used%s)",
			FmtK(in.TokenUsage.Prompt), FmtK(in.InputBudget),
			pct(in.TokenUsage.Prompt, in.InputBudget), windowSuffix))
	}

	if in.TokenUsage.Present {
		if in.PromptBreakdown.Present {
			telemetry = append(telemetry, fmt.Sprintf(
				"Input tokens: ~%s review prompt · ~%s AGENTS.md · ~%s file content",
				Fmt(in.PromptBreakdown.Template), Fmt(in.PromptBreakdown.RepoInstructions),
				Fmt(in.PromptBreakdown.Files)))
		}
		total := in.TokenUsage.Prompt + in.TokenUsage.Completion
		stats := fmt.Sprintf("Model: `%s` · ↑ %s · ↓ %s (%s total)",
			in.ModelLabel, Fmt(in.TokenUsage.Prompt), Fmt(in.TokenUsage.Completion), Fmt(total))
		if in.ElapsedPresent {
			stats += fmt.Sprintf(" · ⏱️ %.1fs", in.ElapsedSeconds)
		}
		telemetry = append(telemetry, stats)
	}

	// The run cost is omitted when the endpoint did not price the run, which
	// beats a misleading "$0.00". The key total is shown regardless: it is
	// the gateway's own gauge and stays valid while one run goes unpriced.
	var costParts []string
	if in.RunCostUSD != nil {
		costParts = append(costParts, fmt.Sprintf("$%.2f this run", *in.RunCostUSD))
		if in.CumulativeCostUSD != nil {
			costParts = append(costParts, fmt.Sprintf("$%.2f PR total / $%.2f limit",
				*in.CumulativeCostUSD, in.MaxPRCostUSD))
		}
	}
	// Zero is suppressed: a proxy that does not track key spend reports 0,
	// and "$0.00 key total" reads as a broken integration.
	if in.KeySpendUSD != 0 {
		costParts = append(costParts, fmt.Sprintf("$%.2f key total", in.KeySpendUSD))
	}
	if len(costParts) > 0 {
		telemetry = append(telemetry, "Cost: "+strings.Join(costParts, ", "))
	}

	all := append(scope, telemetry...)
	if len(all) == 0 {
		return ""
	}
	var b strings.Builder
	b.WriteString("---")
	for _, line := range all {
		b.WriteString("\n- _" + line + "_")
	}
	return b.String()
}

// agentsMDScopeLine reports AGENTS.md usage and, when a warn threshold is
// configured and a token count is known, how close it is to that threshold.
func agentsMDScopeLine(in SummaryInput) string {
	const base = "Using project-specific review guidelines from `AGENTS.md`"
	tokens := 0
	if in.PromptBreakdown.Present {
		tokens = in.PromptBreakdown.RepoInstructions
	}
	if tokens == 0 || in.AgentsMDWarnTokens <= 0 {
		return base + " ✅"
	}
	stats := fmt.Sprintf("(~%d / %d tokens, %d%%)", tokens, in.AgentsMDWarnTokens,
		pct(tokens, in.AgentsMDWarnTokens))
	if tokens > in.AgentsMDWarnTokens {
		return fmt.Sprintf("%s %s — risk of context bloat, consider trimming ⚠️", base, stats)
	}
	return fmt.Sprintf("%s %s ✅", base, stats)
}

func diffParts(added, removed int) string {
	var parts []string
	if added != 0 {
		parts = append(parts, fmt.Sprintf("+%d", added))
	}
	if removed != 0 {
		parts = append(parts, fmt.Sprintf("-%d", removed))
	}
	return strings.Join(parts, " / ")
}

// baseNames renders a comma-separated list of quoted file basenames.
//
// Python's PurePosixPath("").name is "" while Go's path.Base("") is ".", so
// the empty case is guarded rather than left to path.Base.
func baseNames(paths []string) string {
	out := make([]string, len(paths))
	for i, p := range paths {
		name := ""
		if p != "" {
			name = path.Base(p)
		}
		out[i] = "`" + name + "`"
	}
	return strings.Join(out, ", ")
}

func orDefault(s, fallback string) string {
	if s == "" {
		return fallback
	}
	return s
}

func firstLine(s string) string {
	if i := strings.IndexByte(s, '\n'); i >= 0 {
		return s[:i]
	}
	return s
}

func shortSHA(s string, n int) string {
	if len(s) > n {
		return s[:n]
	}
	return s
}
