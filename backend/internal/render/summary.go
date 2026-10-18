package render

import (
	"fmt"
	"math"
	"path"
	"strings"

	"github.com/trick77/noergler/internal/inference"
	"github.com/trick77/noergler/internal/jira"
	"github.com/trick77/noergler/internal/lang"
)

// verdictLabels maps the model's verdict enum to its rendered label. An
// unrecognised value falls back to approve.
var verdictLabels = map[lang.Lang]map[string]string{
	lang.EN: {
		"approve":                "Approve ✅",
		"approve_with_followups": "Approve with follow-ups ⚠️",
		"request_changes":        "Request changes 🛑",
	},
	lang.DE: {
		"approve":                "Freigeben ✅",
		"approve_with_followups": "Freigeben mit Nacharbeiten ⚠️",
		"request_changes":        "Änderungen nötig 🛑",
	},
}

// PromptBreakdown is the per-section input token count the footnote renders.
// RepoInstructions doubles as the AGENTS.md figure on the scope line.
type PromptBreakdown struct {
	Template         int
	RepoInstructions int
	Files            int
	Present          bool
}

// SummaryInput is everything the summary comment renders.
//
// Pointer fields are the ones where absent and zero differ: a nil cost means
// the run was not priced (and the line is omitted entirely), while 0.0 would
// print "$0.00" and read as a broken integration.
type SummaryInput struct {
	// Lang is the PR author's language; the zero value renders German.
	Lang lang.Lang

	Findings  []inference.ReviewFinding
	Truncated bool
	Summary   inference.ReviewSummary

	// FixedEarlier and OpenEarlier are the earlier findings this run listed
	// to the model, retired by it and still open after it. Zero on a full
	// review: it can retire nothing, so it can say nothing about them.
	FixedEarlier int
	OpenEarlier  int
	// Dropped is what validation kept off the PR.
	//
	// Decided, not complete: the line counts only what this run checked.
	// Repeats of earlier findings, findings trimmed from the prompt, the cap
	// and refused inline posts are logged, not counted here. A repeat of a
	// listed finding is already in OpenEarlier and would count twice; the
	// cap has its own bullet; a refused post is still in the list.
	Dropped int

	AgentsMDFound bool
	// SkippedFiles were too large for the size gate; OverBudgetFiles were
	// left out of the prompt because the PR is over the model's token budget
	// (named to the model only), whatever their own size. Both are not
	// reviewed. ContentSkippedFiles were reviewed from their diff without
	// full file context, and are never in either list.
	SkippedFiles        []string
	OverBudgetFiles     []string
	ContentSkippedFiles []string

	PromptBreakdown PromptBreakdown

	Ticket       *jira.Ticket
	ParentTicket *jira.Ticket

	ComplianceRequirements     []inference.ComplianceRequirement
	ComplianceExtractionFailed bool
	TicketComplianceCheck      bool
	JiraEnabled                bool

	ElapsedSeconds  float64
	ElapsedPresent  bool
	ReviewedCommit  string
	IncrementalFrom string
	TotalFiles      int
	FilesCountsSet  bool
	DiffAdded       int
	DiffRemoved     int
	ModelLabel      string
	// RunURL is this run's page on the dashboard; empty (no public URL
	// configured) prints no link rather than one to nowhere.
	RunURL string

	// RunCostUSD is nil when the gateway did not price the run.
	RunCostUSD *float64
	// CumulativeCostUSD is the PR total including this run; only read when
	// RunCostUSD is set, so an unpriced run shows no total.
	CumulativeCostUSD *float64
	MaxPRCostUSD      float64

	// AgentsMDWarnTokens is the soft threshold the scope line reports
	// against; 0 or less disables the stats entirely.
	AgentsMDWarnTokens int
}

// Summary renders the PR summary comment.
//
// Section order is fixed (TestSummarySectionOrder): optional incremental header,
// Overview, Strengths, Issues and suggestions, Security and performance, Test
// coverage, the ticket block when a ticket is linked, Recommendation, and the
// footnote after a horizontal rule.
func Summary(in SummaryInput) string {
	if in.Lang != lang.EN {
		in = swissInput(in)
	}
	var sections []string
	p := in.Lang.Pick

	if in.IncrementalFrom != "" && in.ReviewedCommit != "" {
		sections = append(sections, fmt.Sprintf(
			p("### Review-Zusammenfassung (inkrementell)\n- Geprüfte Änderungen: `%s` .. `%s`",
				"### Review summary (incremental update)\n- Changes reviewed: `%s` .. `%s`"),
			shortSHA(in.IncrementalFrom, 10), shortSHA(in.ReviewedCommit, 10)))
	}

	sections = append(sections, renderOverview(in))
	sections = append(sections, renderStrengths(in))
	sections = append(sections, renderIssues(in))

	// The italic fallbacks mean the model omitted the field, which is a
	// defect worth seeing; the plain sentinels mean it looked and found
	// nothing. Two different states, both rendered.
	sections = append(sections, p("### Sicherheit und Performance\n", "### Security and performance\n")+
		orDefault(WrapProse(strings.TrimSpace(in.Summary.SecurityPerformance)), p("Nichts Auffälliges.", "None notable.")))
	sections = append(sections, p("### Testabdeckung\n", "### Test coverage\n")+
		orDefault(WrapProse(strings.TrimSpace(in.Summary.TestCoverage)), p("_Nicht beurteilt._", "_Not assessed._")))

	if in.Ticket != nil {
		sections = append(sections, renderTicket(in))
	}

	sections = append(sections, renderRecommendation(in))

	if footnote := renderFootnote(in); footnote != "" {
		sections = append(sections, footnote)
	}

	return strings.Join(sections, "\n\n")
}

// swissInput is in with SwissSS applied to every text the model wrote, on
// copies: the caller's findings are stored as they came.
func swissInput(in SummaryInput) SummaryInput {
	s := in.Summary
	s.Overview = SwissSS(s.Overview)
	s.SecurityPerformance = SwissSS(s.SecurityPerformance)
	s.TestCoverage = SwissSS(s.TestCoverage)
	s.VerdictRationale = SwissSS(s.VerdictRationale)
	s.Strengths = make([]string, len(in.Summary.Strengths))
	for i, v := range in.Summary.Strengths {
		s.Strengths[i] = SwissSS(v)
	}
	in.Summary = s
	findings := make([]inference.ReviewFinding, len(in.Findings))
	for i, f := range in.Findings {
		f.Comment = SwissSS(f.Comment)
		if f.Headline != nil {
			h := SwissSS(*f.Headline)
			f.Headline = &h
		}
		findings[i] = f
	}
	in.Findings = findings
	reqs := make([]inference.ComplianceRequirement, len(in.ComplianceRequirements))
	for i, r := range in.ComplianceRequirements {
		r.Requirement = SwissSS(r.Requirement)
		reqs[i] = r
	}
	if in.ComplianceRequirements != nil {
		in.ComplianceRequirements = reqs
	}
	return in
}

func renderOverview(in SummaryInput) string {
	p := in.Lang.Pick
	body := p("_Nicht angegeben._", "_Not provided._")
	if in.Summary.Overview != "" {
		body = WrapProse(strings.TrimSpace(in.Summary.Overview))
	}
	return p("### Überblick\n", "### Overview\n") + body
}

func renderStrengths(in SummaryInput) string {
	p := in.Lang.Pick
	heading := p("### Stärken", "### Strengths")
	if len(in.Summary.Strengths) == 0 {
		return heading + p("\nKeine.", "\nNone.")
	}
	var b strings.Builder
	b.WriteString(heading)
	for _, s := range in.Summary.Strengths {
		b.WriteString("\n- " + s)
	}
	return b.String()
}

// renderAccounting is the line under the issues that says what the list
// leaves out: earlier findings and dropped ones. Empty when there is nothing
// to say, so a first review reads as before.
func renderAccounting(in SummaryInput) string {
	p := in.Lang.Pick
	var parts, earlier []string
	if in.FixedEarlier > 0 {
		earlier = append(earlier, fmt.Sprintf(p("%d mit diesem Push behoben", "%d fixed by this push"), in.FixedEarlier))
	}
	if in.OpenEarlier > 0 {
		earlier = append(earlier, fmt.Sprintf(p("%d noch offen", "%d still open"), in.OpenEarlier))
	}
	if len(earlier) > 0 {
		parts = append(parts, p("Frühere Findings geprüft: ", "Earlier findings checked: ")+strings.Join(earlier, ", ")+".")
	}
	if in.Dropped > 0 {
		parts = append(parts, PluralIn(in.Lang, in.Dropped, "Finding", "Findings", "finding", "findings")+
			p(" von der Validierung verworfen.", " dropped by validation."))
	}
	if len(parts) == 0 {
		return ""
	}
	return "_" + strings.Join(parts, " ") + "_"
}

// renderIssues lists numbered headlines only; the detail lives on the inline
// comments.
func renderIssues(in SummaryInput) string {
	p := in.Lang.Pick
	lines := []string{p("### Probleme und Vorschläge", "### Issues and suggestions")}
	accounting := renderAccounting(in)
	if len(in.Findings) == 0 {
		none := p("\nKeine.", "\nNone.")
		if accounting != "" {
			return lines[0] + none + "\n\n" + accounting
		}
		return lines[0] + none
	}

	securityCount := 0
	for _, f := range in.Findings {
		if IsSecurityFinding(f.Comment) {
			securityCount++
		}
	}
	if securityCount > 0 {
		lines = append(lines, fmt.Sprintf(p("- %s 🔒 — sorgfältig prüfen", "- %s 🔒 — review carefully"),
			PluralIn(in.Lang, securityCount, "mögliches Sicherheitsproblem", "mögliche Sicherheitsprobleme",
				"potential security issue", "potential security issues")))
	}
	if in.Truncated {
		lines = append(lines, fmt.Sprintf(
			p("- Die %d wichtigsten Findings nach Schweregrad. Weitere wurden weggelassen.",
				"- Showing top %d findings by severity. Additional findings were omitted."),
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
			headline = p("(keine Beschreibung)", "(no description)")
			if f.Comment != "" {
				headline = strings.TrimSpace(firstLine(f.Comment))
			}
		}
		lines = append(lines, fmt.Sprintf("%d. %s", i+1, headline))
	}
	if accounting != "" {
		lines = append(lines, "", accounting)
	}
	return strings.Join(lines, "\n")
}

func renderTicket(in SummaryInput) string {
	var reqs []inference.ComplianceRequirement
	if in.TicketComplianceCheck {
		reqs = in.ComplianceRequirements
	}
	hasCompliance := len(reqs) > 0
	p := in.Lang.Pick

	heading := "### Ticket"
	verdictSuffix := ""
	if hasCompliance {
		met := 0
		for _, r := range reqs {
			if r.Met {
				met++
			}
		}
		label, emoji := p("Nicht erfüllt", "Not compliant"), "❌"
		switch {
		case met == len(reqs):
			label, emoji = p("Vollständig erfüllt", "Fully compliant"), "✅"
		case met > 0:
			label, emoji = p("Teilweise erfüllt", "Partially compliant"), "⚠️"
		}
		verdictSuffix = fmt.Sprintf(" · **%s** %s", label, emoji)
		heading = p("### Erfüllung der Anforderungen", "### Requirement compliance")
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
	// checked before LLM-side ones.
	var reason string
	switch {
	case !in.TicketComplianceCheck:
		reason = p("_Compliance-Prüfung in den Einstellungen deaktiviert_", "_Compliance check disabled in config_")
	case in.Ticket.AcceptanceCriteria == "":
		reason = p("_Keine Akzeptanzkriterien im Ticket gefunden_", "_No acceptance criteria found in ticket_")
	case in.ComplianceExtractionFailed:
		reason = p("_Compliance-Auswertung im LLM-Review fehlgeschlagen — siehe Logs_",
			"_Compliance extraction failed during LLM review — check logs_")
	default:
		reason = p("_Keine Akzeptanzkriterien anhand der Code-Änderungen prüfbar_",
			"_No acceptance criteria are verifiable from code changes_")
	}
	return strings.Join(append(lines, reason), "\n")
}

func renderRecommendation(in SummaryInput) string {
	p := in.Lang.Pick
	labels := verdictLabels[in.Lang.Or(lang.DE)]
	label, ok := labels[in.Summary.VerdictDecision]
	if !ok {
		label = labels[inference.DefaultVerdictDecision]
	}
	rationale := orDefault(WrapProse(strings.TrimSpace(in.Summary.VerdictRationale)),
		p("_Keine Begründung angegeben._", "_No rationale provided._"))
	return fmt.Sprintf(p("### Empfehlung\n**%s** — %s", "### Recommendation\n**%s** — %s"), label, rationale)
}

// renderFootnote is the footer under a rule: a bullet for each thing the
// author can act on, then one telemetry line ending in the run's dashboard
// page. Everything else (tokens, prompt parts, cross-file symbols, key
// spend) lives on that page, not in the PR (decided).
func renderFootnote(in SummaryInput) string {
	p := in.Lang.Pick
	var warn []string

	if !in.AgentsMDFound {
		warn = append(warn, p(
			"Tipp: Eine `AGENTS.md` mit projektspezifischen Review-Richtlinien im Repo-Root sorgt "+
				"für gezielteres Feedback. 💡",
			"Tip: Add an `AGENTS.md` to your repository root with project-specific "+
				"review guidelines for more targeted feedback. 💡"))
	} else if line := agentsMDBloatLine(in); line != "" {
		warn = append(warn, line)
	}
	// Only when Jira is on: "Jira is not enabled" is true of every PR of the
	// team and gives the author nothing to do.
	if in.Ticket == nil && in.JiraEnabled {
		warn = append(warn, p("Kein Ticket in Branch-Name oder PR-Titel gefunden ℹ️",
			"No ticket found in branch name or PR title ℹ️"))
	}
	if len(in.SkippedFiles) > 0 {
		warn = append(warn, p("Nicht geprüft (zu gross): ", "Not reviewed (too large): ")+baseNames(in.SkippedFiles)+" ⚠️")
	}
	if len(in.OverBudgetFiles) > 0 {
		warn = append(warn, p("Nicht geprüft (PR über dem Token-Budget des Modells): ",
			"Not reviewed (PR over the model's token budget): ")+baseNames(in.OverBudgetFiles)+" ⚠️")
	}
	if len(in.ContentSkippedFiles) > 0 {
		warn = append(warn, p("Ohne vollen Datei-Kontext geprüft (zu gross): ",
			"Reviewed without full file context (too large): ")+baseNames(in.ContentSkippedFiles)+" ⚠️")
	}

	tele := telemetryLine(in)
	if len(warn) == 0 && tele == "" {
		return ""
	}
	var b strings.Builder
	b.WriteString("---")
	for _, line := range warn {
		b.WriteString("\n- _" + line + "_")
	}
	if tele != "" {
		if len(warn) > 0 {
			b.WriteString("\n")
		}
		b.WriteString("\n_" + tele + "_")
	}
	return b.String()
}

// telemetryLine is the footer's one line of figures: model, files, lines,
// elapsed, cost, and the link to the run's page.
func telemetryLine(in SummaryInput) string {
	p := in.Lang.Pick
	var parts []string
	if in.ModelLabel != "" {
		parts = append(parts, "`"+in.ModelLabel+"`")
	}
	// Every reviewable file, over the same set as the +/- lines. How many of
	// them were not reviewed is counted from the warning lines above, so the
	// two never disagree and "40 files" never reads as all reviewed.
	if in.FilesCountsSet {
		files := PluralIn(in.Lang, in.TotalFiles, "Datei", "Dateien", "file", "files")
		if n := len(in.SkippedFiles) + len(in.OverBudgetFiles); n > 0 {
			files += fmt.Sprintf(p(", %d nicht geprüft", ", %d not reviewed"), n)
		}
		parts = append(parts, files)
	}
	if d := diffParts(in.DiffAdded, in.DiffRemoved); d != "" {
		parts = append(parts, d)
	}
	if in.ElapsedPresent {
		parts = append(parts, Elapsed(in.ElapsedSeconds))
	}
	// A summary is always a review that called the model, so a missing cost
	// is an unpriced call: said so, never $0.00, and with no PR total, which
	// is only read when this run is priced.
	if in.RunCostUSD != nil {
		cost := fmt.Sprintf("$%.2f", *in.RunCostUSD)
		if in.CumulativeCostUSD != nil {
			cost += fmt.Sprintf(p(", PR $%.2f von $%.2f", ", PR $%.2f of $%.2f"), *in.CumulativeCostUSD, in.MaxPRCostUSD)
		}
		parts = append(parts, cost)
	} else {
		parts = append(parts, p("Kosten nicht gemeldet", "cost not reported"))
	}
	if in.RunURL != "" {
		parts = append(parts, fmt.Sprintf(p("[Details zum Run](%s)", "[Run details](%s)"), in.RunURL))
	}
	return strings.Join(parts, " · ")
}

// Elapsed is a run's wall clock as the dashboard prints it: 850ms, 12.3s,
// 1m 13s. Each form is rounded before it is chosen, so 59.96s reads 1m 0s,
// never 60.0s, and 119.5s reads 2m 0s, never 1m 60s.
func Elapsed(seconds float64) string {
	ms := int64(math.Round(seconds * 1000))
	tenths := int64(math.Round(float64(ms) / 100))
	switch {
	case ms < 1000:
		return fmt.Sprintf("%dms", ms)
	case tenths < 600:
		return fmt.Sprintf("%d.%ds", tenths/10, tenths%10)
	}
	total := int64(math.Round(float64(ms) / 1000))
	return fmt.Sprintf("%dm %ds", total/60, total%60)
}

// agentsMDBloatLine warns when AGENTS.md is over the soft threshold, and is
// empty otherwise: "AGENTS.md was used" is true of every reviewed PR and
// moved to the run page.
func agentsMDBloatLine(in SummaryInput) string {
	p := in.Lang.Pick
	if !in.PromptBreakdown.Present || in.AgentsMDWarnTokens <= 0 {
		return ""
	}
	tokens := in.PromptBreakdown.RepoInstructions
	if tokens <= in.AgentsMDWarnTokens {
		return ""
	}
	return fmt.Sprintf(p("`AGENTS.md` belegt ~%d von %d Tokens (%d%%) — droht den Kontext aufzublähen, besser kürzen ⚠️",
		"`AGENTS.md` takes ~%d of %d tokens (%d%%) — risk of context bloat, consider trimming ⚠️"),
		tokens, in.AgentsMDWarnTokens, pct(tokens, in.AgentsMDWarnTokens))
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
// path.Base("") is ".", not "", so the empty case is guarded rather than
// left to path.Base.
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
