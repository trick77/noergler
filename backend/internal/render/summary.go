package render

import (
	"fmt"
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
	// RunCostUSD is set, so an unpriced run shows no total.
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

// renderFootnote builds the scope lines then the telemetry lines, each as an
// italic bullet under a horizontal rule.
func renderFootnote(in SummaryInput) string {
	var scope []string
	p := in.Lang.Pick

	if in.AgentsMDFound {
		scope = append(scope, agentsMDScopeLine(in))
	} else {
		scope = append(scope, p(
			"Tipp: Eine `AGENTS.md` mit projektspezifischen Review-Richtlinien im Repo-Root sorgt "+
				"für gezielteres Feedback. 💡",
			"Tip: Add an `AGENTS.md` to your repository root with project-specific "+
				"review guidelines for more targeted feedback. 💡"))
	}

	if in.Ticket == nil {
		if in.JiraEnabled {
			scope = append(scope, p("Kein Ticket in Branch-Name oder PR-Titel gefunden ℹ️",
				"No ticket found in branch name or PR title ℹ️"))
		} else {
			scope = append(scope, p("Jira ist nicht aktiviert ℹ️", "Jira is not enabled ℹ️"))
		}
	}

	lines := p(" Zeilen", " lines")
	if in.FilesCountsSet {
		filesStr := fmt.Sprintf("Reviewed %d files", in.FilesReviewed)
		if in.Lang != lang.EN {
			filesStr = PluralIn(in.Lang, in.FilesReviewed, "Datei", "Dateien", "", "") + " geprüft"
		}
		if in.FilesReviewed != in.TotalFiles {
			filesStr = fmt.Sprintf(p("%d von %d Dateien geprüft (%d übersprungen: Lock-Files, Binaries, Config)",
				"Reviewed %d of %d files (%d skipped: lock files, binaries, config)"),
				in.FilesReviewed, in.TotalFiles, in.TotalFiles-in.FilesReviewed)
		}
		if diff := diffParts(in.DiffAdded, in.DiffRemoved); diff != "" {
			filesStr += ", " + diff + lines
		}
		scope = append(scope, filesStr)
	} else if in.DiffAdded != 0 || in.DiffRemoved != 0 {
		if diff := diffParts(in.DiffAdded, in.DiffRemoved); diff != "" {
			scope = append(scope, "Diff: "+diff+lines)
		}
	}

	if n := len(in.CrossFileSymbols); n > 0 {
		shown := in.CrossFileSymbols
		suffix := ""
		if n > 5 {
			shown = shown[:5]
			suffix = fmt.Sprintf(p(" und %d weitere", " and %d more"), n-5)
		}
		quoted := make([]string, len(shown))
		for i, s := range shown {
			quoted[i] = "`" + s + "`"
		}
		deps := PluralIn(in.Lang, n, "dateiübergreifende Abhängigkeit", "dateiübergreifende Abhängigkeiten",
			"cross-file dependency", "cross-file dependencies")
		scope = append(scope, fmt.Sprintf(p("%s analysiert (%s%s)", "%s analyzed (%s%s)"),
			deps, strings.Join(quoted, ", "), suffix))
	}

	if len(in.SkippedFiles) > 0 {
		scope = append(scope, p("Nicht geprüft (zu gross): ", "Not reviewed (too large): ")+baseNames(in.SkippedFiles)+" ⚠️")
	}
	if len(in.ContentSkippedFiles) > 0 {
		scope = append(scope, p("Ohne vollen Datei-Kontext geprüft (zu gross): ",
			"Reviewed without full file context (too large): ")+baseNames(in.ContentSkippedFiles)+" ⚠️")
	}

	var telemetry []string
	if in.InputBudget > 0 && in.TokenUsage.Present {
		windowSuffix := ""
		if in.ContextWindow > 0 {
			windowSuffix = fmt.Sprintf(p(", Modell-Max. %s", ", model max %s"), FmtK(in.ContextWindow))
		}
		telemetry = append(telemetry, fmt.Sprintf(p("Tokens: %s von %s verfügbar (%d%% genutzt%s)",
			"Tokens used: %s of %s available (%d%% used%s)"),
			FmtK(in.TokenUsage.Prompt), FmtK(in.InputBudget),
			pct(in.TokenUsage.Prompt, in.InputBudget), windowSuffix))
	}

	if in.TokenUsage.Present {
		if in.PromptBreakdown.Present {
			telemetry = append(telemetry, fmt.Sprintf(p(
				"Input-Tokens: ~%s Review-Prompt · ~%s AGENTS.md · ~%s Dateiinhalt",
				"Input tokens: ~%s review prompt · ~%s AGENTS.md · ~%s file content"),
				Fmt(in.PromptBreakdown.Template), Fmt(in.PromptBreakdown.RepoInstructions),
				Fmt(in.PromptBreakdown.Files)))
		}
		total := in.TokenUsage.Prompt + in.TokenUsage.Completion
		stats := fmt.Sprintf(p("Modell: `%s` · ↑ %s · ↓ %s (%s total)", "Model: `%s` · ↑ %s · ↓ %s (%s total)"),
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
		costParts = append(costParts, fmt.Sprintf(p("$%.2f dieser Lauf", "$%.2f this run"), *in.RunCostUSD))
		if in.CumulativeCostUSD != nil {
			costParts = append(costParts, fmt.Sprintf(p("$%.2f PR gesamt / $%.2f Limit", "$%.2f PR total / $%.2f limit"),
				*in.CumulativeCostUSD, in.MaxPRCostUSD))
		}
	}
	// Zero is suppressed: a proxy that does not track key spend reports 0,
	// and "$0.00 key total" reads as a broken integration.
	if in.KeySpendUSD != 0 {
		costParts = append(costParts, fmt.Sprintf(p("$%.2f Key gesamt", "$%.2f key total"), in.KeySpendUSD))
	}
	if len(costParts) > 0 {
		telemetry = append(telemetry, p("Kosten: ", "Cost: ")+strings.Join(costParts, ", "))
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
	p := in.Lang.Pick
	base := p("Projektspezifische Review-Richtlinien aus `AGENTS.md` verwendet",
		"Using project-specific review guidelines from `AGENTS.md`")
	tokens := 0
	if in.PromptBreakdown.Present {
		tokens = in.PromptBreakdown.RepoInstructions
	}
	if tokens == 0 || in.AgentsMDWarnTokens <= 0 {
		return base + " ✅"
	}
	stats := fmt.Sprintf(p("(~%d / %d Tokens, %d%%)", "(~%d / %d tokens, %d%%)"), tokens, in.AgentsMDWarnTokens,
		pct(tokens, in.AgentsMDWarnTokens))
	if tokens > in.AgentsMDWarnTokens {
		return fmt.Sprintf(p("%s %s — droht den Kontext aufzublähen, besser kürzen ⚠️",
			"%s %s — risk of context bloat, consider trimming ⚠️"), base, stats)
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
