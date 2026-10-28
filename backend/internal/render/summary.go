package render

import (
	"fmt"
	"math"
	"path"
	"slices"
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

// SummaryInput is everything the PR summary comment renders. It is the PR
// summary: the whole PR on every run, never one push. The comment is
// replaced on each run, so what it leaves out of the PR is gone.
//
// Pointer fields are the ones where absent and zero differ: a nil cost means
// the run was not priced (and the line is omitted entirely), while 0.0 would
// print "$0.00" and read as a broken integration.
type SummaryInput struct {
	// Lang is the PR author's language; the zero value renders German.
	Lang lang.Lang

	// Findings are this run's new ones; EarlierOpen the PR's earlier
	// findings still open after it, on any run, a full one included. The
	// issue list is both.
	Findings    []inference.ReviewFinding
	EarlierOpen []inference.ReviewFinding
	// EarlierUnknown: the earlier findings could not be read, so the list
	// is this run's only, and the summary says so rather than read as the
	// whole PR.
	EarlierUnknown bool
	// Cut is the new findings the comment cap kept off the PR. The verdict
	// counts them, so the list says they exist.
	Cut     int
	Summary inference.ReviewSummary

	// FixedEarlier is the earlier findings this run retired. Zero on a full
	// review: it can retire nothing.
	FixedEarlier int
	// Dropped is what validation kept off the PR.
	//
	// Decided, not complete: the line counts only what this run checked.
	// Repeats of earlier findings, the cap and refused inline posts are
	// logged, not counted here. A repeat is already listed as the earlier
	// finding; the cap has its own bullet; a refused post is still in the
	// list.
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
	// PRURL is the PR's page on the dashboard, every run of it included;
	// empty (no public URL configured) prints no link rather than one to
	// nowhere.
	PRURL string

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

// Summary renders the PR summary comment: the whole PR, every open finding,
// on every run.
//
// Section order is fixed (TestSummarySectionOrder): the PR summary heading
// (with the last push on an incremental run), Overview, Strengths, Issues and suggestions, Security and performance, Test
// coverage, the ticket block when a ticket is linked, Recommendation, and the
// footnote after a horizontal rule.
func Summary(in SummaryInput) string {
	if in.Lang != lang.EN {
		in = swissInput(in)
	}
	var sections []string
	p := in.Lang.Pick

	// Named for the PR (user, 2026-10-28): the last push is one line under
	// it, never the scope of the summary.
	header := p("### PR Review-Zusammenfassung", "### PR review summary")
	// The latest push gets its own line, filled from its own field, so the
	// Overview below stays the whole PR's whatever the push was. A full
	// review has no latest push: whatever the model put there is not shown.
	if in.IncrementalFrom != "" && in.ReviewedCommit != "" {
		header += fmt.Sprintf(p("\n- Letzter Push: `%s` .. `%s`", "\n- Last push: `%s` .. `%s`"),
			shortSHA(in.IncrementalFrom, 10), shortSHA(in.ReviewedCommit, 10))
		// One line: a wrapped one would break out of the list item.
		if push := strings.Join(strings.Fields(in.Summary.LatestPush), " "); push != "" {
			header += " — " + push
		}
	}
	sections = append(sections, header)

	sections = append(sections, renderOverview(in))
	sections = append(sections, renderStrengths(in))
	sections = append(sections, renderIssues(in))

	// The italic fallbacks mean the model omitted the field, which is a
	// defect worth seeing; the plain sentinels mean it looked and found
	// nothing. Two different states, both rendered.
	security := strings.TrimSpace(in.Summary.SecurityPerformance)
	if isNoneSentinel(security) {
		security = ""
	}
	coverage := strings.TrimSpace(in.Summary.TestCoverage)
	if strings.EqualFold(coverage, coverageNotAssessable) {
		coverage = p("Testabdeckung allein aus dem Diff nicht beurteilbar.", coverageNotAssessable)
	}
	sections = append(sections, p("### Sicherheit und Performance\n", "### Security and performance\n")+
		orDefault(WrapProse(security), p("Nichts Auffälliges.", "None notable.")))
	sections = append(sections, p("### Testabdeckung\n", "### Test coverage\n")+
		orDefault(WrapProse(coverage), p("_Nicht beurteilt._", "_Not assessed._")))

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
	s.LatestPush = SwissSS(s.LatestPush)
	s.SecurityPerformance = SwissSS(s.SecurityPerformance)
	s.TestCoverage = SwissSS(s.TestCoverage)
	s.VerdictRationale = SwissSS(s.VerdictRationale)
	s.Strengths = make([]string, len(in.Summary.Strengths))
	for i, v := range in.Summary.Strengths {
		s.Strengths[i] = SwissSS(v)
	}
	in.Summary = s
	in.Findings = swissFindings(in.Findings)
	in.EarlierOpen = swissFindings(in.EarlierOpen)
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

// swissFindings copies findings with SwissSS applied to their prose.
func swissFindings(in []inference.ReviewFinding) []inference.ReviewFinding {
	out := make([]inference.ReviewFinding, len(in))
	for i, f := range in {
		f.Comment = SwissSS(f.Comment)
		if f.Headline != nil {
			h := SwissSS(*f.Headline)
			f.Headline = &h
		}
		out[i] = f
	}
	return out
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
	var b strings.Builder
	b.WriteString(heading)
	for _, s := range in.Summary.Strengths {
		if !isNoneSentinel(s) {
			b.WriteString("\n- " + s)
		}
	}
	if b.Len() == len(heading) {
		return heading + p("\nKeine.", "\nNone.")
	}
	return b.String()
}

// The prompt quotes its sentinels in English and asks for them verbatim, so
// a German review would post them in English. The words are the code's: a
// field that is only a sentinel renders as the localized one.
const coverageNotAssessable = "Test coverage not assessable from diff alone."

func isNoneSentinel(s string) bool {
	switch strings.ToLower(strings.TrimSuffix(strings.TrimSpace(s), ".")) {
	case "none", "none notable":
		return true
	}
	return false
}

// renderAccounting is the line under the issues that says what the list
// leaves out: earlier findings this push fixed and dropped ones. Empty when
// there is nothing to say, so a first review reads as before.
func renderAccounting(in SummaryInput) string {
	p := in.Lang.Pick
	var parts []string
	if in.EarlierUnknown {
		parts = append(parts, p("Frühere Findings nicht lesbar: die Liste zeigt nur diesen Lauf.",
			"Earlier findings unreadable: the list shows this run's only."))
	}
	if in.FixedEarlier > 0 {
		parts = append(parts, PluralIn(in.Lang, in.FixedEarlier, "früheres Finding", "frühere Findings",
			"earlier finding", "earlier findings")+p(" mit diesem Push behoben.", " fixed by this push."))
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

// listedFinding is one line of the PR summary's issue list.
type listedFinding struct {
	inference.ReviewFinding
	isNew bool
}

// maxListedFindings caps the PR summary's issue list (user, 2026-10-28);
// the rest are counted under it, issues first in either part.
const maxListedFindings = 30

// prFindings is every open finding of the PR, issues first; within a
// severity the earlier ones come first, in the order they were posted.
func prFindings(in SummaryInput) []listedFinding {
	all := make([]listedFinding, 0, len(in.EarlierOpen)+len(in.Findings))
	for _, f := range in.EarlierOpen {
		all = append(all, listedFinding{ReviewFinding: f})
	}
	for _, f := range in.Findings {
		all = append(all, listedFinding{ReviewFinding: f, isNew: true})
	}
	slices.SortStableFunc(all, func(a, b listedFinding) int {
		return inference.SeverityRank(a.Severity) - inference.SeverityRank(b.Severity)
	})
	return all
}

// renderIssues lists every open finding of the PR as a numbered headline;
// the detail lives on the inline comments. This run's are marked new when
// earlier ones are listed beside them.
func renderIssues(in SummaryInput) string {
	p := in.Lang.Pick
	lines := []string{p("### Probleme und Vorschläge", "### Issues and suggestions")}
	accounting := renderAccounting(in)
	all := prFindings(in)
	if len(all) == 0 && in.Cut == 0 {
		none := p("\nKeine.", "\nNone.")
		if accounting != "" {
			return lines[0] + none + "\n\n" + accounting
		}
		return lines[0] + none
	}

	securityCount := 0
	for _, f := range all {
		if IsSecurityFinding(f.Comment) {
			securityCount++
		}
	}
	if securityCount > 0 {
		lines = append(lines, fmt.Sprintf(p("- %s 🔒 — sorgfältig prüfen", "- %s 🔒 — review carefully"),
			PluralIn(in.Lang, securityCount, "mögliches Sicherheitsproblem", "mögliche Sicherheitsprobleme",
				"potential security issue", "potential security issues")))
	}
	if in.Cut > 0 {
		// Says what was not posted, not what is listed: the list has its own
		// cap. The verdict counts the cut ones, so the summary names them.
		lines = append(lines, "- "+PluralIn(in.Lang, in.Cut,
			"weiteres neues Finding", "weitere neue Findings", "more new finding", "more new findings")+
			p(" über dem Kommentar-Limit nicht gepostet.", " over the comment cap not posted."))
	}
	if securityCount > 0 || in.Cut > 0 {
		lines = append(lines, "")
	}
	markNew := len(in.EarlierOpen) > 0
	shown := all[:min(len(all), maxListedFindings)]
	for i, f := range shown {
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
		if markNew && f.isNew {
			headline += p(" _(neu)_", " _(new)_")
		}
		lines = append(lines, fmt.Sprintf("%d. %s", i+1, headline))
	}
	if more := len(all) - len(shown); more > 0 {
		lines = append(lines, p("- … und ", "- … and ")+PluralIn(in.Lang, more,
			"weiteres offenes Finding.", "weitere offene Findings.", "more open finding.", "more open findings."))
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
// author can act on, then one telemetry line ending in the PR's dashboard
// page. Everything else (tokens, prompt parts, cross-file symbols, key
// spend) lives on that page and its run pages, not in the PR (decided).
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
// elapsed, cost, and the link to the PR's page.
func telemetryLine(in SummaryInput) string {
	p := in.Lang.Pick
	var parts []string
	if in.ModelLabel != "" {
		parts = append(parts, "`"+in.ModelLabel+"`")
	}
	// Every reviewable file, over the same set as the +/- lines. How many of
	// them were not reviewed is counted from the warning lines above (size
	// gate and token budget), so the two never disagree.
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
	if in.PRURL != "" {
		parts = append(parts, fmt.Sprintf(p("[Details zum Review](%s)", "[Review details](%s)"), in.PRURL))
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
