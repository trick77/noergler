package render

import (
	"fmt"
	"strings"

	"github.com/trick77/noergler/internal/inference"
	"github.com/trick77/noergler/internal/lang"
)

// InlineComment builds the body of one inline review comment.
//
// Phase 3's bitbucket.PostInlineComment takes the body as a string and owns
// the anchor, the a/ b/ path strip and the ADDED then CONTEXT retry, so the
// text is all that belongs here.
//
// The severity label is capitalised ("Issue:", "Suggestion:") and the comment
// is wrapped; the suggestion block is appended verbatim, never wrapped, since
// it is code. Criterion labels (CriterionLabels) in the comment go in a code
// span, so Bitbucket does not link them as Jira issues.
func InlineComment(f inference.ReviewFinding, l lang.Lang, labels ...string) string {
	label, comment := capitalize(f.Severity), f.Comment
	if l != lang.EN {
		if de, ok := severityDE[f.Severity]; ok {
			label = de
		}
		comment = lang.SwissSS(comment)
	}
	comment = spanLabels(comment, labels)
	parts := []string{WrapProse(fmt.Sprintf("**%s:** %s", label, comment))}
	if rule := blockingLine(f, l); rule != "" {
		parts = append(parts, WrapProse(rule))
	}

	if f.Suggestion != nil && *f.Suggestion != "" {
		body := strings.Trim(*f.Suggestion, "\n")
		// Bitbucket Data Center renders an "Apply suggestion" button only on
		// a single-line ```suggestion fence. A multi-line block stays plain
		// so the reviewer is not offered a button that would replace just
		// one line.
		fence := "suggestion"
		if strings.Contains(body, "\n") {
			fence = ""
		}
		parts = append(parts, fmt.Sprintf(l.Pick("**Vorgeschlagene Änderung:**\n```%s\n%s\n```",
			"**Suggested change:**\n```%s\n%s\n```"), fence, body))
	}
	return strings.Join(parts, "\n\n")
}

// blockingLine names the blocking rule a finding violates: on a task, what
// it blocks and how it ends; on one code left a plain comment past the
// task cap or with tasks off, why it is no task. Nothing for a claim code
// refused (unknown rule, severity, confidence): the model's say-so alone
// is not shown as a rule violation. The rule is AGENTS.md's text, never
// rewritten (no SwissSS: it is not model prose).
func blockingLine(f inference.ReviewFinding, l lang.Lang) string {
	if f.BlockingText == "" {
		return ""
	}
	head := l.Pick("**Blockierende Regel:** ", "**Blocking rule:** ") + f.BlockingText
	switch f.Blocking {
	case inference.BlockingTask:
		return head + "\n\n" + l.Pick(
			"Dieser Task blockiert den Merge, bis er erledigt ist. Behebt ein Push die Verletzung, erledige ich ihn selbst; "+
				"soll der PR trotzdem so gemergt werden, erledige ihn von Hand.",
			"This task blocks the merge until it is resolved. Once a push fixes the violation, I resolve it myself; "+
				"to merge the PR as it is anyway, resolve it yourself.")
	case inference.BlockingCap:
		return head + fmt.Sprintf(l.Pick(" (kein Task: dieses Review hat seine %d Tasks schon geöffnet)",
			" (no task: this review opened its %d tasks already)"), inference.MaxBlockingTasks)
	case inference.BlockingTasksOff:
		return head + l.Pick(" (kein Task: Tasks sind ausgeschaltet)", " (no task: tasks are switched off)")
	case inference.BlockingTaskFailed:
		return head + l.Pick(" (kein Task: Bitbucket hat ihn abgelehnt)", " (no task: Bitbucket refused it)")
	}
	return ""
}

// severityDE labels the severity enum in German; the enum itself stays
// English everywhere else.
var severityDE = map[string]string{"issue": "Problem", "suggestion": "Vorschlag"}

// capitalize renders the severity enum as a label: upper the first
// character, lower the rest, so "issue" becomes "Issue".
func capitalize(s string) string {
	if s == "" {
		return s
	}
	r := []rune(s)
	return strings.ToUpper(string(r[0])) + strings.ToLower(string(r[1:]))
}
