package inference

import (
	"fmt"
	"regexp"
	"strings"
	"unicode"
	"unicode/utf8"

	"github.com/trick77/noergler/internal/lang"
)

// Blocking rules (user, 2026-11-15): a line of the repo's AGENTS.md marked
// `[blocking]` is a rule whose violation becomes a Bitbucket task on the
// finding's line, which blocks the merge where the repo's "incomplete
// tasks" merge check is on. Code decides which rules block, never the
// model: the model only names the rule a finding violates
// (`blocking_rule`), and CheckBlocking accepts that bluntly or downgrades
// the finding to a plain comment. Never dropped: the finding stands either
// way.

const (
	// MaxBlockingRules caps the rules one AGENTS.md may mark; the rest are
	// left plain guidelines (user, 2026-11-15).
	MaxBlockingRules = 10
	// MaxBlockingTasks caps the blocking tasks one review posts; the rest
	// post as plain comments (user, 2026-11-15).
	MaxBlockingTasks = 3
	// BlockingMinConfidence is the confidence a finding needs to block
	// (user, 2026-11-15): the prompt's "exact line identified, failing
	// input describable". The schema's floor is 80, so 80 filters nothing.
	BlockingMinConfidence = 90
	// maxBlockingRuleRunes caps one rule's text: it rides the prompt, the
	// task and the findings row.
	maxBlockingRuleRunes = 300
)

// BlockingRule is one rule AGENTS.md marks `[blocking]`. ID is its place
// among this review's rules (B1, B2, ...), for the model's answer only:
// it shifts when AGENTS.md changes, so a stored finding keeps Text.
type BlockingRule struct {
	ID   string
	Text string
}

// blockingMarker is `[blocking]` at the start of a line, after optional
// indentation and one list marker (`- `, `* `, `+ `, `1. `, `1) `: the
// ones listItem knows), any case. RE2 classes are ASCII: the indentation
// is matched as Unicode space.
var blockingMarker = regexp.MustCompile(`(?i)^[\s\p{Zs}]*(?:(?:[-*+]|\p{Nd}+[.)])[\s\p{Zs}]+)?\[blocking\][\s\p{Zs}]*(.*)$`)

// fenceLine opens or closes a code fence (``` or ~~~): a marked line inside
// one is an example, not a rule.
var fenceLine = regexp.MustCompile("^[\\s\\p{Zs}]*(```|~~~)")

// listItem starts a list item or a heading: it ends a wrapped rule.
var listItem = regexp.MustCompile(`^[\s\p{Zs}]*(?:[-*+][\s\p{Zs}]|\p{Nd}+[.)][\s\p{Zs}]|#)`)

// ExtractBlockingRules reads the marked rules off agentsMD, in file order,
// numbered from B1. A rule wrapped over indented lines is one rule, its
// whitespace folded (NormalizeRule): a blank line, a list item, a heading,
// a fence or a line at the margin ends it. A repeated text counts once; an
// empty one never. over is the texts of the marked rules past
// MaxBlockingRules, left plain: still marked, so their open tasks stay.
func ExtractBlockingRules(agentsMD string) (rules []BlockingRule, over []string) {
	var texts []string
	open := false // the last text may take a continuation line
	fence := ""
	for line := range strings.SplitSeq(agentsMD, "\n") {
		line = strings.TrimRight(line, "\r")
		if m := fenceLine.FindStringSubmatch(line); m != nil {
			switch fence {
			case "":
				fence = m[1]
			case m[1]:
				fence = ""
			}
			open = false
			continue
		}
		if fence != "" {
			continue
		}
		if m := blockingMarker.FindStringSubmatch(line); m != nil {
			texts, open = append(texts, m[1]), true
			continue
		}
		indented := strings.TrimLeftFunc(line, unicode.IsSpace) != line
		if open && indented && strings.TrimSpace(line) != "" && !listItem.MatchString(line) {
			texts[len(texts)-1] += " " + line
			continue
		}
		open = false
	}
	seen := map[string]bool{}
	for _, t := range texts {
		text := capRunes(NormalizeRule(t), maxBlockingRuleRunes)
		if text == "" || seen[text] {
			continue
		}
		seen[text] = true
		if len(rules) == MaxBlockingRules {
			over = append(over, text)
			continue
		}
		rules = append(rules, BlockingRule{ID: fmt.Sprintf("B%d", len(rules)+1), Text: text})
	}
	return rules, over
}

// NormalizeRule is a rule's text as rules are compared: whitespace runs
// folded, case kept. A rule re-wrapped in AGENTS.md is the same rule; a
// reworded one is another.
func NormalizeRule(text string) string {
	return strings.Join(strings.Fields(text), " ")
}

// capRunes cuts s to n runes, the last an ellipsis.
func capRunes(s string, n int) string {
	if utf8.RuneCountInString(s) <= n {
		return s
	}
	return strings.TrimSpace(lang.TruncateRunes(s, n-1)) + "…"
}

// RenderBlockingRules is the {blocking_rules} block: the rules with their
// ids, "" when there are none (no block, no instruction).
func RenderBlockingRules(rules []BlockingRule) string {
	if len(rules) == 0 {
		return ""
	}
	var b strings.Builder
	b.WriteString("### Blocking rules\n\n")
	b.WriteString("The project marks these guidelines as blocking. A finding that violates one of them names its id in `blocking_rule`:\n\n")
	for _, r := range rules {
		fmt.Fprintf(&b, "- %s: %s\n", r.ID, r.Text)
	}
	return strings.TrimRight(b.String(), "\n")
}

// BlockingOutcome is what became of a finding the model marked with
// `blocking_rule`. The values are log and findings-row vocabulary
// (findings.blocking_outcome): do not reword them.
type BlockingOutcome string

const (
	// BlockingTask is posted as a Bitbucket task.
	BlockingTask BlockingOutcome = "task"
	// BlockingUnknownRule means the id names no rule of this review.
	BlockingUnknownRule BlockingOutcome = "downgraded_unknown_rule"
	// BlockingSeverity means a `suggestion`, which never blocks a merge.
	BlockingSeverity BlockingOutcome = "downgraded_severity"
	// BlockingConfidence means no confidence, or under
	// BlockingMinConfidence.
	BlockingConfidence BlockingOutcome = "downgraded_confidence"
	// BlockingCap means past MaxBlockingTasks in this review.
	BlockingCap BlockingOutcome = "downgraded_cap"
	// BlockingTasksOff means the instance's or the team's task switch is
	// off.
	BlockingTasksOff BlockingOutcome = "downgraded_tasks_off"
	// BlockingTaskFailed means Bitbucket refused the task, so the finding
	// went up as a plain comment rather than not at all.
	BlockingTaskFailed BlockingOutcome = "downgraded_task_failed"
	// BlockingRuleRemoved marks a task noergler resolved because its rule
	// left AGENTS.md (or was reworded): stored on the row afterwards, never
	// a claim's outcome. Such a finding holds back no new task.
	BlockingRuleRemoved BlockingOutcome = "rule_removed"
)

// Marked reports whether the model named a blocking rule on f.
func (f ReviewFinding) Marked() bool {
	return f.BlockingRule != nil && strings.TrimSpace(*f.BlockingRule) != ""
}

// IsTask reports whether f is (to be) posted as a blocking task.
func (f ReviewFinding) IsTask() bool { return f.Blocking == BlockingTask }

// CompareFindings orders findings by severity, a blocking task first
// within one: the comment cap and the summary's list cap must never cut a
// task for a plain finding. For a stable sort.
func CompareFindings(a, b ReviewFinding) int {
	if d := SeverityRank(a.Severity) - SeverityRank(b.Severity); d != 0 {
		return d
	}
	return taskRank(a) - taskRank(b)
}

func taskRank(f ReviewFinding) int {
	if f.IsTask() {
		return 0
	}
	return 1
}

// CheckBlocking is the blunt check of one marked finding: the rule must be
// one of rules, the severity `issue`, the confidence at least
// BlockingMinConfidence. ok false = downgraded, for the reason given. The
// rule is returned whenever its id is known, ok or not. The caps and the
// switches are the caller's.
func CheckBlocking(f ReviewFinding, rules []BlockingRule) (rule *BlockingRule, outcome BlockingOutcome, ok bool) {
	id := strings.ToUpper(strings.TrimSpace(*f.BlockingRule))
	for i := range rules {
		if rules[i].ID == id {
			rule = &rules[i]
			break
		}
	}
	switch {
	case rule == nil:
		return nil, BlockingUnknownRule, false
	case f.Severity != "issue":
		return rule, BlockingSeverity, false
	case f.Confidence == nil || *f.Confidence < BlockingMinConfidence:
		return rule, BlockingConfidence, false
	}
	return rule, BlockingTask, true
}
