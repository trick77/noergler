package render

import (
	"encoding/json"
	"os"
	"strings"
	"testing"

	"github.com/trick77/noergler/internal/inference"
	"github.com/trick77/noergler/internal/lang"
)

// Expected bodies are the golden corpus in testdata, compared byte for byte:
// they are posted verbatim on real PRs.
func TestInlineCommentIsPinned(t *testing.T) {
	blob, err := os.ReadFile("testdata/inline_golden.json")
	if err != nil {
		t.Fatalf("read inline golden: %v", err)
	}
	var cases []struct {
		Severity   string  `json:"severity"`
		Comment    string  `json:"comment"`
		Suggestion *string `json:"suggestion"`
		Out        string  `json:"out"`
	}
	if err := json.Unmarshal(blob, &cases); err != nil {
		t.Fatalf("decode inline golden: %v", err)
	}
	for i, c := range cases {
		f := inference.ReviewFinding{
			File:       "a.go",
			Line:       1,
			Severity:   c.Severity,
			Comment:    c.Comment,
			Suggestion: c.Suggestion,
		}
		if got := InlineComment(f, lang.EN); got != c.Out {
			t.Errorf("case %d\n got:  %q\n want: %q", i, got, c.Out)
		}
	}
}

// A blocking rule shows on the finding: a task says what it blocks and how
// it ends, a downgraded one why it is no task; a claim code refused shows
// nothing. The rule is AGENTS.md's text: never SwissSS'd.
func TestInlineCommentBlockingRule(t *testing.T) {
	f := inference.ReviewFinding{Severity: "issue", Comment: "Der Controller ruft SQL auf.",
		BlockingText: "Kein SQL im Controller (Straße).", Blocking: inference.BlockingTask}
	got := InlineComment(f, lang.DE)
	if !strings.Contains(got, "**Blockierende Regel:** Kein SQL im Controller (Straße).") ||
		!strings.Contains(got, "Dieser Task blockiert den Merge") {
		t.Errorf("task:\n%s", got)
	}
	f.Blocking = inference.BlockingCap
	if got := InlineComment(f, lang.EN); !strings.Contains(got, "(no task: this review opened its 3 tasks already)") ||
		strings.Contains(got, "blocks the merge") {
		t.Errorf("capped:\n%s", got)
	}
	for _, refused := range []inference.BlockingOutcome{inference.BlockingConfidence, inference.BlockingSeverity, inference.BlockingUnknownRule} {
		f.Blocking = refused
		if got := InlineComment(f, lang.EN); strings.Contains(got, "Blocking rule") {
			t.Errorf("%s shows the rule:\n%s", refused, got)
		}
	}
}

// A single-line suggestion gets the ```suggestion fence so Bitbucket renders
// the "Apply suggestion" button; a multi-line one must not, or the button
// would replace only one line.
func TestSuggestionFenceOnlyForSingleLine(t *testing.T) {
	one := "foo := bar()"
	many := "if x {\n\treturn y\n}"

	got := InlineComment(inference.ReviewFinding{Severity: "issue", Comment: "c", Suggestion: &one}, lang.EN)
	if !strings.Contains(got, "```suggestion\n") {
		t.Errorf("single-line suggestion lost its fence: %q", got)
	}

	got = InlineComment(inference.ReviewFinding{Severity: "issue", Comment: "c", Suggestion: &many}, lang.EN)
	if strings.Contains(got, "```suggestion") {
		t.Errorf("multi-line suggestion must not use the suggestion fence: %q", got)
	}
}

// An empty-string suggestion counts as absent, so no empty code block is
// appended.
func TestEmptySuggestionOmitsTheBlock(t *testing.T) {
	empty := ""
	got := InlineComment(inference.ReviewFinding{Severity: "issue", Comment: "c", Suggestion: &empty}, lang.EN)
	if got != "**Issue:** c" {
		t.Errorf("empty suggestion rendered a block: %q", got)
	}
	got = InlineComment(inference.ReviewFinding{Severity: "issue", Comment: "c"}, lang.EN)
	if got != "**Issue:** c" {
		t.Errorf("nil suggestion rendered a block: %q", got)
	}
}
