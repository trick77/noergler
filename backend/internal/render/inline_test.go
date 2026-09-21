package render

import (
	"encoding/json"
	"os"
	"strings"
	"testing"

	"github.com/trick77/noergler/internal/inference"
)

// Expected bodies captured from the Python (app/bitbucket.py
// post_inline_comment text assembly) with the venv.
func TestInlineCommentMatchesPython(t *testing.T) {
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
		if got := InlineComment(f); got != c.Out {
			t.Errorf("case %d\n got:  %q\n want: %q", i, got, c.Out)
		}
	}
}

// A single-line suggestion gets the ```suggestion fence so Bitbucket renders
// the "Apply suggestion" button; a multi-line one must not, or the button
// would replace only one line.
func TestSuggestionFenceOnlyForSingleLine(t *testing.T) {
	one := "foo := bar()"
	many := "if x {\n\treturn y\n}"

	got := InlineComment(inference.ReviewFinding{Severity: "issue", Comment: "c", Suggestion: &one})
	if !strings.Contains(got, "```suggestion\n") {
		t.Errorf("single-line suggestion lost its fence: %q", got)
	}

	got = InlineComment(inference.ReviewFinding{Severity: "issue", Comment: "c", Suggestion: &many})
	if strings.Contains(got, "```suggestion") {
		t.Errorf("multi-line suggestion must not use the suggestion fence: %q", got)
	}
}

// An empty-string suggestion is absent, matching Python's falsy check, so no
// empty code block is appended.
func TestEmptySuggestionOmitsTheBlock(t *testing.T) {
	empty := ""
	got := InlineComment(inference.ReviewFinding{Severity: "issue", Comment: "c", Suggestion: &empty})
	if got != "**Issue:** c" {
		t.Errorf("empty suggestion rendered a block: %q", got)
	}
	got = InlineComment(inference.ReviewFinding{Severity: "issue", Comment: "c"})
	if got != "**Issue:** c" {
		t.Errorf("nil suggestion rendered a block: %q", got)
	}
}
