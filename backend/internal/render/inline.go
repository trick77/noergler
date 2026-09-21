package render

import (
	"fmt"
	"strings"

	"github.com/trick77/noergler/internal/inference"
)

// InlineComment builds the body of one inline review comment.
//
// Phase 3's bitbucket.PostInlineComment takes the body as a string and owns
// the anchor, the a/ b/ path strip and the ADDED then CONTEXT retry, so the
// text is all that belongs here.
//
// The severity label is capitalised ("Issue:", "Suggestion:") and the comment
// is wrapped; the suggestion block is appended verbatim, never wrapped, since
// it is code.
func InlineComment(f inference.ReviewFinding) string {
	label := capitalize(f.Severity)
	parts := []string{WrapProse(fmt.Sprintf("**%s:** %s", label, f.Comment))}

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
		parts = append(parts, fmt.Sprintf("**Suggested change:**\n```%s\n%s\n```", fence, body))
	}
	return strings.Join(parts, "\n\n")
}

// capitalize renders the severity enum as a label: upper the first
// character, lower the rest, so "issue" becomes "Issue".
func capitalize(s string) string {
	if s == "" {
		return s
	}
	r := []rune(s)
	return strings.ToUpper(string(r[0])) + strings.ToLower(string(r[1:]))
}
