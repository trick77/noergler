package prompts

import (
	"strings"
	"testing"
)

// The embed works and each prompt still carries its placeholders.
func TestPromptsAreEmbedded(t *testing.T) {
	for name, c := range map[string]struct{ text, placeholder string }{
		"review":          {Review, "{files}"},
		"review overview": {Review, "{previous_overview}"},
		"mention":         {Mention, "{question}"},
		"mention parent":  {Mention, "{parent_comment}"},
		"finding_reply":   {FindingReply, "{reply}"},
		"reply name":      {FindingReply, "{author_first_name}"},
		"almanach note":   {AlmanachNote, "{note_limit}"},
		"almanach repo":   {AlmanachNote, "{repo}"},
	} {
		if !strings.Contains(c.text, c.placeholder) {
			t.Errorf("%s prompt lacks %s: the embed is broken or the prompt lost it", name, c.placeholder)
		}
	}
}

// The review prompt's tail goes from stable to per-push, for the prefix
// cache: the previous overview changes on every push, so it comes last.
func TestReviewPlaceholderOrder(t *testing.T) {
	order := []string{"{files}", "{cumulative_pr_diff}", "{previously_posted_findings}", "{previous_overview}"}
	last := -1
	for _, ph := range order {
		i := strings.Index(Review, ph)
		if i <= last {
			t.Fatalf("%s at %d, want after %d: order is %v", ph, i, last, order)
		}
		last = i
	}
}
