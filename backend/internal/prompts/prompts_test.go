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

// Dispute signals are consulted on a reply to a finding only: the
// finding-reply prompt carries them once, as untrusted input, and no other
// prompt carries them at all.
func TestDisputeSignalsOnlyInTheFindingReply(t *testing.T) {
	const ph = "{dispute_signals}"
	if strings.Count(FindingReply, ph) != 1 {
		t.Errorf("finding_reply carries %s %d times, want once", ph, strings.Count(FindingReply, ph))
	}
	if u := strings.Index(FindingReply, "## Untrusted input"); u < 0 || strings.Index(FindingReply, ph) < u {
		t.Errorf("%s is not below the untrusted marker", ph)
	}
	for name, text := range map[string]string{"review": Review, "mention": Mention, "almanach note": AlmanachNote} {
		if strings.Contains(text, ph) {
			t.Errorf("%s prompt carries %s", name, ph)
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
