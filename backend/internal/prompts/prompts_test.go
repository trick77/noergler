package prompts

import (
	"strings"
	"testing"
)

// The embed works and each prompt still carries its placeholders.
func TestPromptsAreEmbedded(t *testing.T) {
	for name, c := range map[string]struct{ text, placeholder string }{
		"review":        {Review, "{files}"},
		"mention":       {Mention, "{question}"},
		"finding_reply": {FindingReply, "{reply}"},
		"reply name":    {FindingReply, "{author_first_name}"},
	} {
		if !strings.Contains(c.text, c.placeholder) {
			t.Errorf("%s prompt lacks %s: the embed is broken or the prompt lost it", name, c.placeholder)
		}
	}
}
