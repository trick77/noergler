package inference

import (
	"strings"
	"testing"

	"github.com/trick77/noergler/internal/lang"
	"github.com/trick77/noergler/internal/prompts"
)

// Every template carries the block once, as our own text above the
// untrusted section: below it the model is told to treat everything as data.
func TestOutputLanguageSitsAboveTheUntrustedInput(t *testing.T) {
	for name, tmpl := range map[string]string{
		"review": prompts.Review, "mention": prompts.Mention, "finding_reply": prompts.FindingReply,
	} {
		if strings.Count(tmpl, PlaceholderOutputLanguage) != 1 {
			t.Errorf("%s: want exactly one %s", name, PlaceholderOutputLanguage)
			continue
		}
		if strings.Index(tmpl, PlaceholderOutputLanguage) > strings.Index(tmpl, "## Untrusted input") {
			t.Errorf("%s: the language block is below the untrusted input", name)
		}
	}
}

func TestEveryPromptFollowsTheLanguage(t *testing.T) {
	for _, l := range []lang.Lang{lang.DE, lang.EN} {
		block := OutputLanguageBlock(l)
		review := AssembleReviewPrompt(AssembleRequest{Template: prompts.Review, Language: l}, func(string) int { return 0 }).Prompt
		mention := RenderMentionPrompt(MentionPromptRequest{Template: prompts.Mention, Language: l})
		reply := RenderFindingReplyPrompt(FindingReplyPromptRequest{Template: prompts.FindingReply, Language: l})
		for name, p := range map[string]string{"review": review, "mention": mention, "reply": reply} {
			if !strings.Contains(p, block) || strings.Contains(p, PlaceholderOutputLanguage) {
				t.Errorf("%s/%s: block not substituted", name, l)
			}
		}
	}
	if !strings.Contains(OutputLanguageBlock(""), "Swiss Standard German") {
		t.Error("the zero value must write German")
	}
}

// A question or reply containing the placeholder stays literal: one pass.
func TestOutputLanguagePlaceholderInAuthorTextStaysLiteral(t *testing.T) {
	p := RenderMentionPrompt(MentionPromptRequest{Template: prompts.Mention, Question: "what is {output_language}?", Language: lang.EN})
	if !strings.Contains(p, "what is {output_language}?") {
		t.Error("the question's placeholder was expanded")
	}
}
