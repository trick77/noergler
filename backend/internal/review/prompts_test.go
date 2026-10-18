package review

import (
	"testing"

	"github.com/trick77/noergler/internal/prompts"
)

// No Reviewer can send a blank prompt: an empty template option falls back to
// the compiled-in one, and a set one wins.
func TestNewDefaultsTheCompiledInPrompts(t *testing.T) {
	r := New(Options{Log: quietLogger()})
	if r.template != prompts.Review || r.mentionTmpl != prompts.Mention || r.replyTmpl != prompts.FindingReply {
		t.Error("empty template options must fall back to the compiled-in prompts")
	}
	if prompts.Review == "" || prompts.Mention == "" || prompts.FindingReply == "" {
		t.Error("a compiled-in prompt is empty: the embed is broken")
	}
	if got := New(Options{Template: "T", Log: quietLogger()}); got.template != "T" {
		t.Errorf("template = %q, want the option", got.template)
	}
}
