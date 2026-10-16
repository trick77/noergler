package review

import (
	"bytes"
	"context"
	"log/slog"
	"strings"
	"testing"

	"github.com/trick77/noergler/internal/inference"
	"github.com/trick77/noergler/internal/webhook"
)

func TestRenderPRDescription(t *testing.T) {
	if got := renderPRDescription("T", "  \n "); got != "" {
		t.Errorf("blank description rendered %q, want nothing", got)
	}
	if got, want := renderPRDescription("Add X", " Why.\n"), "**Title:** Add X\n**Description:**\nWhy."; got != want {
		t.Errorf("got %q, want %q", got, want)
	}
	// Runes, not bytes: a byte cut would split the umlaut.
	long := strings.Repeat("ü", maxPRDescriptionRunes+10)
	got := renderPRDescription("T", long)
	if !strings.HasSuffix(got, "ü...") || strings.Count(got, "ü") != maxPRDescriptionRunes {
		t.Errorf("cap not applied in runes: %d ü, suffix %q", strings.Count(got, "ü"), got[len(got)-8:])
	}
}

func describedPayload(event string) *webhook.Payload {
	p := prPayload(event)
	p.PullRequest.Description = "The retry is deliberate: the gateway drops the first call."
	return p
}

// On by default, the description reaches the review prompt; switched off by
// the team, the sentinel does. Both are logged.
func TestReviewPromptCarriesThePRDescription(t *testing.T) {
	t.Run("on", func(t *testing.T) {
		h := newHarness(t, nil)
		var buf bytes.Buffer
		h.r.log = slog.New(slog.NewTextHandler(&buf, nil))
		h.r.ReviewPullRequest(context.Background(), describedPayload(webhook.EventOpened), false)
		if len(h.llm.Reviews) != 1 {
			t.Fatalf("expected one LLM call, got %d", len(h.llm.Reviews))
		}
		prompt := h.llm.Reviews[0].Prompt
		for _, want := range []string{"**Title:** Add new feature", "The retry is deliberate"} {
			if !strings.Contains(prompt, want) {
				t.Errorf("prompt is missing %q", want)
			}
		}
		wantLogged(t, buf.String(), "PR description sent (")
	})

	t.Run("disabled by team", func(t *testing.T) {
		h := newHarness(t, nil)
		h.r.SetUsePRDescription(false)
		var buf bytes.Buffer
		h.r.log = slog.New(slog.NewTextHandler(&buf, nil))
		h.r.ReviewPullRequest(context.Background(), describedPayload(webhook.EventOpened), false)
		prompt := h.llm.Reviews[0].Prompt
		if strings.Contains(prompt, "The retry is deliberate") {
			t.Error("description sent although the team switched it off")
		}
		if !strings.Contains(prompt, inference.NoPRDescription) {
			t.Error("sentinel missing")
		}
		wantLogged(t, buf.String(), "PR description not sent (disabled by team)")
	})

	t.Run("empty", func(t *testing.T) {
		logs := reviewLogs(t, nil, nil)
		wantLogged(t, logs, "PR description not sent (empty)")
	})
}

func TestMentionPromptCarriesThePRDescription(t *testing.T) {
	h := newHarness(t, nil)
	p := mentionPayload("@noergler why the retry?", "alice")
	p.PullRequest.Description = "The retry is deliberate."
	h.r.HandleMention(context.Background(), p, "platform", nil)
	if len(h.llm.Mentions) != 1 {
		t.Fatalf("expected one mention call, got %d", len(h.llm.Mentions))
	}
	if !strings.Contains(h.llm.Mentions[0].Prompt, "The retry is deliberate.") {
		t.Errorf("mention prompt lacks the description:\n%s", h.llm.Mentions[0].Prompt)
	}

	h.r.SetUsePRDescription(false)
	h.r.HandleMention(context.Background(), p, "platform", nil)
	if strings.Contains(h.llm.Mentions[1].Prompt, "The retry is deliberate.") {
		t.Error("mention sent the description although the team switched it off")
	}
}
