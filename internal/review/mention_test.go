package review

import (
	"context"
	"strings"
	"testing"
	"time"

	"github.com/trick77/noergler-go/internal/inference"
	"github.com/trick77/noergler-go/internal/store"
	"github.com/trick77/noergler-go/internal/webhook"
)

func mentionPayload(text, author string) *webhook.Payload {
	p := prPayload(webhook.EventCommentAdded)
	p.Comment = &webhook.Comment{ID: 77, Text: text, Author: webhook.User{Name: author}}
	return p
}

func TestMentionIgnoresOwnCommentAndNonOpenPR(t *testing.T) {
	t.Run("the bot's own comment", func(t *testing.T) {
		h := newHarness(t, nil)
		h.r.HandleMention(context.Background(), mentionPayload("@noergler review", "noergler"))
		if len(h.llm.Reviews)+len(h.llm.Mentions) != 0 {
			t.Error("the bot must not answer itself")
		}
	})

	t.Run("a non-open PR", func(t *testing.T) {
		h := newHarness(t, nil)
		p := mentionPayload("@noergler what is this?", "alice")
		p.PullRequest.State = "MERGED"
		h.r.HandleMention(context.Background(), p)
		if len(h.llm.Reviews)+len(h.llm.Mentions) != 0 {
			t.Error("a mention on a non-open PR must be ignored")
		}
	})

	t.Run("an absent state is treated as open", func(t *testing.T) {
		h := newHarness(t, nil)
		p := mentionPayload("@noergler what is this?", "alice")
		p.PullRequest.State = ""
		h.r.HandleMention(context.Background(), p)
		if len(h.llm.Mentions) != 1 {
			t.Error("a payload without a state must still be answered")
		}
	})
}

// An empty question or a review keyword re-runs the review; anything else is
// answered as Q&A.
func TestMentionRouting(t *testing.T) {
	cases := []struct {
		text       string
		wantReview bool
	}{
		{"@noergler", true},
		{"@noergler review", true},
		{"@noergler review this", true},
		{"@noergler re-review", true},
		{"@noergler rereview", true},
		{"@noergler REVIEW", true},
		{"@noergler what does this do?", false},
		{"@noergler is the lock held here?", false},
	}
	for _, c := range cases {
		t.Run(c.text, func(t *testing.T) {
			h := newHarness(t, nil)
			h.r.HandleMention(context.Background(), mentionPayload(c.text, "alice"))

			if c.wantReview {
				if len(h.llm.Reviews) != 1 {
					t.Errorf("expected a full review, got %d reviews and %d mentions",
						len(h.llm.Reviews), len(h.llm.Mentions))
				}
				return
			}
			if len(h.llm.Mentions) != 1 {
				t.Errorf("expected a Q&A call, got %d reviews and %d mentions",
					len(h.llm.Reviews), len(h.llm.Mentions))
			}
		})
	}
}

// A mention bypasses the author gates, which is the whole point of the
// skip_author_check flag.
func TestMentionReviewBypassesAuthorGates(t *testing.T) {
	h := newHarness(t, nil)
	h.r.cfg.AutoReviewAuthors = []string{"nobody"}

	h.r.HandleMention(context.Background(), mentionPayload("@noergler review", "alice"))

	if len(h.llm.Reviews) != 1 {
		t.Error("a mention must review even when the author is not in the allow list")
	}
}

func TestMentionReactivatesIgnoredPR(t *testing.T) {
	h := newHarness(t, nil)
	now := time.Now()
	h.st.skipState = &store.SkipState{IgnoredAt: &now}

	h.r.HandleMention(context.Background(), mentionPayload("@noergler what is this?", "alice"))

	if h.st.Reactived != 1 {
		t.Errorf("Reactivate called %d times, want 1", h.st.Reactived)
	}
}

func TestMentionQAPostsAThreadedReply(t *testing.T) {
	h := newHarness(t, nil)
	h.llm.mention = inference.MentionResult{Outcome: inference.OutcomeOK, Answer: "It locks the map."}

	h.r.HandleMention(context.Background(), mentionPayload("@noergler is the lock held?", "alice"))

	if len(h.bb.Replies) != 1 {
		t.Fatalf("expected one reply, got %d", len(h.bb.Replies))
	}
	if h.bb.Replies[0].Text != "It locks the map." {
		t.Errorf("reply = %q", h.bb.Replies[0].Text)
	}
	if h.bb.Replies[0].Line != 77 {
		t.Errorf("reply threaded under comment %d, want 77", h.bb.Replies[0].Line)
	}
	// Q&A writes nothing: no run row, no findings, no summary.
	if len(h.st.Runs) != 0 || len(h.bb.Posted) != 0 || len(h.bb.Inline) != 0 {
		t.Error("Q&A must not write a run row or post comments")
	}
}

func TestMentionQAFailureReplies(t *testing.T) {
	cases := []struct {
		name   string
		result inference.MentionResult
		want   string
	}{
		{"timeout", inference.MentionResult{Outcome: inference.OutcomeTimedOut}, "No response from the model within"},
		{"too large", inference.MentionResult{Outcome: inference.OutcomeTooLarge}, "too large to answer"},
		{"empty answer", inference.MentionResult{Outcome: inference.OutcomeOK, Answer: ""}, "couldn't process this PR"},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			h := newHarness(t, nil)
			h.llm.mention = c.result

			h.r.HandleMention(context.Background(), mentionPayload("@noergler explain this", "alice"))

			if len(h.bb.Replies) != 1 {
				t.Fatalf("expected one reply, got %d", len(h.bb.Replies))
			}
			if !strings.Contains(h.bb.Replies[0].Text, c.want) {
				t.Errorf("reply = %q, want it to mention %q", h.bb.Replies[0].Text, c.want)
			}
		})
	}

	t.Run("a hard error posts nothing", func(t *testing.T) {
		h := newHarness(t, nil)
		h.llm.mention = inference.MentionResult{Outcome: inference.OutcomeError}
		h.r.HandleMention(context.Background(), mentionPayload("@noergler explain this", "alice"))
		if len(h.bb.Replies) != 0 {
			t.Errorf("an errored mention must not reply, got %q", h.bb.Replies[0].Text)
		}
	})
}

func TestMentionQAWithOversizedDiffReplies(t *testing.T) {
	h := newHarness(t, nil)
	h.bb.prDiffErr = errDiffTooLarge

	h.r.HandleMention(context.Background(), mentionPayload("@noergler explain this", "alice"))

	if len(h.bb.Replies) != 1 || !strings.Contains(h.bb.Replies[0].Text, "exceeds 10 MiB") {
		t.Errorf("expected the oversized-diff reply, got %+v", h.bb.Replies)
	}
	if len(h.llm.Mentions) != 0 {
		t.Error("an oversized diff must not reach the model")
	}
}

func TestMentionQAWithNoReviewableFilesReplies(t *testing.T) {
	h := newHarness(t, func(h *harness) {
		h.bb.prDiff = "diff --git a/go.sum b/go.sum\n--- a/go.sum\n+++ b/go.sum\n@@ -1 +1 @@\n-a\n+b\n"
	})

	h.r.HandleMention(context.Background(), mentionPayload("@noergler explain this", "alice"))

	if len(h.bb.Replies) != 1 || !strings.Contains(h.bb.Replies[0].Text, "No reviewable files") {
		t.Errorf("expected the no-files reply, got %+v", h.bb.Replies)
	}
}

// The mention prompt carries the question and the file group, and a mention
// review is recorded as a mention run rather than an auto one.
func TestMentionReviewIsRecordedAsAMentionRun(t *testing.T) {
	h := newHarness(t, nil)
	h.llm.review = okResultWith()

	h.r.HandleMention(context.Background(), mentionPayload("@noergler review", "alice"))

	if len(h.st.Runs) != 1 {
		t.Fatalf("expected one run row, got %d", len(h.st.Runs))
	}
	if h.st.Runs[0].Kind != store.RunMention {
		t.Errorf("run kind = %q, want %q", h.st.Runs[0].Kind, store.RunMention)
	}
}

func TestMentionPromptCarriesTheQuestion(t *testing.T) {
	h := newHarness(t, nil)
	h.llm.mention = inference.MentionResult{Outcome: inference.OutcomeOK, Answer: "answer"}

	h.r.HandleMention(context.Background(), mentionPayload("@noergler why is a.go locked?", "alice"))

	if len(h.llm.Mentions) != 1 {
		t.Fatalf("expected one mention call, got %d", len(h.llm.Mentions))
	}
	prompt := h.llm.Mentions[0].Prompt
	if !strings.Contains(prompt, "why is a.go locked?") {
		t.Errorf("prompt is missing the question: %q", prompt)
	}
	if !strings.Contains(prompt, "a.go") {
		t.Errorf("prompt is missing the file group: %q", prompt)
	}
}
