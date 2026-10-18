package review

import (
	"context"
	"strings"
	"testing"
	"time"

	"github.com/trick77/noergler/internal/bitbucket"
	"github.com/trick77/noergler/internal/inference"
	"github.com/trick77/noergler/internal/lang"
	"github.com/trick77/noergler/internal/render"
	"github.com/trick77/noergler/internal/store"
	"github.com/trick77/noergler/internal/webhook"
)

func mentionPayload(text, author string) *webhook.Payload {
	p := prPayload(webhook.EventCommentAdded)
	p.Comment = &webhook.Comment{ID: 77, Text: text, Author: webhook.User{Name: author}}
	return p
}

func TestMentionIgnoresOwnCommentAndNonOpenPR(t *testing.T) {
	t.Run("the bot's own comment", func(t *testing.T) {
		h := newHarness(t, nil)
		h.r.HandleMention(context.Background(), mentionPayload("@noergler review", "noergler"), "platform", nil)
		if len(h.llm.Reviews)+len(h.llm.Mentions) != 0 {
			t.Error("the bot must not answer itself")
		}
	})

	// The gate matches the trigger case-insensitively, so the self check must
	// too: otherwise a BITBUCKET_USERNAME cased differently from Bitbucket's
	// name turns every bot notice (they all mention the bot) into a paid call.
	t.Run("the bot's own comment, cased differently", func(t *testing.T) {
		h := newHarness(t, nil)
		h.r.HandleMention(context.Background(), mentionPayload("@noergler review", "Noergler"), "platform", nil)
		if len(h.llm.Reviews)+len(h.llm.Mentions) != 0 {
			t.Error("the bot must not answer itself")
		}
	})

	t.Run("a non-open PR", func(t *testing.T) {
		h := newHarness(t, nil)
		p := mentionPayload("@noergler what is this?", "alice")
		p.PullRequest.State = "MERGED"
		h.r.HandleMention(context.Background(), p, "platform", nil)
		if len(h.llm.Reviews)+len(h.llm.Mentions) != 0 {
			t.Error("a mention on a non-open PR must be ignored")
		}
	})

	t.Run("an absent state is treated as open", func(t *testing.T) {
		h := newHarness(t, nil)
		p := mentionPayload("@noergler what is this?", "alice")
		p.PullRequest.State = ""
		h.r.HandleMention(context.Background(), p, "platform", nil)
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
			h.r.HandleMention(context.Background(), mentionPayload(c.text, "alice"), "platform", nil)

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
// skipAuthorCheck flag.
func TestMentionReviewBypassesAuthorGates(t *testing.T) {
	h := newHarness(t, nil)
	h.r.SetAuthorLists([]string{"nobody"}, nil)

	h.r.HandleMention(context.Background(), mentionPayload("@noergler review", "alice"), "platform", nil)

	if len(h.llm.Reviews) != 1 {
		t.Error("a mention must review even when the author is not in the allow list")
	}
}

func TestMentionReactivatesIgnoredPR(t *testing.T) {
	h := newHarness(t, nil)
	now := time.Now()
	h.st.skipState = &store.SkipState{IgnoredAt: &now}

	h.r.HandleMention(context.Background(), mentionPayload("@noergler what is this?", "alice"), "platform", nil)

	if h.st.Reactived != 1 {
		t.Errorf("Reactivate called %d times, want 1", h.st.Reactived)
	}
}

func TestMentionQAPostsAThreadedReply(t *testing.T) {
	h := newHarness(t, nil)
	h.llm.mention = inference.MentionResult{Outcome: inference.OutcomeOK, Answer: "It locks the map."}

	h.r.HandleMention(context.Background(), mentionPayload("@noergler is the lock held?", "alice"), "platform", nil)

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

			h.r.HandleMention(context.Background(), mentionPayload("@noergler explain this", "alice"), "platform", nil)

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
		h.r.HandleMention(context.Background(), mentionPayload("@noergler explain this", "alice"), "platform", nil)
		if len(h.bb.Replies) != 0 {
			t.Errorf("an errored mention must not reply, got %q", h.bb.Replies[0].Text)
		}
	})
}

func TestMentionQAWithOversizedDiffReplies(t *testing.T) {
	h := newHarness(t, nil)
	h.bb.prDiffErr = errDiffTooLarge

	h.r.HandleMention(context.Background(), mentionPayload("@noergler explain this", "alice"), "platform", nil)

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

	h.r.HandleMention(context.Background(), mentionPayload("@noergler explain this", "alice"), "platform", nil)

	if len(h.bb.Replies) != 1 || !strings.Contains(h.bb.Replies[0].Text, "No reviewable files") {
		t.Errorf("expected the no-files reply, got %+v", h.bb.Replies)
	}
}

// The mention prompt carries the question and the file group, and a mention
// review is recorded as a mention run rather than an auto one.
func TestMentionReviewIsRecordedAsAMentionRun(t *testing.T) {
	h := newHarness(t, nil)
	h.llm.review = okResultWith()

	h.r.HandleMention(context.Background(), mentionPayload("@noergler review", "alice"), "platform", nil)

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

	h.r.HandleMention(context.Background(), mentionPayload("@noergler why is a.go locked?", "alice"), "platform", nil)

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

// At the cap a question gets no answer and no gateway call: the PR gets the
// cap banner, its one notice (decided: a mention no longer overrides the
// cap).
func TestMentionQAStopsAtTheCap(t *testing.T) {
	h := newHarness(t, nil)
	h.st.prCost = ptri(6_000_000_000)

	h.r.HandleMention(context.Background(), mentionPayload("@noergler is the lock held?", "alice"), "platform", nil)

	if len(h.llm.Mentions) != 0 || len(h.bb.Replies) != 0 {
		t.Errorf("calls = %d, replies = %d: want neither at the cap", len(h.llm.Mentions), len(h.bb.Replies))
	}
	if len(h.bb.Posted) != 1 || !strings.Contains(h.bb.Posted[0].Text, "Cost limit exceeded") {
		t.Errorf("want the cap banner, got %+v", h.bb.Posted)
	}
}

// Every round of a Q&A tool loop is recorded as a question call under the
// comment it answers; the rows sum to the loop's total.
func TestMentionQARecordsEveryRound(t *testing.T) {
	h := newHarness(t, nil)
	a, b := int64(1_000_000), int64(3_000_000)
	h.llm.mention = inference.MentionResult{Outcome: inference.OutcomeOK, Answer: "ok",
		Rounds: []inference.CallCost{{Answered: true, NanoUSD: &a}, {Answered: true, NanoUSD: &b}}}

	h.r.HandleMention(context.Background(), mentionPayload("@noergler why?", "alice"), "platform", nil)

	if len(h.st.CommentCalls) != 2 {
		t.Fatalf("comment calls = %+v, want one per round", h.st.CommentCalls)
	}
	for _, c := range h.st.CommentCalls {
		if c.Kind != store.CallQuestion || c.CommentID != 77 {
			t.Errorf("call = %+v, want a question under comment 77", c)
		}
	}
	if total, _ := h.st.PRCost(context.Background(), store.PRKey{}); total == nil || *total != a+b {
		t.Errorf("PR cost = %v, want the rounds summed", total)
	}
}

// A summary that already carries the cap banner is left alone: no second
// Bitbucket write per refused comment, and a blocked push's "not reviewed"
// sentence stays.
func TestCapBannerIsWrittenOnce(t *testing.T) {
	h := newHarness(t, nil)
	h.st.prCost = ptri(6_000_000_000)
	body := render.CostLimitBanner(lang.EN, 6, 5, true) + "\n\nsummary"
	h.st.summary = &store.SummaryComment{ID: 900, Version: 1}
	h.bb.comments = map[int]*bitbucket.Comment{900: {ID: 900, Version: 1, Text: body}}

	h.r.HandleMention(context.Background(), mentionPayload("@noergler why?", "alice"), "platform", nil)

	if len(h.bb.Updates) != 0 || len(h.bb.Posted) != 0 {
		t.Errorf("updates = %d, posts = %d: the banner was there already", len(h.bb.Updates), len(h.bb.Posted))
	}
}

// An ignored PR (its author removed the summary) gets nothing posted at the
// cap, and stays ignored.
func TestCapOnAnIgnoredPRPostsNothing(t *testing.T) {
	h := newHarness(t, nil)
	h.st.prCost = ptri(6_000_000_000)
	at := time.Now()
	h.st.skipState = &store.SkipState{IgnoredAt: &at}

	h.r.HandleMention(context.Background(), mentionPayload("@noergler review", "alice"), "platform", nil)

	if len(h.bb.Posted) != 0 || len(h.bb.Updates) != 0 || h.st.Reactived != 0 || len(h.llm.Reviews) != 0 {
		t.Errorf("posts = %d, updates = %d, reactivated = %d, reviews = %d; want nothing",
			len(h.bb.Posted), len(h.bb.Updates), h.st.Reactived, len(h.llm.Reviews))
	}
}

// A Q&A on a PR never reviewed writes its row first, keeping any pointer, so
// its calls are counted and capped from the first one on.
func TestMentionQAWritesThePRRow(t *testing.T) {
	h := newHarness(t, nil)
	h.llm.mention = inference.MentionResult{Outcome: inference.OutcomeOK, Answer: "ok"}

	h.r.HandleMention(context.Background(), mentionPayload("@noergler why?", "alice"), "platform", nil)

	if len(h.st.Upserts) != 1 || !h.st.Upserts[0].KeepPointer || h.st.Upserts[0].LastReviewedCommit != nil {
		t.Errorf("upserts = %+v, want one that keeps the pointer", h.st.Upserts)
	}
}
