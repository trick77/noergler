package review

import (
	"context"
	"strings"
	"testing"
	"time"

	"github.com/trick77/noergler/internal/bitbucket"
	"github.com/trick77/noergler/internal/inference"
	"github.com/trick77/noergler/internal/lang"
	"github.com/trick77/noergler/internal/prompts"
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

// A review asked for at the cap is a skipped attempt on the dashboard, like a
// blocked push, and a banner from a refused reply gains the "not reviewed"
// sentence it did not have.
func TestCapRefusedMentionReviewIsRecorded(t *testing.T) {
	h := newHarness(t, nil)
	h.st.prCost = ptri(6_000_000_000)
	body := render.CostLimitBanner(lang.EN, 6, 5, false) + "\n\nsummary"
	h.st.summary = &store.SummaryComment{ID: 900, Version: 1}
	h.bb.comments = map[int]*bitbucket.Comment{900: {ID: 900, Version: 1, Text: body}}

	h.r.HandleMention(context.Background(), mentionPayload("@noergler review", "alice"), "platform", nil)

	if len(h.st.Attempts) != 1 || h.st.Attempts[0].Reason != string(SkipCostCap) || h.st.Attempts[0].Kind != store.RunMention {
		t.Errorf("attempts = %+v, want one skipped mention attempt", h.st.Attempts)
	}
	if len(h.bb.Updates) != 1 || !render.CostBannerSaysBlocked(h.bb.Updates[0].Text) {
		t.Errorf("updates = %+v, want the banner to say the review was not run", h.bb.Updates)
	}
}

// A mention replying to someone's comment carries that comment, its author
// and where it sits, so "answer Peter's question" has the question; the
// answer then names no one.
func TestMentionCarriesTheRepliedToComment(t *testing.T) {
	h := newHarness(t, nil)
	h.r.mentionTmpl = prompts.Mention
	h.r.displayName = defaultNameRE
	h.llm.mention = inference.MentionResult{Outcome: inference.OutcomeOK, Answer: "Three retries cap the backoff."}
	parent := &bitbucket.Comment{ID: 41, Text: "Why is the retry loop capped at 3?",
		Anchor: &bitbucket.CommentAnchor{Path: "a.go", Line: 2}}
	parent.Author.Name, parent.Author.DisplayName = "pmuster", "Muster Peter (PMU)"
	h.bb.comments[41] = parent
	p := mentionPayload("@noergler can you answer Peter's question?", "jsaner")
	p.Comment.Author.DisplayName = "Saner Jan (SJH)"
	p.CommentParentID = 41

	h.r.HandleMention(context.Background(), p, "platform", nil)

	if len(h.llm.Mentions) != 1 {
		t.Fatalf("expected one mention call, got %d", len(h.llm.Mentions))
	}
	prompt := h.llm.Mentions[0].Prompt
	for _, want := range []string{"Author: Muster Peter (PMU)", "Location: a.go:2", "Why is the retry loop capped at 3?",
		"## The developer's first name\n\n_Unknown._"} {
		if !strings.Contains(prompt, want) {
			t.Errorf("prompt lacks %q:\n%s", want, prompt)
		}
	}
	if len(h.bb.Replies) != 1 || h.bb.Replies[0].Line != 77 {
		t.Fatalf("replies = %+v, want one under the mention", h.bb.Replies)
	}
}

// Under one of the bot's finding comments the finding is the context, and
// the asker is still named.
func TestMentionUnderABotCommentKeepsTheAskersName(t *testing.T) {
	h := newHarness(t, nil)
	h.r.mentionTmpl = prompts.Mention
	h.r.displayName = defaultNameRE
	h.llm.mention = inference.MentionResult{Outcome: inference.OutcomeOK, Answer: "ok"}
	parent := &bitbucket.Comment{ID: 41, Text: "**Bug**: nil map write"}
	parent.Author.Name = "noergler"
	h.bb.comments[41] = parent
	p := mentionPayload("@noergler is this right?", "jsaner")
	p.Comment.Author.DisplayName = "Saner Jan (SJH)"
	p.CommentParentID = 41

	h.r.HandleMention(context.Background(), p, "platform", nil)

	prompt := h.llm.Mentions[0].Prompt
	if !strings.Contains(prompt, "Author: noergler (this bot)") || !strings.Contains(prompt, "nil map write") {
		t.Errorf("prompt lacks the finding:\n%s", prompt)
	}
	if !strings.Contains(prompt, "## The developer's first name\n\nJan\n") {
		t.Errorf("prompt lacks the asker's name:\n%s", prompt)
	}
}

// No parent to show: top-level, under the summary, or unreadable. The Q&A
// runs as before.
func TestMentionWithoutARepliedToComment(t *testing.T) {
	for name, c := range map[string]struct {
		parent  int
		summary bool
		fetched bool
	}{
		"top-level":   {parent: 0},
		"summary":     {parent: 555, summary: true},
		"fetch fails": {parent: 41, fetched: true},
	} {
		t.Run(name, func(t *testing.T) {
			h := newHarness(t, nil)
			h.r.mentionTmpl = prompts.Mention
			if c.summary {
				h.st.skipState = &store.SkipState{Summary: &store.SummaryComment{ID: 555, Version: 1}}
			}
			h.llm.mention = inference.MentionResult{Outcome: inference.OutcomeOK, Answer: "ok"}
			p := mentionPayload("@noergler why?", "alice")
			p.CommentParentID = c.parent

			h.r.HandleMention(context.Background(), p, "platform", nil)

			if len(h.llm.Mentions) != 1 || !strings.Contains(h.llm.Mentions[0].Prompt, inference.NoParentComment) {
				t.Fatalf("want one answer without a parent, got %+v", h.llm.Mentions)
			}
			if got := len(h.bb.FetchedC) > 0; got != c.fetched {
				t.Errorf("fetched = %v (%v), want %v", got, h.bb.FetchedC, c.fetched)
			}
		})
	}
}

func TestMentionCapsALongRepliedToComment(t *testing.T) {
	h := newHarness(t, nil)
	h.r.mentionTmpl = prompts.Mention
	h.llm.mention = inference.MentionResult{Outcome: inference.OutcomeOK, Answer: "ok"}
	parent := &bitbucket.Comment{ID: 41, Text: strings.Repeat("ü", maxParentRunes+10)}
	parent.Author.Name = "pmuster"
	h.bb.comments[41] = parent
	p := mentionPayload("@noergler answer that", "alice")
	p.CommentParentID = 41

	h.r.HandleMention(context.Background(), p, "platform", nil)

	prompt := h.llm.Mentions[0].Prompt
	if strings.Contains(prompt, strings.Repeat("ü", maxParentRunes+1)) || !strings.Contains(prompt, "[... cut]") {
		t.Error("the parent comment was not capped")
	}
}

// Under the asker's own comment the asker is the one answered: named.
func TestMentionUnderTheAskersOwnCommentKeepsTheName(t *testing.T) {
	h := newHarness(t, nil)
	h.r.mentionTmpl = prompts.Mention
	h.r.displayName = defaultNameRE
	h.llm.mention = inference.MentionResult{Outcome: inference.OutcomeOK, Answer: "ok"}
	parent := &bitbucket.Comment{ID: 41, Text: "Why is the retry loop capped at 3?"}
	parent.Author.Name, parent.Author.DisplayName = "JSaner", "Saner Jan (SJH)"
	h.bb.comments[41] = parent
	p := mentionPayload("@noergler see my question above", "jsaner")
	p.Comment.Author.DisplayName = "Saner Jan (SJH)"
	p.CommentParentID = 41

	h.r.HandleMention(context.Background(), p, "platform", nil)

	if prompt := h.llm.Mentions[0].Prompt; !strings.Contains(prompt, "## The developer's first name\n\nJan\n") {
		t.Errorf("prompt lacks the asker's name:\n%s", prompt)
	}
}

// A comment on the old side carries an old-file line number, which the
// diff's new-file gutter would misplace: it says which side.
func TestMentionParentOnTheOldSideSaysSo(t *testing.T) {
	h := newHarness(t, nil)
	h.r.mentionTmpl = prompts.Mention
	h.llm.mention = inference.MentionResult{Outcome: inference.OutcomeOK, Answer: "ok"}
	parent := &bitbucket.Comment{ID: 41, Text: "why was this removed?",
		Anchor: &bitbucket.CommentAnchor{Path: "a.go", Line: 7, FileType: "FROM"}}
	parent.Author.Name = "pmuster"
	h.bb.comments[41] = parent
	p := mentionPayload("@noergler answer that", "alice")
	p.CommentParentID = 41

	h.r.HandleMention(context.Background(), p, "platform", nil)

	prompt := h.llm.Mentions[0].Prompt
	if !strings.Contains(prompt, "Location: a.go, line 7 of the old version") || strings.Contains(prompt, "a.go:7") {
		t.Errorf("prompt misplaces the old-side line:\n%s", prompt)
	}
}

// A cut inside a code block closes it, so the question below is not code.
func TestMentionCapClosesAnOpenFence(t *testing.T) {
	h := newHarness(t, nil)
	h.r.mentionTmpl = prompts.Mention
	h.llm.mention = inference.MentionResult{Outcome: inference.OutcomeOK, Answer: "ok"}
	parent := &bitbucket.Comment{ID: 41, Text: "see log:\n```\n" + strings.Repeat("x", maxParentRunes) + "\n```"}
	parent.Author.Name = "pmuster"
	h.bb.comments[41] = parent
	p := mentionPayload("@noergler answer that", "alice")
	p.CommentParentID = 41

	h.r.HandleMention(context.Background(), p, "platform", nil)

	if prompt := h.llm.Mentions[0].Prompt; !strings.Contains(prompt, "x\n```\n[... cut]") {
		t.Errorf("the cut left the fence open:\n%s", prompt)
	}
}
