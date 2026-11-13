package review

import (
	"context"
	"strings"
	"testing"

	"github.com/trick77/noergler/internal/bitbucket"
	"github.com/trick77/noergler/internal/inference"
	"github.com/trick77/noergler/internal/lang"
	"github.com/trick77/noergler/internal/prompts"
	"github.com/trick77/noergler/internal/store"
	"github.com/trick77/noergler/internal/webhook"
)

func anchored(id int, author, text, path string, line int) *bitbucket.Comment {
	c := &bitbucket.Comment{ID: id, Text: text, Anchor: &bitbucket.CommentAnchor{Path: path, Line: line, FileType: "TO"}}
	c.Author.Name = author
	return c
}

// A question on a line of code keeps that line with noergler's answer, the
// cheapest way: its own (one fetch, a root), its parent's, or the answer
// it replies to.
func TestQAAnswerKeepsTheQuestionsLine(t *testing.T) {
	want := &store.BotReplyAnchor{Path: "a.go", Line: 2, Commit: "src1"}
	for name, setup := range map[string]func(*harness, *webhook.Payload){
		"root question": func(h *harness, _ *webhook.Payload) {
			h.bb.comments[77] = anchored(77, "alice", "@noergler thread-safe?", "a.go", 2)
		},
		"reply to an inline root": func(h *harness, p *webhook.Payload) {
			h.bb.comments[41] = anchored(41, "bob", "is this right?", "a.go", 2)
			p.CommentParentID = 41
		},
		"under noergler's answer": func(h *harness, p *webhook.Payload) {
			bot := &bitbucket.Comment{ID: 950, Text: "Use a mutex."}
			bot.Author.Name = "noergler"
			h.bb.comments[950] = bot
			h.st.botReplies = map[int]*store.BotReply{950: {CommentID: 950, Kind: store.BotReplyQuestion,
				Anchor: &store.BotReplyAnchor{Path: "a.go", Line: 2, Commit: "src1"}}}
			p.CommentParentID = 950
		},
	} {
		t.Run(name, func(t *testing.T) {
			h := newHarness(t, nil)
			h.r.mentionTmpl = prompts.Mention
			h.llm.mention = inference.MentionResult{Outcome: inference.OutcomeOK, Answer: "Yes."}
			p := mentionPayload("@noergler why?", "alice")
			setup(h, p)

			h.r.HandleMention(context.Background(), p, "platform", &inlineScheduler{})

			last := h.st.BotReplies[len(h.st.BotReplies)-1]
			if last.Anchor == nil || *last.Anchor != *want {
				t.Errorf("answer kept %+v, want %+v", last.Anchor, want)
			}
		})
	}
	t.Run("old side or general", func(t *testing.T) {
		h := newHarness(t, nil)
		h.r.mentionTmpl = prompts.Mention
		h.llm.mention = inference.MentionResult{Outcome: inference.OutcomeOK, Answer: "Yes."}
		c := anchored(77, "alice", "q", "a.go", 2)
		c.Anchor.FileType = "FROM"
		h.bb.comments[77] = c

		h.r.HandleMention(context.Background(), mentionPayload("@noergler why?", "alice"), "platform", &inlineScheduler{})

		if a := h.st.BotReplies[len(h.st.BotReplies)-1].Anchor; a != nil {
			t.Errorf("an old-side line kept %+v", a)
		}
	})
}

// answerHarness: alice, the PR author, replies "setz das um" under
// noergler's Q&A answer 950 on a.go:2, and the model reads it as `apply`.
func answerHarness(t *testing.T) (*harness, *webhook.Payload) {
	t.Helper()
	h := newHarness(t, nil)
	h.r.mentionTmpl = prompts.Mention
	h.st.languages = map[string]lang.Lang{"alice": lang.EN}
	bot := &bitbucket.Comment{ID: 950, Text: "Validate the input in new()."}
	bot.Author.Name = "noergler"
	h.bb.comments[950] = bot
	h.st.botReplies = map[int]*store.BotReply{950: {CommentID: 950, Kind: store.BotReplyQuestion,
		Anchor: &store.BotReplyAnchor{Path: "a.go", Line: 2, Commit: "src1"}}}
	h.llm.mention = inference.MentionResult{Outcome: inference.OutcomeOK, Answer: "On it.", Apply: true}
	h.llm.fix = inference.FindingFixResult{Outcome: inference.OutcomeOK, Fix: inference.FindingFix{
		Feasible: true, Old: "func new() {}", New: "func new() { validate() }", CommitSubject: "validate the input",
	}, Cost: inference.CallCost{NanoUSD: ptri(2_000_000), Answered: true}}
	p := mentionPayload("gute Idee, setz das um", "alice")
	p.CommentParentID = 950
	return h, p
}

func TestApplyQAAnswerCommits(t *testing.T) {
	h, p := answerHarness(t)

	h.r.HandleMention(context.Background(), p, "platform", &inlineScheduler{})

	if len(h.llm.Mentions) != 1 || h.llm.Mentions[0].Fix == nil ||
		!strings.Contains(h.llm.Mentions[0].Fix.Prompt, "Validate the input in new().") {
		t.Fatalf("want one Q&A call carrying the fix of the answer, got %+v", h.llm.Mentions)
	}
	if len(h.bb.Edits) != 1 || h.bb.Edits[0].Content != "package a\nfunc new() { validate() }\n" ||
		!strings.HasSuffix(h.bb.Edits[0].Message, "commentId=950") {
		t.Fatalf("edits = %+v", h.bb.Edits)
	}
	if len(h.bb.Replies) != 1 || !strings.Contains(h.bb.Replies[0].Text, "Applied in") || strings.Contains(h.bb.Replies[0].Text, "On it.") {
		t.Errorf("replies = %+v, want code's confirmation, not the model's sentence", h.bb.Replies)
	}
	if len(h.st.Fixes) != 1 || h.st.Fixes[0] != (store.FindingFix{AnswerCommentID: 950, TeamSlug: "platform", Outcome: store.FixCommitted, CommitSHA: "c0ffee1234567890"}) {
		t.Errorf("fixes = %+v", h.st.Fixes)
	}
	// The confirmation keeps no code line: no second fix starts from it.
	if last := h.st.BotReplies[len(h.st.BotReplies)-1]; last.Anchor != nil {
		t.Errorf("confirmation kept %+v", last.Anchor)
	}
	kinds := []string{}
	for _, c := range h.st.CommentCalls {
		kinds = append(kinds, c.Kind)
	}
	if strings.Join(kinds, ",") != "fix" {
		t.Errorf("recorded calls = %v, want the fix (the fake Q&A has no rounds)", kinds)
	}
}

// Only the PR author may have it applied; anyone else is told so, no call
// is paid and nothing is recorded (user, 2026-11-13).
func TestApplyQAAnswerOnlyForThePRAuthor(t *testing.T) {
	h, p := answerHarness(t)
	p.Comment.Author.Name = "bob"
	h.st.languages = map[string]lang.Lang{"bob": lang.DE}

	h.r.HandleMention(context.Background(), p, "platform", &inlineScheduler{})

	if h.llm.Mentions[0].Fix != nil || len(h.bb.Edits) != 0 || len(h.st.Fixes) != 0 {
		t.Errorf("fix request %v, edits %d, fixes %d: want none", h.llm.Mentions[0].Fix != nil, len(h.bb.Edits), len(h.st.Fixes))
	}
	if len(h.bb.Replies) != 1 || !strings.Contains(h.bb.Replies[0].Text, "Umsetzen lasse ich mich nur vom PR-Autor") {
		t.Errorf("replies = %+v", h.bb.Replies)
	}
}

func TestApplyQAAnswerSaysWhyNot(t *testing.T) {
	for name, c := range map[string]struct {
		tweak func(*harness)
		say   string
		fixes int
	}{
		"no code line": {func(h *harness) { h.st.botReplies[950].Anchor = nil }, "not on a line of code", 1},
		"applied already": {func(h *harness) {
			h.st.Fixes = []store.FindingFix{{AnswerCommentID: 950, Outcome: store.FixCommitted, CommitSHA: "abcdef1234"}}
		}, "Already applied in `abcdef12`", 1},
		"no write permission": {func(h *harness) {
			h.bb.editErr = &bitbucket.StatusError{Method: "PUT", Path: "/browse/a.go", Status: 403}
		}, "no write permission", 1},
	} {
		t.Run(name, func(t *testing.T) {
			h, p := answerHarness(t)
			c.tweak(h)

			h.r.HandleMention(context.Background(), p, "platform", &inlineScheduler{})

			if len(h.bb.Replies) != 1 || !strings.Contains(h.bb.Replies[0].Text, c.say) {
				t.Errorf("replies = %+v, want %q", h.bb.Replies, c.say)
			}
			if len(h.st.Fixes) != c.fixes {
				t.Errorf("fixes = %+v", h.st.Fixes)
			}
		})
	}
}

// Without `apply` it is a question like any other: answered, nothing
// committed, even though the fix was ready.
func TestQAAnswerWithoutApplyIsAQuestion(t *testing.T) {
	h, p := answerHarness(t)
	h.llm.mention.Apply = false
	h.llm.mention.Answer = "Because new() is public."

	h.r.HandleMention(context.Background(), p, "platform", &inlineScheduler{})

	if len(h.bb.Edits) != 0 || len(h.bb.Replies) != 1 || !strings.Contains(h.bb.Replies[0].Text, "Because new() is public.") {
		t.Errorf("edits %d, replies %+v", len(h.bb.Edits), h.bb.Replies)
	}
}
