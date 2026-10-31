package review

import (
	"context"
	"errors"
	"strings"
	"testing"
	"time"

	"github.com/trick77/noergler/internal/bitbucket"
	"github.com/trick77/noergler/internal/inference"
	"github.com/trick77/noergler/internal/lang"
	"github.com/trick77/noergler/internal/prompts"
	"github.com/trick77/noergler/internal/store"
	"github.com/trick77/noergler/internal/webhook"
)

// replyHarness has one finding (row 11, comment 501) on a.go and a reply
// (comment 900) under it; the model calls it a dispute and suggests a line.
func replyHarness(t *testing.T) (*harness, *webhook.Payload) {
	t.Helper()
	h := newHarness(t, nil)
	cid := 501
	h.st.byComment = map[int]*store.Finding{501: {
		ID: 11, FilePath: "a.go", LineNumber: 2, Severity: "issue",
		CommentText: "new() does not validate the business rule", BitbucketCommentID: &cid,
	}}
	h.llm.reply = inference.FindingReplyResult{Outcome: inference.OutcomeOK, Reply: inference.FindingReply{
		Intent: "disputes", Reply: "Understood, the UI owns that check.",
		AgentsMDLine: "- Business rule validations live in the UI; the backend does not re-validate them.",
	}}
	p := prPayload(webhook.EventCommentAdded)
	// prPayload's author is alice: this reply is the author's.
	p.Comment = &webhook.Comment{ID: 900, Text: "We do this in the UI", Author: webhook.User{Name: "alice"}}
	p.CommentParentID = 501
	return h, p
}

// A first dispute with a project-wide reason is answered without any
// AGENTS.md text and recorded as a new signal (user, 2026-10-31: one
// dispute is weak evidence).
func TestFindingReplyAnswersADisputeAndRecordsTheSignal(t *testing.T) {
	h, p := replyHarness(t)

	h.r.HandleFindingReply(context.Background(), p, "payments", &inlineScheduler{})

	if len(h.llm.Replies) != 1 {
		t.Fatalf("model calls = %d", len(h.llm.Replies))
	}
	prompt := h.llm.Replies[0].Prompt
	for _, want := range []string{"a.go:2 [issue]", "does not validate the business rule", "We do this in the UI",
		"    2  func new() {}"} {
		if !strings.Contains(prompt, want) {
			t.Errorf("prompt lacks %q:\n%s", want, prompt)
		}
	}
	if len(h.bb.Replies) != 1 || h.bb.Replies[0].Line != 900 {
		t.Fatalf("replies = %+v, want one under the author's comment 900", h.bb.Replies)
	}
	body := h.bb.Replies[0].Text
	if !strings.Contains(body, "Understood, the UI owns that check.") || strings.Contains(body, "AGENTS.md") {
		t.Errorf("reply body:\n%s", body)
	}
	if len(h.st.Replied) != 1 || h.st.Replied[0].FindingID != 11 ||
		h.st.Replied[0].Dispute == nil || *h.st.Replied[0].Dispute != "We do this in the UI" {
		t.Errorf("replied = %+v, want row 11 marked disputed with the author's text", h.st.Replied)
	}
	want := occurrence{Team: "payments", SignalID: 0, FindingID: 11,
		Rule: "- Business rule validations live in the UI; the backend does not re-validate them."}
	if len(h.st.Occurrences) != 1 || h.st.Occurrences[0] != want {
		t.Errorf("occurrences = %+v, want a new signal %+v", h.st.Occurrences, want)
	}
	if len(h.st.SignalReads) != 1 || h.st.SignalReads[0] != (signalRead{"payments", "PROJ", "my-repo", 30, 20}) {
		t.Errorf("signal reads = %+v", h.st.SignalReads)
	}
}

// signalAt3 is a reply matching open signal [2] (id 5), which two earlier
// PRs of two authors already disputed: this one is the third PR.
func signalAt3(t *testing.T) (*harness, *webhook.Payload) {
	t.Helper()
	h, p := replyHarness(t)
	h.st.signals = []store.DisputeSignal{{ID: 4, Rule: "- Errors wrap with %w."}, {ID: 5, Rule: "- Validation lives in the UI."}}
	c12, c31 := 1201, 3101
	h.st.signalPRs = map[int64][]store.DisputePR{5: {
		{PRID: 31, CommentID: &c31, Author: "bob"},
		{PRID: 12, CommentID: &c12, Author: "Alice"},
	}}
	h.llm.reply.Reply.MatchesSignal = 2
	return h, p
}

// The third PR inside the window: the answer says so, links the earlier
// threads, counts the authors and proposes the merged bullet; the
// occurrence goes under the matched signal.
func TestFindingReplyProposesAtTheThreshold(t *testing.T) {
	h, p := signalAt3(t)

	h.r.HandleFindingReply(context.Background(), p, "payments", &inlineScheduler{})

	if prompt := h.llm.Replies[0].Prompt; !strings.Contains(prompt, "[1] - Errors wrap with %w.\n[2] - Validation lives in the UI.") {
		t.Errorf("prompt lacks the numbered signals:\n%s", prompt)
	}
	body := h.bb.Replies[0].Text
	for _, want := range []string{
		"This is the 3rd PR within 30 days disputing this for the same reason",
		"[#31](https://bb.test/projects/PROJ/repos/my-repo/pull-requests/31/overview?commentId=3101)",
		"[#12](https://bb.test/projects/PROJ/repos/my-repo/pull-requests/12/overview?commentId=1201)",
		"2 authors", "Proposed `AGENTS.md` change",
		"```markdown\n- Business rule validations live in the UI; the backend does not re-validate them.\n```",
	} {
		if !strings.Contains(strings.ReplaceAll(body, "\n", " "), strings.ReplaceAll(want, "\n", " ")) {
			t.Errorf("reply lacks %q:\n%s", want, body)
		}
	}
	if len(h.st.Occurrences) != 1 || h.st.Occurrences[0].SignalID != 5 {
		t.Errorf("occurrences = %+v, want one under signal 5", h.st.Occurrences)
	}
}

// German: the ordinal is "3.", the counts are German.
func TestFindingReplyProposesInGerman(t *testing.T) {
	h, p := signalAt3(t)
	h.st.languages = map[string]lang.Lang{"alice": lang.DE}

	h.r.HandleFindingReply(context.Background(), p, "payments", &inlineScheduler{})

	body := strings.ReplaceAll(h.bb.Replies[0].Text, "\n", " ")
	for _, want := range []string{"Das ist der 3. PR innert 30 Tagen mit diesem Einwand", "2 Verfasser", "Vorschlag für `AGENTS.md`"} {
		if !strings.Contains(body, want) {
			t.Errorf("reply lacks %q:\n%s", want, body)
		}
	}
}

// Below the threshold, off the list, not a dispute, a dispute without a
// bullet, a failed read: no statement. Only a checked dispute bullet is
// recorded, and only once the answer is posted.
func TestFindingReplyHoldsTheStatement(t *testing.T) {
	cases := map[string]struct {
		tweak     func(h *harness)
		occurs    bool
		signalID  int64
		postFails bool
	}{
		"one earlier PR":       {tweak: func(h *harness) { h.st.signalPRs[5] = h.st.signalPRs[5][:1] }, occurs: true, signalID: 5},
		"a match off the list": {tweak: func(h *harness) { h.llm.reply.Reply.MatchesSignal = 3 }, occurs: true},
		"not a dispute":        {tweak: func(h *harness) { h.llm.reply.Reply.Intent = "question" }},
		"no bullet": {tweak: func(h *harness) {
			h.llm.reply.Reply.AgentsMDLine = ""
		}},
		"bullet AGENTS.md has": {tweak: func(h *harness) {
			h.bb.files["src1:AGENTS.md"] = "- Business rule validations live in the UI; the backend does not re-validate them.\n"
		}},
		"signal reads fail": {tweak: func(h *harness) { h.st.signalErr = errors.New("boom") }, occurs: true},
		"post fails":        {tweak: func(h *harness) { h.bb.replyErr = errors.New("boom") }, postFails: true},
	}
	for name, c := range cases {
		t.Run(name, func(t *testing.T) {
			h, p := signalAt3(t)
			c.tweak(h)

			h.r.HandleFindingReply(context.Background(), p, "payments", &inlineScheduler{})

			if c.postFails {
				if len(h.st.Occurrences) != 0 {
					t.Errorf("recorded %+v without a posted answer", h.st.Occurrences)
				}
				return
			}
			if len(h.bb.Replies) != 1 || strings.Contains(h.bb.Replies[0].Text, "AGENTS.md") {
				t.Errorf("replies = %+v, want one without AGENTS.md text", h.bb.Replies)
			}
			if got := len(h.st.Occurrences); (got == 1) != c.occurs || got > 1 {
				t.Fatalf("occurrences = %+v, want recorded: %v", h.st.Occurrences, c.occurs)
			}
			if c.occurs && h.st.Occurrences[0].SignalID != c.signalID {
				t.Errorf("signal = %d, want %d", h.st.Occurrences[0].SignalID, c.signalID)
			}
		})
	}
}

// More earlier PRs than the statement links: the rest is a count.
func TestDisputeStatementCapsTheLinks(t *testing.T) {
	h, p := signalAt3(t)
	var prs []store.DisputePR
	for i := 1; i <= 12; i++ {
		prs = append(prs, store.DisputePR{PRID: 100 + i, Author: "bob"})
	}
	h.st.signalPRs[5] = prs

	h.r.HandleFindingReply(context.Background(), p, "payments", &inlineScheduler{})

	body := strings.ReplaceAll(h.bb.Replies[0].Text, "\n", " ")
	if !strings.Contains(body, "This is the 13th PR") || !strings.Contains(body, "#110, +2 more; 2 authors") ||
		strings.Contains(body, "#111") {
		t.Errorf("reply:\n%s", body)
	}
}

// A valid match whose merged bullet fails the format check still counts,
// under the listed wording, and that wording is what the statement shows.
func TestFindingReplyKeepsAValidMatchWithABadBullet(t *testing.T) {
	h, p := signalAt3(t)
	h.llm.reply.Reply.AgentsMDLine = "- " + strings.Repeat("x", 250)

	h.r.HandleFindingReply(context.Background(), p, "payments", &inlineScheduler{})

	if len(h.st.Occurrences) != 1 || h.st.Occurrences[0].SignalID != 5 || h.st.Occurrences[0].Rule != "- Validation lives in the UI." {
		t.Errorf("occurrences = %+v, want signal 5 under its listed wording", h.st.Occurrences)
	}
	if body := h.bb.Replies[0].Text; !strings.Contains(body, "```markdown\n- Validation lives in the UI.\n```") || strings.Contains(body, "xxxx") {
		t.Errorf("reply:\n%s", body)
	}
}

// A German bullet is Swiss-spelled before it is posted or stored; a one-day
// window reads singular.
func TestDisputeStatementSpelling(t *testing.T) {
	h, p := signalAt3(t)
	h.st.languages = map[string]lang.Lang{"alice": lang.DE}
	h.llm.reply.Reply.AgentsMDLine = "- Validierung erfolgt ausschließlich im UI."
	h.r.cfg.DisputeSignalWindowDays = 1

	h.r.HandleFindingReply(context.Background(), p, "payments", &inlineScheduler{})

	body := strings.ReplaceAll(h.bb.Replies[0].Text, "\n", " ")
	if !strings.Contains(body, "ausschliesslich") || strings.Contains(body, "ß") || !strings.Contains(body, "innert 1 Tag mit") {
		t.Errorf("reply:\n%s", body)
	}
	if len(h.st.Occurrences) != 1 || strings.Contains(h.st.Occurrences[0].Rule, "ß") {
		t.Errorf("stored %+v", h.st.Occurrences)
	}
}

func TestOrdinalEN(t *testing.T) {
	for n, want := range map[int]string{1: "1st", 2: "2nd", 3: "3rd", 4: "4th", 11: "11th", 12: "12th", 13: "13th", 21: "21st", 112: "112th"} {
		if got := ordinalEN(n); got != want {
			t.Errorf("ordinalEN(%d) = %q, want %q", n, got, want)
		}
	}
}

// What never reaches the model: each returns before any Bitbucket or gateway
// call.
func TestFindingReplyIgnores(t *testing.T) {
	cases := map[string]func(h *harness, p *webhook.Payload){
		"a reply not on a finding": func(_ *harness, p *webhook.Payload) { p.CommentParentID = 777 },
		"a top-level comment":      func(_ *harness, p *webhook.Payload) { p.CommentParentID = 0 },
		"the bot's own reply":      func(h *harness, p *webhook.Payload) { p.Comment.Author.Name = h.bb.bot },
		"an empty reply":           func(_ *harness, p *webhook.Payload) { p.Comment.Text = "  " },
		"a closed PR":              func(_ *harness, p *webhook.Payload) { p.PullRequest.State = "MERGED" },
		// No author name on either side is no proof of authorship.
		"no author to compare": func(_ *harness, p *webhook.Payload) {
			p.PullRequest.Author.User.Name, p.Comment.Author.Name = "", ""
		},
		// The summary was removed: only a mention brings the PR back.
		"an ignored PR": func(h *harness, _ *webhook.Payload) {
			at := time.Now()
			h.st.skipState = &store.SkipState{IgnoredAt: &at}
		},
		"a finding answered already": func(h *harness, _ *webhook.Payload) {
			at := time.Now()
			h.st.byComment[501].RepliedAt = &at
		},
	}
	for name, tweak := range cases {
		t.Run(name, func(t *testing.T) {
			h, p := replyHarness(t)
			tweak(h, p)

			h.r.HandleFindingReply(context.Background(), p, "payments", &inlineScheduler{})

			if len(h.llm.Replies) != 0 || len(h.bb.Replies) != 0 || len(h.st.Replied) != 0 {
				t.Errorf("model %d, replies %d, marked %d: want nothing", len(h.llm.Replies), len(h.bb.Replies), len(h.st.Replied))
			}
		})
	}
}

// A line on anything but a dispute, or one AGENTS.md already says, is not
// shown; a non-dispute records no dispute.
func TestFindingReplyDropsTheAgentsMDLine(t *testing.T) {
	t.Run("not a dispute", func(t *testing.T) {
		h, p := replyHarness(t)
		h.llm.reply.Reply.Intent = "acknowledges"

		h.r.HandleFindingReply(context.Background(), p, "payments", &inlineScheduler{})

		if strings.Contains(h.bb.Replies[0].Text, "AGENTS.md") {
			t.Errorf("reply suggests a line on an acknowledgement:\n%s", h.bb.Replies[0].Text)
		}
		if len(h.st.Replied) != 1 || h.st.Replied[0].Dispute != nil {
			t.Errorf("replied = %+v, want answered without a dispute", h.st.Replied)
		}
	})
	t.Run("already in AGENTS.md", func(t *testing.T) {
		h, p := replyHarness(t)
		h.bb.files["src1:AGENTS.md"] = "# Rules\n- Business rule validations live in the UI; the backend does not re-validate them.\n"

		h.r.HandleFindingReply(context.Background(), p, "payments", &inlineScheduler{})

		if strings.Contains(h.bb.Replies[0].Text, "AGENTS.md") {
			t.Errorf("reply repeats a line AGENTS.md has:\n%s", h.bb.Replies[0].Text)
		}
		// Nothing AGENTS.md states counts toward a signal.
		if len(h.st.Occurrences) != 0 {
			t.Errorf("occurrences = %+v", h.st.Occurrences)
		}
	})
}

// Nothing posted, nothing stamped: a later reply on the thread may try again.
func TestFindingReplyFailuresLeaveTheFindingUnanswered(t *testing.T) {
	t.Run("model failed", func(t *testing.T) {
		h, p := replyHarness(t)
		h.llm.reply = inference.FindingReplyResult{Outcome: inference.OutcomeUnparseable}

		h.r.HandleFindingReply(context.Background(), p, "payments", &inlineScheduler{})

		if len(h.bb.Replies) != 0 || len(h.st.Replied) != 0 {
			t.Errorf("replies %d, marked %d", len(h.bb.Replies), len(h.st.Replied))
		}
	})
	t.Run("post failed", func(t *testing.T) {
		h, p := replyHarness(t)
		h.bb.replyErr = errors.New("boom")

		h.r.HandleFindingReply(context.Background(), p, "payments", &inlineScheduler{})

		if len(h.st.Replied) != 0 {
			t.Errorf("marked %+v after a failed post", h.st.Replied)
		}
	})
	// The file narrows the context; without it the finding still stands.
	t.Run("file unavailable", func(t *testing.T) {
		h, p := replyHarness(t)
		h.bb.fileErr["src1:a.go"] = errors.New("boom")

		h.r.HandleFindingReply(context.Background(), p, "payments", &inlineScheduler{})

		if len(h.bb.Replies) != 1 || !strings.Contains(h.llm.Replies[0].Prompt, "could not be read") {
			t.Errorf("replies %d; want an answer without the file", len(h.bb.Replies))
		}
	})
}

// A stored dispute reaches the next review's prompt beside its finding.
func TestDisputeReachesTheNextReview(t *testing.T) {
	h := newHarness(t, nil)
	d := "we do this in the UI"
	h.st.existing = []store.Finding{{ID: 11, FilePath: "a.go", LineNumber: 2, Severity: "issue",
		CommentText: "unchecked", DisputeText: &d}}

	h.r.reviewInline(context.Background(), prPayload(webhook.EventOpened), false)

	if p := h.llm.Reviews[0].Prompt; !strings.Contains(p, "unchecked (author replied: we do this in the UI)") {
		t.Errorf("prompt lacks the dispute:\n%s", p)
	}
}

// Only the PR author's reply is the feature's: a reviewer's plain reply is
// not answered and does not use up the author's one answer; a reviewer's
// @mention is the Q&A's.
func TestFindingReplyIsTheAuthorsOnly(t *testing.T) {
	h, p := replyHarness(t)
	p.Comment.Author.Name = "carol"

	if handled, _ := h.r.HandleFindingReply(context.Background(), p, "payments", &inlineScheduler{}); !handled {
		t.Error("a plain reply by someone else is handled: ignored, nothing else to do")
	}
	p.Comment.Text = "@noergler why?"
	if handled, _ := h.r.HandleFindingReply(context.Background(), p, "payments", &inlineScheduler{}); handled {
		t.Error("someone else's @mention belongs to the Q&A")
	}
	if len(h.llm.Replies) != 0 || len(h.st.Replied) != 0 {
		t.Errorf("model %d, marked %d: a reviewer must not use up the author's answer", len(h.llm.Replies), len(h.st.Replied))
	}
}

// An @mention on a finding thread is about that finding; asking for a
// review, or a thread that is no finding's, is the mention path's.
func TestFindingReplyWithAMention(t *testing.T) {
	t.Run("question about the finding", func(t *testing.T) {
		h, p := replyHarness(t)
		p.Comment.Text = "@noergler we do this in the UI"

		handled, _ := h.r.HandleFindingReply(context.Background(), p, "payments", &inlineScheduler{})

		if !handled || !strings.Contains(h.llm.Replies[0].Prompt, "## The reply") ||
			strings.Contains(h.llm.Replies[0].Prompt, "@noergler") {
			t.Errorf("handled = %v; the reply must reach the prompt without the mention", handled)
		}
	})
	for name, tweak := range map[string]func(*harness, *webhook.Payload){
		"asks for a review": func(_ *harness, p *webhook.Payload) { p.Comment.Text = "@noergler review" },
		"not a finding":     func(_ *harness, p *webhook.Payload) { p.Comment.Text = "@noergler why?"; p.CommentParentID = 777 },
		"answered already": func(h *harness, p *webhook.Payload) {
			p.Comment.Text = "@noergler why?"
			at := time.Now()
			h.st.byComment[501].RepliedAt = &at
		},
	} {
		t.Run(name, func(t *testing.T) {
			h, p := replyHarness(t)
			tweak(h, p)
			if handled, _ := h.r.HandleFindingReply(context.Background(), p, "payments", &inlineScheduler{}); handled || len(h.llm.Replies) != 0 {
				t.Errorf("handled = %v, calls = %d: want it left to the mention path", handled, len(h.llm.Replies))
			}
		})
	}
}

// At the cap a reply gets no answer, mention or not (decided): the PR gets
// the cap banner instead, its one notice. A mention is left to the Q&A path,
// which refuses it the same way, and reactivates nothing on the way.
func TestFindingReplyRespectsTheCostCap(t *testing.T) {
	h, p := replyHarness(t)
	h.st.prCost = ptri(6_000_000_000)

	h.r.HandleFindingReply(context.Background(), p, "payments", &inlineScheduler{})

	if len(h.llm.Replies) != 0 {
		t.Error("a PR at the cap must not buy a plain reply")
	}
	if len(h.bb.Posted) != 1 || !strings.Contains(h.bb.Posted[0].Text, "Cost limit exceeded") {
		t.Errorf("want the cap banner as the one notice, got %+v", h.bb.Posted)
	}
	if strings.Contains(h.bb.Posted[0].Text, "not** reviewed") {
		t.Error("a refused reply blocked no review; the banner must not say one was")
	}

	at := time.Now()
	h.st.skipState = &store.SkipState{IgnoredAt: &at}
	p.Comment.Text = "@noergler we do this in the UI"
	handled, _ := h.r.HandleFindingReply(context.Background(), p, "payments", &inlineScheduler{})

	if handled || len(h.llm.Replies) != 0 {
		t.Errorf("handled = %v, replies = %d: a mention at the cap is no reply either", handled, len(h.llm.Replies))
	}
	if h.st.Reactived != 0 {
		t.Errorf("reactivated = %d: a mention at the cap must leave an ignored PR ignored", h.st.Reactived)
	}
}

// A reply's call is recorded, and the PR's cost counts it: the next reply
// at the cap is refused because of the replies before it.
func TestFindingReplyIsRecordedAndCounted(t *testing.T) {
	h, p := replyHarness(t)
	cost := int64(2_000_000)
	h.llm.reply.Cost = inference.CallCost{Answered: true, NanoUSD: &cost}
	h.r.HandleFindingReply(context.Background(), p, "payments", &inlineScheduler{})

	if len(h.st.CommentCalls) != 1 || h.st.CommentCalls[0].Kind != store.CallReply ||
		h.st.CommentCalls[0].CommentID != int64(p.Comment.ID) {
		t.Fatalf("comment calls = %+v, want the reply's", h.st.CommentCalls)
	}
	total, _ := h.st.PRCost(context.Background(), store.PRKey{})
	if total == nil || *total != cost {
		t.Errorf("PR cost = %v, want the reply counted", total)
	}
}

// A row stored with the model's b/ still finds its file; a line the push
// cut off says so instead of claiming the file is unreadable.
func TestFindingReplyFileExcerpt(t *testing.T) {
	t.Run("side prefix", func(t *testing.T) {
		h, p := replyHarness(t)
		h.st.byComment[501].FilePath = "b/a.go"

		h.r.HandleFindingReply(context.Background(), p, "payments", &inlineScheduler{})

		if !strings.Contains(h.llm.Replies[0].Prompt, "    2  func new() {}") {
			t.Errorf("prompt lacks the file:\n%s", h.llm.Replies[0].Prompt)
		}
	})
	// Just past the end, inside the window: still said, never the file's
	// tail passed off as the finding's context.
	t.Run("line just past the end", func(t *testing.T) {
		h, p := replyHarness(t)
		h.st.byComment[501].LineNumber = 5

		h.r.HandleFindingReply(context.Background(), p, "payments", &inlineScheduler{})

		if !strings.Contains(h.llm.Replies[0].Prompt, "Line 5 is past the end of the file (2 lines)") {
			t.Errorf("prompt:\n%s", h.llm.Replies[0].Prompt)
		}
	})
	t.Run("line past the end", func(t *testing.T) {
		h, p := replyHarness(t)
		h.st.byComment[501].LineNumber = 400

		h.r.HandleFindingReply(context.Background(), p, "payments", &inlineScheduler{})

		if !strings.Contains(h.llm.Replies[0].Prompt, "Line 400 is past the end") {
			t.Errorf("prompt:\n%s", h.llm.Replies[0].Prompt)
		}
	})
}

// Staged like a review: the call goes to the pool, the post comes back.
func TestFindingReplyIsStaged(t *testing.T) {
	h, p := replyHarness(t)
	s := &deferredScheduler{}

	handled, handedOff := h.r.HandleFindingReply(context.Background(), p, "payments", s)

	if !handled || !handedOff || s.post == nil {
		t.Fatalf("handled %v, handed off %v, staged %v", handled, handedOff, s.post != nil)
	}
	if len(h.bb.Replies) != 0 {
		t.Error("nothing is posted before the staged post runs")
	}
	s.finish()
	if len(h.bb.Replies) != 1 || len(h.st.Replied) != 1 {
		t.Errorf("after the stages: replies %d, marked %d", len(h.bb.Replies), len(h.st.Replied))
	}
}

// The excerpt reads the file at the commit the finding was posted at: a later
// push moves the lines its number names.
func TestFindingReplyReadsTheFileAsReviewed(t *testing.T) {
	h, p := replyHarness(t)
	h.st.byComment[501].RunCommit = "old1"
	h.bb.files["old1:a.go"] = "package a\nfunc old() {}\n"

	h.r.HandleFindingReply(context.Background(), p, "payments", &inlineScheduler{})

	if pr := h.llm.Replies[0].Prompt; !strings.Contains(pr, "    2  func old() {}") || strings.Contains(pr, "func new()") {
		t.Errorf("prompt shows the head, not the reviewed file:\n%s", pr)
	}
}

// The reviewed commit is gone (a rebase): the head stands in, labelled.
func TestFindingReplyFallsBackToTheHead(t *testing.T) {
	h, p := replyHarness(t)
	h.st.byComment[501].RunCommit = "gone1"

	h.r.HandleFindingReply(context.Background(), p, "payments", &inlineScheduler{})

	if pr := h.llm.Replies[0].Prompt; !strings.Contains(pr, "reviewed commit is gone") || !strings.Contains(pr, "    2  func new() {}") {
		t.Errorf("prompt:\n%s", pr)
	}
}

// Outside the allow-list, a plain reply is not automatic either.
func TestFindingReplyRespectsAutoReviewAuthors(t *testing.T) {
	h, p := replyHarness(t)
	h.r.SetAuthorLists([]string{"bob"}, nil)

	h.r.HandleFindingReply(context.Background(), p, "payments", &inlineScheduler{})

	if len(h.llm.Replies) != 0 {
		t.Error("an author outside auto_review_authors must not buy a plain reply")
	}
}

// A plain reply on an ignored author's PR buys no call; a mention still does.
func TestFindingReplyRespectsIgnoreAuthors(t *testing.T) {
	h, p := replyHarness(t)
	h.r.SetAuthorLists(nil, []string{"alice"})

	h.r.HandleFindingReply(context.Background(), p, "payments", &inlineScheduler{})
	if len(h.llm.Replies) != 0 {
		t.Error("an ignored author's plain reply must not buy a call")
	}

	p.Comment.Text = "@noergler we do this in the UI"
	h.r.HandleFindingReply(context.Background(), p, "payments", &inlineScheduler{})
	if len(h.llm.Replies) != 1 {
		t.Error("a mention overrides the author lists")
	}
}

func TestCodeFenceOutlastsTheLine(t *testing.T) {
	cases := map[string]string{
		"- plain":                          "```",
		"- uses `%w`":                      "```",
		"- x ``` [click](http://evil) ```": "````",
		"- x ```` y":                       "`````",
	}
	for line, want := range cases {
		if got := codeFence(line); got != want {
			t.Errorf("codeFence(%q) = %q, want %q", line, got, want)
		}
	}
}

func TestExcerptAroundNumbersTheWindow(t *testing.T) {
	got := excerptAround("a\nb\nc\nd\ne", 3, 1)
	if got != "    2  b\n    3  c\n    4  d\n" {
		t.Errorf("got %q", got)
	}
	if got := excerptAround("a", 9, 2); got != "" {
		t.Errorf("a line past the end: %q", got)
	}
	// A final newline ends the last line; it is no phantom line 3.
	if got := excerptAround("a\nb\n", 2, 5); got != "    1  a\n    2  b\n" {
		t.Errorf("trailing newline: %q", got)
	}
}

// A reply on the summary is addressed to the bot: a Q&A, whoever asks. A
// reply elsewhere off a finding stays unanswered, and a mention on the
// summary is left to the mention route.
func TestReplyOnTheSummaryIsAnsweredLikeAMention(t *testing.T) {
	summaryReply := func(text string, parent int) (*harness, *webhook.Payload) {
		h := newHarness(t, nil)
		h.st.skipState = &store.SkipState{Summary: &store.SummaryComment{ID: 555, Version: 1}}
		h.llm.mention = inference.MentionResult{Outcome: inference.OutcomeOK, Answer: "Three settings."}
		p := prPayload(webhook.EventCommentAdded)
		p.Comment = &webhook.Comment{ID: 900, Text: text, Author: webhook.User{Name: "bob"}}
		p.CommentParentID = parent
		return h, p
	}

	h, p := summaryReply("Was für Team-Einstellungen gibt es?", 555)
	if handled, _ := h.r.HandleFindingReply(context.Background(), p, "platform", &inlineScheduler{}); !handled {
		t.Error("a reply on the summary must be handled here")
	}
	if len(h.llm.Mentions) != 1 || len(h.bb.Replies) != 1 || h.bb.Replies[0].Line != 900 {
		t.Fatalf("mentions = %d, replies = %+v, want one answer under comment 900", len(h.llm.Mentions), h.bb.Replies)
	}

	h, p = summaryReply("Was für Team-Einstellungen gibt es?", 777)
	h.r.HandleFindingReply(context.Background(), p, "platform", &inlineScheduler{})
	if len(h.llm.Mentions)+len(h.bb.Replies) != 0 {
		t.Error("a reply on another comment must stay unanswered")
	}

	h, p = summaryReply("@noergler Was für Team-Einstellungen gibt es?", 555)
	if handled, _ := h.r.HandleFindingReply(context.Background(), p, "platform", &inlineScheduler{}); handled || len(h.llm.Mentions) != 0 {
		t.Error("a mention on the summary goes to HandleMention from the route, not from here")
	}
}

// A plain reply directly under any reply the bot posted (a Q&A answer, a
// finding-thread answer) is addressed to the bot: a Q&A, whoever asks, with
// the bot's own answer as the parent the model sees. Not on an ignored PR:
// only an @mention brings a PR back.
func TestReplyUnderNoerglersAnswerIsAnsweredLikeAMention(t *testing.T) {
	under := func(kind, author string) (*harness, *webhook.Payload) {
		h := newHarness(t, nil)
		h.r.mentionTmpl = prompts.Mention
		// A bot reply implies the PR's row: the mention path writes it first.
		h.st.skipState = &store.SkipState{}
		h.st.botReplies = map[int]*store.BotReply{600: {CommentID: 600, Kind: kind}}
		answer := &bitbucket.Comment{ID: 600, Version: 1, Text: "Three settings: a, b and c."}
		answer.Author.Name = "noergler"
		h.bb.comments[600] = answer
		h.llm.mention = inference.MentionResult{Outcome: inference.OutcomeOK, Answer: "b is the ignore list."}
		p := prPayload(webhook.EventCommentAdded)
		p.Comment = &webhook.Comment{ID: 900, Text: "Und was ist b?", Author: webhook.User{Name: author}}
		p.CommentParentID = 600
		return h, p
	}
	answered := func(t *testing.T, h *harness, p *webhook.Payload) {
		t.Helper()
		if handled, _ := h.r.HandleFindingReply(context.Background(), p, "platform", &inlineScheduler{}); !handled {
			t.Error("a reply under the bot's answer must be handled here")
		}
		if len(h.llm.Mentions) != 1 || len(h.llm.Replies) != 0 || len(h.bb.Replies) != 1 || h.bb.Replies[0].Line != 900 {
			t.Fatalf("mentions = %d, finding replies = %d, replies = %+v, want one Q&A answer under comment 900",
				len(h.llm.Mentions), len(h.llm.Replies), h.bb.Replies)
		}
		if prompt := h.llm.Mentions[0].Prompt; !strings.Contains(prompt, "Three settings: a, b and c.") ||
			!strings.Contains(prompt, "this bot") {
			t.Errorf("the model must see the bot's own answer as the parent:\n%s", prompt)
		}
	}

	// Not the author: "bob" asked, the bot answered, bob asks on.
	h, p := under(store.BotReplyQuestion, "bob")
	answered(t, h, p)

	// The author, under the bot's answer on a finding thread (decided):
	// a Q&A, not a second finding reply.
	h, p = under(store.BotReplyReply, "alice")
	answered(t, h, p)

	// An ignored PR stays ignored: no answer, no reactivation.
	h, p = under(store.BotReplyQuestion, "bob")
	at := time.Now()
	h.st.skipState = &store.SkipState{IgnoredAt: &at}
	if handled, _ := h.r.HandleFindingReply(context.Background(), p, "platform", &inlineScheduler{}); !handled {
		t.Error("a reply under the bot's answer on an ignored PR is handled: refused, not passed on")
	}
	if len(h.llm.Mentions)+len(h.bb.Replies) != 0 || h.st.Reactived != 0 {
		t.Errorf("mentions = %d, replies = %d, reactivated = %d: a plain reply never brings a PR back",
			len(h.llm.Mentions), len(h.bb.Replies), h.st.Reactived)
	}

	// The store down: the lookup fails open to "not the bot's", unanswered.
	h, p = under(store.BotReplyQuestion, "bob")
	h.st.failAll = true
	h.r.HandleFindingReply(context.Background(), p, "platform", &inlineScheduler{})
	if len(h.llm.Mentions)+len(h.bb.Replies) != 0 {
		t.Error("with the store down a reply under the bot's answer stays unanswered")
	}

	// The skip state unread but the bot reply found: not answered, so an
	// ignored PR cannot be reactivated behind a failed read.
	h, p = under(store.BotReplyQuestion, "bob")
	h.st.skipStateErr = errors.New("db hiccup")
	if handled, _ := h.r.HandleFindingReply(context.Background(), p, "platform", &inlineScheduler{}); !handled {
		t.Error("handled: refused, not passed on")
	}
	if len(h.llm.Mentions)+len(h.bb.Replies) != 0 || h.st.Reactived != 0 {
		t.Error("with the skip state unread a reply under the bot's answer stays unanswered and reactivates nothing")
	}

	// A closed PR: nothing is read, nothing answered.
	h, p = under(store.BotReplyQuestion, "bob")
	p.PullRequest.State = "MERGED"
	h.r.HandleFindingReply(context.Background(), p, "platform", &inlineScheduler{})
	if len(h.llm.Mentions)+len(h.bb.Replies) != 0 {
		t.Error("a reply on a merged PR stays unanswered")
	}
}
