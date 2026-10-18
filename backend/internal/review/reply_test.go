package review

import (
	"context"
	"errors"
	"strings"
	"testing"
	"time"

	"github.com/trick77/noergler/internal/inference"
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

func TestFindingReplyAnswersADisputeWithAnAgentsMDLine(t *testing.T) {
	h, p := replyHarness(t)

	h.r.HandleFindingReply(context.Background(), p, "payments", nil)

	if len(h.llm.Replies) != 1 {
		t.Fatalf("model calls = %d", len(h.llm.Replies))
	}
	prompt := h.llm.Replies[0].Prompt
	for _, want := range []string{"a.go:2 [issue]", "does not validate the business rule", "We do this in the UI",
		"    2  func new() {}", "the PR author"} {
		if !strings.Contains(prompt, want) {
			t.Errorf("prompt lacks %q:\n%s", want, prompt)
		}
	}
	if len(h.bb.Replies) != 1 || h.bb.Replies[0].Line != 900 {
		t.Fatalf("replies = %+v, want one under the author's comment 900", h.bb.Replies)
	}
	body := h.bb.Replies[0].Text
	if !strings.Contains(body, "Understood, the UI owns that check.") ||
		!strings.Contains(body, "consider adding to `AGENTS.md`") ||
		!strings.Contains(body, "- Business rule validations live in the UI") {
		t.Errorf("reply body:\n%s", body)
	}
	if len(h.st.Replied) != 1 || h.st.Replied[0].FindingID != 11 ||
		h.st.Replied[0].Dispute == nil || *h.st.Replied[0].Dispute != "We do this in the UI" {
		t.Errorf("replied = %+v, want row 11 marked disputed with the author's text", h.st.Replied)
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
		"a finding answered already": func(h *harness, _ *webhook.Payload) {
			at := time.Now()
			h.st.byComment[501].RepliedAt = &at
		},
	}
	for name, tweak := range cases {
		t.Run(name, func(t *testing.T) {
			h, p := replyHarness(t)
			tweak(h, p)

			h.r.HandleFindingReply(context.Background(), p, "payments", nil)

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

		h.r.HandleFindingReply(context.Background(), p, "payments", nil)

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

		h.r.HandleFindingReply(context.Background(), p, "payments", nil)

		if strings.Contains(h.bb.Replies[0].Text, "consider adding") {
			t.Errorf("reply repeats a line AGENTS.md has:\n%s", h.bb.Replies[0].Text)
		}
	})
}

// Nothing posted, nothing stamped: a later reply on the thread may try again.
func TestFindingReplyFailuresLeaveTheFindingUnanswered(t *testing.T) {
	t.Run("model failed", func(t *testing.T) {
		h, p := replyHarness(t)
		h.llm.reply = inference.FindingReplyResult{Outcome: inference.OutcomeUnparseable}

		h.r.HandleFindingReply(context.Background(), p, "payments", nil)

		if len(h.bb.Replies) != 0 || len(h.st.Replied) != 0 {
			t.Errorf("replies %d, marked %d", len(h.bb.Replies), len(h.st.Replied))
		}
	})
	t.Run("post failed", func(t *testing.T) {
		h, p := replyHarness(t)
		h.bb.replyErr = errors.New("boom")

		h.r.HandleFindingReply(context.Background(), p, "payments", nil)

		if len(h.st.Replied) != 0 {
			t.Errorf("marked %+v after a failed post", h.st.Replied)
		}
	})
	// The file narrows the context; without it the finding still stands.
	t.Run("file unavailable", func(t *testing.T) {
		h, p := replyHarness(t)
		h.bb.fileErr["src1:a.go"] = errors.New("boom")

		h.r.HandleFindingReply(context.Background(), p, "payments", nil)

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

	h.r.ReviewPullRequest(context.Background(), prPayload(webhook.EventOpened), false)

	if p := h.llm.Reviews[0].Prompt; !strings.Contains(p, "unchecked (author replied: we do this in the UI)") {
		t.Errorf("prompt lacks the dispute:\n%s", p)
	}
}

// Someone other than the PR author is answered, but their dispute is not
// stored: only the author's case keeps the issue out of later reviews.
func TestFindingReplyStoresOnlyTheAuthorsDispute(t *testing.T) {
	h, p := replyHarness(t)
	p.Comment.Author.Name = "carol"

	h.r.HandleFindingReply(context.Background(), p, "payments", nil)

	if !strings.Contains(h.llm.Replies[0].Prompt, "a reviewer, not the PR author") {
		t.Errorf("the prompt must say who replied:\n%s", h.llm.Replies[0].Prompt)
	}
	if len(h.bb.Replies) != 1 || strings.Contains(h.bb.Replies[0].Text, "AGENTS.md") {
		t.Errorf("replies = %+v, want an answer without a suggested line", h.bb.Replies)
	}
	if len(h.st.Replied) != 1 || h.st.Replied[0].Dispute != nil {
		t.Errorf("replied = %+v, want answered, no dispute stored", h.st.Replied)
	}
}

// An @mention on a finding thread is about that finding; asking for a
// review, or a thread that is no finding's, is the mention path's.
func TestFindingReplyWithAMention(t *testing.T) {
	t.Run("question about the finding", func(t *testing.T) {
		h, p := replyHarness(t)
		p.Comment.Text = "@noergler we do this in the UI"

		handled, _ := h.r.HandleFindingReply(context.Background(), p, "payments", nil)

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
			if handled, _ := h.r.HandleFindingReply(context.Background(), p, "payments", nil); handled || len(h.llm.Replies) != 0 {
				t.Errorf("handled = %v, calls = %d: want it left to the mention path", handled, len(h.llm.Replies))
			}
		})
	}
}

// A reply is automatic, so the per-PR cost cap holds; an unpriced PR never
// blocks.
func TestFindingReplyRespectsTheCostCap(t *testing.T) {
	h, p := replyHarness(t)
	h.st.prCost = ptri(6_000_000_000)

	h.r.HandleFindingReply(context.Background(), p, "payments", nil)

	if len(h.llm.Replies) != 0 {
		t.Error("a PR at the cap must not buy a reply")
	}
}

// No template, no feature: the team keeps reviewing.
func TestFindingReplyIsOffWithoutATemplate(t *testing.T) {
	h, p := replyHarness(t)
	h.r.replyTmpl = ""

	if handled, _ := h.r.HandleFindingReply(context.Background(), p, "payments", nil); handled || len(h.llm.Replies) != 0 {
		t.Errorf("handled = %v with the feature off", handled)
	}
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
}
