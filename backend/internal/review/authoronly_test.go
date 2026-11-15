package review

import (
	"context"
	"strings"
	"testing"
	"time"

	"github.com/trick77/noergler/internal/store"
)

func TestChangeRequest(t *testing.T) {
	for text, want := range map[string]bool{
		"fix it": true, "bitte beheben": true, "apply": true, "please fix this": true, "übernimm das": true,
		"Fix!": true, "korrigiere das bitte": true,
		"how would you fix this?": false, "I'll fix it": false, "ich fixe das": false, "the fix looks good": false,
		"don't fix it": false, "LGTM": false, "fixed in the next commit, thanks for spotting this one": false,
		"this change is fine": false, "this change is intended": false, "the fix please": true,
	} {
		if got := changeRequest(text); got != want {
			t.Errorf("changeRequest(%q) = %v, want %v", text, got, want)
		}
	}
}

// Someone other than the author orders a fix straight under noergler's
// finding: told kindly, no model, no fix (user, 2026-11-15).
func TestChangeAskedUnderAFindingByAnother(t *testing.T) {
	h, p := replyHarness(t)
	h.st.skipState = &store.SkipState{ID: 1}
	p.Comment.Author.Name = "bob"
	p.Comment.Text = "fix it"

	handled, _ := h.r.HandleFindingReply(context.Background(), p, "payments", &inlineScheduler{})

	if !handled || len(h.llm.Replies) != 0 || len(h.bb.Replies) != 1 ||
		!strings.Contains(h.bb.Replies[0].Text, "only change the PR when its author asks") {
		t.Fatalf("handled = %t, model = %d, replies = %+v", handled, len(h.llm.Replies), h.bb.Replies)
	}
}

// On a PR left alone (its summary deleted) a plain reply is not answered,
// an order to change it neither.
func TestChangeAskedUnderAFindingOfAnIgnoredPR(t *testing.T) {
	h, p := replyHarness(t)
	now := time.Now()
	h.st.skipState = &store.SkipState{ID: 1, IgnoredAt: &now, IgnoreReason: store.IgnoreSummaryDeleted}
	p.Comment.Author.Name = "bob"
	p.Comment.Text = "fix it"

	h.r.HandleFindingReply(context.Background(), p, "payments", &inlineScheduler{})

	if len(h.bb.Replies) != 0 {
		t.Fatalf("replies = %+v", h.bb.Replies)
	}
}

// A reviewer talking to the author under the finding stays unanswered.
func TestDiscussionUnderAFindingByAnotherStaysSilent(t *testing.T) {
	h, p := replyHarness(t)
	p.Comment.Author.Name = "bob"
	p.Comment.Text = "I agree with this one, the UI does not validate either"

	h.r.HandleFindingReply(context.Background(), p, "payments", &inlineScheduler{})

	if len(h.bb.Replies) != 0 || len(h.llm.Replies) != 0 {
		t.Fatalf("replies = %+v", h.bb.Replies)
	}
}

// With a mention, anywhere: the same kind answer, no Q&A call.
func TestChangeAskedByMentionByAnother(t *testing.T) {
	h := newHarness(t, nil)

	h.r.HandleMention(context.Background(), mentionPayload("@noergler bitte beheben", "bob"), "payments", &inlineScheduler{})

	if len(h.llm.Mentions) != 0 || len(h.bb.Replies) != 1 || !strings.Contains(h.bb.Replies[0].Text, "its author asks") {
		t.Fatalf("model = %d, replies = %+v", len(h.llm.Mentions), h.bb.Replies)
	}
}

// The author's own order is not this path's.
func TestChangeAskedByTheAuthorIsNotRefused(t *testing.T) {
	h := newHarness(t, nil)

	h.r.HandleMention(context.Background(), mentionPayload("@noergler bitte beheben", "alice"), "payments", &inlineScheduler{})

	for _, r := range h.bb.Replies {
		if strings.Contains(r.Text, "its author asks") {
			t.Fatalf("the author was refused: %q", r.Text)
		}
	}
}
