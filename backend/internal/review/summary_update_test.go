package review

import (
	"context"
	"testing"
	"time"

	"github.com/trick77/noergler/internal/store"
	"github.com/trick77/noergler/internal/webhook"
)

// A successful review that finds a tracked summary comment EDITS it rather
// than posting a second one. Only the timeout path reached this branch before,
// and that one posts through failureNotice.
func TestSuccessfulReviewUpdatesTheTrackedSummary(t *testing.T) {
	h := newHarness(t, func(h *harness) {
		h.st.summary = &store.SummaryComment{ID: 55, Version: 3}
		h.bb.setComment(55, "the previous summary", 3)
	})

	h.r.ReviewPullRequest(context.Background(), prPayload(webhook.EventFromRefUpdated), false)

	if len(h.bb.Updates) != 1 {
		t.Fatalf("expected one comment update, got %d", len(h.bb.Updates))
	}
	if len(h.bb.Posted) != 0 {
		t.Errorf("a tracked summary must be edited, not duplicated; %d fresh comments posted", len(h.bb.Posted))
	}
	if h.bb.Updates[0].Line != 55 {
		t.Errorf("updated comment %d, want 55", h.bb.Updates[0].Line)
	}
	// The version travels with the update: Bitbucket rejects a stale one, so
	// sending the wrong number would break every re-review.
	if len(h.bb.UpdateVersions) != 1 || h.bb.UpdateVersions[0] != 3 {
		t.Errorf("update sent versions %v, want [3]", h.bb.UpdateVersions)
	}
	// And the new version is tracked, or the next update would send a stale
	// one.
	if len(h.st.Summaries) == 0 {
		t.Fatal("the new summary version was not recorded")
	}
	last := h.st.Summaries[len(h.st.Summaries)-1]
	if last.ID != 55 || last.Version != 4 {
		t.Errorf("recorded summary %+v, want id 55 version 4", last)
	}
}

// A REVIEW-keyword mention on an ignored PR reactivates it AND runs a review,
// which then posts a fresh summary. The existing test covers only the Q&A
// path, which reactivates without reviewing.
func TestReviewKeywordMentionReactivatesAndReviewsFresh(t *testing.T) {
	now := time.Now()
	h := newHarness(t, func(h *harness) {
		h.st.skipState = &store.SkipState{IgnoredAt: &now}
	})

	h.r.HandleMention(context.Background(), mentionPayload("@noergler review", "alice"))

	if h.st.Reactived != 1 {
		t.Errorf("Reactivate called %d times, want 1", h.st.Reactived)
	}
	if len(h.llm.Reviews) != 1 {
		t.Fatalf("the keyword should trigger a full review, got %d LLM calls", len(h.llm.Reviews))
	}
	if len(h.llm.Mentions) != 0 {
		t.Errorf("a review keyword must not go to Q&A, got %d mention calls", len(h.llm.Mentions))
	}
	// Reactivate clears the stale summary id, so the review posts a fresh
	// comment rather than trying to edit one that is gone.
	if len(h.bb.Posted) != 1 {
		t.Errorf("expected a fresh summary comment, got %d", len(h.bb.Posted))
	}
	if len(h.bb.Updates) != 0 {
		t.Errorf("a reactivated PR should not edit a prior summary, got %d updates", len(h.bb.Updates))
	}
}
