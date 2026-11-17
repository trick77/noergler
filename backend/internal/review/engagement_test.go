package review

import (
	"context"
	"testing"

	"github.com/trick77/noergler/internal/inference"
	"github.com/trick77/noergler/internal/store"
)

// Every comment that tags noergler or replies to one of its comments counts
// once as an engagement, by what came of it (user, 2026-11-17); a fix
// request counts as a fix and a review asked by mention as a run, never
// also as an engagement.
func TestEngagementIsCountedOnceByOutcome(t *testing.T) {
	mention := func(t *testing.T, res inference.MentionResult, text string) *harness {
		h := newHarness(t, nil)
		h.llm.mention = res
		h.r.HandleMention(context.Background(), mentionPayload(text, "alice"), "platform", &inlineScheduler{})
		return h
	}
	for name, c := range map[string]struct {
		run  func(t *testing.T) *harness
		want []engagement
	}{
		"Q&A answered": {func(t *testing.T) *harness {
			return mention(t, inference.MentionResult{Outcome: inference.OutcomeOK, Answer: "It locks."}, "@noergler why?")
		}, []engagement{{77, store.EngagementMention, store.EngagementAnswered}}},
		"Q&A failed": {func(t *testing.T) *harness {
			return mention(t, inference.MentionResult{Outcome: inference.OutcomeError}, "@noergler why?")
		}, []engagement{{77, store.EngagementMention, store.EngagementFailed}}},
		"Q&A timed out": {func(t *testing.T) *harness {
			return mention(t, inference.MentionResult{Outcome: inference.OutcomeTimedOut}, "@noergler why?")
		}, []engagement{{77, store.EngagementMention, store.EngagementFailed}}},
		"review by mention is a run": {func(t *testing.T) *harness {
			h := newHarness(t, nil)
			h.llm.review = okResultWith()
			h.r.HandleMention(context.Background(), mentionPayload("@noergler review", "alice"), "platform", &inlineScheduler{})
			return h
		}, nil},
		"finding reply answered": {func(t *testing.T) *harness {
			h, p := replyHarness(t)
			h.r.HandleFindingReply(context.Background(), p, "payments", &inlineScheduler{})
			return h
		}, []engagement{{900, store.EngagementReply, store.EngagementAnswered}}},
		"order by another under a finding": {func(t *testing.T) *harness {
			h, p := replyHarness(t)
			h.st.skipState = &store.SkipState{ID: 1}
			p.Comment.Author.Name, p.Comment.Text = "bob", "fix it"
			h.r.HandleFindingReply(context.Background(), p, "payments", &inlineScheduler{})
			return h
		}, []engagement{{900, store.EngagementReply, store.EngagementAuthorOnly}}},
		"fix request is a fix": {func(t *testing.T) *harness {
			h, p := fixHarness(t)
			h.r.HandleFindingReply(context.Background(), p, "payments", &inlineScheduler{})
			if len(h.st.Fixes) != 1 {
				t.Errorf("fixes = %+v, want the request counted as a fix", h.st.Fixes)
			}
			return h
		}, nil},
	} {
		t.Run(name, func(t *testing.T) {
			h := c.run(t)
			if len(h.st.Engagements) != len(c.want) {
				t.Fatalf("engagements = %+v, want %+v", h.st.Engagements, c.want)
			}
			for i, e := range c.want {
				if h.st.Engagements[i] != e {
					t.Errorf("engagement %d = %+v, want %+v", i, h.st.Engagements[i], e)
				}
			}
		})
	}
}

// A mention on a PR with no row yet still counts: the row is written, as
// the Q&A writes it, then the comment counted; one row, one upsert.
func TestEngagementOnAPRWithoutARowWritesTheRow(t *testing.T) {
	h := newHarness(t, nil)
	h.st.noPRRow = true

	h.r.HandleMention(context.Background(), mentionPayload("@noergler english", "alice"), "platform", &inlineScheduler{})

	if len(h.st.Upserts) != 1 || len(h.st.Engagements) != 1 ||
		h.st.Engagements[0] != (engagement{77, store.EngagementMention, store.EngagementAnswered}) {
		t.Errorf("upserts = %d, engagements = %+v", len(h.st.Upserts), h.st.Engagements)
	}
}

// A merge-conflict request from anyone but the PR's author is told kindly
// and counts as an engagement: no fix row is written for it.
func TestConflictRequestByAnotherIsAnEngagement(t *testing.T) {
	h := newHarness(t, nil)
	p := mentionPayload("@noergler behebe den Merge-Konflikt", "bob")

	h.r.HandleMention(context.Background(), p, "platform", &inlineScheduler{})

	if len(h.st.Fixes) != 0 || len(h.st.Engagements) != 1 ||
		h.st.Engagements[0] != (engagement{77, store.EngagementMention, store.EngagementAuthorOnly}) {
		t.Errorf("fixes = %+v, engagements = %+v", h.st.Fixes, h.st.Engagements)
	}
}

// A SNAPSHOT task counts once as a task, when it is posted; a later push
// that keeps it posts and counts nothing more.
func TestSnapshotTaskCountedOnce(t *testing.T) {
	h := snapshotHarness(t, snap1)

	h.push(t)
	if len(h.st.snapshotTasks) != 1 || len(h.st.SnapshotCounted) != 1 ||
		h.st.SnapshotCounted[0] != h.st.snapshotTasks[0].ID {
		t.Fatalf("stored = %+v, counted = %v", h.st.snapshotTasks, h.st.SnapshotCounted)
	}
	h.push(t)
	if len(h.st.SnapshotCounted) != 1 {
		t.Errorf("counted = %v after a second push, want the one task", h.st.SnapshotCounted)
	}
}
