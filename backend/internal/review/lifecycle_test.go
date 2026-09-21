package review

import (
	"context"
	"testing"
	"time"

	"github.com/trick77/noergler/internal/store"
	"github.com/trick77/noergler/internal/webhook"
)

func snapshot() *store.RollupSnapshot {
	return &store.RollupSnapshot{
		Runs: 3, PromptTokens: 3000, CompletionTokens: 600, ElapsedMS: 12000, Findings: 4,
		CostNanoUSD: ptri(360_000_000), Models: []string{"gpt-5.5"},
		FirstReviewAt: time.Date(2026, 9, 1, 10, 0, 0, 0, time.UTC),
		SourceCommit:  ptrs("src1"),
	}
}

func TestCommentDeletedMarksIgnoredOnlyForOurSummary(t *testing.T) {
	t.Run("our summary comment", func(t *testing.T) {
		h := newHarness(t, nil)
		h.st.skipState = &store.SkipState{Summary: &store.SummaryComment{ID: 555, Version: 1}}
		p := prPayload(webhook.EventCommentDeleted)
		p.Comment = &webhook.Comment{ID: 555, Text: "gone", Author: webhook.User{Name: "alice"}}

		h.r.HandleCommentDeleted(context.Background(), p)

		if h.st.Ignored != 1 {
			t.Errorf("MarkIgnored called %d times, want 1", h.st.Ignored)
		}
	})

	t.Run("somebody else's comment", func(t *testing.T) {
		h := newHarness(t, nil)
		h.st.skipState = &store.SkipState{Summary: &store.SummaryComment{ID: 555, Version: 1}}
		p := prPayload(webhook.EventCommentDeleted)
		p.Comment = &webhook.Comment{ID: 999, Text: "gone", Author: webhook.User{Name: "alice"}}

		h.r.HandleCommentDeleted(context.Background(), p)

		if h.st.Ignored != 0 {
			t.Error("deleting an unrelated comment must not ignore the PR")
		}
	})

	t.Run("no row at all", func(t *testing.T) {
		h := newHarness(t, nil)
		p := prPayload(webhook.EventCommentDeleted)
		p.Comment = &webhook.Comment{ID: 555, Text: "gone", Author: webhook.User{Name: "alice"}}

		h.r.HandleCommentDeleted(context.Background(), p)

		if h.st.Ignored != 0 {
			t.Error("a PR we never tracked must not be marked ignored")
		}
	})
}

func TestMergedFreezesCostAndEmitsRollup(t *testing.T) {
	h := newHarness(t, nil)
	h.st.rollup = snapshot()
	h.st.prCost = ptri(360_000_000)
	p := prPayload(webhook.EventMerged)
	p.PullRequest.Properties = &webhook.Properties{MergeCommit: &webhook.MergeCommit{ID: "merge1"}}

	h.r.HandlePRMerged(context.Background(), p)

	if h.st.Merged != 1 {
		t.Errorf("MarkMerged called %d times, want 1", h.st.Merged)
	}
	if h.st.Frozen != 1 {
		t.Errorf("FreezeFinalCost called %d times, want 1", h.st.Frozen)
	}
	if len(h.rt.Emitted) != 1 {
		t.Fatalf("expected one rollup, got %d", len(h.rt.Emitted))
	}

	got := h.rt.Emitted[0]
	if got.Outcome != "merged" {
		t.Errorf("outcome = %q", got.Outcome)
	}
	if got.PRKey != "PROJ/my-repo#42" || got.Repo != "PROJ/my-repo" {
		t.Errorf("pr key = %q, repo = %q", got.PRKey, got.Repo)
	}
	if got.SourceCommitSHA != "src1" {
		t.Errorf("source commit = %q", got.SourceCommitSHA)
	}
	if got.ReviewerHandle != "noergler" {
		t.Errorf("reviewer handle = %q", got.ReviewerHandle)
	}
	if got.TotalRuns != 3 || got.TotalFindingsCount != 4 {
		t.Errorf("run totals = %d runs, %d findings", got.TotalRuns, got.TotalFindingsCount)
	}
	// The merge commit goes in through the claim, not the snapshot the fake
	// returns, so the claim is what carries it.
	if len(h.st.Claims) != 1 || h.st.Claims[0].MergeCommit == nil || *h.st.Claims[0].MergeCommit != "merge1" {
		t.Errorf("claim did not carry the merge commit: %+v", h.st.Claims)
	}
}

func TestDeclinedAndDeletedEmitRollups(t *testing.T) {
	t.Run("declined", func(t *testing.T) {
		h := newHarness(t, nil)
		h.st.rollup = snapshot()

		h.r.HandlePRDeclined(context.Background(), prPayload(webhook.EventDeclined))

		if h.st.Declined != 1 {
			t.Errorf("MarkDeclined called %d times, want 1", h.st.Declined)
		}
		if len(h.rt.Emitted) != 1 || h.rt.Emitted[0].Outcome != "declined" {
			t.Errorf("expected one declined rollup, got %+v", h.rt.Emitted)
		}
	})

	t.Run("deleted skips the final diff fetch", func(t *testing.T) {
		h := newHarness(t, nil)
		h.st.rollup = snapshot()
		// A deleted PR's diff is gone; fetching it would log an error for
		// nothing, so the fetch is skipped entirely.
		h.bb.prDiffErr = errDiffTooLarge

		h.r.HandlePRDeleted(context.Background(), prPayload(webhook.EventDeleted))

		if h.st.Deleted != 1 {
			t.Errorf("MarkDeleted called %d times, want 1", h.st.Deleted)
		}
		if len(h.rt.Emitted) != 1 || h.rt.Emitted[0].Outcome != "deleted" {
			t.Errorf("expected one deleted rollup, got %+v", h.rt.Emitted)
		}
		if len(h.st.Claims) != 1 {
			t.Fatalf("expected one claim, got %d", len(h.st.Claims))
		}
		if h.st.Claims[0].LinesAdded != nil || h.st.Claims[0].FilesChanged != nil {
			t.Error("a deleted PR must not carry refreshed diff stats")
		}
	})
}

// ClaimRollup returning nil means already emitted or no runs: either way
// nothing is forwarded. This is what stops a redelivered pr:merged from
// producing a second event.
func TestRollupNotEmittedWhenClaimReturnsNil(t *testing.T) {
	h := newHarness(t, nil)
	h.st.rollup = nil

	h.r.HandlePRMerged(context.Background(), prPayload(webhook.EventMerged))

	if len(h.rt.Emitted) != 0 {
		t.Errorf("expected no rollup, got %d", len(h.rt.Emitted))
	}
}

// A snapshot without a source commit cannot be emitted: riptide's validator
// rejects it, so the warning here is better than a 422 downstream.
func TestRollupSkippedWithoutSourceCommit(t *testing.T) {
	h := newHarness(t, nil)
	snap := snapshot()
	snap.SourceCommit = nil
	h.st.rollup = snap

	h.r.HandlePRMerged(context.Background(), prPayload(webhook.EventMerged))

	if len(h.rt.Emitted) != 0 {
		t.Errorf("expected no rollup without a source commit, got %d", len(h.rt.Emitted))
	}
}

func TestRollupIsANoOpWhenRiptideIsOff(t *testing.T) {
	t.Run("disabled emitter", func(t *testing.T) {
		h := newHarness(t, func(h *harness) { h.rt = &fakeRiptide{enabled: false} })
		h.st.rollup = snapshot()

		h.r.HandlePRMerged(context.Background(), prPayload(webhook.EventMerged))

		if len(h.rt.Emitted) != 0 {
			t.Error("a disabled emitter must not be called")
		}
		if len(h.st.Claims) != 0 {
			t.Error("a disabled emitter must not claim the rollup either")
		}
		// The merge itself is still recorded.
		if h.st.Merged != 1 {
			t.Error("the PR must still be marked merged")
		}
	})

	t.Run("no emitter at all", func(t *testing.T) {
		h := newHarness(t, func(h *harness) { h.rt = nil })
		h.st.rollup = snapshot()

		h.r.HandlePRMerged(context.Background(), prPayload(webhook.EventMerged))

		if h.st.Merged != 1 {
			t.Error("the PR must still be marked merged without riptide")
		}
	})
}

// files_changed counts the REVIEWABLE files, as AGENTS.md pins: counting
// every diff --git header would include files that were never reviewed.
func TestRollupCountsReviewableFilesOnly(t *testing.T) {
	h := newHarness(t, nil)
	h.st.rollup = snapshot()
	h.bb.prDiff = sampleDiff + `diff --git a/go.sum b/go.sum
index 3333333..4444444 100644
--- a/go.sum
+++ b/go.sum
@@ -1 +1 @@
-old
+new
`

	h.r.HandlePRMerged(context.Background(), prPayload(webhook.EventMerged))

	if len(h.st.Claims) != 1 {
		t.Fatalf("expected one claim, got %d", len(h.st.Claims))
	}
	got := h.st.Claims[0]
	if got.FilesChanged == nil {
		t.Fatal("the claim carries no files_changed")
	}
	if *got.FilesChanged != 1 {
		t.Errorf("files_changed = %d, want 1: go.sum is not reviewable", *got.FilesChanged)
	}
}

// A payload with no repository on either ref names no PR, so every handler
// returns without touching anything.
func TestLifecycleHandlersIgnoreAPayloadWithoutARepository(t *testing.T) {
	h := newHarness(t, nil)
	p := &webhook.Payload{EventKey: webhook.EventMerged}
	p.PullRequest.ID = 42

	h.r.HandlePRMerged(context.Background(), p)
	h.r.HandlePRDeclined(context.Background(), p)
	h.r.HandlePRDeleted(context.Background(), p)

	if h.st.Merged+h.st.Declined+h.st.Deleted != 0 {
		t.Error("a payload without a repository must not be recorded")
	}
	if len(h.rt.Emitted) != 0 {
		t.Error("a payload without a repository must not emit a rollup")
	}
}
