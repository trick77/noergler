package review

import (
	"context"
	"fmt"
	"strings"
	"time"

	"github.com/trick77/noergler/internal/diff"
	"github.com/trick77/noergler/internal/logging"
	"github.com/trick77/noergler/internal/riptide"
	"github.com/trick77/noergler/internal/store"
	"github.com/trick77/noergler/internal/webhook"
)

// HandleCommentDeleted is the primary opt-out signal: if the user deleted our
// summary comment, mark the PR ignored immediately (reviewer.py:1202).
//
// The payload carries the deleted comment id directly, which is robust where
// the pull-time guard's 404 check is fragile: a summary with replies may be
// soft-deleted and return 200 with a tombstone. The pull-time guard stays as
// a backstop for repos not subscribed to this event.
func (r *Reviewer) HandleCommentDeleted(ctx context.Context, payload *webhook.Payload) {
	comment := payload.Comment
	if comment == nil {
		return
	}
	project, repo := payload.ProjectRepo()
	if project == "" || repo == "" {
		return
	}
	prID := payload.PullRequest.ID
	prTag := fmt.Sprintf("%s/%s#%d", project, repo, prID)
	ctx = logging.With(ctx, "pr_tag", prTag, "repo", project+"/"+repo, "pr_id", prID)
	key := prKey(project, repo, prID)

	state := safeDB(ctx, r.log, "GetSkipState", func() (*store.SkipState, error) {
		return r.store.GetSkipState(ctx, key)
	})
	// A different comment was deleted: none of our business.
	if state == nil || state.Summary == nil || state.Summary.ID != comment.ID {
		return
	}

	safeDBErr(ctx, r.log, "MarkIgnored", func() error { return r.store.MarkIgnored(ctx, key) })
	r.log.InfoContext(ctx, fmt.Sprintf("%s: summary comment %d deleted (webhook) - ignoring PR from now on",
		prTag, comment.ID))
}

// HandlePRMerged marks the PR merged, freezes its cost and emits the rollup.
func (r *Reviewer) HandlePRMerged(ctx context.Context, payload *webhook.Payload) {
	ctx, key, prTag, ok := r.lifecycleContext(ctx, payload)
	if !ok {
		return
	}

	safeDBErr(ctx, r.log, "MarkMerged", func() error { return r.store.MarkMerged(ctx, key) })
	if frozen := safeDB(ctx, r.log, "FreezeFinalCost", func() (*int64, error) {
		return r.store.FreezeFinalCost(ctx, key)
	}); frozen != nil {
		// The sum of what the endpoint reported per run. Unpriced runs
		// contributed nothing, so the total is a floor, not the bill.
		r.log.InfoContext(ctx, fmt.Sprintf("%s merged - frozen LLM cost $%.4f",
			prTag, float64(*frozen)/nanoPerUSD))
	}

	r.emitRollup(ctx, key, prTag, "merged", payload.MergeCommitSHA())
}

// HandlePRDeclined marks the PR declined and emits the rollup. The data is
// retained for metrics.
func (r *Reviewer) HandlePRDeclined(ctx context.Context, payload *webhook.Payload) {
	ctx, key, prTag, ok := r.lifecycleContext(ctx, payload)
	if !ok {
		return
	}
	safeDBErr(ctx, r.log, "MarkDeclined", func() error { return r.store.MarkDeclined(ctx, key) })
	r.log.InfoContext(ctx, prTag+" declined - marked, data retained")
	r.emitRollup(ctx, key, prTag, "declined", "")
}

// HandlePRDeleted marks the PR deleted and emits the rollup.
func (r *Reviewer) HandlePRDeleted(ctx context.Context, payload *webhook.Payload) {
	ctx, key, prTag, ok := r.lifecycleContext(ctx, payload)
	if !ok {
		return
	}
	safeDBErr(ctx, r.log, "MarkDeleted", func() error { return r.store.MarkDeleted(ctx, key) })
	r.log.InfoContext(ctx, prTag+" deleted - marked, data retained")
	r.emitRollup(ctx, key, prTag, "deleted", "")
}

// lifecycleContext resolves the key and binds the log context, reporting
// whether the payload names a usable PR.
func (r *Reviewer) lifecycleContext(ctx context.Context, payload *webhook.Payload) (context.Context, store.PRKey, string, bool) {
	project, repo := payload.ProjectRepo()
	if project == "" || repo == "" {
		return ctx, store.PRKey{}, "", false
	}
	prID := payload.PullRequest.ID
	prTag := fmt.Sprintf("%s/%s#%d", project, repo, prID)
	ctx = logging.With(ctx, "pr_tag", prTag, "repo", project+"/"+repo, "pr_id", prID)
	return ctx, prKey(project, repo, prID), prTag, true
}

// emitRollup aggregates the per-run stats and posts one pr_completed event.
//
// A no-op when riptide is off, when no review ever ran, or when the rollup
// was already emitted: ClaimRollup stamps riptide_emitted_at in the same
// statement that reads the snapshot, so a redelivered pr:merged produces no
// second event. The claim happens BEFORE the POST, so a failed emission is
// never retried.
func (r *Reviewer) emitRollup(ctx context.Context, key store.PRKey, prTag, outcome, mergeCommit string) {
	if r.riptide == nil || !r.riptide.Enabled() {
		return
	}

	// Refresh the final cumulative diff at close time: the per-run
	// accumulators may only have seen incremental diffs. For a deleted PR
	// the diff is gone, so the fetch is skipped rather than always logged
	// and excepted.
	final := store.RollupFinal{}
	if mergeCommit != "" {
		final.MergeCommit = &mergeCommit
	}
	if outcome != "deleted" {
		if fullDiff, err := r.bitbucket.FetchPRDiff(ctx, key.Project, key.Repo, key.PRID, 0); err != nil {
			r.log.InfoContext(ctx, prTag+": final PR diff unavailable at close - using last-known per-run stats")
		} else if strings.TrimSpace(fullDiff) != "" {
			added, removed := countDiffLines(fullDiff)
			// files_changed counts the reviewable files, which is the
			// deliberate divergence AGENTS.md pins: Python counts every
			// "diff --git" header including the ones it never reviewed.
			changed := countReviewableFiles(fullDiff)
			final.LinesAdded, final.LinesRemoved, final.FilesChanged = &added, &removed, &changed
		}
	}

	snapshot := safeDB(ctx, r.log, "ClaimRollup", func() (*store.RollupSnapshot, error) {
		return r.store.ClaimRollup(ctx, key, final)
	})
	if snapshot == nil {
		r.log.DebugContext(ctx, prTag+": rollup not emitted (already emitted or no review runs)")
		return
	}
	if snapshot.SourceCommit == nil || *snapshot.SourceCommit == "" {
		r.log.WarnContext(ctx, prTag+
			": rollup has no source_commit_sha - skipping emit (should not happen if a run was recorded)")
		return
	}

	firstReview := snapshot.FirstReviewAt
	if firstReview.IsZero() {
		firstReview = time.Now().UTC()
	}

	r.riptide.EmitPRCompleted(ctx, riptide.Rollup{
		Outcome:               outcome,
		PRKey:                 prTag,
		Repo:                  key.Project + "/" + key.Repo,
		SourceCommitSHA:       *snapshot.SourceCommit,
		MergeCommitSHA:        derefString(snapshot.MergeCommit),
		LinesAdded:            derefInt(snapshot.LinesAdded),
		LinesRemoved:          derefInt(snapshot.LinesRemoved),
		FilesChanged:          derefInt(snapshot.FilesChanged),
		TotalRuns:             snapshot.Runs,
		TotalPromptTokens:     snapshot.PromptTokens,
		TotalCompletionTokens: snapshot.CompletionTokens,
		TotalElapsedMS:        snapshot.ElapsedMS,
		TotalFindingsCount:    snapshot.Findings,
		TotalCostNanoUSD:      snapshot.CostNanoUSD,
		ModelsUsed:            snapshot.Models,
		FirstReviewAt:         firstReview,
		ClosedAt:              time.Now().UTC(),
		ReviewerHandle:        r.bitbucket.BotUsername(),
	})
}

// countReviewableFiles counts the files in a diff the reviewer would look at.
//
// Python counts every "diff --git" header, including lock files and vendored
// bundles it never reviewed. AGENTS.md pins counting the reviewable ones as a
// deliberate divergence, so the number riptide records matches the number the
// summary reports.
func countReviewableFiles(rawDiff string) int {
	n := 0
	for _, fd := range diff.SplitByFile(rawDiff) {
		if diff.IsReviewable(fd) {
			n++
		}
	}
	return n
}

func derefString(p *string) string {
	if p == nil {
		return ""
	}
	return *p
}

func derefInt(p *int) int {
	if p == nil {
		return 0
	}
	return *p
}
