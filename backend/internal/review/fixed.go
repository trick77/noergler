package review

import (
	"context"
	"fmt"

	"github.com/trick77/noergler/internal/inference"
	"github.com/trick77/ship/kit/logging"
)

// retireFixed closes the threads of earlier findings this run's diff fixed.
//
// The model's claims go through inference.ValidateFixed first: a wrong
// "fixed" resolves the thread on a live bug, so a claim needs a changed line
// of the finding's own file. Each accepted one gets a reply, then its thread
// resolved, in that order: Bitbucket takes no reply on a resolved thread.
// Neither failing fails the run. The row is stamped either way, because the
// fix is what the diff shows; the log says what reached the PR.
func (r *Reviewer) retireFixed(ctx context.Context, plan *reviewPlan, claims []inference.FixedClaim, runID int64) {
	if len(claims) == 0 {
		return
	}
	v := inference.ValidateFixed(claims, plan.posted, plan.anchors)
	for _, d := range v.Dropped {
		r.log.InfoContext(ctx, fmt.Sprintf("%s: fixed claim %s not accepted (%s)", plan.prTag, d.Claim.ID, d.Reason))
	}
	commentOf := make(map[int64]*int, len(plan.existing))
	for _, f := range plan.existing {
		commentOf[f.ID] = f.BitbucketCommentID
	}
	for _, k := range v.Kept {
		f := k.Finding
		replied, resolved := false, false
		if cid := commentOf[f.ID]; cid != nil {
			reply := "Fixed in `" + shortSHA(plan.sourceCommit, 8) + "`."
			if err := r.bitbucket.ReplyToComment(ctx, plan.project, plan.repo, plan.prID, *cid, reply); err != nil {
				r.log.WarnContext(ctx, plan.prTag+": reply on a fixed finding failed", logging.Err(err))
			} else {
				replied = true
			}
			if err := r.bitbucket.ResolveCommentThread(ctx, plan.project, plan.repo, plan.prID, *cid); err != nil {
				r.log.WarnContext(ctx, plan.prTag+": resolving a fixed finding's thread failed", logging.Err(err))
			} else {
				resolved = true
			}
		}
		if runID != 0 {
			safeDBErr(ctx, r.log, "MarkFindingFixed", func() error { return r.store.MarkFindingFixed(ctx, f.ID, runID) })
		}
		r.log.InfoContext(ctx, fmt.Sprintf("%s: earlier finding on %s (%s) fixed: reply posted %s, thread resolved %s",
			plan.prTag, location(f), f.Severity, yesNo(replied), yesNo(resolved)))
	}
}

func location(f inference.PostedFinding) string {
	if f.LineNumber == nil {
		return f.FilePath
	}
	return fmt.Sprintf("%s:%d", f.FilePath, *f.LineNumber)
}

func yesNo(b bool) string {
	if b {
		return "yes"
	}
	return "no"
}
