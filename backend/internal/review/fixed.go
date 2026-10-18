package review

import (
	"context"
	"fmt"

	"github.com/trick77/noergler/internal/inference"
	"github.com/trick77/noergler/internal/store"
	"github.com/trick77/ship/kit/logging"
)

// validateFixed checks the model's `fixed` claims against the findings the
// prompt listed by id and logs every refused one. A wrong "fixed" resolves
// the thread on a live bug, so a claim needs a changed line of the
// finding's own file (inference.ValidateFixed).
func (r *Reviewer) validateFixed(ctx context.Context, plan *reviewPlan, claims []inference.FixedClaim) []store.Finding {
	if len(claims) == 0 {
		return nil
	}
	var sent []inference.PostedFinding
	byID := map[int64]store.Finding{}
	for _, f := range plan.existing {
		if !plan.listed[f.ID] {
			continue
		}
		line := f.LineNumber
		sent = append(sent, inference.PostedFinding{ID: f.ID, FilePath: f.FilePath, LineNumber: &line, Severity: f.Severity})
		byID[f.ID] = f
	}
	v := inference.ValidateFixed(claims, sent, plan.anchors)
	for _, d := range v.Dropped {
		r.log.InfoContext(ctx, fmt.Sprintf("%s: fixed claim %s not accepted (%s)", plan.prTag, d.Claim.ID, d.Reason))
	}
	out := make([]store.Finding, 0, len(v.Kept))
	for _, k := range v.Kept {
		out = append(out, byID[k.Finding.ID])
	}
	return out
}

// withoutFixed is existing less the findings this run retires.
func withoutFixed(existing, fixed []store.Finding) []store.Finding {
	if len(fixed) == 0 {
		return existing
	}
	gone := make(map[int64]bool, len(fixed))
	for _, f := range fixed {
		gone[f.ID] = true
	}
	out := make([]store.Finding, 0, len(existing))
	for _, f := range existing {
		if !gone[f.ID] {
			out = append(out, f)
		}
	}
	return out
}

// retireFixed closes the threads of the earlier findings validateFixed
// accepted: a reply, then the resolve, in that order, because Bitbucket
// takes no reply on a resolved thread, and the resolve only after the reply
// landed, so no thread is closed without saying why. Neither failing fails
// the run. The row is stamped either way: the fix is what the diff shows,
// and an unstamped row would be listed and claimed again.
func (r *Reviewer) retireFixed(ctx context.Context, plan *reviewPlan, fixed []store.Finding, runID int64) {
	for _, f := range fixed {
		replied, resolved := false, false
		if cid := f.BitbucketCommentID; cid != nil {
			reply := "Fixed in `" + shortSHA(plan.sourceCommit, 8) + "`."
			if err := r.bitbucket.ReplyToComment(ctx, plan.project, plan.repo, plan.prID, *cid, reply); err != nil {
				r.log.WarnContext(ctx, plan.prTag+": reply on a fixed finding failed, thread left open", logging.Err(err))
			} else {
				replied = true
				if err := r.bitbucket.ResolveCommentThread(ctx, plan.project, plan.repo, plan.prID, *cid); err != nil {
					r.log.WarnContext(ctx, plan.prTag+": resolving a fixed finding's thread failed", logging.Err(err))
				} else {
					resolved = true
				}
			}
		}
		safeDBErr(ctx, r.log, "MarkFindingFixed", func() error { return r.store.MarkFindingFixed(ctx, f.ID, runID) })
		r.log.InfoContext(ctx, fmt.Sprintf("%s: earlier finding on %s:%d (%s) fixed: reply posted %s, thread resolved %s",
			plan.prTag, f.FilePath, f.LineNumber, f.Severity, yesNo(replied), yesNo(resolved)))
	}
}

func yesNo(b bool) string {
	if b {
		return "yes"
	}
	return "no"
}
