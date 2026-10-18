package review

import (
	"context"
	"fmt"

	"github.com/trick77/noergler/internal/inference"
	"github.com/trick77/noergler/internal/store"
	"github.com/trick77/ship/kit/logging"
)

// asPosted is a stored finding as the prompt lists it. The one conversion:
// what the prompt showed and what a claim is checked against must not drift.
func asPosted(f store.Finding) inference.PostedFinding {
	line := f.LineNumber
	p := inference.PostedFinding{
		ID: f.ID, FilePath: f.FilePath, LineNumber: &line, Severity: f.Severity, CommentText: f.CommentText,
	}
	if f.DisputeText != nil {
		p.Dispute = *f.DisputeText
	}
	return p
}

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
		if plan.listed[f.ID] {
			sent = append(sent, asPosted(f))
			byID[f.ID] = f
		}
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
// the run.
//
// The row is stamped whatever Bitbucket answered. Decided: no later run can
// retry, because the next push's diff no longer holds the fix, so an
// unstamped row would only dedup a real recurrence for the life of the PR.
// A failed reply leaves an open thread on fixed code, and the log says so.
func (r *Reviewer) retireFixed(ctx context.Context, plan *reviewPlan, fixed []store.Finding, runID int64) {
	for _, f := range fixed {
		replied, resolved := false, false
		if cid := f.BitbucketCommentID; cid != nil {
			reply := plan.lang.Pick("Behoben in `", "Fixed in `") + shortSHA(plan.sourceCommit, 8) + "`."
			if err := r.bitbucket.ReplyToComment(ctx, plan.project, plan.repo, plan.prID, *cid, reply); err != nil {
				// Usually the author resolved the thread already: Bitbucket
				// takes no reply on a resolved one.
				r.log.WarnContext(ctx, plan.prTag+": reply on a fixed finding failed (thread may be resolved already)", logging.Err(err))
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
