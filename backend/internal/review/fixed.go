package review

import (
	"context"
	"fmt"

	"github.com/trick77/noergler/internal/diff"
	"github.com/trick77/noergler/internal/inference"
	"github.com/trick77/noergler/internal/store"
	"github.com/trick77/ship/kit/logging"
)

// validateFixed checks the model's `fixed` claims against the findings the
// prompt listed by id and logs every refused one. A wrong "fixed" resolves
// the thread on a live bug, so a claim needs a changed line of the
// finding's own file (inference.ValidateFixed).
//
// A claim is also refused when the same response reports that finding again
// at its file, line and severity: the model contradicts itself, and the
// thread stays open with the new report deduped against it.
func (r *Reviewer) validateFixed(ctx context.Context, plan *reviewPlan, claims []inference.FixedClaim, reported []inference.ReviewFinding) []store.Finding {
	if len(claims) == 0 {
		return nil
	}
	v := inference.ValidateFixed(claims, plan.posted, plan.anchors)
	for _, d := range v.Dropped {
		r.log.InfoContext(ctx, fmt.Sprintf("%s: fixed claim %s not accepted (%s)", plan.prTag, d.Claim.ID, d.Reason))
	}
	byID := make(map[int64]store.Finding, len(plan.existing))
	for _, f := range plan.existing {
		byID[f.ID] = f
	}
	out := make([]store.Finding, 0, len(v.Kept))
	for _, k := range v.Kept {
		f := byID[k.Finding.ID]
		if reportedAgain(f, reported, plan.anchors) {
			r.log.InfoContext(ctx, fmt.Sprintf("%s: fixed claim F%d not accepted (reported again on %s:%d)",
				plan.prTag, f.ID, f.FilePath, f.LineNumber))
			continue
		}
		out = append(out, f)
	}
	return out
}

// reportedAgain is dedupe's key, asked of one stored finding.
func reportedAgain(f store.Finding, reported []inference.ReviewFinding, idx diff.AnchorIndex) bool {
	path := inference.ShownPath(f.FilePath, idx)
	for _, n := range reported {
		if n.File == path && n.Line == f.LineNumber && n.Severity == f.Severity {
			return true
		}
	}
	return false
}

// retireFixed closes the threads of the earlier findings validateFixed
// accepted: a reply, then the resolve, in that order, because Bitbucket
// takes no reply on a resolved thread, and the resolve only after the reply
// landed, so no thread is closed without saying why. Neither failing fails
// the run.
//
// The row is stamped once the reply landed, resolved or not: the thread then
// says it is fixed, and an unstamped row would be listed and claimed again.
// A failed reply leaves the row open like the thread, so the two agree.
func (r *Reviewer) retireFixed(ctx context.Context, plan *reviewPlan, fixed []store.Finding, runID int64) {
	for _, f := range fixed {
		replied, resolved := false, false
		if cid := f.BitbucketCommentID; cid != nil {
			reply := "Fixed in `" + shortSHA(plan.sourceCommit, 8) + "`."
			if err := r.bitbucket.ReplyToComment(ctx, plan.project, plan.repo, plan.prID, *cid, reply); err != nil {
				r.log.WarnContext(ctx, plan.prTag+": reply on a fixed finding failed, not retired", logging.Err(err))
			} else {
				replied = true
				if err := r.bitbucket.ResolveCommentThread(ctx, plan.project, plan.repo, plan.prID, *cid); err != nil {
					r.log.WarnContext(ctx, plan.prTag+": resolving a fixed finding's thread failed", logging.Err(err))
				} else {
					resolved = true
				}
			}
		}
		if replied {
			safeDBErr(ctx, r.log, "MarkFindingFixed", func() error { return r.store.MarkFindingFixed(ctx, f.ID, runID) })
		}
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
