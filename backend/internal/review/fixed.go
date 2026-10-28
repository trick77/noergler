package review

import (
	"context"
	"fmt"
	"net/http"
	"slices"
	"time"

	"github.com/trick77/noergler/internal/bitbucket"
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
	// Settled on the PR: shown so it is not raised again, with no id, so it
	// can never be claimed fixed.
	if f.ClosedAt != nil {
		p.ID, p.Closed = 0, true
	}
	return p
}

// asFindings are stored findings as the PR summary lists them and the
// verdict counts them: still open, so still part of the PR.
func asFindings(stored []store.Finding) []inference.ReviewFinding {
	out := make([]inference.ReviewFinding, 0, len(stored))
	for _, f := range stored {
		out = append(out, inference.ReviewFinding{
			File: f.FilePath, Line: f.LineNumber, Severity: f.Severity, Comment: f.CommentText,
			Confidence: f.Confidence, Headline: f.Headline, Suggestion: f.Suggestion,
		})
	}
	return out
}

// closeThreads returns findings, in their order, with ClosedAt set on each
// whose Bitbucket thread is resolved or deleted: settled (user, 2026-10-28),
// out of the PR summary and its verdict, never posted again. A newly closed
// one is stamped, so its thread is never read again; a reopened thread is
// not noticed (accepted). Open = ClosedAt nil. A thread that cannot be read
// stays open, failing toward showing it; so does a row with no comment id
// (legacy: none to read). posted counts the findings with a thread.
//
// One GET per unsettled finding, on the worker. limit > 0 reads at most
// that many, issues first (user, 2026-10-28: every review pays this, on the
// shared worker); the rest stay open unread. 0 reads all, for the rare
// handoff. A cancelled ctx stops the reads. Each outcome is one log line.
func (r *Reviewer) closeThreads(ctx context.Context, key store.PRKey, prTag string, findings []store.Finding, limit int) (out []store.Finding, posted int) {
	out = slices.Clone(findings)
	var toRead []int // indexes into out, issues first when capped
	for i, f := range out {
		if f.BitbucketCommentID != nil {
			posted++
			if f.ClosedAt == nil {
				toRead = append(toRead, i)
			}
		}
	}
	if limit > 0 {
		slices.SortStableFunc(toRead, func(a, b int) int {
			return inference.SeverityRank(out[a].Severity) - inference.SeverityRank(out[b].Severity)
		})
	}
	unread, unreadable, closed := 0, 0, 0
	var lastErr error
	for n, i := range toRead {
		if (limit > 0 && n >= limit) || ctx.Err() != nil {
			unread++
			continue
		}
		f := &out[i]
		c, err := r.bitbucket.FetchPRComment(ctx, key.Project, key.Repo, key.PRID, *f.BitbucketCommentID)
		switch {
		case bitbucket.Status(err) == http.StatusNotFound, err == nil && c.ThreadResolved:
			now := time.Now()
			f.ClosedAt = &now
			closed++
			safeDBErr(ctx, r.log, "MarkFindingClosed", func() error { return r.store.MarkFindingClosed(ctx, f.ID) })
		case err != nil && ctx.Err() == nil:
			unreadable++
			lastErr = err
		case err != nil:
			unread++
		}
	}
	if closed > 0 {
		r.log.InfoContext(ctx, fmt.Sprintf("%s: %d finding thread(s) resolved or deleted on the PR, settled", prTag, closed))
	}
	if unreadable > 0 {
		r.log.WarnContext(ctx, fmt.Sprintf("%s: %d finding thread(s) unreadable, kept open", prTag, unreadable), logging.Err(lastErr))
	}
	if unread > 0 {
		why := "a cancelled review"
		if limit > 0 {
			why = fmt.Sprintf("the %d-read limit or a cancelled review", limit)
		}
		r.log.InfoContext(ctx, fmt.Sprintf("%s: %d finding thread(s) not read (%s), kept open", prTag, unread, why))
	}
	return out, posted
}

// openOnly is the findings closeThreads left open.
func openOnly(findings []store.Finding) []store.Finding {
	out := make([]store.Finding, 0, len(findings))
	for _, f := range findings {
		if f.ClosedAt == nil {
			out = append(out, f)
		}
	}
	return out
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
			if posted, err := r.bitbucket.ReplyToComment(ctx, plan.project, plan.repo, plan.prID, *cid, reply); err != nil {
				// Usually the author resolved the thread already: Bitbucket
				// takes no reply on a resolved one.
				if bitbucket.ThreadResolved(err) {
					r.log.InfoContext(ctx, plan.prTag+": reply on a fixed finding not posted: thread resolved already")
				} else {
					r.log.WarnContext(ctx, plan.prTag+": reply on a fixed finding failed", logging.Err(err))
				}
			} else {
				replied = true
				r.recordBotReply(ctx, plan.key, posted, store.BotReplyFixed)
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
