package review

import (
	"context"
	"fmt"
	"net/http"

	"github.com/trick77/noergler/internal/bitbucket"
	"github.com/trick77/noergler/internal/render"
	"github.com/trick77/noergler/internal/store"
	"github.com/trick77/noergler/internal/webhook"
	"github.com/trick77/ship/kit/logging"
)

// HandleReadyForReview runs when a draft PR goes ready: the author hands it
// from noergler to human reviewers (user, 2026-10-22). When noergler's
// finding threads are still open, the summary says so at the top, with
// links, so the reviewers see the bot's round is not closed. All closed:
// nothing is written, and a banner from an earlier flip is removed.
//
// No model call, no run, no attempt, no cost. Open = a stored finding that
// is not fixed and whose Bitbucket thread is not resolved; a thread that
// cannot be read counts as open (fail toward telling the humans), a
// deleted one as closed; findings unreadable from the DB leave the summary
// as it is. One GET per unfixed finding on the worker, accepted: the event
// is rare and bounded by what was posted, as the fixed-thread retirement
// already does per finding.
func (r *Reviewer) HandleReadyForReview(ctx context.Context, payload *webhook.Payload) {
	ctx, key, prTag, ok := prContext(ctx, payload)
	if !ok {
		r.log.ErrorContext(ctx, "Could not extract project/repo from webhook payload")
		return
	}
	pr := payload.PullRequest
	if pr.State != "" && pr.State != "OPEN" {
		r.log.DebugContext(ctx, fmt.Sprintf("%s: draft -> ready ignored: PR is %s", prTag, pr.State))
		return
	}
	if r.store == nil {
		return
	}
	st := safeDB(ctx, r.log, "GetSkipState", func() (*store.SkipState, error) {
		return r.store.GetSkipState(ctx, key)
	})
	if st == nil || st.IgnoredAt != nil || st.Summary == nil {
		r.log.DebugContext(ctx, prTag+": draft -> ready, no summary to annotate (never reviewed, or ignored)")
		return
	}

	// The summary first: unreadable, there is nothing to annotate, and no
	// thread needs reading.
	summary, err := r.bitbucket.FetchPRComment(ctx, key.Project, key.Repo, key.PRID, st.Summary.ID)
	if err != nil {
		// A deleted summary is the comment-deleted handler's to act on.
		r.log.ErrorContext(ctx, "Failed to fetch the summary for the handoff banner: "+err.Error())
		return
	}
	// Not safeDB: its fallback is "no findings", which would read as "all
	// closed" and remove a banner that is still true. A fault leaves the
	// summary as it is.
	findings, err := r.store.ExistingFindings(ctx, key)
	if err != nil {
		r.log.WarnContext(ctx, prTag+": draft -> ready, findings unreadable, summary left as it is", logging.Err(err))
		return
	}
	var open []render.HandoffThread
	posted := 0
	for _, f := range findings {
		if f.BitbucketCommentID == nil {
			continue
		}
		posted++
		c, err := r.bitbucket.FetchPRComment(ctx, key.Project, key.Repo, key.PRID, *f.BitbucketCommentID)
		switch {
		case bitbucket.Status(err) == http.StatusNotFound:
			continue
		case err != nil:
			r.log.WarnContext(ctx, fmt.Sprintf("%s: finding thread %d unreadable, counted as open", prTag, *f.BitbucketCommentID),
				logging.Err(err))
		case c.ThreadResolved:
			continue
		}
		open = append(open, render.HandoffThread{
			File: f.FilePath, Line: f.LineNumber,
			URL: r.bitbucket.CommentURL(key.Project, key.Repo, key.PRID, *f.BitbucketCommentID),
		})
	}
	r.log.InfoContext(ctx, fmt.Sprintf("%s: draft -> ready, %d of %d noergler threads open", prTag, len(open), posted))

	banner := ""
	if len(open) > 0 {
		banner = render.HandoffBanner(r.languageFor(ctx, pr.Author.User.Name), open)
	}
	body := render.PlaceHandoffBanner(summary.Text, banner)
	if body == summary.Text {
		return
	}
	version := summary.Version
	if version == 0 {
		version = st.Summary.Version
	}
	newVersion, err := r.bitbucket.UpdatePRComment(ctx, key.Project, key.Repo, key.PRID, st.Summary.ID, version, body)
	if err != nil {
		if !r.repostOtherAuthors(ctx, key.Project, key.Repo, key.PRID, st.ID, err, body, "the summary with the handoff banner") {
			r.log.ErrorContext(ctx, "Failed to update the summary with the handoff banner: "+err.Error())
		}
		return
	}
	safeDBErr(ctx, r.log, "SetSummaryComment", func() error {
		return r.store.SetSummaryComment(ctx, st.ID, st.Summary.ID, newVersion)
	})
}
