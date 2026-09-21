package review

import (
	"context"
	"errors"
	"fmt"
	"net/http"

	"github.com/trick77/noergler/internal/bitbucket"
	"github.com/trick77/noergler/internal/inference"
	"github.com/trick77/noergler/internal/render"
	"github.com/trick77/noergler/internal/store"
)

// timeoutMinutes is the wall-clock cap in whole minutes, as the notices
// phrase it.
var timeoutMinutes = int(inference.CallTimeout.Minutes())

// postOrUpdateSummary posts a new summary comment or updates the existing
// one, tracking it in the store.
//
// Python falls back to a fresh post whenever update_pr_comment returns None.
// Go returns errors instead, so the mapping is explicit: a 409 version
// conflict (the adapter already logs "falling back to a new comment") and a
// 404 (the comment was deleted) both mean post fresh. Anything else is a real
// failure and is logged without posting a duplicate.
func (r *Reviewer) postOrUpdateSummary(ctx context.Context, project, repo string, prID int, prReviewID int64, summary string) {
	var existing *store.SummaryComment
	if prReviewID != 0 {
		existing = safeDB(ctx, r.log, "GetSummaryComment", func() (*store.SummaryComment, error) {
			return r.store.GetSummaryComment(ctx, prReviewID)
		})
	}

	commentID, version := 0, 0
	if existing != nil {
		newVersion, err := r.bitbucket.UpdatePRComment(ctx, project, repo, prID, existing.ID, existing.Version, summary)
		switch {
		case err == nil:
			commentID, version = existing.ID, newVersion
		case shouldRepost(err):
			// The tracked comment is gone or was edited underneath us; a
			// fresh one becomes the noergler comment for this PR.
			id, v, postErr := r.bitbucket.PostPRComment(ctx, project, repo, prID, summary)
			if postErr != nil {
				r.log.ErrorContext(ctx, "Failed to post summary comment: "+postErr.Error())
				return
			}
			commentID, version = id, v
		default:
			r.log.ErrorContext(ctx, "Failed to post summary comment: "+err.Error())
			return
		}
	} else {
		id, v, err := r.bitbucket.PostPRComment(ctx, project, repo, prID, summary)
		if err != nil {
			r.log.ErrorContext(ctx, "Failed to post summary comment: "+err.Error())
			return
		}
		commentID, version = id, v
	}

	if prReviewID != 0 && commentID != 0 {
		safeDBErr(ctx, r.log, "SetSummaryComment", func() error {
			return r.store.SetSummaryComment(ctx, prReviewID, commentID, version)
		})
	}
}

// shouldRepost reports whether a failed update means "post a fresh comment"
// rather than "give up".
func shouldRepost(err error) bool {
	return errors.Is(err, bitbucket.ErrVersionConflict) ||
		bitbucket.Status(err) == http.StatusNotFound
}

// failureNotice posts a review-failure notice or prepends a staleness banner.
//
// No prior summary tracked: post freshBody as a new comment, which becomes
// the noergler comment for this PR. A prior summary tracked: fetch its body,
// strip any previous staleness banner (so repeated failures do not stack),
// and prepend a fresh one. The original body is always preserved; the next
// successful review replaces the whole comment, banner and all.
func (r *Reviewer) failureNotice(ctx context.Context, project, repo string, prID int, prReviewID int64, freshBody, bannerLine, label string) {
	var existing *store.SummaryComment
	if prReviewID != 0 {
		existing = safeDB(ctx, r.log, "GetSummaryComment", func() (*store.SummaryComment, error) {
			return r.store.GetSummaryComment(ctx, prReviewID)
		})
	}

	if existing == nil {
		id, version, err := r.bitbucket.PostPRComment(ctx, project, repo, prID, freshBody)
		if err != nil {
			r.log.ErrorContext(ctx, "Failed to post "+label+": "+err.Error())
			return
		}
		if prReviewID != 0 {
			safeDBErr(ctx, r.log, "SetSummaryComment", func() error {
				return r.store.SetSummaryComment(ctx, prReviewID, id, version)
			})
		}
		return
	}

	comment, err := r.bitbucket.FetchPRComment(ctx, project, repo, prID, existing.ID)
	if err != nil {
		r.log.ErrorContext(ctx, "Failed to fetch existing summary for staleness banner: "+err.Error())
		return
	}
	// Bitbucket's optimistic locking: prefer the live version over the one
	// cached in the DB to avoid a 409.
	liveVersion := comment.Version
	if liveVersion == 0 {
		liveVersion = existing.Version
	}

	newBody := render.StaleBanner(bannerLine) + "\n\n" + render.StripStaleBanner(comment.Text)
	newVersion, err := r.bitbucket.UpdatePRComment(ctx, project, repo, prID, existing.ID, liveVersion, newBody)
	if err != nil {
		r.log.ErrorContext(ctx, "Failed to update existing summary with staleness banner: "+err.Error())
		return
	}
	if prReviewID != 0 {
		safeDBErr(ctx, r.log, "SetSummaryComment", func() error {
			return r.store.SetSummaryComment(ctx, prReviewID, existing.ID, newVersion)
		})
	}
}

// timeoutNotice surfaces a deadline-exceeded run on the summary comment.
func (r *Reviewer) timeoutNotice(ctx context.Context, project, repo string, prID int, prReviewID int64, newCommit, priorCommit string) {
	short := shortOrUnknown(newCommit)
	fresh := fmt.Sprintf("⚠️ **Review skipped** — no response from the model within %d minutes for commit `%s`. "+
		"The review did not complete. Push a new commit or `@%s` to retry.",
		timeoutMinutes, short, r.bitbucket.BotUsername())

	banner := fmt.Sprintf("⚠️ No response from the model within %d minutes on commit `%s` — findings below reflect an earlier commit.",
		timeoutMinutes, short)
	if priorCommit != "" {
		banner = fmt.Sprintf("⚠️ No response from the model within %d minutes on commit `%s` — findings below reflect the earlier commit `%s`.",
			timeoutMinutes, short, shortSHA(priorCommit, 8))
	}
	r.failureNotice(ctx, project, repo, prID, prReviewID, fresh, banner, "timeout notice")
}

// unparseableNotice surfaces a refused or unparseable model response.
func (r *Reviewer) unparseableNotice(ctx context.Context, project, repo string, prID int, prReviewID int64, newCommit, priorCommit string) {
	short := shortOrUnknown(newCommit)
	fresh := fmt.Sprintf("⚠️ **Review skipped** — the model returned a response that could not be processed "+
		"(it may have refused this diff) for commit `%s`. The review did not complete. "+
		"Push a new commit or `@%s` to retry.", short, r.bitbucket.BotUsername())

	banner := fmt.Sprintf("⚠️ The model returned an unprocessable response on commit `%s` — findings below reflect an earlier commit.", short)
	if priorCommit != "" {
		banner = fmt.Sprintf("⚠️ The model returned an unprocessable response on commit `%s` — findings below reflect the earlier commit `%s`.",
			short, shortSHA(priorCommit, 8))
	}
	r.failureNotice(ctx, project, repo, prID, prReviewID, fresh, banner, "unparseable-response notice")
}

// tooLargeNotice surfaces a PR that does not fit the model's context window.
func (r *Reviewer) tooLargeNotice(ctx context.Context, project, repo string, prID int, prReviewID int64, newCommit, priorCommit string) {
	short := shortOrUnknown(newCommit)
	fresh := fmt.Sprintf("⚠️ **Review skipped** — this PR is too large to review within the model's context window "+
		"(commit `%s`). Split it into smaller PRs, then `@%s` to retry.", short, r.bitbucket.BotUsername())

	banner := fmt.Sprintf("⚠️ Commit `%s` is too large to review within the model's context window — findings below reflect an earlier commit.", short)
	if priorCommit != "" {
		banner = fmt.Sprintf("⚠️ Commit `%s` is too large to review within the model's context window — findings below reflect the earlier commit `%s`.",
			short, shortSHA(priorCommit, 8))
	}
	r.failureNotice(ctx, project, repo, prID, prReviewID, fresh, banner, "too-large notice")
}

// costLimitNotice surfaces a blocked auto-review on the summary comment.
//
// Same shape as failureNotice but with the cost banner and its own stripper,
// so repeated blocked pushes replace the banner instead of stacking it.
func (r *Reviewer) costLimitNotice(ctx context.Context, project, repo string, prID int, prReviewID int64, cumulativeUSD, limitUSD float64) {
	banner := render.CostLimitBanner(cumulativeUSD, limitUSD, r.bitbucket.BotUsername(), true)

	var existing *store.SummaryComment
	if prReviewID != 0 {
		existing = safeDB(ctx, r.log, "GetSummaryComment", func() (*store.SummaryComment, error) {
			return r.store.GetSummaryComment(ctx, prReviewID)
		})
	}

	if existing == nil {
		id, version, err := r.bitbucket.PostPRComment(ctx, project, repo, prID, banner)
		if err != nil {
			r.log.ErrorContext(ctx, "Failed to post cost-limit notice: "+err.Error())
			return
		}
		if prReviewID != 0 {
			safeDBErr(ctx, r.log, "SetSummaryComment", func() error {
				return r.store.SetSummaryComment(ctx, prReviewID, id, version)
			})
		}
		return
	}

	comment, err := r.bitbucket.FetchPRComment(ctx, project, repo, prID, existing.ID)
	if err != nil {
		r.log.ErrorContext(ctx, "Failed to fetch existing summary for cost-limit banner: "+err.Error())
		return
	}
	liveVersion := comment.Version
	if liveVersion == 0 {
		liveVersion = existing.Version
	}

	newBody := banner + "\n\n" + render.StripCostBanner(comment.Text)
	newVersion, err := r.bitbucket.UpdatePRComment(ctx, project, repo, prID, existing.ID, liveVersion, newBody)
	if err != nil {
		r.log.ErrorContext(ctx, "Failed to update existing summary with cost-limit banner: "+err.Error())
		return
	}
	if prReviewID != 0 {
		safeDBErr(ctx, r.log, "SetSummaryComment", func() error {
			return r.store.SetSummaryComment(ctx, prReviewID, existing.ID, newVersion)
		})
	}
}

// shortOrUnknown is the 8-character commit prefix the notices use, or
// "unknown" when the commit is missing.
func shortOrUnknown(commit string) string {
	if s := shortSHA(commit, 8); s != "" {
		return s
	}
	return "unknown"
}
