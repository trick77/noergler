package review

import (
	"cmp"
	"context"
	"errors"
	"fmt"
	"net/http"

	"github.com/trick77/noergler/internal/bitbucket"
	"github.com/trick77/noergler/internal/inference"
	"github.com/trick77/noergler/internal/lang"
	"github.com/trick77/noergler/internal/render"
	"github.com/trick77/noergler/internal/store"
)

// timeoutMinutes is the wall-clock cap in whole minutes, as the notices
// phrase it.
var timeoutMinutes = int(inference.CallTimeout.Minutes())

// postOrUpdateSummary posts a new summary comment or updates the existing
// one, tracking it in the store.
//
// The fallback to a fresh post is mapped from the specific errors, not from
// any failure: a 409 version conflict (the adapter already logs "falling
// back to a new comment") and a 404 (the comment was deleted) both mean post
// fresh. Anything else is a real failure and is logged without posting a
// duplicate.
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

	if commentID != 0 {
		r.trackSummary(ctx, prReviewID, commentID, version)
	}
}

// trackSummary records the PR's summary comment, when the PR has a row.
func (r *Reviewer) trackSummary(ctx context.Context, prReviewID int64, commentID, version int) {
	if prReviewID == 0 {
		return
	}
	safeDBErr(ctx, r.log, "SetSummaryComment", func() error {
		return r.store.SetSummaryComment(ctx, prReviewID, commentID, version)
	})
}

// shouldRepost reports whether a failed update means "post a fresh comment"
// rather than "give up". 403: the summary is another account's, the shared
// bot's before the team pushed its own (or the reverse), and Bitbucket lets
// only its author edit it.
func shouldRepost(err error) bool {
	return errors.Is(err, bitbucket.ErrVersionConflict) ||
		bitbucket.Status(err) == http.StatusNotFound ||
		bitbucket.Status(err) == http.StatusForbidden
}

// bannerSummary puts a banner on the PR's summary comment.
//
// No prior summary tracked: post fresh as a new comment, which becomes the
// noergler comment for this PR. A prior summary tracked: fetch its live body
// and hand it to edit, which returns the new body (its own banner on top of
// the body with the previous one stripped, so repeated notices do not
// stack) or false to leave the comment as it is. The original body is always
// preserved; the next successful review replaces the whole comment, banner
// and all. An edit Bitbucket refuses with 403 is reposted
// (repostOtherAuthors). label names the notice in the log.
func (r *Reviewer) bannerSummary(ctx context.Context, project, repo string, prID int, prReviewID int64,
	fresh string, edit func(live string) (string, bool), label string) {
	var existing *store.SummaryComment
	if prReviewID != 0 {
		existing = safeDB(ctx, r.log, "GetSummaryComment", func() (*store.SummaryComment, error) {
			return r.store.GetSummaryComment(ctx, prReviewID)
		})
	}

	if existing == nil {
		id, version, err := r.bitbucket.PostPRComment(ctx, project, repo, prID, fresh)
		if err != nil {
			r.log.ErrorContext(ctx, "Failed to post "+label+": "+err.Error())
			return
		}
		r.trackSummary(ctx, prReviewID, id, version)
		return
	}

	comment, err := r.bitbucket.FetchPRComment(ctx, project, repo, prID, existing.ID)
	if err != nil {
		r.log.ErrorContext(ctx, "Failed to fetch existing summary for "+label+": "+err.Error())
		return
	}
	newBody, ok := edit(comment.Text)
	if !ok {
		return
	}
	// Bitbucket's optimistic locking: prefer the live version over the one
	// cached in the DB to avoid a 409.
	liveVersion := cmp.Or(comment.Version, existing.Version)
	newVersion, err := r.bitbucket.UpdatePRComment(ctx, project, repo, prID, existing.ID, liveVersion, newBody)
	if err != nil {
		if !r.repostOtherAuthors(ctx, project, repo, prID, prReviewID, err, newBody, label) {
			r.log.ErrorContext(ctx, "Failed to update existing summary with "+label+": "+err.Error())
		}
		return
	}
	r.trackSummary(ctx, prReviewID, existing.ID, newVersion)
}

// failureNotice posts a review-failure notice or prepends a staleness banner
// to the tracked summary (bannerSummary).
func (r *Reviewer) failureNotice(ctx context.Context, project, repo string, prID int, prReviewID int64, freshBody, bannerLine, label string) {
	r.bannerSummary(ctx, project, repo, prID, prReviewID, freshBody, func(live string) (string, bool) {
		return render.StaleBanner(bannerLine) + "\n\n" + render.StripStaleBanner(live), true
	}, label)
}

// repostOtherAuthors posts body as the PR's new noergler comment when an edit
// of the tracked summary failed with 403: the summary is another account's
// (the shared bot's before the team pushed its own, or the reverse), and
// Bitbucket lets only its author edit it. Reports whether it took the error
// over; anything but a 403 is the caller's to report.
func (r *Reviewer) repostOtherAuthors(ctx context.Context, project, repo string, prID int, prReviewID int64, err error, body, label string) bool {
	if bitbucket.Status(err) != http.StatusForbidden {
		return false
	}
	id, version, postErr := r.bitbucket.PostPRComment(ctx, project, repo, prID, body)
	if postErr != nil {
		r.log.ErrorContext(ctx, "Failed to post "+label+": "+postErr.Error())
		return true
	}
	r.trackSummary(ctx, prReviewID, id, version)
	return true
}

// failureNoticeFor surfaces a run that produced no review on the summary
// comment: a timeout, a refused or unparseable answer, or a PR over the
// context window. Any other outcome posts nothing.
func (r *Reviewer) failureNoticeFor(ctx context.Context, outcome inference.Outcome, l lang.Lang, project, repo string, prID int, prReviewID int64, newCommit, priorCommit string) {
	short := shortOrUnknown(newCommit, l)
	bot := r.bitbucket.BotUsername()
	var fresh, lead, label string
	switch outcome {
	case inference.OutcomeTimedOut:
		fresh = fmt.Sprintf(l.Pick(
			"⚠️ **Review übersprungen** — keine Antwort vom Modell innerhalb von %d Minuten für Commit `%s`. "+
				"Das Review wurde nicht abgeschlossen. Neuen Commit pushen oder `@%s` für einen neuen Versuch.",
			"⚠️ **Review skipped** — no response from the model within %d minutes for commit `%s`. "+
				"The review did not complete. Push a new commit or `@%s` to retry."),
			timeoutMinutes, short, bot)
		lead = fmt.Sprintf(l.Pick("Keine Antwort vom Modell innerhalb von %d Minuten auf Commit `%s`",
			"No response from the model within %d minutes on commit `%s`"), timeoutMinutes, short)
		label = "timeout notice"
	case inference.OutcomeUnparseable:
		fresh = fmt.Sprintf(l.Pick(
			"⚠️ **Review übersprungen** — das Modell hat eine Antwort geliefert, die nicht verarbeitet werden konnte "+
				"(eventuell hat es diesen Diff abgelehnt), für Commit `%s`. Das Review wurde nicht abgeschlossen. "+
				"Neuen Commit pushen oder `@%s` für einen neuen Versuch.",
			"⚠️ **Review skipped** — the model returned a response that could not be processed "+
				"(it may have refused this diff) for commit `%s`. The review did not complete. "+
				"Push a new commit or `@%s` to retry."), short, bot)
		lead = fmt.Sprintf(l.Pick("Das Modell hat auf Commit `%s` eine unverarbeitbare Antwort geliefert",
			"The model returned an unprocessable response on commit `%s`"), short)
		label = "unparseable-response notice"
	case inference.OutcomeTooLarge:
		fresh = fmt.Sprintf(l.Pick(
			"⚠️ **Review übersprungen** — dieser PR ist zu gross für das Kontextfenster des Modells "+
				"(Commit `%s`). In kleinere PRs aufteilen, dann `@%s` für einen neuen Versuch.",
			"⚠️ **Review skipped** — this PR is too large to review within the model's context window "+
				"(commit `%s`). Split it into smaller PRs, then `@%s` to retry."), short, bot)
		lead = fmt.Sprintf(l.Pick("Commit `%s` ist zu gross für das Kontextfenster des Modells",
			"Commit `%s` is too large to review within the model's context window"), short)
		label = "too-large notice"
	default:
		return
	}
	r.failureNotice(ctx, project, repo, prID, prReviewID, fresh, staleBanner(l, lead, priorCommit), label)
}

// staleBanner is the banner row every failure notice shares. Each language
// carries its marker phrase (render.StaleBannerMarkerDE, "findings below
// reflect"), which the stripper falls back on when the sentinel is gone.
func staleBanner(l lang.Lang, lead, priorCommit string) string {
	if priorCommit == "" {
		return "⚠️ " + lead + l.Pick(" — "+render.StaleBannerMarkerDE+" einen früheren Commit.",
			" — findings below reflect an earlier commit.")
	}
	return fmt.Sprintf(l.Pick("⚠️ %s — "+render.StaleBannerMarkerDE+" den früheren Commit `%s`.",
		"⚠️ %s — findings below reflect the earlier commit `%s`."), lead, shortSHA(priorCommit, 8))
}

// costLimitNotice surfaces a blocked auto-review on the summary comment.
//
// Same shape as failureNotice but with the cost banner and its own stripper,
// so repeated blocked pushes replace the banner instead of stacking it. once
// leaves a summary already carrying a cost banner untouched, unless this one
// must add that a review was not run and the old one does not say so.
func (r *Reviewer) costLimitNotice(ctx context.Context, l lang.Lang, project, repo string, prID int, prReviewID int64, cumulativeUSD, limitUSD float64, blocked, once bool) {
	banner := render.CostLimitBanner(l, cumulativeUSD, limitUSD, blocked)
	r.bannerSummary(ctx, project, repo, prID, prReviewID, banner, func(live string) (string, bool) {
		stripped := render.StripCostBanner(live)
		if once && stripped != live && (!blocked || render.CostBannerSaysBlocked(live)) {
			return "", false
		}
		return banner + "\n\n" + stripped, true
	}, "cost-limit notice")
}

// shortOrUnknown is the 8-character commit prefix the notices use, or the
// word for unknown in l when the commit is missing.
func shortOrUnknown(commit string, l lang.Lang) string {
	if s := shortSHA(commit, 8); s != "" {
		return s
	}
	return l.Pick("unbekannt", "unknown")
}
