package review

import (
	"context"
	"fmt"

	"github.com/trick77/noergler/internal/inference"
	"github.com/trick77/noergler/internal/store"
	"github.com/trick77/ship/kit/logging"
)

// overCap reads the PR's cost against the per-PR cap. At the cap everything
// stops: reviews, replies and Q&A answers, mention or not (decided; a
// mention used to override it). rowID is the PR row the cap notice edits the
// summary of. Fails open: an unpriced PR, or a failed read, never blocks.
func (r *Reviewer) overCap(ctx context.Context, key store.PRKey) (rowID int64, cumulativeUSD float64, over bool) {
	if r.store == nil {
		return 0, 0, false
	}
	var cost *int64
	safeDBErr(ctx, r.log, "PRCapState", func() error {
		var err error
		rowID, cost, err = r.store.PRCapState(ctx, key)
		return err
	})
	if cost == nil {
		return rowID, 0, false
	}
	cumulativeUSD = float64(*cost) / nanoPerUSD
	return rowID, cumulativeUSD, cumulativeUSD >= r.cfg.MaxPRCostUSD
}

// refuseOverCap is a reply's or a mention's stop at the cap: one log line,
// and the cap banner on the summary, which is the PR's one notice; the
// comment itself gets no answer. A summary that already carries a banner is
// left as it is, so refused comments cost no further Bitbucket write, unless
// a refused review must add the "not reviewed" sentence the banner lacks.
// An ignored PR gets nothing posted:
// its author removed the summary to be left alone. blocked: a review was
// asked for and not run. The banner is in the PR author's language, as the
// summary is.
func (r *Reviewer) refuseOverCap(ctx context.Context, key store.PRKey, prTag, prAuthor, what string, rowID int64, cumulativeUSD float64, blocked bool) {
	r.log.InfoContext(ctx, fmt.Sprintf("%s: PR cost $%.2f >= limit $%.2f - %s not answered",
		prTag, cumulativeUSD, r.cfg.MaxPRCostUSD, what))
	if state := safeDB(ctx, r.log, "GetSkipState", func() (*store.SkipState, error) {
		return r.store.GetSkipState(ctx, key)
	}); state != nil && state.IgnoredAt != nil {
		return
	}
	r.costLimitNotice(ctx, r.languageFor(ctx, prAuthor), key.Project, key.Repo, key.PRID, rowID,
		cumulativeUSD, r.cfg.MaxPRCostUSD, blocked, true)
}

// ensurePRRow writes the PR's row without touching its review pointer, so a
// Q&A on a PR never reviewed has a row its calls are counted under from the
// first one on (prCostSQL counts calls from the row's creation). Returns the
// row id, 0 when the write failed.
func (r *Reviewer) ensurePRRow(ctx context.Context, key store.PRKey, author, title string, openedAt int64) int64 {
	if r.store == nil {
		return 0
	}
	return r.upsert(ctx, store.PRUpsert{Key: key, TeamSlug: r.TeamSlug, KeepPointer: true,
		Author: &author, Title: &title, OpenedAt: epochMSToTime(openedAt)}, "")
}

// recordCommentCall writes a reply's or a question's gateway call. It is the
// only record of that call and the PR's cost counts it, so a refused write
// is a WARN: that money is then missing from the cap. Nothing to write when
// the gateway never answered.
func (r *Reviewer) recordCommentCall(ctx context.Context, key store.PRKey, kind string, commentID int, c inference.CallCost) {
	row := r.llmCall(c)
	if row == nil || r.store == nil {
		return
	}
	if err := r.store.InsertCommentCall(ctx, r.TeamSlug, key, kind, int64(commentID), *row); err != nil {
		r.log.WarnContext(ctx, key.Tag()+": "+kind+" call not recorded, its cost is missing from the PR total",
			logging.Err(err))
	}
}
