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

// refuseOverCap is a reply's or a Q&A's stop at the cap: one log line, and
// the cap banner on the summary, which is the PR's one notice; the comment
// itself gets no answer. The banner is in the PR author's language, as the
// summary is.
func (r *Reviewer) refuseOverCap(ctx context.Context, key store.PRKey, prTag, prAuthor, what string, rowID int64, cumulativeUSD float64) {
	r.log.InfoContext(ctx, fmt.Sprintf("%s: PR cost $%.2f >= limit $%.2f - %s not answered",
		prTag, cumulativeUSD, r.cfg.MaxPRCostUSD, what))
	r.costLimitNotice(ctx, r.languageFor(ctx, prAuthor), key.Project, key.Repo, key.PRID, rowID,
		cumulativeUSD, r.cfg.MaxPRCostUSD, false)
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
