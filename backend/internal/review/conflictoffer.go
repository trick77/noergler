package review

import (
	"context"
	"fmt"
	"net/http"
	"sync"

	"github.com/trick77/noergler/internal/bitbucket"
	"github.com/trick77/noergler/internal/lang"
	"github.com/trick77/noergler/internal/render"
	"github.com/trick77/noergler/internal/store"
	"github.com/trick77/noergler/internal/webhook"
)

// A review that finds a merge conflict noergler could resolve offers it
// (user, 2026-11-14): a comment of its own that mentions the author, at
// most one open at a time, resolved once no offer holds. It only offers;
// the author's command resolves (conflict.go). Seen at a review only, the
// first included: Bitbucket sends no event when the target branch moves
// (BSERV-14136), so a conflict that arises later shows at the next push.

// offerKind is what a review's check found.
type offerKind int

const (
	// offerNone: no conflict, past merge's bounds, a fork; or the PR is at
	// its cost cap, where noergler offers nothing (user, 2026-11-14).
	offerNone offerKind = iota
	// offerCode: code resolves every conflict, no model.
	offerCode
	// offerModel: within bounds, the model must answer some.
	offerModel
	// offerUnknown: the check failed (merge status, fetch) or another one
	// was running. An open offer is left as it is: a flaky check must not
	// withdraw it and post it again next time.
	offerUnknown
)

func (k offerKind) String() string {
	return [...]string{"none", "code", "model", "unknown"}[k]
}

// offerSlot lets one offer check fetch at a time, process-wide: a check
// may hold up to conflictMaxInflated, and every team's reviews may end at
// once. A check that finds it taken is unknown, tried at the next review.
var offerSlot = make(chan struct{}, 1)

// offerPR is what the check needs of the PR, taken in prepare: the plan
// keeps no payload.
type offerPR struct {
	author, source, target string
	sourceHead, targetHead string
	fork                   bool
}

func offerPROf(pr webhook.PullRequest, key store.PRKey) offerPR {
	return offerPR{author: pr.Author.User.Name, source: pr.FromRef.Branch(), target: pr.ToRef.Branch(),
		sourceHead: pr.FromRef.LatestCommit, targetHead: pr.ToRef.LatestCommit, fork: conflictFromFork(pr, key)}
}

// offerCacheMax bounds the check's memory of PRs: past it, it starts over
// (a check is repeated, nothing more).
const offerCacheMax = 1000

// offerCache is each PR's last check, by the heads it ran on: a review on
// the same two heads fetches nothing again. Memory only; the open comment
// is in the store.
type offerCache struct {
	mu sync.Mutex
	m  map[store.PRKey]offerEntry
}

type offerEntry struct {
	sourceHead, targetHead string
	kind                   offerKind
}

func (c *offerCache) get(k store.PRKey, p offerPR) (offerKind, bool) {
	c.mu.Lock()
	defer c.mu.Unlock()
	e, ok := c.m[k]
	if !ok || e.sourceHead != p.sourceHead || e.targetHead != p.targetHead {
		return offerNone, false
	}
	return e.kind, true
}

func (c *offerCache) put(k store.PRKey, p offerPR, kind offerKind) {
	if p.sourceHead == "" || p.targetHead == "" {
		return
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.m == nil || len(c.m) >= offerCacheMax {
		c.m = map[store.PRKey]offerEntry{}
	}
	c.m[k] = offerEntry{sourceHead: p.sourceHead, targetHead: p.targetHead, kind: kind}
}

// conflictOffer checks the PR for a conflict noergler could resolve: the
// merge status, then, on a conflict, what a request does before the model
// (fetch, merge, bounds, code's resolutions). No model, no commit; the
// request's fetch depths, so a long-lived branch is offered as a request
// would resolve it; bounded like a request's stage. A failed check is
// unknown and is not remembered.
func (r *Reviewer) conflictOffer(ctx context.Context, key store.PRKey, prTag string, p offerPR) offerKind {
	if p.fork || p.source == "" || p.target == "" {
		return offerNone
	}
	if kind, ok := r.offers.get(key, p); ok {
		return kind
	}
	ctx, cancel := context.WithTimeout(ctx, conflictStageTimeout)
	defer cancel()
	status, err := r.bitbucket.MergeStatus(ctx, key.Project, key.Repo, key.PRID)
	if err != nil {
		r.log.WarnContext(ctx, fmt.Sprintf("%s: merge conflict offer: merge status failed: %v", prTag, err))
		return offerUnknown
	}
	kind := offerNone
	if status.Conflicted {
		select {
		case offerSlot <- struct{}{}:
			defer func() { <-offerSlot }()
		default:
			r.log.InfoContext(ctx, prTag+": merge conflict offer: another check is fetching, left for the next review")
			return offerUnknown
		}
		j := &conflictJob{key: key, prTag: prTag, team: r.TeamSlug, lang: lang.EN, source: p.source, target: p.target, offer: true}
		outcome, _ := r.mergeConflict(ctx, j, false)
		j.repo = nil
		switch {
		case outcome == store.FixError:
			return offerUnknown
		case outcome != "":
		case len(j.pending) == 0:
			kind = offerCode
		default:
			kind = offerModel
		}
	}
	r.offers.put(key, p, kind)
	return kind
}

// noteConflictOffer keeps the PR's offer comment in step with kind: posted
// when an offer holds and none is open, resolved once none holds, left as
// it is when the check failed. prID is the PR row.
func (r *Reviewer) noteConflictOffer(ctx context.Context, key store.PRKey, prID int64, prTag string, l lang.Lang, p offerPR, kind offerKind) {
	if kind == offerUnknown {
		return
	}
	// Unread, the open offer is unknown: posting could post it twice.
	open, err := r.store.ConflictOffer(ctx, prID)
	if err != nil {
		r.log.WarnContext(ctx, fmt.Sprintf("%s: merge conflict offer: reading the open offer failed: %v", prTag, err))
		return
	}
	switch {
	case kind != offerNone && open == 0:
		text := render.ConflictOffer(l, kind == offerCode, p.author, r.bitbucket.BotUsername(), p.target)
		id, _, err := r.bitbucket.PostPRComment(ctx, key.Project, key.Repo, key.PRID, text)
		if err != nil {
			r.log.ErrorContext(ctx, fmt.Sprintf("%s: merge conflict offer: post failed: %v", prTag, err))
			return
		}
		safeDBErr(ctx, r.log, "SetConflictOffer", func() error { return r.store.SetConflictOffer(ctx, prID, id) })
		// A plain reply under it is addressed to the bot, as under any
		// notice: "resolve the merge conflict" there is the command.
		r.recordBotReply(ctx, key, id, store.BotReplyQuestion)
		r.log.InfoContext(ctx, fmt.Sprintf("%s: merge conflict offer: posted %s (comment %d)", prTag, kind, id))
	case kind == offerNone && open != 0:
		r.withdrawConflictOffer(ctx, key, prID, prTag, open)
	}
}

// withdrawConflictOffer resolves the open offer's thread and clears it. One
// a human deleted is cleared too; a failed resolve keeps it for the next
// review.
func (r *Reviewer) withdrawConflictOffer(ctx context.Context, key store.PRKey, prID int64, prTag string, open int) {
	err := r.bitbucket.ResolveCommentThread(ctx, key.Project, key.Repo, key.PRID, open)
	if err != nil && bitbucket.Status(err) != http.StatusNotFound {
		r.log.WarnContext(ctx, fmt.Sprintf("%s: merge conflict offer: resolving comment %d failed: %v", prTag, open, err))
		return
	}
	safeDBErr(ctx, r.log, "SetConflictOffer", func() error { return r.store.SetConflictOffer(ctx, prID, 0) })
	r.log.InfoContext(ctx, fmt.Sprintf("%s: merge conflict offer: comment %d resolved, no offer holds", prTag, open))
}

// conflictResolvedOffer withdraws the PR's open offer once noergler pushed
// the merge: the next review would, but a push it skips (no reviewable
// file) would leave the offer open.
func (r *Reviewer) conflictResolvedOffer(ctx context.Context, key store.PRKey, prTag string) {
	rowID, _, err := r.store.PRCapState(ctx, key)
	if err != nil || rowID == 0 {
		return
	}
	if open, err := r.store.ConflictOffer(ctx, rowID); err == nil && open != 0 {
		r.withdrawConflictOffer(ctx, key, rowID, prTag, open)
	}
}
