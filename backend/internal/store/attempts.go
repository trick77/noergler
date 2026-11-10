package store

import (
	"context"
	"errors"
	"fmt"
	"time"

	"github.com/jackc/pgx/v5"
)

// InsertAttempt records one attempt, and its gateway call when it made one.
// Returns the attempt id.
//
// The caller swallows the error: an attempt row is a dashboard record, and
// failing a review over one would trade a working review for a log line.
//
// Two writes, not one transaction: the attempt carries billed cost that
// PRCost and the cap read, and a call row the database refuses (a header
// it cannot store) must never take that cost down with it. A failed call
// write returns the attempt id and ErrCallNotRecorded, so the caller logs it
// as what it is.
func (s *Store) InsertAttempt(ctx context.Context, a Attempt) (int64, error) {
	var reason *string
	if a.Reason != "" {
		r := a.Reason
		reason = &r
	}
	var id int64
	if err := s.pool.QueryRow(ctx, `
		INSERT INTO review_attempts
		    (team_slug, project_key, repo_slug, pr_id, kind,
		     outcome, reason, elapsed_ms, review_run_id, cost_nano_usd)
		VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
		RETURNING id`,
		a.TeamSlug, a.Key.Project, a.Key.Repo, a.Key.PRID, a.Kind,
		a.Outcome, reason, a.ElapsedMS, a.RunID, a.CostNanoUSD).Scan(&id); err != nil {
		return 0, err
	}
	if a.Call != nil {
		if err := s.insertLLMCall(ctx, id, a, *a.Call); err != nil {
			return id, fmt.Errorf("attempt %d: %w: %w", id, ErrCallNotRecorded, err)
		}
	}
	return id, nil
}

// ErrCallNotRecorded is InsertAttempt's error when the attempt (and any
// billed cost on it) was written and only its gateway call was not.
var ErrCallNotRecorded = errors.New("gateway call not recorded (attempt written)")

// AttemptFilter narrows the feed by outcome. The zero value is every row.
//
// Failures are "not ok and not skipped" rather than a list of the failing
// outcomes: the set grows (timed_out, too_large, unparseable, error), and a
// filter that enumerates them silently drops whatever a newer binary writes.
type AttemptFilter string

const (
	// AttemptsAll is every outcome.
	AttemptsAll AttemptFilter = ""
	// AttemptsFailed is every outcome that is neither ok nor skipped.
	AttemptsFailed AttemptFilter = "failed"
	// AttemptsSkipped is the pre-flight exits.
	AttemptsSkipped AttemptFilter = "skipped"
)

// maxAttempts caps the feed. A request may ask for less, never for more: the
// rows are unauthenticated and the page shows a window, not an archive.
const maxAttempts = 100

// RecentAttempts is the feed, newest first. team empty means every team.
//
// The outcome filter runs in SQL, not over the returned page: filtering a
// 100-row window client-side can come back empty while failures sit just
// past its edge, which reads as "nothing failed" rather than "look further".
// The dashboard's Runs page no longer asks for it: it filters the latest
// runs it shows, on purpose (user, 2026-11-10). The API keeps ?outcome= for
// a failure older than those.
func (s *Store) RecentAttempts(ctx context.Context, team string, outcome AttemptFilter, limit int) ([]AttemptRow, error) {
	if limit <= 0 || limit > maxAttempts {
		limit = maxAttempts
	}
	return queryAll(ctx, s.pool, `
		SELECT a.id, a.team_slug, a.project_key, a.repo_slug, a.pr_id, a.kind,
		       a.outcome, COALESCE(a.reason, ''), a.elapsed_ms, a.created_at,
		       r.findings_posted, r.cost_nano_usd
		  FROM review_attempts a
		  LEFT JOIN review_runs r ON r.id = a.review_run_id
		 WHERE ($1 = '' OR a.team_slug = $1)
		   AND ($2 = ''
		        OR ($2 = 'skipped' AND a.outcome = 'skipped')
		        OR ($2 = 'failed' AND a.outcome NOT IN ('ok', 'skipped')))
		 ORDER BY a.created_at DESC
		 LIMIT $3`, []any{team, string(outcome), limit},
		func(row pgx.CollectableRow, a *AttemptRow) error {
			return row.Scan(&a.ID, &a.TeamSlug, &a.Key.Project, &a.Key.Repo, &a.Key.PRID,
				&a.Kind, &a.Outcome, &a.Reason, &a.ElapsedMS, &a.CreatedAt,
				&a.Findings, &a.CostNanoUSD)
		})
}

// OutcomeBreakdown counts attempts by outcome since a point in time. Skips
// are split by reason; every other outcome carries an empty Reason.
func (s *Store) OutcomeBreakdown(ctx context.Context, since time.Time) ([]OutcomeCount, error) {
	return queryAll(ctx, s.pool, `
		SELECT outcome, COALESCE(reason, ''), COUNT(*)
		  FROM review_attempts
		 WHERE created_at >= $1
		 GROUP BY outcome, COALESCE(reason, '')
		 ORDER BY COUNT(*) DESC`, []any{since},
		func(row pgx.CollectableRow, c *OutcomeCount) error {
			return row.Scan(&c.Outcome, &c.Reason, &c.Count)
		})
}
