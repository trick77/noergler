package store

import (
	"context"
	"errors"
	"fmt"
	"time"
)

// Attempt is one terminal outcome, including the ones that write no run row.
// See migrations/0002_review_attempts.sql for why this is its own table.
type Attempt struct {
	Key      PRKey
	TeamSlug string
	Kind     RunKind
	// Outcome is inference.Outcome's String(), or "skipped" for a
	// pre-flight exit that never reached the gateway.
	Outcome string
	// Reason is a review.SkipReason. Empty unless Outcome is "skipped".
	Reason    string
	ElapsedMS *int64
	// RunID is set only when the attempt produced a run row.
	RunID *int64
	// CostNanoUSD is a billed call that wrote no run row (unparseable). A
	// successful attempt leaves it nil: its cost is on the run.
	CostNanoUSD *int64
	// Call is the gateway call this attempt made, written to llm_calls right
	// after the attempt and apart from it, so a missing call row can mean a
	// refused write, not only no call. Nil when the gateway never answered (a
	// pre-flight skip, a timeout); a late repo-unclaimed skip carries the call
	// it paid for.
	Call *LLMCall
	// LinkToken is the run page's key (NewLinkToken). Empty when the
	// attempt posts no summary to link it from: no token, no page.
	LinkToken string
}

// InsertAttempt records one attempt, and its gateway call when it made one.
// Returns the attempt id; the run detail page is keyed by LinkToken.
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
	var token *string
	if a.LinkToken != "" {
		token = &a.LinkToken
	}
	var id int64
	if err := s.pool.QueryRow(ctx, `
		INSERT INTO review_attempts
		    (team_slug, project_key, repo_slug, pr_id, kind,
		     outcome, reason, elapsed_ms, review_run_id, cost_nano_usd, link_token)
		VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
		RETURNING id`,
		a.TeamSlug, a.Key.Project, a.Key.Repo, a.Key.PRID, a.Kind,
		a.Outcome, reason, a.ElapsedMS, a.RunID, a.CostNanoUSD, token).Scan(&id); err != nil {
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

// AttemptRow is one row of the dashboard's feed: the attempt, plus the run
// figures where the attempt produced a run.
type AttemptRow struct {
	Attempt
	// ID is the attempt's row id. Never its link token: the feed links to
	// no run page.
	ID        int64
	CreatedAt time.Time
	// Findings and CostNanoUSD come from the joined run. Both nil when the
	// attempt produced none, and CostNanoUSD is nil for an unpriced run
	// too: the two cases are distinguished by Outcome, never by a zero.
	Findings    *int
	CostNanoUSD *int64
}

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
func (s *Store) RecentAttempts(ctx context.Context, team string, outcome AttemptFilter, limit int) ([]AttemptRow, error) {
	if limit <= 0 || limit > maxAttempts {
		limit = maxAttempts
	}
	rows, err := s.pool.Query(ctx, `
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
		 LIMIT $3`, team, string(outcome), limit)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	var out []AttemptRow
	for rows.Next() {
		var a AttemptRow
		if err := rows.Scan(&a.ID, &a.TeamSlug, &a.Key.Project, &a.Key.Repo, &a.Key.PRID,
			&a.Kind, &a.Outcome, &a.Reason, &a.ElapsedMS, &a.CreatedAt,
			&a.Findings, &a.CostNanoUSD); err != nil {
			return nil, err
		}
		out = append(out, a)
	}
	return out, rows.Err()
}

// OutcomeCount is one bucket of the outcome breakdown.
type OutcomeCount struct {
	Outcome string
	Reason  string
	Count   int
}

// OutcomeBreakdown counts attempts by outcome since a point in time. Skips
// are split by reason; every other outcome carries an empty Reason.
func (s *Store) OutcomeBreakdown(ctx context.Context, since time.Time) ([]OutcomeCount, error) {
	rows, err := s.pool.Query(ctx, `
		SELECT outcome, COALESCE(reason, ''), COUNT(*)
		  FROM review_attempts
		 WHERE created_at >= $1
		 GROUP BY outcome, COALESCE(reason, '')
		 ORDER BY COUNT(*) DESC`, since)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	var out []OutcomeCount
	for rows.Next() {
		var c OutcomeCount
		if err := rows.Scan(&c.Outcome, &c.Reason, &c.Count); err != nil {
			return nil, err
		}
		out = append(out, c)
	}
	return out, rows.Err()
}
