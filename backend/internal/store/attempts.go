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

// Acts in the feed beside the reviews (AttemptRow.Act).
const (
	ActReview     = "review"
	ActEngagement = "engagement"
	ActFix        = "fix"
	ActTask       = "task" // a SNAPSHOT task; a review's tasks ride its row
)

// RecentAttempts is the feed, newest first: every review attempt and every
// other act the metrics count (user, 2026-11-17: "eigene Einträge"), an
// engagement, a fix or merge-conflict request, a SNAPSHOT task, each its
// own row. team empty means every team.
//
// The outcome filter runs in SQL, not over the returned page: filtering a
// 100-row window client-side can come back empty while failures sit just
// past its edge, which reads as "nothing failed" rather than "look further".
// The dashboard's Runs page no longer asks for it: it filters the latest
// runs it shows, on purpose (user, 2026-11-10). The API keeps ?outcome= for
// a failure older than those. Skipped is a review's only; failed is a
// review's failure, a failed engagement and a fix that went wrong: its call
// failed, its change was rejected as unsafe, or Bitbucket refused the
// commit. "Not feasible" is an answer, not a failure.
//
// An engagement's duration and cost are its own calls' (llm_calls by its
// comment, the fix call aside). One unpriced call makes the whole cost
// unpriced (NULL with Unpriced set), never a partial sum passed off as
// the price: cost fails open. A fix's call row names no request, so a fix
// row shows none.
//
// Each source is filtered, sorted and cut to the limit on its own, by its
// time index, before the merge: the page polls every 10 s, and an unbounded
// branch would sort its whole history each time.
func (s *Store) RecentAttempts(ctx context.Context, team string, outcome AttemptFilter, limit int) ([]AttemptRow, error) {
	if limit <= 0 || limit > maxAttempts {
		limit = maxAttempts
	}
	return queryAll(ctx, s.pool, `
		SELECT act, id, team_slug, project_key, repo_slug, pr_id, kind, outcome, reason, elapsed_ms,
		       created_at, findings, tasks, cost, unpriced, detail, commit_sha
		  FROM (
			(SELECT 'review' AS act, a.id, a.team_slug, a.project_key, a.repo_slug, a.pr_id, a.kind,
			        a.outcome, COALESCE(a.reason, '') AS reason, a.elapsed_ms, a.created_at,
			        r.findings_posted AS findings, r.tasks_posted AS tasks, r.cost_nano_usd AS cost,
			        false AS unpriced, '' AS detail, NULL::text AS commit_sha
			   FROM review_attempts a
			   LEFT JOIN review_runs r ON r.id = a.review_run_id
			  WHERE ($1 = '' OR a.team_slug = $1)
			    AND ($2 = '' OR ($2 = 'skipped' AND a.outcome = 'skipped')
			         OR ($2 = 'failed' AND a.outcome NOT IN ('ok', 'skipped')))
			  ORDER BY a.created_at DESC
			  LIMIT $3)
			UNION ALL
			(SELECT CASE WHEN e.kind = 'engagement' THEN 'engagement' ELSE 'task' END, e.comment_id,
			        p.team_slug, p.project_key, p.repo_slug, p.pr_id, '',
			        COALESCE(e.outcome, 'posted'), '', c.ms, e.created_at, NULL, NULL,
			        CASE WHEN c.unpriced THEN NULL ELSE c.cost END, COALESCE(c.unpriced, false),
			        COALESCE(e.trigger, ''), NULL
			   FROM activity_events e
			   JOIN pull_requests p ON p.id = e.pull_request_id
			   LEFT JOIN LATERAL (
				SELECT SUM(l.total_ms)::bigint AS ms, SUM(l.cost_nano_usd)::bigint AS cost,
				       bool_or(l.cost_nano_usd IS NULL) AS unpriced
				  FROM llm_calls l
				 WHERE e.kind = 'engagement' AND l.project_key = p.project_key AND l.repo_slug = p.repo_slug
				   AND l.pr_id = p.pr_id AND l.comment_id = e.comment_id AND l.kind <> 'fix'
			   ) c ON true
			  WHERE ($1 = '' OR p.team_slug = $1)
			    AND $2 <> 'skipped'
			    AND ($2 <> 'failed' OR e.outcome = 'failed')
			  ORDER BY e.created_at DESC
			  LIMIT $3)
			UNION ALL
			(SELECT 'fix', f.id, f.team_slug, p.project_key, p.repo_slug, p.pr_id, '',
			        f.outcome, '', NULL, f.created_at, NULL, NULL, NULL, false,
			        CASE WHEN f.conflict_comment_id IS NOT NULL THEN 'conflict'
			             WHEN f.answer_comment_id IS NOT NULL THEN 'answer' ELSE 'finding' END,
			        f.commit_sha
			   FROM finding_fixes f
			   LEFT JOIN findings fd ON fd.id = f.finding_id
			   JOIN pull_requests p ON p.id = COALESCE(f.pull_request_id, fd.pull_request_id)
			  WHERE ($1 = '' OR f.team_slug = $1)
			    AND $2 <> 'skipped'
			    AND ($2 <> 'failed' OR f.outcome IN ('error', 'conflict', 'refused', 'no_write', 'rejected'))
			  ORDER BY f.created_at DESC
			  LIMIT $3)
		  ) acts
		 ORDER BY created_at DESC
		 LIMIT $3`, []any{team, string(outcome), limit},
		func(row pgx.CollectableRow, a *AttemptRow) error {
			return row.Scan(&a.Act, &a.ID, &a.TeamSlug, &a.Key.Project, &a.Key.Repo, &a.Key.PRID,
				&a.Kind, &a.Outcome, &a.Reason, &a.ElapsedMS, &a.CreatedAt,
				&a.Findings, &a.Tasks, &a.CostNanoUSD, &a.Unpriced, &a.Detail, &a.Commit)
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
