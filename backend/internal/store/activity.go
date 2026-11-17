package store

import (
	"context"
	"time"

	"github.com/jackc/pgx/v5"
)

// Activity kinds: what noergler did, each act counted under exactly one
// (user, 2026-11-17: a post is a finding or a task, never both). The
// metrics page charts one kind at a time; a new kind is one more branch in
// ActivitySince, never a new chart or field.
const (
	ActivityFinding = "finding" // an inline comment a review posted
	ActivityTask    = "task"    // a blocking task a review posted
	// A fix request split by what came of it: the Fixes view stacks the two.
	ActivityFixCommitted = "fix_committed"
	ActivityFixOther     = "fix_other"
)

// ActivitySince counts noergler's acts in the window per day, team and
// kind (the Activity kinds above). Days, teams and kinds with nothing are
// absent; the caller fills the window.
//
// Findings and tasks come off the runs that posted them (findings_posted,
// tasks_posted), fix requests off finding_fixes: every request, whatever
// came of it. The team is the PR's (webhook-authenticated) for a run, the
// request's own for a fix.
func (s *Store) ActivitySince(ctx context.Context, since time.Time) ([]ActivityCount, error) {
	return queryAll(ctx, s.pool, `
		SELECT day, team_slug, kind, SUM(n)::int
		  FROM (
			SELECT date_trunc('day', r.created_at) AS day, p.team_slug, 'finding' AS kind, r.findings_posted AS n
			  FROM review_runs r JOIN pull_requests p ON p.id = r.pull_request_id
			 WHERE r.created_at >= $1 AND r.findings_posted > 0
			UNION ALL
			SELECT date_trunc('day', r.created_at), p.team_slug, 'task', r.tasks_posted
			  FROM review_runs r JOIN pull_requests p ON p.id = r.pull_request_id
			 WHERE r.created_at >= $1 AND r.tasks_posted > 0
			UNION ALL
			SELECT date_trunc('day', created_at), team_slug,
			       CASE WHEN outcome = 'committed' THEN 'fix_committed' ELSE 'fix_other' END, 1
			  FROM finding_fixes
			 WHERE created_at >= $1
		  ) acts
		 GROUP BY day, team_slug, kind
		 ORDER BY day, team_slug, kind`, []any{since},
		func(row pgx.CollectableRow, c *ActivityCount) error {
			return row.Scan(&c.Day, &c.TeamSlug, &c.Kind, &c.Count)
		})
}
