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
	ActivityTask    = "task"    // a blocking task: a review's or a SNAPSHOT task
	// A fix request split by what came of it: the Fixes view stacks the two.
	ActivityFixCommitted = "fix_committed"
	ActivityFixOther     = "fix_other"
	// ActivityEngagement is a comment that tags noergler or replies to one
	// of its comments, whatever came of it; a fix request or a review asked
	// by mention is not one, it counts as a fix or a run.
	ActivityEngagement = "engagement"
)

// Engagement triggers and outcomes (activity_events.trigger, .outcome).
const (
	EngagementMention = "mention" // the comment tags noergler
	EngagementReply   = "reply"   // the comment replies to one of noergler's

	EngagementAnswered   = "answered"
	EngagementAuthorOnly = "author_only" // an order only the PR's author may give
	EngagementRefused    = "refused"     // the PR is at its cost cap
	EngagementFailed     = "failed"
)

// RecordEngagement counts comment commentID on k's PR as an engagement. A
// replayed comment is kept once; a failed one handled again (a redelivery)
// takes the later outcome, any other stays as first recorded. noRow
// reports a PR without a row, which records nothing: the caller writes the
// row and asks again.
func (s *Store) RecordEngagement(ctx context.Context, k PRKey, commentID int, trigger, outcome string) (noRow bool, err error) {
	err = s.pool.QueryRow(ctx, `
		WITH pr AS (SELECT id FROM pull_requests WHERE project_key = $1 AND repo_slug = $2 AND pr_id = $3),
		ins AS (
			INSERT INTO activity_events (pull_request_id, kind, comment_id, trigger, outcome)
			SELECT id, 'engagement', $4, $5, $6 FROM pr
			ON CONFLICT (pull_request_id, kind, comment_id) DO UPDATE SET outcome = EXCLUDED.outcome
			 WHERE activity_events.outcome = 'failed'
		)
		SELECT NOT EXISTS (SELECT 1 FROM pr)`, k.Project, k.Repo, k.PRID, int64(commentID), trigger, outcome).Scan(&noRow)
	return noRow, err
}

// RecordSnapshotTasks counts the SNAPSHOT tasks ids, posted or taken over
// on the PR row prID, as tasks. One already counted is kept once.
func (s *Store) RecordSnapshotTasks(ctx context.Context, prID int64, ids []int) error {
	if len(ids) == 0 {
		return nil
	}
	ids64 := make([]int64, len(ids))
	for i, id := range ids {
		ids64[i] = int64(id)
	}
	_, err := s.pool.Exec(ctx, `
		INSERT INTO activity_events (pull_request_id, kind, comment_id)
		SELECT $1, 'snapshot_task', unnest($2::bigint[])
		ON CONFLICT DO NOTHING`, prID, ids64)
	return err
}

// ActivitySince counts noergler's acts in the window per day, team and
// kind (the Activity kinds above). Days, teams and kinds with nothing are
// absent; the caller fills the window.
//
// Findings and tasks come off the runs that posted them (findings_posted,
// tasks_posted), fix requests off finding_fixes: every request, whatever
// came of it. SNAPSHOT tasks and engagements come off activity_events, a
// SNAPSHOT task counted as a task. The team is the PR's
// (webhook-authenticated), the fix request's own for a fix.
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
			UNION ALL
			SELECT date_trunc('day', e.created_at), p.team_slug,
			       CASE WHEN e.kind = 'snapshot_task' THEN 'task' ELSE 'engagement' END, 1
			  FROM activity_events e JOIN pull_requests p ON p.id = e.pull_request_id
			 WHERE e.created_at >= $1
		  ) acts
		 GROUP BY day, team_slug, kind
		 ORDER BY day, team_slug, kind`, []any{since},
		func(row pgx.CollectableRow, c *ActivityCount) error {
			return row.Scan(&c.Day, &c.TeamSlug, &c.Kind, &c.Count)
		})
}
