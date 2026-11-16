package store

import (
	"context"
	"time"

	"github.com/jackc/pgx/v5"
)

// TaskCountsSince counts the blocking tasks posted in the window per day,
// for the metrics page's chart beside the fix requests. rule_removed
// counts: MarkRuleRemoved overwrites a posted task's 'task' once its rule
// leaves AGENTS.md, and the task was posted all the same.
// downgraded_task_failed does not: no task reached Bitbucket. SNAPSHOT
// tasks are not here: snapshot_tasks keeps no time. Days without a task
// are absent. idx_findings_task_time keeps this off a seq scan of every
// finding ever posted.
func (s *Store) TaskCountsSince(ctx context.Context, since time.Time) ([]TaskCount, error) {
	return queryAll(ctx, s.pool, `
		SELECT date_trunc('day', created_at) AS day, COUNT(*)
		  FROM findings
		 WHERE created_at >= $1 AND blocking_outcome IN ('task', 'rule_removed')
		 GROUP BY day
		 ORDER BY day`, []any{since},
		func(row pgx.CollectableRow, c *TaskCount) error {
			return row.Scan(&c.Day, &c.Count)
		})
}
