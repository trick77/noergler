package store

import (
	"context"
	"time"

	"github.com/jackc/pgx/v5"
)

// Fix outcomes: what came of the PR author's request to have noergler apply
// a finding. One row per request; the metrics page counts them.
const (
	FixCommitted   = "committed"    // the change is on the source branch
	FixNotFeasible = "not_feasible" // the model said it cannot be done in one file
	FixRejected    = "rejected"     // the change failed ValidateFix
	FixNoWrite     = "no_write"     // Bitbucket refused the commit: 401/403
	FixConflict    = "conflict"     // the branch moved since the file was read: 409
	FixRefused     = "refused"      // any other refusal (a pre-receive hook)
	FixError       = "error"        // the call failed or nothing usable came back
)

// RecordFindingFix writes one fix request. A second committed row for the
// same finding is dropped (the partial unique index): inserted is false
// then, and the caller logs it.
func (s *Store) RecordFindingFix(ctx context.Context, x FindingFix) (inserted bool, err error) {
	var sha any
	if x.CommitSHA != "" {
		sha = x.CommitSHA
	}
	tag, err := s.pool.Exec(ctx, `
		INSERT INTO finding_fixes (finding_id, team_slug, outcome, commit_sha)
		VALUES ($1, $2, $3, $4)
		ON CONFLICT DO NOTHING`, x.FindingID, x.TeamSlug, x.Outcome, sha)
	if err != nil {
		return false, err
	}
	return tag.RowsAffected() == 1, nil
}

// FixCountsSince counts the fix requests in the window per team and
// outcome, for the metrics page's Auto-fixes column.
func (s *Store) FixCountsSince(ctx context.Context, since time.Time) ([]FixCount, error) {
	return queryAll(ctx, s.pool, `
		SELECT team_slug, outcome, COUNT(*)
		  FROM finding_fixes
		 WHERE created_at >= $1
		 GROUP BY team_slug, outcome
		 ORDER BY team_slug, outcome`, []any{since},
		func(row pgx.CollectableRow, c *FixCount) error {
			return row.Scan(&c.TeamSlug, &c.Outcome, &c.Count)
		})
}
