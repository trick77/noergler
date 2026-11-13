package store

import (
	"context"
	"errors"
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

// RecordFindingFix writes one fix request, of a finding or of a Q&A answer
// on k's PR. A second committed row for the same source is dropped (the
// partial unique indexes): inserted is false then, and the caller logs it.
// An answer's request on a PR without a row records nothing.
func (s *Store) RecordFindingFix(ctx context.Context, k PRKey, x FindingFix) (inserted bool, err error) {
	var sha, finding, answer any
	if x.CommitSHA != "" {
		sha = x.CommitSHA
	}
	if x.FindingID != 0 {
		finding = x.FindingID
	}
	if x.AnswerCommentID != 0 {
		answer = int64(x.AnswerCommentID)
	}
	tag, err := s.pool.Exec(ctx, `
		INSERT INTO finding_fixes (finding_id, answer_comment_id, pull_request_id, team_slug, outcome, commit_sha)
		SELECT $1::bigint, $2::bigint,
		       CASE WHEN $2::bigint IS NULL THEN NULL ELSE
		            (SELECT id FROM pull_requests WHERE project_key = $6 AND repo_slug = $7 AND pr_id = $8) END,
		       $3, $4, $5
		 WHERE $2::bigint IS NULL
		    OR EXISTS (SELECT 1 FROM pull_requests WHERE project_key = $6 AND repo_slug = $7 AND pr_id = $8)
		ON CONFLICT DO NOTHING`, finding, answer, x.TeamSlug, x.Outcome, sha, k.Project, k.Repo, k.PRID)
	if err != nil {
		return false, err
	}
	return tag.RowsAffected() == 1, nil
}

// AnswerFixCommit is the commit made for noergler's Q&A answer
// answerCommentID, "" when none.
func (s *Store) AnswerFixCommit(ctx context.Context, answerCommentID int) (string, error) {
	var sha string
	err := s.pool.QueryRow(ctx, `
		SELECT commit_sha FROM finding_fixes
		 WHERE answer_comment_id = $1 AND outcome = 'committed'`, int64(answerCommentID)).Scan(&sha)
	if errors.Is(err, pgx.ErrNoRows) {
		return "", nil
	}
	return sha, err
}

// FixCountsSince counts the fix requests in the window per day, team and
// outcome: one read for both the metrics page's fix columns (summed per
// team) and its chart (summed per day). Days without a request are absent;
// the caller fills the window, as for DailyByTeam.
func (s *Store) FixCountsSince(ctx context.Context, since time.Time) ([]FixCount, error) {
	return queryAll(ctx, s.pool, `
		SELECT date_trunc('day', created_at) AS day, team_slug, outcome, COUNT(*)
		  FROM finding_fixes
		 WHERE created_at >= $1
		 GROUP BY day, team_slug, outcome
		 ORDER BY day, team_slug, outcome`, []any{since},
		func(row pgx.CollectableRow, c *FixCount) error {
			return row.Scan(&c.Day, &c.TeamSlug, &c.Outcome, &c.Count)
		})
}
