package store

import (
	"context"
	"errors"
	"time"

	"github.com/jackc/pgx/v5"
)

// PRStatus is what noergler knows about one PR: the answer to "why was my
// PR (not) reviewed?". Read-only, and nothing in it is author text: no
// finding, no dispute.
type PRStatus struct {
	// Known is false when noergler holds no row for the PR yet; the
	// attempts may still say why (a skip decided before any row).
	Known              bool
	LastReviewedCommit *string
	IgnoredAt          *time.Time
	MergedAt           *time.Time
	DeclinedAt         *time.Time
	Runs               int
	LastRun            *RunBrief
	// Attempts are the latest outcomes, newest first.
	Attempts []AttemptBrief
}

// RunBrief is one review run, without its tokens or cost.
type RunBrief struct {
	At          time.Time
	Kind        RunKind
	Incremental bool
	Findings    int
	ModelLabel  string
}

// AttemptBrief is one terminal outcome.
type AttemptBrief struct {
	At      time.Time
	Kind    RunKind
	Outcome string
	// Reason is the review.SkipReason value, empty unless Outcome is
	// "skipped".
	Reason string
}

// statusAttempts is how many outcomes PRStatus reads: enough to explain the
// latest push, not a history.
const statusAttempts = 5

// PRStatus reads the PR's state, its runs and its latest attempts, the
// team's only and from this row's life only, as prCostSQL bounds them: a
// purged repo claimed again, maybe by another team, starts from nothing.
func (s *Store) PRStatus(ctx context.Context, k PRKey, team string) (*PRStatus, error) {
	st := &PRStatus{}
	var id int64
	// No row: every attempt of the team's on this key counts.
	since := time.Time{}
	err := s.pool.QueryRow(ctx, `
		SELECT id, created_at, last_reviewed_commit, ignored_at, merged_at, declined_at FROM pull_requests
		WHERE project_key = $1 AND repo_slug = $2 AND pr_id = $3`,
		k.Project, k.Repo, k.PRID).Scan(&id, &since, &st.LastReviewedCommit, &st.IgnoredAt, &st.MergedAt, &st.DeclinedAt)
	switch {
	case errors.Is(err, pgx.ErrNoRows):
	case err != nil:
		return nil, err
	default:
		st.Known = true
		if err := s.pool.QueryRow(ctx, `SELECT count(*) FROM review_runs WHERE pull_request_id = $1`, id).
			Scan(&st.Runs); err != nil {
			return nil, err
		}
		var r RunBrief
		err := s.pool.QueryRow(ctx, `
			SELECT created_at, kind, incremental, findings_posted, model_label FROM review_runs
			WHERE pull_request_id = $1 ORDER BY created_at DESC, id DESC LIMIT 1`, id).
			Scan(&r.At, &r.Kind, &r.Incremental, &r.Findings, &r.ModelLabel)
		if err != nil && !errors.Is(err, pgx.ErrNoRows) {
			return nil, err
		}
		if err == nil {
			st.LastRun = &r
		}
	}

	rows, err := s.pool.Query(ctx, `
		SELECT created_at, kind, outcome, COALESCE(reason, '') FROM review_attempts
		WHERE project_key = $1 AND repo_slug = $2 AND pr_id = $3 AND team_slug = $4 AND created_at >= $5
		ORDER BY created_at DESC, id DESC LIMIT $6`,
		k.Project, k.Repo, k.PRID, team, since, statusAttempts)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	for rows.Next() {
		var a AttemptBrief
		if err := rows.Scan(&a.At, &a.Kind, &a.Outcome, &a.Reason); err != nil {
			return nil, err
		}
		st.Attempts = append(st.Attempts, a)
	}
	return st, rows.Err()
}
