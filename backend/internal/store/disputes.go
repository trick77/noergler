package store

import (
	"context"
	"errors"

	"github.com/jackc/pgx/v5"
)

// Dispute signals, migrations/0025_dispute_signals.sql. Read and written on
// the finding-reply path only. Every read counts inside the team's window
// (days); an occurrence outside it is kept and ignored.

// DisputeSignal is one open signal of a repo: its id and the rule as the
// newest occurrence in the window worded it.
type DisputeSignal struct {
	ID   int64
	Rule string
}

// DisputePR is one PR a signal occurred on inside the window: the Bitbucket
// PR number, the disputed finding's comment (nil if never posted) and the PR
// author.
type DisputePR struct {
	PRID      int
	CommentID *int
	Author    string
}

// OpenDisputeSignals is the repo's signals with an occurrence in the window,
// most recently disputed first, at most limit. Scoped by the route's team
// as well as the repo: a repo that changed hands starts without signals.
func (s *Store) OpenDisputeSignals(ctx context.Context, team, project, repo string, windowDays, limit int) ([]DisputeSignal, error) {
	rows, err := s.pool.Query(ctx, `
		SELECT s.id, o.rule
		FROM dispute_signals s
		JOIN LATERAL (
			SELECT rule, created_at FROM dispute_occurrences
			WHERE signal_id = s.id AND created_at > now() - make_interval(days => $4::int)
			ORDER BY created_at DESC LIMIT 1
		) o ON true
		WHERE s.team_slug = $1 AND s.project_key = $2 AND s.repo_slug = $3
		ORDER BY o.created_at DESC, s.id DESC
		LIMIT $5`,
		team, project, repo, windowDays, limit)
	if err != nil {
		return nil, err
	}
	return pgx.CollectRows(rows, func(r pgx.CollectableRow) (DisputeSignal, error) {
		var d DisputeSignal
		err := r.Scan(&d.ID, &d.Rule)
		return d, err
	})
}

// DisputeSignalPRs is every PR but exclude the signal occurred on inside the
// window, most recent first: several disputes in one PR are one PR. Merged
// PRs count; the evidence is the dispute, not the PR's state.
func (s *Store) DisputeSignalPRs(ctx context.Context, signalID int64, windowDays int, exclude PRKey) ([]DisputePR, error) {
	rows, err := s.pool.Query(ctx, `
		SELECT p.pr_id,
			(array_agg(f.bitbucket_comment_id ORDER BY o.created_at DESC))[1],
			COALESCE(p.author, '')
		FROM dispute_occurrences o
		JOIN findings f ON f.id = o.finding_id
		JOIN pull_requests p ON p.id = f.pull_request_id
		WHERE o.signal_id = $1 AND o.created_at > now() - make_interval(days => $2::int)
			AND NOT (p.project_key = $3 AND p.repo_slug = $4 AND p.pr_id = $5)
		GROUP BY p.id, p.pr_id, p.author
		ORDER BY max(o.created_at) DESC, p.pr_id DESC`,
		signalID, windowDays, exclude.Project, exclude.Repo, exclude.PRID)
	if err != nil {
		return nil, err
	}
	return pgx.CollectRows(rows, func(r pgx.CollectableRow) (DisputePR, error) {
		var d DisputePR
		err := r.Scan(&d.PRID, &d.CommentID, &d.Author)
		return d, err
	})
}

// RecordDisputeOccurrence stores one dispute of a finding under signalID, or
// under a new signal of the team's repo when signalID is 0, in one
// transaction. A signalID that is not the team's repo's records nothing. A
// finding is disputed once per signal. Returns the signal's id, 0 when
// nothing was recorded.
func (s *Store) RecordDisputeOccurrence(ctx context.Context, team string, k PRKey, signalID, findingID int64, rule string) (int64, error) {
	var id int64
	err := s.inTx(ctx, func(tx pgx.Tx) error {
		if signalID == 0 {
			if err := tx.QueryRow(ctx, `
				INSERT INTO dispute_signals (team_slug, project_key, repo_slug)
				VALUES ($1, $2, $3) RETURNING id`, team, k.Project, k.Repo).Scan(&id); err != nil {
				return err
			}
		} else {
			err := tx.QueryRow(ctx, `
				SELECT id FROM dispute_signals
				WHERE id = $1 AND team_slug = $2 AND project_key = $3 AND repo_slug = $4`,
				signalID, team, k.Project, k.Repo).Scan(&id)
			if errors.Is(err, pgx.ErrNoRows) {
				id = 0
				return nil
			}
			if err != nil {
				return err
			}
		}
		_, err := tx.Exec(ctx, `
			INSERT INTO dispute_occurrences (signal_id, finding_id, rule)
			VALUES ($1, $2, $3) ON CONFLICT DO NOTHING`, id, findingID, rule)
		return err
	})
	if err != nil {
		return 0, err
	}
	return id, nil
}
