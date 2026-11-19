package store

import (
	"context"
	"time"

	"github.com/jackc/pgx/v5"
)

// What became of a finding, first match wins: a fix (a later review's
// checked fixed claim, or a fix noergler committed), the author's dispute,
// its thread resolved without either, or still open.
//
// Every state but a committed fix is what the last review saw: a thread is
// stamped resolved only when a review reads it, so one resolved after the
// last review reads open. A deleted thread is never stamped (a 404 settles
// it for one run only) and reads open too. A dispute is the first answered
// reply's; a later one is not stored.
const (
	OutcomeFixed    = "fixed"
	OutcomeDisputed = "disputed"
	OutcomeClosed   = "closed"
	OutcomeOpen     = "open"
)

// Confidence bands. The review prompt reports nothing under 80, so the
// lowest band is everything below 90; BandUnknown is a row with no
// confidence stored.
const (
	BandLow     = "<90"
	BandMid     = "90-94"
	BandHigh    = "95+"
	BandUnknown = "unknown"
)

// FindingOutcomesSince counts the findings of PRs merged in the window by
// team, severity, confidence band and outcome. Merged only: an open PR's
// findings may still be fixed, and a declined or deleted PR says nothing
// about whether a finding was right.
//
// Inline findings only, never a blocking task (a post is a finding or a
// task, never both, in every count): a task's state is Bitbucket's task
// list, not a thread. A legacy row with no comment id is left out (no
// thread anyone could resolve), as is a task noergler resolved because its
// rule left AGENTS.md.
//
// Read only, for the metrics page: never a prompt, never a filter on
// findings, never a cost.
func (s *Store) FindingOutcomesSince(ctx context.Context, since time.Time) ([]FindingOutcomeCount, error) {
	return queryAll(ctx, s.pool, `
		SELECT p.team_slug, f.severity,
		       CASE WHEN f.confidence IS NULL THEN $2
		            WHEN f.confidence >= 95 THEN $3
		            WHEN f.confidence >= 90 THEN $4
		            ELSE $5 END AS band,
		       CASE WHEN f.fixed_at IS NOT NULL
		              OR EXISTS (SELECT 1 FROM finding_fixes x
		                          WHERE x.finding_id = f.id AND x.outcome = 'committed') THEN $6
		            WHEN f.dispute_text IS NOT NULL THEN $7
		            WHEN f.closed_at IS NOT NULL THEN $8
		            ELSE $9 END AS outcome,
		       COUNT(*)
		  FROM findings f
		  JOIN pull_requests p ON p.id = f.pull_request_id
		 WHERE p.merged_at >= $1
		   AND f.bitbucket_comment_id IS NOT NULL
		   AND f.blocking_outcome IS DISTINCT FROM 'task'
		   AND f.blocking_outcome IS DISTINCT FROM 'rule_removed'
		 GROUP BY 1, 2, 3, 4
		 ORDER BY 1, 2, 3, 4`,
		[]any{since, BandUnknown, BandHigh, BandMid, BandLow, OutcomeFixed, OutcomeDisputed, OutcomeClosed, OutcomeOpen},
		func(row pgx.CollectableRow, c *FindingOutcomeCount) error {
			return row.Scan(&c.TeamSlug, &c.Severity, &c.Band, &c.Outcome, &c.Count)
		})
}
