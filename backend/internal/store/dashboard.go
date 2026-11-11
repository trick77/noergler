package store

import (
	"context"
	"time"

	"github.com/jackc/pgx/v5"
)

// Every aggregate the store had before this file is keyed to one PRKey, and
// ClaimRollup is destructive: it stamps riptide_emitted_at in the statement
// that reads the snapshot. The dashboard needs read-only, time-windowed
// siblings, so they live here rather than being bolted onto those.
//
// Cost is BIGINT nano-USD throughout and NULL means unpriced. SUM() skips
// NULLs, so Unpriced is counted separately and NEVER folded in as zero: a
// run the gateway did not price is not a free run.
//
// Spend is runs plus billed run-less attempts (an unparseable response, a
// review stopped by a removed claim) plus reply and question calls: the same
// money prCostSQL counts per PR. Read off runs alone, the monthly figure fell
// short of the key's real spend. The attempts and calls add cost only, never
// a run, a token or a finding.
//
// Lines reviewed (lines_added, lines_removed) are the run's
// reviewed_lines_*, the files the prompt carried, counted once per PR: an
// incremental run's lines are new, and so are the PR's first run's; any
// other full run (a mention, a rebase or force push, an incremental diff
// that would not fetch) re-reads lines already counted and adds nothing.
// The gap is known and leans low, never high: new commits under a rebase's
// full re-review are not counted. NULL (a run from before the column)
// counts nothing; lines_uncounted counts those runs, so the page can say a
// window reaching back past migration 0029 reads low.
//
// k.counted: incremental, or the PR's first run ever (not merely the
// window's). OFFSET 0 keeps the planner from pulling the subquery up and
// pasting its NOT EXISTS into both CASEs, which would probe twice a row.
const spendCTE = `
	WITH spend AS (
		SELECT p.team_slug, r.created_at, r.cost_nano_usd, 1 AS run, r.prompt_tokens,
		       r.cached_tokens, r.completion_tokens, r.findings_posted,
		       CASE WHEN k.counted THEN COALESCE(r.reviewed_lines_added, 0) ELSE 0 END AS lines_added,
		       CASE WHEN k.counted THEN COALESCE(r.reviewed_lines_removed, 0) ELSE 0 END AS lines_removed,
		       CASE WHEN r.reviewed_lines_added IS NULL THEN 1 ELSE 0 END AS lines_uncounted
		  FROM review_runs r
		  JOIN pull_requests p ON p.id = r.pull_request_id
		 CROSS JOIN LATERAL (
		       SELECT r.incremental OR NOT EXISTS (
		              SELECT 1 FROM review_runs e
		               WHERE e.pull_request_id = r.pull_request_id AND e.id < r.id) AS counted
		       OFFSET 0) k
		UNION ALL
		SELECT a.team_slug, a.created_at, a.cost_nano_usd, 0, 0, 0, 0, 0, 0, 0, 0
		  FROM review_attempts a
		 WHERE a.review_run_id IS NULL AND a.cost_nano_usd IS NOT NULL
		UNION ALL
		SELECT c.team_slug, c.created_at, c.cost_nano_usd, 0, 0, 0, 0, 0, 0, 0, 0
		  FROM llm_calls c
		 WHERE c.attempt_id IS NULL AND c.cost_nano_usd IS NOT NULL
	)`

const totalsCols = `
	COALESCE(SUM(run), 0),
	COALESCE(SUM(prompt_tokens), 0),
	COALESCE(SUM(cached_tokens), 0),
	COALESCE(SUM(completion_tokens), 0),
	COALESCE(SUM(findings_posted), 0),
	SUM(cost_nano_usd),
	COUNT(*) FILTER (WHERE run = 1 AND cost_nano_usd IS NULL),
	COALESCE(SUM(lines_added), 0),
	COALESCE(SUM(lines_removed), 0),
	COALESCE(SUM(lines_uncounted), 0)`

// TotalsSince aggregates every team's runs in the window.
func (s *Store) TotalsSince(ctx context.Context, since time.Time) (Totals, error) {
	var t Totals
	err := s.pool.QueryRow(ctx, spendCTE+`
		SELECT`+totalsCols+`
		  FROM spend
		 WHERE created_at >= $1`, since).
		Scan(&t.Runs, &t.PromptTokens, &t.CachedTokens, &t.CompletionTokens,
			&t.FindingsPosted, &t.CostNanoUSD, &t.Unpriced, &t.LinesAdded, &t.LinesRemoved,
			&t.LinesUncounted)
	return t, err
}

// TotalsByTeam is TotalsSince grouped by the team that owns the PR. The slug
// comes off pull_requests, which the webhook route authenticated.
func (s *Store) TotalsByTeam(ctx context.Context, since time.Time) ([]TeamTotals, error) {
	return queryAll(ctx, s.pool, spendCTE+`
		SELECT team_slug,`+totalsCols+`
		  FROM spend
		 WHERE created_at >= $1
		 GROUP BY team_slug
		 ORDER BY SUM(cost_nano_usd) DESC NULLS LAST`, []any{since},
		func(row pgx.CollectableRow, t *TeamTotals) error {
			return row.Scan(&t.TeamSlug, &t.Runs, &t.PromptTokens, &t.CachedTokens,
				&t.CompletionTokens, &t.FindingsPosted, &t.CostNanoUSD, &t.Unpriced,
				&t.LinesAdded, &t.LinesRemoved, &t.LinesUncounted)
		})
}

// PRCountsSince counts distinct PRs in the window: reviewed is those with at
// least one run (an LLM review that came back), seen is every PR noergler
// touched, whatever the outcome. Runs count per push and per mention, so a
// busy PR inflated "reviewed" while one skipped for a missing precondition
// never showed up at all.
//
// Runs join the seen side too: the attempt write fails open, so a reviewed
// PR can lack an attempt row in the window. Read off attempts alone,
// reviewed could exceed seen.
func (s *Store) PRCountsSince(ctx context.Context, since time.Time) (reviewed, seen int, err error) {
	err = s.pool.QueryRow(ctx, `
		WITH prs AS (
			SELECT project_key, repo_slug, pr_id, false AS reviewed
			  FROM review_attempts
			 WHERE created_at >= $1
			UNION ALL
			SELECT p.project_key, p.repo_slug, p.pr_id, true
			  FROM review_runs r
			  JOIN pull_requests p ON p.id = r.pull_request_id
			 WHERE r.created_at >= $1
		)
		SELECT COUNT(DISTINCT (project_key, repo_slug, pr_id)) FILTER (WHERE reviewed),
		       COUNT(DISTINCT (project_key, repo_slug, pr_id))
		  FROM prs`, since).Scan(&reviewed, &seen)
	return reviewed, seen, err
}

// DailyByTeam is runs, findings, cost, unpriced runs and lines reviewed
// (once per PR, as spendCTE counts them) per day per team, for the charts
// and the tiles' trends. Days with no runs are absent rather than zero-filled: the
// caller knows the window it asked for and fills the gaps, which keeps this
// query from inventing rows.
func (s *Store) DailyByTeam(ctx context.Context, since time.Time) ([]Bucket, error) {
	return queryAll(ctx, s.pool, spendCTE+`
		SELECT date_trunc('day', created_at) AS day,
		       team_slug,
		       COALESCE(SUM(run), 0),
		       COALESCE(SUM(findings_posted), 0),
		       SUM(cost_nano_usd),
		       COUNT(*) FILTER (WHERE run = 1 AND cost_nano_usd IS NULL),
		       COALESCE(SUM(lines_added + lines_removed), 0)
		  FROM spend
		 WHERE created_at >= $1
		 GROUP BY day, team_slug
		 ORDER BY day`, []any{since},
		func(row pgx.CollectableRow, b *Bucket) error {
			return row.Scan(&b.Day, &b.TeamSlug, &b.Runs, &b.FindingsPosted, &b.CostNanoUSD, &b.Unpriced, &b.Lines)
		})
}

// DailyAttempts is attempts per day per outcome, for the throughput chart.
// It reads review_attempts, so a day's failures and skips are visible where
// DailyByTeam sees only the runs that succeeded.
func (s *Store) DailyAttempts(ctx context.Context, since time.Time) ([]AttemptBucket, error) {
	return queryAll(ctx, s.pool, `
		SELECT date_trunc('day', created_at) AS day, outcome, COUNT(*)
		  FROM review_attempts
		 WHERE created_at >= $1
		 GROUP BY day, outcome
		 ORDER BY day`, []any{since},
		func(row pgx.CollectableRow, b *AttemptBucket) error {
			return row.Scan(&b.Day, &b.Outcome, &b.Count)
		})
}

// ActivityByTeam is one row per team that has ever been seen, whether or not
// it is currently enabled: a team disabled an hour ago still has history.
//
// FULL OUTER over both tables: a skip can be decided before any
// pull_requests row exists, so a team whose every run was skipped has
// attempts and no PRs, and must still report its last run.
func (s *Store) ActivityByTeam(ctx context.Context) ([]TeamActivity, error) {
	return queryAll(ctx, s.pool, `
		WITH prs AS (
			SELECT p.team_slug, COUNT(DISTINCT p.id) AS prs, MAX(r.created_at) AS last_reviewed
			  FROM pull_requests p
			  LEFT JOIN review_runs r ON r.pull_request_id = p.id
			 GROUP BY p.team_slug
		), last AS (
			SELECT DISTINCT ON (team_slug) team_slug, created_at, outcome, COALESCE(reason, '') AS reason
			  FROM review_attempts
			 ORDER BY team_slug, created_at DESC, id DESC
		)
		SELECT COALESCE(prs.team_slug, last.team_slug), COALESCE(prs.prs, 0), prs.last_reviewed,
		       last.created_at, COALESCE(last.outcome, ''), COALESCE(last.reason, '')
		  FROM prs
		  FULL OUTER JOIN last ON last.team_slug = prs.team_slug
		 ORDER BY 1`, nil,
		func(row pgx.CollectableRow, a *TeamActivity) error {
			if err := row.Scan(&a.TeamSlug, &a.PRs, &a.LastReviewed, &a.LastRun, &a.LastOutcome, &a.LastReason); err != nil {
				return err
			}
			// A success newer than every attempt row IS the last run: the
			// attempt write fails open, and runs older than review_attempts
			// have none. Without this a team shows a last run older than its
			// last review.
			if a.LastReviewed != nil && (a.LastRun == nil || a.LastReviewed.After(*a.LastRun)) {
				a.LastRun, a.LastOutcome, a.LastReason = a.LastReviewed, "ok", ""
			}
			return nil
		})
}
