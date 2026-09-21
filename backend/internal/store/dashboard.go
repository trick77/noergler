package store

import (
	"context"
	"time"
)

// Every aggregate the store had before this file is keyed to one PRKey, and
// ClaimRollup is destructive: it stamps riptide_emitted_at in the statement
// that reads the snapshot. The dashboard needs read-only, time-windowed
// siblings, so they live here rather than being bolted onto those.
//
// Cost is BIGINT nano-USD throughout and NULL means unpriced. SUM() skips
// NULLs, so Unpriced is counted separately and NEVER folded in as zero: a
// run the gateway did not price is not a free run.

// Totals is the headline figure set over a time window.
type Totals struct {
	Runs             int
	PromptTokens     int64
	CachedTokens     int64
	CompletionTokens int64
	FindingsPosted   int64
	// CostNanoUSD sums the priced runs only. Nil when none were priced.
	CostNanoUSD *int64
	// Unpriced is how many runs carried no cost. Shown beside the total,
	// never added to it.
	Unpriced int
}

const totalsCols = `
	COUNT(*),
	COALESCE(SUM(prompt_tokens), 0),
	COALESCE(SUM(cached_tokens), 0),
	COALESCE(SUM(completion_tokens), 0),
	COALESCE(SUM(findings_posted), 0),
	SUM(cost_nano_usd),
	COUNT(*) FILTER (WHERE cost_nano_usd IS NULL)`

// TotalsSince aggregates every team's runs in the window.
func (s *Store) TotalsSince(ctx context.Context, since time.Time) (Totals, error) {
	var t Totals
	err := s.pool.QueryRow(ctx, `
		SELECT`+totalsCols+`
		  FROM review_runs
		 WHERE created_at >= $1`, since).
		Scan(&t.Runs, &t.PromptTokens, &t.CachedTokens, &t.CompletionTokens,
			&t.FindingsPosted, &t.CostNanoUSD, &t.Unpriced)
	return t, err
}

// TeamTotals is Totals for one team.
type TeamTotals struct {
	TeamSlug string
	Totals
}

// TotalsByTeam is TotalsSince grouped by the team that owns the PR. The slug
// comes off pull_requests, which the webhook route authenticated.
func (s *Store) TotalsByTeam(ctx context.Context, since time.Time) ([]TeamTotals, error) {
	rows, err := s.pool.Query(ctx, `
		SELECT p.team_slug,`+totalsCols+`
		  FROM review_runs r
		  JOIN pull_requests p ON p.id = r.pull_request_id
		 WHERE r.created_at >= $1
		 GROUP BY p.team_slug
		 ORDER BY SUM(r.cost_nano_usd) DESC NULLS LAST`, since)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	var out []TeamTotals
	for rows.Next() {
		var t TeamTotals
		if err := rows.Scan(&t.TeamSlug, &t.Runs, &t.PromptTokens, &t.CachedTokens,
			&t.CompletionTokens, &t.FindingsPosted, &t.CostNanoUSD, &t.Unpriced); err != nil {
			return nil, err
		}
		out = append(out, t)
	}
	return out, rows.Err()
}

// Bucket is one day of the timeseries.
type Bucket struct {
	Day         time.Time
	TeamSlug    string
	Runs        int
	CostNanoUSD *int64
}

// DailyByTeam is runs and cost per day per team, for the charts. Days with no
// runs are absent rather than zero-filled: the caller knows the window it
// asked for and fills the gaps, which keeps this query from inventing rows.
func (s *Store) DailyByTeam(ctx context.Context, since time.Time) ([]Bucket, error) {
	rows, err := s.pool.Query(ctx, `
		SELECT date_trunc('day', r.created_at) AS day,
		       p.team_slug,
		       COUNT(*),
		       SUM(r.cost_nano_usd)
		  FROM review_runs r
		  JOIN pull_requests p ON p.id = r.pull_request_id
		 WHERE r.created_at >= $1
		 GROUP BY day, p.team_slug
		 ORDER BY day`, since)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	var out []Bucket
	for rows.Next() {
		var b Bucket
		if err := rows.Scan(&b.Day, &b.TeamSlug, &b.Runs, &b.CostNanoUSD); err != nil {
			return nil, err
		}
		out = append(out, b)
	}
	return out, rows.Err()
}

// DailyAttempts is attempts per day per outcome, for the throughput chart.
// It reads review_attempts, so a day's failures and skips are visible where
// DailyByTeam sees only the runs that succeeded.
func (s *Store) DailyAttempts(ctx context.Context, since time.Time) ([]AttemptBucket, error) {
	rows, err := s.pool.Query(ctx, `
		SELECT date_trunc('day', created_at) AS day, outcome, COUNT(*)
		  FROM review_attempts
		 WHERE created_at >= $1
		 GROUP BY day, outcome
		 ORDER BY day`, since)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	var out []AttemptBucket
	for rows.Next() {
		var b AttemptBucket
		if err := rows.Scan(&b.Day, &b.Outcome, &b.Count); err != nil {
			return nil, err
		}
		out = append(out, b)
	}
	return out, rows.Err()
}

// AttemptBucket is one day's count for one outcome.
type AttemptBucket struct {
	Day     time.Time
	Outcome string
	Count   int
}

// TeamActivity is the per-team roster figure the dashboard shows beside a
// team's state: when it last produced a run, and how many PRs it has touched.
type TeamActivity struct {
	TeamSlug string
	PRs      int
	LastRun  *time.Time
}

// ActivityByTeam is one row per team that has ever been seen, whether or not
// it is currently enabled: a team disabled an hour ago still has history.
func (s *Store) ActivityByTeam(ctx context.Context) ([]TeamActivity, error) {
	rows, err := s.pool.Query(ctx, `
		SELECT p.team_slug, COUNT(DISTINCT p.id), MAX(r.created_at)
		  FROM pull_requests p
		  LEFT JOIN review_runs r ON r.pull_request_id = p.id
		 GROUP BY p.team_slug
		 ORDER BY p.team_slug`)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	var out []TeamActivity
	for rows.Next() {
		var a TeamActivity
		if err := rows.Scan(&a.TeamSlug, &a.PRs, &a.LastRun); err != nil {
			return nil, err
		}
		out = append(out, a)
	}
	return out, rows.Err()
}
