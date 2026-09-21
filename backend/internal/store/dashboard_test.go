package store

import (
	"context"
	"testing"
	"time"
)

func attempt(t *testing.T, s *Store, k PRKey, outcome, reason string, runID *int64) {
	t.Helper()
	ms := int64(1500)
	err := s.InsertAttempt(context.Background(), Attempt{
		Key: k, TeamSlug: "platform", Kind: RunAuto,
		Outcome: outcome, Reason: reason, ElapsedMS: &ms, RunID: runID,
	})
	if err != nil {
		t.Fatal(err)
	}
}

// The feed's whole reason for existing: a failure and a skip write no run
// row, so before this table they were invisible to everything but the log.
func TestRecentAttemptsReturnsFailuresAndSkips(t *testing.T) {
	s := testStore(t)
	ctx := context.Background()
	prID := upsert(t, s, key, "c1")
	runID := run(t, s, prID, "c1", nano(2_000_000_000), "gpt-5.5")

	attempt(t, s, key, "ok", "", &runID)
	attempt(t, s, key, "timed_out", "", nil)
	attempt(t, s, key, "skipped", "head_unchanged", nil)

	rows, err := s.RecentAttempts(ctx, "", 50)
	if err != nil {
		t.Fatal(err)
	}
	if len(rows) != 3 {
		t.Fatalf("rows = %d, want 3", len(rows))
	}
	// Newest first.
	if rows[0].Outcome != "skipped" || rows[0].Reason != "head_unchanged" {
		t.Errorf("newest row = %+v, want the skip", rows[0])
	}

	var ok AttemptRow
	for _, r := range rows {
		if r.Outcome == "ok" {
			ok = r
		}
	}
	// The join carries the run's figures onto the successful attempt.
	if ok.Findings == nil || *ok.Findings != 2 {
		t.Errorf("findings = %v, want 2 from the joined run", ok.Findings)
	}
	if ok.CostNanoUSD == nil || *ok.CostNanoUSD != 2_000_000_000 {
		t.Errorf("cost = %v, want the joined run's", ok.CostNanoUSD)
	}
	if ok.Key != key {
		t.Errorf("key = %+v, want %+v", ok.Key, key)
	}
}

func TestRecentAttemptsFiltersByTeam(t *testing.T) {
	s := testStore(t)
	ctx := context.Background()
	upsert(t, s, key, "c1")
	attempt(t, s, key, "ok", "", nil)

	other := PRKey{Project: "PAY", Repo: "ledger", PRID: 1}
	if err := s.InsertAttempt(ctx, Attempt{
		Key: other, TeamSlug: "payments", Kind: RunAuto, Outcome: "ok",
	}); err != nil {
		t.Fatal(err)
	}

	rows, err := s.RecentAttempts(ctx, "payments", 50)
	if err != nil {
		t.Fatal(err)
	}
	if len(rows) != 1 || rows[0].TeamSlug != "payments" {
		t.Fatalf("rows = %+v, want just the payments one", rows)
	}

	all, err := s.RecentAttempts(ctx, "", 50)
	if err != nil {
		t.Fatal(err)
	}
	if len(all) != 2 {
		t.Errorf("unfiltered rows = %d, want 2", len(all))
	}
}

// Skips group by reason; every other outcome groups by itself. A breakdown
// that lumped every skip together would answer the question with the
// question.
func TestOutcomeBreakdownSplitsSkipsByReason(t *testing.T) {
	s := testStore(t)
	upsert(t, s, key, "c1")
	attempt(t, s, key, "skipped", "head_unchanged", nil)
	attempt(t, s, key, "skipped", "head_unchanged", nil)
	attempt(t, s, key, "skipped", "empty_diff", nil)
	attempt(t, s, key, "error", "", nil)

	rows, err := s.OutcomeBreakdown(context.Background(), time.Now().Add(-time.Hour))
	if err != nil {
		t.Fatal(err)
	}
	got := map[string]int{}
	for _, r := range rows {
		got[r.Outcome+"/"+r.Reason] = r.Count
	}
	if got["skipped/head_unchanged"] != 2 {
		t.Errorf("head_unchanged = %d, want 2", got["skipped/head_unchanged"])
	}
	if got["skipped/empty_diff"] != 1 {
		t.Errorf("empty_diff = %d, want 1", got["skipped/empty_diff"])
	}
	if got["error/"] != 1 {
		t.Errorf("error = %d, want 1", got["error/"])
	}
}

// An unpriced run is not a free run. SUM skips NULLs, so the count of
// unpriced runs is carried separately and must never be folded into the
// total as a zero.
func TestTotalsKeepsUnpricedRunsOutOfTheCost(t *testing.T) {
	s := testStore(t)
	prID := upsert(t, s, key, "c1")
	run(t, s, prID, "c1", nano(1_500_000_000), "gpt-5.5")
	run(t, s, prID, "c2", nil, "gpt-5.5")
	run(t, s, prID, "c3", nano(500_000_000), "gpt-5.5")

	got, err := s.TotalsSince(context.Background(), time.Now().Add(-time.Hour))
	if err != nil {
		t.Fatal(err)
	}
	if got.Runs != 3 {
		t.Errorf("runs = %d, want 3", got.Runs)
	}
	if got.CostNanoUSD == nil || *got.CostNanoUSD != 2_000_000_000 {
		t.Errorf("cost = %v, want 2e9 (the two priced runs only)", got.CostNanoUSD)
	}
	if got.Unpriced != 1 {
		t.Errorf("unpriced = %d, want 1", got.Unpriced)
	}
	if got.PromptTokens != 300 {
		t.Errorf("prompt tokens = %d, want 300", got.PromptTokens)
	}
}

// Every run unpriced means no cost at all, which is nil rather than zero:
// "$0.000 spent" and "nothing was priced" are different facts.
func TestTotalsCostIsNilWhenNothingWasPriced(t *testing.T) {
	s := testStore(t)
	prID := upsert(t, s, key, "c1")
	run(t, s, prID, "c1", nil, "gpt-5.5")

	got, err := s.TotalsSince(context.Background(), time.Now().Add(-time.Hour))
	if err != nil {
		t.Fatal(err)
	}
	if got.CostNanoUSD != nil {
		t.Errorf("cost = %v, want nil when no run was priced", *got.CostNanoUSD)
	}
	if got.Unpriced != 1 {
		t.Errorf("unpriced = %d, want 1", got.Unpriced)
	}
}

func TestTotalsByTeamGroupsOnTheAuthenticatedSlug(t *testing.T) {
	s := testStore(t)
	ctx := context.Background()
	prID := upsert(t, s, key, "c1")
	run(t, s, prID, "c1", nano(1_000_000_000), "gpt-5.5")

	other := PRKey{Project: "PAY", Repo: "ledger", PRID: 1}
	otherID, err := s.UpsertPullRequest(ctx, PRUpsert{
		Key: other, TeamSlug: "payments", LastReviewedCommit: str("d1"),
	})
	if err != nil {
		t.Fatal(err)
	}
	run(t, s, otherID, "d1", nano(3_000_000_000), "gpt-5.5")

	rows, err := s.TotalsByTeam(ctx, time.Now().Add(-time.Hour))
	if err != nil {
		t.Fatal(err)
	}
	if len(rows) != 2 {
		t.Fatalf("rows = %d, want one per team", len(rows))
	}
	// Ordered by spend, descending.
	if rows[0].TeamSlug != "payments" {
		t.Errorf("first team = %q, want payments (the bigger spender)", rows[0].TeamSlug)
	}
}

func TestDailySeriesBucketByDay(t *testing.T) {
	s := testStore(t)
	ctx := context.Background()
	prID := upsert(t, s, key, "c1")
	run(t, s, prID, "c1", nano(1_000_000_000), "gpt-5.5")
	run(t, s, prID, "c2", nano(2_000_000_000), "gpt-5.5")
	attempt(t, s, key, "ok", "", nil)
	attempt(t, s, key, "skipped", "empty_diff", nil)

	since := time.Now().Add(-24 * time.Hour)

	days, err := s.DailyByTeam(ctx, since)
	if err != nil {
		t.Fatal(err)
	}
	if len(days) != 1 {
		t.Fatalf("day buckets = %d, want 1 (both runs are today)", len(days))
	}
	if days[0].Runs != 2 {
		t.Errorf("runs = %d, want 2", days[0].Runs)
	}
	if days[0].CostNanoUSD == nil || *days[0].CostNanoUSD != 3_000_000_000 {
		t.Errorf("cost = %v, want 3e9", days[0].CostNanoUSD)
	}

	att, err := s.DailyAttempts(ctx, since)
	if err != nil {
		t.Fatal(err)
	}
	if len(att) != 2 {
		t.Fatalf("attempt buckets = %d, want one per outcome", len(att))
	}
}

func TestActivityByTeamReportsLastRun(t *testing.T) {
	s := testStore(t)
	prID := upsert(t, s, key, "c1")
	run(t, s, prID, "c1", nano(1_000_000_000), "gpt-5.5")

	rows, err := s.ActivityByTeam(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	if len(rows) != 1 {
		t.Fatalf("rows = %d, want 1", len(rows))
	}
	if rows[0].TeamSlug != "platform" || rows[0].PRs != 1 {
		t.Errorf("row = %+v, want platform with 1 PR", rows[0])
	}
	if rows[0].LastRun == nil {
		t.Error("last run must be set once a run exists")
	}
}

// A PR that has never been reviewed still belongs to its team: the roster
// must list it rather than hiding the team until its first success.
func TestActivityByTeamIncludesTeamsWithNoRuns(t *testing.T) {
	s := testStore(t)
	upsert(t, s, key, "c1")

	rows, err := s.ActivityByTeam(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	if len(rows) != 1 {
		t.Fatalf("rows = %d, want 1", len(rows))
	}
	if rows[0].LastRun != nil {
		t.Errorf("last run = %v, want nil with no runs", rows[0].LastRun)
	}
}

// ON DELETE SET NULL, not CASCADE: deleting a PR's runs must not erase the
// record that the attempt happened.
func TestAttemptSurvivesItsRun(t *testing.T) {
	s := testStore(t)
	ctx := context.Background()
	prID := upsert(t, s, key, "c1")
	runID := run(t, s, prID, "c1", nano(1_000_000_000), "gpt-5.5")
	attempt(t, s, key, "ok", "", &runID)

	if _, err := s.pool.Exec(ctx, `DELETE FROM review_runs WHERE id = $1`, runID); err != nil {
		t.Fatal(err)
	}

	rows, err := s.RecentAttempts(ctx, "", 50)
	if err != nil {
		t.Fatal(err)
	}
	if len(rows) != 1 {
		t.Fatalf("rows = %d, want the attempt to survive", len(rows))
	}
	if rows[0].RunID != nil {
		t.Errorf("run id = %v, want nil once the run is gone", rows[0].RunID)
	}
}

// Every read here is a Query/Scan pair, and a closed pool is the cheapest
// way to prove each one propagates its error instead of returning an empty
// result that reads as "nothing happened".
func TestDashboardReadsPropagateErrors(t *testing.T) {
	s := testStore(t)
	ctx := context.Background()
	s.pool.Close()

	if _, err := s.RecentAttempts(ctx, "", 10); err == nil {
		t.Error("RecentAttempts must report a dead pool")
	}
	if _, err := s.OutcomeBreakdown(ctx, time.Now()); err == nil {
		t.Error("OutcomeBreakdown must report a dead pool")
	}
	if _, err := s.TotalsSince(ctx, time.Now()); err == nil {
		t.Error("TotalsSince must report a dead pool")
	}
	if _, err := s.TotalsByTeam(ctx, time.Now()); err == nil {
		t.Error("TotalsByTeam must report a dead pool")
	}
	if _, err := s.DailyByTeam(ctx, time.Now()); err == nil {
		t.Error("DailyByTeam must report a dead pool")
	}
	if _, err := s.DailyAttempts(ctx, time.Now()); err == nil {
		t.Error("DailyAttempts must report a dead pool")
	}
	if _, err := s.ActivityByTeam(ctx); err == nil {
		t.Error("ActivityByTeam must report a dead pool")
	}
	if err := s.InsertAttempt(ctx, Attempt{Key: key, TeamSlug: "x", Kind: RunAuto, Outcome: "ok"}); err == nil {
		t.Error("InsertAttempt must report a dead pool")
	}
}

// The limit is clamped, so a caller cannot ask for the whole table and a
// zero still means "a sensible page" rather than no rows at all.
func TestRecentAttemptsClampsTheLimit(t *testing.T) {
	s := testStore(t)
	ctx := context.Background()
	upsert(t, s, key, "c1")
	attempt(t, s, key, "ok", "", nil)

	for _, limit := range []int{0, -1, 10000} {
		rows, err := s.RecentAttempts(ctx, "", limit)
		if err != nil {
			t.Fatalf("limit %d: %v", limit, err)
		}
		if len(rows) != 1 {
			t.Errorf("limit %d returned %d rows, want 1", limit, len(rows))
		}
	}
}
