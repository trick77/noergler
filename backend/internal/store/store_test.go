package store

import (
	"context"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"os"
	"reflect"
	"sync"
	"testing"
	"time"

	"github.com/jackc/pgx/v5/pgxpool"

	"github.com/trick77/noergler/internal/config"
)

// testStore opens a store on a fresh schema, migrates it and drops the
// schema at cleanup. Gated on NOERGLER_TEST_DSN; CI provides a Postgres
// service container, locally `docker compose up -d postgres` and
// NOERGLER_TEST_DSN=postgres://noergler:changeme@localhost:5432/noergler?sslmode=disable.
func testStore(t *testing.T) *Store {
	t.Helper()
	dsn := os.Getenv("NOERGLER_TEST_DSN")
	if dsn == "" {
		t.Skip("NOERGLER_TEST_DSN not set")
	}
	ctx := context.Background()
	schema := fmt.Sprintf("t_%d_%d", time.Now().UnixNano(), os.Getpid())
	cfg, err := pgxpool.ParseConfig(dsn)
	if err != nil {
		t.Fatal(err)
	}
	admin, err := pgxpool.NewWithConfig(ctx, cfg)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := admin.Exec(ctx, "CREATE SCHEMA "+schema); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		_, _ = admin.Exec(ctx, "DROP SCHEMA "+schema+" CASCADE")
		admin.Close()
	})
	cfg2, _ := pgxpool.ParseConfig(dsn)
	cfg2.ConnConfig.RuntimeParams["search_path"] = schema
	s, err := openConfig(ctx, cfg2, slog.New(slog.NewTextHandler(io.Discard, nil)))
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(s.Close)
	if err := s.Migrate(ctx); err != nil {
		t.Fatalf("migrate: %v", err)
	}
	return s
}

func str(s string) *string { return &s }
func num(n int) *int       { return &n }
func nano(n int64) *int64  { return &n }

var key = PRKey{Project: "PLAT", Repo: "api", PRID: 7}

func upsert(t *testing.T, s *Store, k PRKey, commit string) int64 {
	t.Helper()
	id, err := s.UpsertPullRequest(context.Background(), PRUpsert{Key: k, TeamSlug: "platform", LastReviewedCommit: str(commit), Author: str("alice"), Title: str("t")})
	if err != nil {
		t.Fatal(err)
	}
	return id
}

func run(t *testing.T, s *Store, prID int64, to string, cost *int64, model string) int64 {
	t.Helper()
	id, err := s.InsertRun(context.Background(), Run{PullRequestID: prID, Kind: RunAuto, ToCommit: to, ModelLabel: model,
		PromptTokens: 100, CachedTokens: 10, CompletionTokens: 20, CostNanoUSD: cost, ElapsedMS: 1500, FindingsPosted: 2,
		LinesAdded: 10, LinesRemoved: 3, FilesChanged: 2})
	if err != nil {
		t.Fatal(err)
	}
	return id
}

func TestMigrate_IsIdempotent(t *testing.T) {
	s := testStore(t)
	var before int
	if err := s.pool.QueryRow(context.Background(), `SELECT COUNT(*) FROM schema_migrations`).Scan(&before); err != nil {
		t.Fatalf("count before: %v", err)
	}
	if before == 0 {
		t.Fatal("migrate recorded nothing")
	}

	if err := s.Migrate(context.Background()); err != nil {
		t.Fatalf("second migrate: %v", err)
	}

	// Compared against the first count, not a literal: the point is that a
	// second Migrate applies nothing, and a hardcoded number makes every
	// future migration fail a test about idempotency.
	var after int
	if err := s.pool.QueryRow(context.Background(), `SELECT COUNT(*) FROM schema_migrations`).Scan(&after); err != nil {
		t.Fatalf("count after: %v", err)
	}
	if after != before {
		t.Errorf("schema_migrations rows = %d after a second migrate, want %d", after, before)
	}
}

func TestMigrate_ConcurrentCallersSerialise(t *testing.T) {
	s := testStore(t)
	var wg sync.WaitGroup
	errs := make(chan error, 4)
	for i := 0; i < 4; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			errs <- s.Migrate(context.Background())
		}()
	}
	wg.Wait()
	close(errs)
	for err := range errs {
		if err != nil {
			t.Errorf("migrate: %v", err)
		}
	}
}

func TestPullRequest_UpsertStickyOpenedAtAndTeamFollows(t *testing.T) {
	s := testStore(t)
	ctx := context.Background()
	opened := time.Date(2026, 9, 1, 10, 0, 0, 0, time.UTC)
	id, err := s.UpsertPullRequest(ctx, PRUpsert{Key: key, TeamSlug: "platform", OpenedAt: &opened, LastReviewedCommit: str("aaa")})
	if err != nil {
		t.Fatal(err)
	}
	later := opened.Add(time.Hour)
	id2, err := s.UpsertPullRequest(ctx, PRUpsert{Key: key, TeamSlug: "payments", OpenedAt: &later, LastReviewedCommit: str("bbb")})
	if err != nil || id2 != id {
		t.Fatalf("second upsert id = %d, err = %v", id2, err)
	}
	var team, commit string
	var got time.Time
	if err := s.pool.QueryRow(ctx, `SELECT team_slug, last_reviewed_commit, opened_at FROM pull_requests WHERE id = $1`, id).Scan(&team, &commit, &got); err != nil {
		t.Fatal(err)
	}
	if team != "payments" || commit != "bbb" || !got.Equal(opened) {
		t.Errorf("team = %s commit = %s opened_at = %v", team, commit, got)
	}
}

func TestPullRequest_LastReviewedCommitExcludesClosedPRs(t *testing.T) {
	s := testStore(t)
	ctx := context.Background()
	if _, ok, _ := s.GetLastReviewedCommit(ctx, key); ok {
		t.Error("no row must be not-found")
	}
	upsert(t, s, key, "abc")
	if c, ok, err := s.GetLastReviewedCommit(ctx, key); err != nil || !ok || c != "abc" {
		t.Errorf("got %q %v %v", c, ok, err)
	}
	for name, mark := range map[string]func(context.Context, PRKey) error{"merged": s.MarkMerged, "declined": s.MarkDeclined, "deleted": s.MarkDeleted} {
		k := PRKey{Project: "P", Repo: name, PRID: 1}
		upsert(t, s, k, "abc")
		if err := mark(ctx, k); err != nil {
			t.Fatal(err)
		}
		if err := mark(ctx, k); err != nil {
			t.Fatalf("%s must be idempotent: %v", name, err)
		}
		if _, ok, _ := s.GetLastReviewedCommit(ctx, k); ok {
			t.Errorf("%s PR must not report a reviewed commit", name)
		}
	}
}

func TestPullRequest_SkipStateIgnoreReactivate(t *testing.T) {
	s := testStore(t)
	ctx := context.Background()
	if st, err := s.GetSkipState(ctx, key); err != nil || st != nil {
		t.Fatalf("first review: %v %v", st, err)
	}
	id := upsert(t, s, key, "abc")
	if err := s.SetSummaryComment(ctx, id, 42, 3); err != nil {
		t.Fatal(err)
	}
	st, err := s.GetSkipState(ctx, key)
	if err != nil || st == nil || st.ID != id || st.IgnoredAt != nil || st.Summary == nil || st.Summary.ID != 42 || st.Summary.Version != 3 {
		t.Fatalf("skip state = %+v, err = %v", st, err)
	}
	if sc, _ := s.GetSummaryComment(ctx, id); sc == nil || sc.ID != 42 {
		t.Errorf("summary comment = %+v", sc)
	}
	if err := s.MarkIgnored(ctx, key); err != nil {
		t.Fatal(err)
	}
	st, _ = s.GetSkipState(ctx, key)
	if st.IgnoredAt == nil {
		t.Fatal("ignored_at not set")
	}
	first := *st.IgnoredAt
	_ = s.MarkIgnored(ctx, key)
	st, _ = s.GetSkipState(ctx, key)
	if !st.IgnoredAt.Equal(first) {
		t.Error("MarkIgnored must not re-stamp")
	}
	if err := s.Reactivate(ctx, key); err != nil {
		t.Fatal(err)
	}
	st, _ = s.GetSkipState(ctx, key)
	if st.IgnoredAt != nil || st.Summary != nil {
		t.Errorf("reactivate must clear ignored_at and the summary: %+v", st)
	}
	if sc, _ := s.GetSummaryComment(ctx, id); sc != nil {
		t.Errorf("summary comment after reactivate = %+v", sc)
	}
}

func TestRuns_CostAggregationWithNulls(t *testing.T) {
	s := testStore(t)
	ctx := context.Background()
	if c, err := s.PRCost(ctx, key); err != nil || c != nil {
		t.Fatalf("no row: %v %v", c, err)
	}
	id := upsert(t, s, key, "a")
	if c, _ := s.PRCost(ctx, key); c != nil {
		t.Fatal("no runs: cost must be nil")
	}
	run(t, s, id, "a", nil, "gpt-5.5-high")
	if c, _ := s.PRCost(ctx, key); c != nil {
		t.Fatal("only unpriced runs: cost must be nil, never zero")
	}
	run(t, s, id, "b", nano(1_500_000_000), "gpt-5.5-high")
	run(t, s, id, "c", nano(500_000_000), "gpt-5.4-medium")
	if c, _ := s.PRCost(ctx, key); c == nil || *c != 2_000_000_000 {
		t.Errorf("cost = %v, want 2 USD", c)
	}
	frozen, err := s.FreezeFinalCost(ctx, key)
	if err != nil || frozen == nil || *frozen != 2_000_000_000 {
		t.Errorf("frozen = %v, err = %v", frozen, err)
	}
	if f, err := s.FreezeFinalCost(ctx, PRKey{Project: "none", Repo: "x", PRID: 1}); err != nil || f != nil {
		t.Errorf("freeze on a missing row = %v %v", f, err)
	}
	var final *int64
	_ = s.pool.QueryRow(ctx, `SELECT final_cost_nano_usd FROM pull_requests WHERE id = $1`, id).Scan(&final)
	if final == nil || *final != 2_000_000_000 {
		t.Errorf("final_cost_nano_usd = %v", final)
	}
	// The latest run's commit and diff size ride on the PR.
	var src string
	var added int
	_ = s.pool.QueryRow(ctx, `SELECT final_source_commit, final_lines_added FROM pull_requests WHERE id = $1`, id).Scan(&src, &added)
	if src != "c" || added != 10 {
		t.Errorf("final_source_commit = %s, lines = %d", src, added)
	}
}

func TestRollup_ClaimIsAtomicAndOnce(t *testing.T) {
	s := testStore(t)
	ctx := context.Background()
	if snap, err := s.ClaimRollup(ctx, key, RollupFinal{}); err != nil || snap != nil {
		t.Fatalf("no row: %v %v", snap, err)
	}
	id := upsert(t, s, key, "a")
	if snap, _ := s.ClaimRollup(ctx, key, RollupFinal{}); snap != nil {
		t.Fatal("no runs: nothing to forward")
	}
	var emitted *time.Time
	_ = s.pool.QueryRow(ctx, `SELECT riptide_emitted_at FROM pull_requests WHERE id = $1`, id).Scan(&emitted)
	if emitted != nil {
		t.Fatal("a PR without runs must not be claimed")
	}
	run(t, s, id, "a", nil, "gpt-5.5-high")
	run(t, s, id, "b", nano(700), "gpt-5.5-high")
	run(t, s, id, "c", nano(300), "gpt-5.4-medium")
	snap, err := s.ClaimRollup(ctx, key, RollupFinal{MergeCommit: str("m1"), LinesAdded: num(99), FilesChanged: num(5)})
	if err != nil || snap == nil {
		t.Fatalf("claim: %v %v", snap, err)
	}
	if snap.Runs != 3 || snap.PromptTokens != 300 || snap.CompletionTokens != 60 || snap.ElapsedMS != 4500 || snap.Findings != 6 {
		t.Errorf("aggregates: %+v", snap)
	}
	if snap.CostNanoUSD == nil || *snap.CostNanoUSD != 1000 {
		t.Errorf("cost = %v", snap.CostNanoUSD)
	}
	if !reflect.DeepEqual(snap.Models, []string{"gpt-5.4-medium", "gpt-5.5-high"}) {
		t.Errorf("models = %v", snap.Models)
	}
	if snap.SourceCommit == nil || *snap.SourceCommit != "c" || snap.MergeCommit == nil || *snap.MergeCommit != "m1" {
		t.Errorf("commits: %v %v", snap.SourceCommit, snap.MergeCommit)
	}
	// Final refresh overrides what the run recorded; nil keeps it.
	if *snap.LinesAdded != 99 || *snap.LinesRemoved != 3 || *snap.FilesChanged != 5 {
		t.Errorf("lines: %d %d %d", *snap.LinesAdded, *snap.LinesRemoved, *snap.FilesChanged)
	}
	if snap.FirstReviewAt.IsZero() {
		t.Error("first_review_at")
	}
	if again, _ := s.ClaimRollup(ctx, key, RollupFinal{}); again != nil {
		t.Error("a second claim must return nil")
	}
	// Unpriced PR: nil cost, still claimed.
	k2 := PRKey{Project: "PLAT", Repo: "api", PRID: 8}
	id2 := upsert(t, s, k2, "x")
	run(t, s, id2, "x", nil, "m")
	snap2, _ := s.ClaimRollup(ctx, k2, RollupFinal{})
	if snap2 == nil || snap2.CostNanoUSD != nil {
		t.Errorf("unpriced snapshot = %+v", snap2)
	}
}

func TestFindings_ExistingOnlyForOpenPR(t *testing.T) {
	s := testStore(t)
	ctx := context.Background()
	id := upsert(t, s, key, "a")
	runID := run(t, s, id, "a", nil, "m")
	for i, f := range []Finding{
		{FilePath: "b.go", LineNumber: 2, Severity: "major", CommentText: "second", Confidence: num(80), Headline: str("h")},
		{FilePath: "a.go", LineNumber: 1, Severity: "minor", CommentText: "first", Suggestion: str("fix"), BitbucketCommentID: num(5)},
	} {
		f.PullRequestID, f.RunID = id, runID
		if err := s.InsertFinding(ctx, f); err != nil {
			t.Fatalf("insert %d: %v", i, err)
		}
	}
	got, err := s.ExistingFindings(ctx, key)
	if err != nil || len(got) != 2 || got[0].CommentText != "second" || got[1].CommentText != "first" {
		t.Fatalf("findings = %+v, err = %v", got, err)
	}
	if *got[0].Confidence != 80 || *got[0].Headline != "h" || *got[1].Suggestion != "fix" || *got[1].BitbucketCommentID != 5 {
		t.Errorf("fields: %+v", got)
	}
	if err := s.MarkMerged(ctx, key); err != nil {
		t.Fatal(err)
	}
	if got, _ := s.ExistingFindings(ctx, key); len(got) != 0 {
		t.Error("a merged PR's findings are history")
	}
	// Purge cascades.
	if n, err := s.PurgeProject(ctx, "platform", "PLAT", nil); err != nil || n != 1 {
		t.Errorf("purge = %d %v", n, err)
	}
	var runs, findings int
	_ = s.pool.QueryRow(ctx, `SELECT (SELECT COUNT(*) FROM review_runs), (SELECT COUNT(*) FROM findings)`).Scan(&runs, &findings)
	if runs != 0 || findings != 0 {
		t.Errorf("cascade left %d runs, %d findings", runs, findings)
	}
}

func TestClaims_ListGroupsRepoClaimsAndKeepsWholeProjectsFirst(t *testing.T) {
	s := testStore(t)
	ctx := context.Background()
	for _, ins := range [][]any{{"platform", "INFRA", "a"}, {"platform", "PLAT", nil}, {"platform", "INFRA", "b"}, {"payments", "PAY", "billing"}} {
		if _, err := s.pool.Exec(ctx, `INSERT INTO team_claims (team_slug, project_key, repo_slug, claimed_by) VALUES ($1, $2, $3, 'seed')`, ins...); err != nil {
			t.Fatal(err)
		}
	}
	got, err := s.ListClaims(ctx, "platform")
	want := []config.ProjectScope{{Key: "PLAT"}, {Key: "INFRA", Repos: []string{"a", "b"}}}
	if err != nil || !reflect.DeepEqual(got, want) {
		t.Errorf("ListClaims = %+v, err = %v", got, err)
	}
	all, err := s.ListAllClaims(ctx)
	if err != nil || !reflect.DeepEqual(all["payments"], []config.ProjectScope{{Key: "PAY", Repos: []string{"billing"}}}) || len(all["platform"]) != 2 {
		t.Errorf("ListAllClaims = %+v, err = %v", all, err)
	}
	if empty, _ := s.ListClaims(ctx, "nobody"); len(empty) != 0 {
		t.Errorf("unknown team = %v", empty)
	}
}

func conflict(t *testing.T, err error, want string) {
	t.Helper()
	var c *ClaimConflict
	if !errors.As(err, &c) || c.Error() != want {
		t.Errorf("err = %v, want ClaimConflict %q", err, want)
	}
}

func TestClaims_AddRules(t *testing.T) {
	s := testStore(t)
	ctx := context.Background()

	// A whole-project claim conflicts with another team's repo claim.
	if _, err := s.AddClaims(ctx, "payments", []config.ProjectScope{{Key: "PAY", Repos: []string{"billing"}}}, "jan"); err != nil {
		t.Fatal(err)
	}
	_, err := s.AddClaims(ctx, "platform", []config.ProjectScope{{Key: "PAY"}}, "jan")
	conflict(t, err, "PAY/billing is claimed by team payments")
	if n, _ := s.CountProjectPRs(ctx, "platform", "PAY", nil); n != 0 {
		t.Error("count")
	}

	// A whole-project claim replaces the team's own repo claims.
	if _, err := s.AddClaims(ctx, "platform", []config.ProjectScope{{Key: "PLAT", Repos: []string{"a"}}}, "jan"); err != nil {
		t.Fatal(err)
	}
	added, err := s.AddClaims(ctx, "platform", []config.ProjectScope{{Key: "PLAT"}}, "jan")
	if err != nil || !reflect.DeepEqual(added, []string{"PLAT"}) {
		t.Fatalf("added = %v, err = %v", added, err)
	}
	if got, _ := s.ListClaims(ctx, "platform"); !reflect.DeepEqual(got, []config.ProjectScope{{Key: "PLAT"}}) {
		t.Errorf("claims after whole = %+v", got)
	}
	// Already ours as a whole: nothing added, repo claims under it neither.
	if added, _ := s.AddClaims(ctx, "platform", []config.ProjectScope{{Key: "PLAT"}}, "jan"); len(added) != 0 {
		t.Errorf("re-claim added %v", added)
	}
	if added, _ := s.AddClaims(ctx, "platform", []config.ProjectScope{{Key: "PLAT", Repos: []string{"x"}}}, "jan"); len(added) != 0 {
		t.Errorf("repo under own whole added %v", added)
	}
	// Another team holds the whole project.
	_, err = s.AddClaims(ctx, "payments", []config.ProjectScope{{Key: "PLAT", Repos: []string{"x"}}}, "jan")
	conflict(t, err, "PLAT is claimed by team platform")

	// One repo ours, one new, one foreign: all or nothing.
	if _, err := s.AddClaims(ctx, "platform", []config.ProjectScope{{Key: "SHARED", Repos: []string{"a"}}}, "jan"); err != nil {
		t.Fatal(err)
	}
	if _, err := s.AddClaims(ctx, "payments", []config.ProjectScope{{Key: "SHARED", Repos: []string{"c"}}}, "jan"); err != nil {
		t.Fatal(err)
	}
	_, err = s.AddClaims(ctx, "platform", []config.ProjectScope{{Key: "SHARED", Repos: []string{"a", "b", "c"}}}, "jan")
	conflict(t, err, "SHARED/c is claimed by team payments")
	if got, _ := s.ListClaims(ctx, "platform"); !reflect.DeepEqual(got, []config.ProjectScope{{Key: "PLAT"}, {Key: "SHARED", Repos: []string{"a"}}}) {
		t.Errorf("nothing must be written on conflict: %+v", got)
	}
	added, err = s.AddClaims(ctx, "platform", []config.ProjectScope{{Key: "SHARED", Repos: []string{"a", "b"}}}, "jan")
	if err != nil || !reflect.DeepEqual(added, []string{"SHARED/b"}) {
		t.Errorf("added = %v, err = %v", added, err)
	}
}

func TestClaims_RemoveReportsWhatWent(t *testing.T) {
	s := testStore(t)
	ctx := context.Background()
	for _, ins := range [][]any{{"platform", "PLAT", nil}, {"platform", "PLAT", "a"}, {"platform", "INFRA", "x"}} {
		if _, err := s.pool.Exec(ctx, `INSERT INTO team_claims (team_slug, project_key, repo_slug, claimed_by) VALUES ($1, $2, $3, 'seed')`, ins...); err != nil {
			t.Fatal(err)
		}
	}
	removed, err := s.RemoveClaims(ctx, "platform", []config.ProjectScope{{Key: "PLAT"}, {Key: "INFRA", Repos: []string{"x", "y"}}})
	if err != nil || !reflect.DeepEqual(removed, []string{"PLAT", "PLAT/a", "INFRA/x"}) {
		t.Errorf("removed = %v, err = %v", removed, err)
	}
	if got, _ := s.ListClaims(ctx, "platform"); len(got) != 0 {
		t.Errorf("left: %+v", got)
	}
}

func TestClaims_ConcurrentFirstClaimsSerialise(t *testing.T) {
	s := testStore(t)
	ctx := context.Background()
	// Two first claims on the same project: both FOR UPDATE reads would see
	// no rows. The advisory lock makes the second wait for the first to
	// commit and then see its row, so the loser gets the ordinary conflict.
	tx1, err := s.pool.Begin(ctx)
	if err != nil {
		t.Fatal(err)
	}
	added := []string{}
	if err := s.addClaims(ctx, tx1, "platform", []config.ProjectScope{{Key: "RACE"}}, "jan", &added); err != nil {
		t.Fatal(err)
	}
	result := make(chan error, 1)
	go func() {
		_, err := s.AddClaims(ctx, "payments", []config.ProjectScope{{Key: "RACE", Repos: []string{"x"}}}, "jan")
		result <- err
	}()
	select {
	case err := <-result:
		t.Fatalf("the second claim must block behind the first, got %v", err)
	case <-time.After(300 * time.Millisecond):
	}
	if err := tx1.Commit(ctx); err != nil {
		t.Fatal(err)
	}
	conflict(t, <-result, "RACE is claimed by team platform")
	if got, _ := s.ListClaims(ctx, "platform"); !reflect.DeepEqual(got, []config.ProjectScope{{Key: "RACE"}}) {
		t.Errorf("winner = %+v", got)
	}
	if got, _ := s.ListClaims(ctx, "payments"); len(got) != 0 {
		t.Errorf("loser wrote %+v", got)
	}
}

func TestPullRequest_ReopenedAfterDeclineStartsFreshButDedups(t *testing.T) {
	s := testStore(t)
	ctx := context.Background()
	id := upsert(t, s, key, "a")
	runID := run(t, s, id, "a", nil, "m")
	if err := s.InsertFinding(ctx, Finding{PullRequestID: id, RunID: runID, FilePath: "a.go", LineNumber: 1, Severity: "minor", CommentText: "x"}); err != nil {
		t.Fatal(err)
	}
	if err := s.MarkDeclined(ctx, key); err != nil {
		t.Fatal(err)
	}
	if _, ok, _ := s.GetLastReviewedCommit(ctx, key); ok {
		t.Error("a declined PR has no pointer: the next review is full")
	}
	if got, _ := s.ExistingFindings(ctx, key); len(got) != 1 {
		t.Error("its findings are still on Bitbucket, so they still count for dedup")
	}
	upsert(t, s, key, "b")
	if c, ok, _ := s.GetLastReviewedCommit(ctx, key); !ok || c != "b" {
		t.Errorf("reopened PR pointer = %q %v, want b", c, ok)
	}
}

func TestSettings_RoundTrip(t *testing.T) {
	s := testStore(t)
	ctx := context.Background()
	if got, err := s.GetSettings(ctx, "platform"); err != nil || got != nil {
		t.Fatalf("missing row = %v %v", got, err)
	}
	if err := s.PutSettings(ctx, "platform", TeamSettings{AutoReviewAuthors: []string{"a"}, IgnoreAuthors: nil, ExcludeRepos: []string{"*-infra"}}, "jan"); err != nil {
		t.Fatal(err)
	}
	got, _ := s.GetSettings(ctx, "platform")
	if !reflect.DeepEqual(got, &TeamSettings{AutoReviewAuthors: []string{"a"}, IgnoreAuthors: []string{}, ExcludeRepos: []string{"*-infra"}}) {
		t.Errorf("settings = %+v", got)
	}
	if !got.Excludes("Platform-INFRA") || got.Excludes("infra-tools") {
		t.Error("exclude glob")
	}
	_ = s.PutSettings(ctx, "platform", TeamSettings{AutoReviewAuthors: []string{"b"}, ExcludeRepos: []string{}}, "jan")
	all, _ := s.GetAllSettings(ctx)
	if !reflect.DeepEqual(all["platform"].AutoReviewAuthors, []string{"b"}) || len(all["platform"].ExcludeRepos) != 0 {
		t.Errorf("all = %+v", all)
	}
}
