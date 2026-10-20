package store

import (
	"context"
	"testing"
)

// prToken is the link token of the PR row id.
func prToken(t *testing.T, s *Store, id int64) string {
	t.Helper()
	tok, err := s.PRLinkToken(context.Background(), id)
	if err != nil {
		t.Fatal(err)
	}
	return tok
}

// addAttempt inserts a and returns its id.
func addAttempt(t *testing.T, s *Store, a Attempt) int64 {
	t.Helper()
	id, err := s.InsertAttempt(context.Background(), a)
	if err != nil {
		t.Fatal(err)
	}
	return id
}

func TestPRPageAndRuns(t *testing.T) {
	s := testStore(t)
	ctx := context.Background()
	prID := upsert(t, s, key, "a")
	token := prToken(t, s, prID)

	earlier := run(t, s, prID, "a", nano(600), "m")
	if err := s.InsertFinding(ctx, Finding{PullRequestID: prID, RunID: earlier, FilePath: "a.go", LineNumber: 3,
		Severity: "issue", Headline: strp("old one"), CommentText: "SECRET BODY"}); err != nil {
		t.Fatal(err)
	}
	runID := run(t, s, prID, "b", nano(900), "m")
	if err := s.InsertFinding(ctx, Finding{PullRequestID: prID, RunID: runID, FilePath: "b.go", LineNumber: 7,
		Severity: "suggestion", Headline: strp("new one"), CommentText: "the body", Suggestion: strp("do this")}); err != nil {
		t.Fatal(err)
	}
	if _, err := s.pool.Exec(ctx, `UPDATE findings SET fixed_at = now(), fixed_run_id = $1 WHERE review_run_id = $2`, runID, earlier); err != nil {
		t.Fatal(err)
	}
	first := addAttempt(t, s, Attempt{Key: key, TeamSlug: "platform", Kind: RunAuto, Outcome: "ok", RunID: &earlier})
	skip := addAttempt(t, s, Attempt{Key: key, TeamSlug: "platform", Kind: RunAuto, Outcome: "skipped", Reason: "head_unchanged"})
	last := addAttempt(t, s, Attempt{Key: key, TeamSlug: "platform", Kind: RunMention, Outcome: "ok",
		RunID: &runID, Call: &LLMCall{Profile: "m", CallID: "c1"}})
	if err := s.InsertRunDetail(ctx, last, Pipeline{Stages: []StageTime{{Name: "gateway", MS: 40}},
		PRScope: &PRScope{Added: 12, Removed: 4, Files: 3}}); err != nil {
		t.Fatal(err)
	}
	// Another team's attempt on the same PR key is not this PR's history.
	addAttempt(t, s, Attempt{Key: key, TeamSlug: "other", Kind: RunAuto, Outcome: "error"})

	pg, ok, err := s.PRPage(ctx, token)
	if err != nil || !ok || pg.ID != prID || pg.Title != "t" || pg.TeamSlug != "platform" || pg.More {
		t.Fatalf("PRPage = %+v, %v, %v", pg, ok, err)
	}
	if len(pg.Runs) != 3 || pg.Runs[0].N != 1 || pg.Runs[1].N != 2 || pg.Runs[2].N != 3 {
		t.Fatalf("runs = %+v", pg.Runs)
	}
	if r := pg.Runs[0]; r.Run == nil || r.Run.ID != earlier || r.Fixed != 0 || r.CostNanoUSD == nil || *r.CostNanoUSD != 600 {
		t.Errorf("run 1 = %+v", r)
	}
	if r := pg.Runs[1]; r.Run != nil || r.Outcome != "skipped" || r.Reason != "head_unchanged" || r.CostNanoUSD != nil {
		t.Errorf("run 2 = %+v", r)
	}
	if r := pg.Runs[2]; r.Run == nil || r.Run.ID != runID || r.Fixed != 1 || r.Kind != RunMention ||
		r.Detail == nil || r.Detail.PRScope == nil || r.Detail.PRScope.Files != 3 {
		t.Errorf("run 3 = %+v", r)
	}
	if len(pg.Findings) != 2 || pg.Findings[0].RunID != earlier || pg.Findings[0].FixedRunID == nil ||
		*pg.Findings[0].FixedRunID != runID || pg.Findings[1].RunID != runID || pg.Findings[1].FixedRunID != nil {
		t.Errorf("findings = %+v", pg.Findings)
	}

	d, ok, err := s.PRRun(ctx, token, 3)
	if err != nil || !ok || d.ID != last || d.N != 3 || d.PRTitle != "t" || d.Key != key {
		t.Fatalf("PRRun 3 = %+v, %v, %v", d, ok, err)
	}
	if d.Run == nil || d.Run.ID != runID || d.Run.CostNanoUSD == nil || *d.Run.CostNanoUSD != 900 {
		t.Errorf("run = %+v", d.Run)
	}
	if d.Call == nil || d.Call.CallID != "c1" || d.Detail == nil || len(d.Detail.Stages) != 1 {
		t.Errorf("call = %+v, detail = %+v", d.Call, d.Detail)
	}
	// The full text: the page is reached only by its PR's token.
	if len(d.Findings) != 1 || d.Findings[0].Headline != "new one" || d.Findings[0].Line != 7 ||
		d.Findings[0].CommentText != "the body" || d.Findings[0].Suggestion == nil || *d.Findings[0].Suggestion != "do this" {
		t.Errorf("findings = %+v", d.Findings)
	}
	if len(d.Fixed) != 1 || d.Fixed[0].Headline != "old one" || d.Fixed[0].CommentText != "SECRET BODY" {
		t.Errorf("fixed = %+v", d.Fixed)
	}

	// A skip has no run, no call, no findings, and is still a page.
	d, ok, err = s.PRRun(ctx, token, 2)
	if err != nil || !ok || d.ID != skip || d.Run != nil || d.Call != nil || len(d.Findings) != 0 || d.Reason != "head_unchanged" {
		t.Errorf("skip = %+v, %v, %v", d, ok, err)
	}
	if d, ok, err := s.PRRun(ctx, token, 1); err != nil || !ok || d.ID != first {
		t.Errorf("PRRun 1 = %d, %v, %v", d.ID, ok, err)
	}
	for _, n := range []int{0, -1, 4} {
		if _, ok, err := s.PRRun(ctx, token, n); ok || err != nil {
			t.Errorf("run %d = %v, %v; want not found", n, ok, err)
		}
	}
	for _, tok := range []string{NewLinkToken(), ""} {
		if _, ok, err := s.PRPage(ctx, tok); ok || err != nil {
			t.Errorf("PRPage(%q) = %v, %v; want not found", tok, ok, err)
		}
		if _, ok, err := s.PRRun(ctx, tok, 1); ok || err != nil {
			t.Errorf("PRRun(%q) = %v, %v; want not found", tok, ok, err)
		}
	}
}

// The history is the PR row's life: an attempt from before the row (a
// repo purged and claimed again) is not this PR's, as prCostSQL does not
// count it.
func TestPRPageStartsWithTheRow(t *testing.T) {
	s := testStore(t)
	ctx := context.Background()
	old := addAttempt(t, s, Attempt{Key: key, TeamSlug: "platform", Kind: RunAuto, Outcome: "skipped", Reason: "draft"})
	if _, err := s.pool.Exec(ctx, `UPDATE review_attempts SET created_at = now() - interval '1 day' WHERE id = $1`, old); err != nil {
		t.Fatal(err)
	}
	prID := upsert(t, s, key, "a")
	addAttempt(t, s, Attempt{Key: key, TeamSlug: "platform", Kind: RunAuto, Outcome: "error"})
	pg, _, err := s.PRPage(ctx, prToken(t, s, prID))
	if err != nil || len(pg.Runs) != 1 || pg.Runs[0].Outcome != "error" || pg.Runs[0].N != 1 {
		t.Errorf("runs = %+v, %v", pg.Runs, err)
	}
}

// A long history keeps the newest, numbered as in the whole history, and
// says it was cut.
func TestPRPageHistoryCap(t *testing.T) {
	s := testStore(t)
	ctx := context.Background()
	prID := upsert(t, s, key, "a")
	extra := 5
	for range prRunsLimit + extra {
		addAttempt(t, s, Attempt{Key: key, TeamSlug: "platform", Kind: RunAuto, Outcome: "skipped", Reason: "head_unchanged"})
	}
	token := prToken(t, s, prID)
	pg, _, err := s.PRPage(ctx, token)
	if err != nil || !pg.More || len(pg.Runs) != prRunsLimit || pg.Runs[0].N != extra+1 ||
		pg.Runs[len(pg.Runs)-1].N != prRunsLimit+extra {
		t.Fatalf("runs = %d (first %d), more = %v, %v", len(pg.Runs), pg.Runs[0].N, pg.More, err)
	}
	// A cut run still has its page.
	if _, ok, err := s.PRRun(ctx, token, 1); !ok || err != nil {
		t.Errorf("run 1 = %v, %v", ok, err)
	}
}

// A PR row draws its token once: an update keeps it, rows never share one,
// and migration 0015's default fills an insert that names none.
func TestPRLinkToken(t *testing.T) {
	s := testStore(t)
	ctx := context.Background()
	id := upsert(t, s, key, "a")
	tok := prToken(t, s, id)
	if !LinkTokenRE.MatchString(tok) {
		t.Fatalf("token %q malformed", tok)
	}
	if upsert(t, s, key, "b") != id || prToken(t, s, id) != tok {
		t.Error("an update changed the PR's token")
	}
	other := upsert(t, s, PRKey{Project: "PLAT", Repo: "api", PRID: 8}, "a")
	if prToken(t, s, other) == tok {
		t.Error("two PRs share a token")
	}
	var def string
	if err := s.pool.QueryRow(ctx, `INSERT INTO pull_requests (project_key, repo_slug, pr_id, team_slug)
		VALUES ('P', 'r', 1, 't') RETURNING link_token`).Scan(&def); err != nil {
		t.Fatal(err)
	}
	if !LinkTokenRE.MatchString(def) || def == tok {
		t.Errorf("default token %q malformed or repeated", def)
	}
}

// Migration 0013's column default still fills every attempt: the column is
// NOT NULL and nothing names it any more.
func TestLinkTokenColumnDefault(t *testing.T) {
	s := testStore(t)
	ctx := context.Background()
	seen := map[string]bool{}
	for range 20 {
		var tok string
		if err := s.pool.QueryRow(ctx, `INSERT INTO review_attempts (team_slug, project_key, repo_slug, pr_id, kind, outcome)
			VALUES ('t', 'P', 'r', 1, 'auto', 'error') RETURNING link_token`).Scan(&tok); err != nil {
			t.Fatal(err)
		}
		if !LinkTokenRE.MatchString(tok) || seen[tok] {
			t.Fatalf("default token %q: malformed or repeated", tok)
		}
		seen[tok] = true
	}
}

func strp(s string) *string { return &s }

// The detail record round-trips as the review wrote it; an attempt without
// one reads none, a second write replaces the first, and a record in a shape
// this binary cannot read leaves the detail empty instead of failing.
func TestRunDetailRecordRoundTrip(t *testing.T) {
	s := testStore(t)
	ctx := context.Background()
	token := prToken(t, s, upsert(t, s, key, "a"))
	id := addAttempt(t, s, Attempt{Key: key, TeamSlug: "platform", Kind: RunAuto, Outcome: "ok"})
	if d, _, err := s.PRRun(ctx, token, 1); err != nil || d.Detail != nil {
		t.Fatalf("no record yet: %+v, %v", d.Detail, err)
	}
	for _, rec := range []Pipeline{{}, {Stages: []StageTime{{Name: "diff", MS: 3}},
		HTTP: []HTTPCount{{Upstream: "bitbucket", Calls: 4}}}} {
		if err := s.InsertRunDetail(ctx, id, rec); err != nil {
			t.Fatal(err)
		}
	}
	d, _, err := s.PRRun(ctx, token, 1)
	if err != nil || d.Detail == nil || len(d.Detail.Stages) != 1 || d.Detail.HTTP[0].Calls != 4 {
		t.Errorf("detail = %+v, %v; want the second record", d.Detail, err)
	}
	if _, err := s.pool.Exec(ctx, `UPDATE run_details SET detail = '{"http":{"bitbucket":4}}' WHERE attempt_id = $1`, id); err != nil {
		t.Fatal(err)
	}
	if d, ok, err := s.PRRun(ctx, token, 1); err != nil || !ok || d.Detail != nil || d.DetailErr == nil {
		t.Errorf("old shape = %+v, %v, %v; want the page without the record", d.Detail, ok, err)
	}
	// The PR page lists the run without it.
	if pg, _, err := s.PRPage(ctx, token); err != nil || len(pg.Runs) != 1 || pg.Runs[0].Detail != nil {
		t.Errorf("PR page = %+v, %v", pg.Runs, err)
	}
}
