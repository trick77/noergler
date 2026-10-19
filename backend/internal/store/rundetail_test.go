package store

import (
	"context"
	"testing"
)

func TestRunDetail(t *testing.T) {
	s := testStore(t)
	ctx := context.Background()
	prID := upsert(t, s, key, "a")
	earlier := run(t, s, prID, "a", nano(600), "m")
	if err := s.InsertFinding(ctx, Finding{PullRequestID: prID, RunID: earlier, FilePath: "a.go", LineNumber: 3,
		Severity: "medium", Headline: strp("old one"), CommentText: "SECRET BODY"}); err != nil {
		t.Fatal(err)
	}
	runID := run(t, s, prID, "b", nano(900), "m")
	if err := s.InsertFinding(ctx, Finding{PullRequestID: prID, RunID: runID, FilePath: "b.go", LineNumber: 7,
		Severity: "high", Headline: strp("new one"), CommentText: "the body", Suggestion: strp("do this")}); err != nil {
		t.Fatal(err)
	}
	if _, err := s.pool.Exec(ctx, `UPDATE findings SET fixed_at = now(), fixed_run_id = $1 WHERE review_run_id = $2`, runID, earlier); err != nil {
		t.Fatal(err)
	}
	earlierAttempt, _ := tokenAttempt(t, s, Attempt{Key: key, TeamSlug: "platform", Kind: RunAuto, Outcome: "ok", RunID: &earlier})
	skip, skipToken := tokenAttempt(t, s, Attempt{Key: key, TeamSlug: "platform", Kind: RunAuto, Outcome: "skipped", Reason: "head_unchanged"})
	id, token := tokenAttempt(t, s, Attempt{Key: key, TeamSlug: "platform", Kind: RunAuto, Outcome: "ok",
		RunID: &runID, Call: &LLMCall{Profile: "m", CallID: "c1"}})
	// Another team's attempt on the same PR key is not this PR's history.
	if _, err := s.InsertAttempt(ctx, Attempt{Key: key, TeamSlug: "other", Kind: RunAuto, Outcome: "error"}); err != nil {
		t.Fatal(err)
	}

	d, ok, err := s.RunDetail(ctx, token)
	if err != nil || !ok || d.ID != id {
		t.Fatalf("RunDetail = %d, %v, %v", d.ID, ok, err)
	}
	if d.Run == nil || d.Run.ID != runID || d.Run.CostNanoUSD == nil || *d.Run.CostNanoUSD != 900 {
		t.Errorf("run = %+v", d.Run)
	}
	if d.Call == nil || d.Call.CallID != "c1" {
		t.Errorf("call = %+v", d.Call)
	}
	// The full text: the page is reached only by its token.
	if len(d.Findings) != 1 || d.Findings[0].Headline != "new one" || d.Findings[0].Line != 7 ||
		d.Findings[0].CommentText != "the body" || d.Findings[0].Suggestion == nil || *d.Findings[0].Suggestion != "do this" {
		t.Errorf("findings = %+v", d.Findings)
	}
	if len(d.Fixed) != 1 || d.Fixed[0].Headline != "old one" || d.Fixed[0].CommentText != "SECRET BODY" {
		t.Errorf("fixed = %+v", d.Fixed)
	}
	if len(d.PRAttempts) != 3 || d.PRAttempts[0].ID != earlierAttempt || d.PRAttempts[1].ID != skip || d.PRAttempts[2].ID != id {
		t.Errorf("pr attempts = %+v", d.PRAttempts)
	}
	if c := d.PRAttempts[2].CostNanoUSD; c == nil || *c != 900 {
		t.Errorf("the ok attempt's cost = %v, want its run's 900", c)
	}

	// A skip has no run, no call, no findings, and is still a page.
	d, ok, err = s.RunDetail(ctx, skipToken)
	if err != nil || !ok || d.ID != skip || d.Run != nil || d.Call != nil || len(d.Findings) != 0 || d.Reason != "head_unchanged" {
		t.Errorf("skip detail = %+v, %v, %v", d, ok, err)
	}

	if _, ok, err := s.RunDetail(ctx, NewLinkToken()); ok || err != nil {
		t.Errorf("unknown token = %v, %v; want not found", ok, err)
	}
	if _, ok, err := s.RunDetail(ctx, ""); ok || err != nil {
		t.Errorf("empty token = %v, %v; want not found", ok, err)
	}
}

// An attempt inserted without a token gets one drawn, never a NULL or a
// shared value.
func TestInsertAttemptDrawsAToken(t *testing.T) {
	s := testStore(t)
	ctx := context.Background()
	seen := map[string]bool{}
	for range 3 {
		id, err := s.InsertAttempt(ctx, Attempt{Key: key, TeamSlug: "platform", Kind: RunAuto, Outcome: "error"})
		if err != nil {
			t.Fatal(err)
		}
		var tok string
		if err := s.pool.QueryRow(ctx, `SELECT link_token FROM review_attempts WHERE id = $1`, id).Scan(&tok); err != nil {
			t.Fatal(err)
		}
		if !LinkTokenRE.MatchString(tok) || seen[tok] {
			t.Errorf("token %q: malformed or repeated", tok)
		}
		seen[tok] = true
	}
}

// Migration 0013's column default fills an insert that names no token: it
// must yield what the routes accept, distinct per row.
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

// tokenAttempt inserts a with a fresh link token and returns its id and token.
func tokenAttempt(t *testing.T, s *Store, a Attempt) (int64, string) {
	t.Helper()
	a.LinkToken = NewLinkToken()
	id, err := s.InsertAttempt(context.Background(), a)
	if err != nil {
		t.Fatal(err)
	}
	return id, a.LinkToken
}

func strp(s string) *string { return &s }

// The detail record round-trips as the review wrote it; an attempt without
// one reads none, a second write replaces the first, and a record in a shape
// this binary cannot read leaves the detail empty instead of failing.
func TestRunDetailRecordRoundTrip(t *testing.T) {
	s := testStore(t)
	ctx := context.Background()
	id, token := tokenAttempt(t, s, Attempt{Key: key, TeamSlug: "platform", Kind: RunAuto, Outcome: "ok"})
	if d, _, err := s.RunDetail(ctx, token); err != nil || d.Detail != nil {
		t.Fatalf("no record yet: %+v, %v", d.Detail, err)
	}
	for _, rec := range []Pipeline{{}, {Stages: []StageTime{{Name: "diff", MS: 3}},
		HTTP: []HTTPCount{{Upstream: "bitbucket", Calls: 4}}}} {
		if err := s.InsertRunDetail(ctx, id, rec); err != nil {
			t.Fatal(err)
		}
	}
	d, _, err := s.RunDetail(ctx, token)
	if err != nil || d.Detail == nil || len(d.Detail.Stages) != 1 || d.Detail.HTTP[0].Calls != 4 {
		t.Errorf("detail = %+v, %v; want the second record", d.Detail, err)
	}
	if _, err := s.pool.Exec(ctx, `UPDATE run_details SET detail = '{"http":{"bitbucket":4}}' WHERE attempt_id = $1`, id); err != nil {
		t.Fatal(err)
	}
	if d, ok, err := s.RunDetail(ctx, token); err != nil || !ok || d.Detail != nil {
		t.Errorf("old shape = %+v, %v, %v; want the page without the record", d.Detail, ok, err)
	}
}

// A long PR's history is a window around the attempt shown, so the attempt
// and its neighbours are always in it, whichever end of the PR it is at.
func TestRunDetailHistoryWindow(t *testing.T) {
	s := testStore(t)
	ctx := context.Background()
	var ids []int64
	var tokens []string
	for i := 0; i < 2*prAttemptsEachSide+10; i++ {
		id, token := tokenAttempt(t, s, Attempt{Key: key, TeamSlug: "platform", Kind: RunAuto, Outcome: "skipped", Reason: "head_unchanged"})
		ids, tokens = append(ids, id), append(tokens, token)
	}
	// The ends, the middle, and both sides of each cut-off boundary.
	n := len(ids)
	for _, at := range []int{0, 2, prAttemptsEachSide, prAttemptsEachSide + 1, n / 2,
		n - 1 - prAttemptsEachSide, n - 2 - prAttemptsEachSide, n - 1} {
		d, ok, err := s.RunDetail(ctx, tokens[at])
		if err != nil || !ok {
			t.Fatalf("RunDetail = %v, %v", ok, err)
		}
		found := -1
		for i, a := range d.PRAttempts {
			if a.ID == ids[at] {
				found = i
			}
			if i > 0 && d.PRAttempts[i-1].ID >= a.ID {
				t.Fatalf("history not oldest first at %d", i)
			}
		}
		if found < 0 || len(d.PRAttempts) > 2*prAttemptsEachSide+1 {
			t.Errorf("attempt %d: found at %d of %d", at, found, len(d.PRAttempts))
		}
		if at > 0 && at < len(ids)-1 && (found == 0 || found == len(d.PRAttempts)-1) {
			t.Errorf("attempt %d lost a neighbour", at)
		}
		// The page is told when rows are cut, on the side they are cut.
		if wantBefore := at > prAttemptsEachSide; d.MoreBefore != wantBefore {
			t.Errorf("attempt %d: MoreBefore = %v, want %v", at, d.MoreBefore, wantBefore)
		}
		if wantAfter := len(ids)-1-at > prAttemptsEachSide; d.MoreAfter != wantAfter {
			t.Errorf("attempt %d: MoreAfter = %v, want %v", at, d.MoreAfter, wantAfter)
		}
	}
}
