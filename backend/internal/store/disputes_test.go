package store

import (
	"context"
	"testing"
)

// disputedFinding stores one posted finding on the PR k (upserting it) and
// returns its id.
func disputedFinding(t *testing.T, s *Store, k PRKey, comment int) int64 {
	t.Helper()
	ctx := context.Background()
	id := upsert(t, s, k, "a")
	r := run(t, s, id, "a", nil, "m")
	if err := s.InsertFinding(ctx, Finding{PullRequestID: id, RunID: r, FilePath: "a.go", LineNumber: comment,
		Severity: "issue", CommentText: "unchecked", BitbucketCommentID: &comment}); err != nil {
		t.Fatal(err)
	}
	var fid int64
	if err := s.pool.QueryRow(ctx, `SELECT id FROM findings WHERE pull_request_id = $1 AND line_number = $2`, id, comment).Scan(&fid); err != nil {
		t.Fatal(err)
	}
	return fid
}

// Occurrences count per distinct PR inside the window, the current PR left
// out; an old one drops out; a dormant signal is not offered; the newest
// wording stands; another team's repo neither reads nor extends it.
func TestDisputeSignals_WindowPerPRAndTeam(t *testing.T) {
	s := testStore(t)
	ctx := context.Background()
	pr := func(n int) PRKey { k := key; k.PRID = n; return k }

	f1 := disputedFinding(t, s, pr(1), 101)
	sig, err := s.RecordDisputeOccurrence(ctx, "platform", pr(1), 0, f1, "- Validation lives in the UI.")
	if err != nil || sig == 0 {
		t.Fatalf("new signal = %d, %v", sig, err)
	}
	// A second finding of the same PR: still one PR.
	f1b := disputedFinding(t, s, pr(1), 102)
	f2 := disputedFinding(t, s, pr(2), 201)
	f3 := disputedFinding(t, s, pr(3), 301)
	for _, c := range []struct {
		k    PRKey
		f    int64
		rule string
	}{{pr(1), f1b, "- Validation lives in the UI."}, {pr(2), f2, "- Validation lives in the UI only."},
		{pr(3), f3, "- Request payloads are validated in the UI; handlers do not re-validate."}} {
		if got, err := s.RecordDisputeOccurrence(ctx, "platform", c.k, sig, c.f, c.rule); err != nil || got != sig {
			t.Fatalf("record on %v = %d, %v", c.k, got, err)
		}
	}

	open, err := s.OpenDisputeSignals(ctx, "platform", key.Project, key.Repo, 30, 20)
	if err != nil || len(open) != 1 || open[0].ID != sig ||
		open[0].Rule != "- Request payloads are validated in the UI; handlers do not re-validate." {
		t.Fatalf("open = %+v, %v", open, err)
	}
	prs, err := s.DisputeSignalPRs(ctx, sig, 30, pr(3))
	if err != nil || len(prs) != 2 || prs[0].PRID != 2 || prs[1].PRID != 1 ||
		prs[0].CommentID == nil || *prs[0].CommentID != 201 || prs[0].Author != "alice" {
		t.Fatalf("prs without #3 = %+v, %v", prs, err)
	}

	// Another team, another repo: nothing to read, nothing to extend.
	if got, _ := s.OpenDisputeSignals(ctx, "payments", key.Project, key.Repo, 30, 20); len(got) != 0 {
		t.Errorf("another team reads %+v", got)
	}
	if got, err := s.RecordDisputeOccurrence(ctx, "payments", pr(3), sig, f3, "- x"); err != nil || got != 0 {
		t.Errorf("another team extends the signal: %d, %v", got, err)
	}

	// PR 1's occurrences fall out of the window: two PRs left.
	if _, err := s.pool.Exec(ctx, `UPDATE dispute_occurrences SET created_at = now() - interval '31 days'
		WHERE finding_id = ANY($1)`, []int64{f1, f1b}); err != nil {
		t.Fatal(err)
	}
	if prs, _ := s.DisputeSignalPRs(ctx, sig, 30, PRKey{}); len(prs) != 2 {
		t.Errorf("after expiry = %+v, want PRs 3 and 2", prs)
	}
	// All of them: dormant, not offered.
	if _, err := s.pool.Exec(ctx, `UPDATE dispute_occurrences SET created_at = now() - interval '31 days'`); err != nil {
		t.Fatal(err)
	}
	if got, _ := s.OpenDisputeSignals(ctx, "platform", key.Project, key.Repo, 30, 20); len(got) != 0 {
		t.Errorf("a dormant signal is offered: %+v", got)
	}
	// A wider window sees them again: expiry is a read, not a delete.
	if got, _ := s.OpenDisputeSignals(ctx, "platform", key.Project, key.Repo, 60, 20); len(got) != 1 {
		t.Errorf("60-day window = %+v", got)
	}
}

// The limit holds, newest first; occurrences go with their PR.
func TestDisputeSignals_LimitAndCascade(t *testing.T) {
	s := testStore(t)
	ctx := context.Background()
	f := disputedFinding(t, s, key, 1)
	var ids []int64
	for _, rule := range []string{"- a", "- b", "- c"} {
		id, err := s.RecordDisputeOccurrence(ctx, "platform", key, 0, f, rule)
		if err != nil {
			t.Fatal(err)
		}
		ids = append(ids, id)
	}
	if _, err := s.pool.Exec(ctx, `UPDATE dispute_occurrences SET created_at = now() - interval '1 hour' WHERE signal_id = $1`, ids[2]); err != nil {
		t.Fatal(err)
	}
	got, err := s.OpenDisputeSignals(ctx, "platform", key.Project, key.Repo, 30, 2)
	if err != nil || len(got) != 2 || got[0].ID != ids[1] || got[1].ID != ids[0] {
		t.Fatalf("limited = %+v, %v; ids %v", got, err, ids)
	}

	if _, err := s.pool.Exec(ctx, `DELETE FROM pull_requests`); err != nil {
		t.Fatal(err)
	}
	var n int
	if err := s.pool.QueryRow(ctx, `SELECT count(*) FROM dispute_occurrences`).Scan(&n); err != nil || n != 0 {
		t.Errorf("occurrences after the PR's delete = %d, %v", n, err)
	}
	if got, _ := s.OpenDisputeSignals(ctx, "platform", key.Project, key.Repo, 30, 20); len(got) != 0 {
		t.Errorf("signals without occurrences are offered: %+v", got)
	}
}

// A purge takes the scope's signals with its PRs; another repo keeps its own.
func TestDisputeSignals_GoWithAPurge(t *testing.T) {
	s := testStore(t)
	ctx := context.Background()
	other := PRKey{Project: key.Project, Repo: "web", PRID: 1}
	for _, k := range []PRKey{key, other} {
		f := disputedFinding(t, s, k, 1)
		if _, err := s.RecordDisputeOccurrence(ctx, "platform", k, 0, f, "- a"); err != nil {
			t.Fatal(err)
		}
	}
	repo := key.Repo
	if _, err := s.PurgeProject(ctx, "platform", key.Project, &repo); err != nil {
		t.Fatal(err)
	}
	var n int
	if err := s.pool.QueryRow(ctx, `SELECT count(*) FROM dispute_signals WHERE repo_slug = $1`, key.Repo).Scan(&n); err != nil || n != 0 {
		t.Errorf("purged repo keeps %d signals, %v", n, err)
	}
	if got, _ := s.OpenDisputeSignals(ctx, "platform", other.Project, other.Repo, 30, 20); len(got) != 1 {
		t.Errorf("another repo lost its signal: %+v", got)
	}
}
