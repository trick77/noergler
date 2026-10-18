package store

import (
	"context"
	"testing"
)

// pr_status reads this row's life and this team's attempts only: a purged
// repo claimed again, maybe by another team, starts from nothing.
func TestPRStatus_BoundedToTheRowAndTeam(t *testing.T) {
	s := testStore(t)
	ctx := context.Background()
	k := PRKey{Project: "PROJ", Repo: "r", PRID: 7}

	if st, err := s.PRStatus(ctx, k, "payments"); err != nil || st.Known || len(st.Attempts) != 0 {
		t.Fatalf("empty = %+v, %v", st, err)
	}
	// Before the row: an older life of the key, and another team's.
	for _, team := range []string{"payments", "platform"} {
		if _, err := s.pool.Exec(ctx, `INSERT INTO review_attempts (team_slug, project_key, repo_slug, pr_id, kind, outcome, reason, created_at)
			VALUES ($1, 'PROJ', 'r', 7, 'auto', 'skipped', 'old', now() - interval '1 day')`, team); err != nil {
			t.Fatal(err)
		}
	}
	if st, _ := s.PRStatus(ctx, k, "payments"); len(st.Attempts) != 1 || st.Attempts[0].Reason != "old" {
		t.Fatalf("no row yet: attempts = %+v, want the team's one", st.Attempts)
	}

	author := "alice"
	if _, err := s.UpsertPullRequest(ctx, PRUpsert{Key: k, TeamSlug: "payments", Author: &author}); err != nil {
		t.Fatal(err)
	}
	if _, err := s.InsertAttempt(ctx, Attempt{Key: k, TeamSlug: "payments", Kind: RunAuto, Outcome: "skipped", Reason: "pr_cost_cap"}); err != nil {
		t.Fatal(err)
	}
	st, err := s.PRStatus(ctx, k, "payments")
	if err != nil || !st.Known {
		t.Fatalf("after upsert = %+v, %v", st, err)
	}
	if len(st.Attempts) != 1 || st.Attempts[0].Reason != "pr_cost_cap" {
		t.Errorf("attempts = %+v, want only the one from this row's life", st.Attempts)
	}
	// Another team asking sees neither the row nor its attempts.
	if other, err := s.PRStatus(ctx, k, "platform"); err != nil || other.Known || len(other.Attempts) != 1 {
		t.Errorf("another team = %+v, %v, want no row and only its own attempt", other, err)
	}
}
