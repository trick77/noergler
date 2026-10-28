package store

import (
	"context"
	"testing"
)

// A pair's line is read back for its PR only, the first stored line stands,
// a merged PR reads nothing, and the lines go with the PR row.
func TestVisualChanges_CachedPerPairAndGoneWithThePR(t *testing.T) {
	s := testStore(t)
	ctx := context.Background()
	id := upsert(t, s, key, "a")

	added := VisualPair{Path: "t/a-snapshots/new.png", AfterID: "bb"}
	changed := VisualPair{Path: "t/a-snapshots/mod.png", BeforeID: "aa", AfterID: "cc"}
	for _, c := range []struct {
		p    VisualPair
		text string
	}{{added, "new login form"}, {changed, "button moved"}, {changed, "a rerun must not replace it"}} {
		if err := s.InsertVisualChange(ctx, key, c.p, c.text, "m"); err != nil {
			t.Fatal(err)
		}
	}
	got, err := s.VisualChanges(ctx, key)
	if err != nil {
		t.Fatal(err)
	}
	if len(got) != 2 || got[added] != "new login form" || got[changed] != "button moved" {
		t.Fatalf("visual changes = %v", got)
	}

	other := key
	other.PRID++
	if got, err := s.VisualChanges(ctx, other); err != nil || len(got) != 0 {
		t.Errorf("another PR reads %v, %v", got, err)
	}
	// A PR without a row stores nothing and is no error.
	if err := s.InsertVisualChange(ctx, other, added, "x", "m"); err != nil {
		t.Errorf("insert for a PR without a row = %v, want none", err)
	}

	if err := s.MarkMerged(ctx, key); err != nil {
		t.Fatal(err)
	}
	if got, _ := s.VisualChanges(ctx, key); len(got) != 0 {
		t.Errorf("a merged PR still reads %v", got)
	}

	if _, err := s.pool.Exec(ctx, `DELETE FROM pull_requests WHERE id = $1`, id); err != nil {
		t.Fatal(err)
	}
	var n int
	if err := s.pool.QueryRow(ctx, `SELECT count(*) FROM visual_changes`).Scan(&n); err != nil || n != 0 {
		t.Errorf("visual_changes after the PR's delete = %d, %v, want none", n, err)
	}
}
