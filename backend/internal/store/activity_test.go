package store

import (
	"context"
	"testing"
	"time"
)

// A SNAPSHOT task counts as a task, an engagement as an engagement, each
// comment once; a PR without a row records nothing; the schema refuses an
// engagement without its trigger and outcome.
func TestActivityEventsCountOnce(t *testing.T) {
	s := testStore(t)
	ctx := context.Background()
	id := upsert(t, s, key, "a")

	if err := s.RecordSnapshotTasks(ctx, id, []int{11, 12}); err != nil {
		t.Fatal(err)
	}
	if err := s.RecordSnapshotTasks(ctx, id, []int{12}); err != nil {
		t.Fatal(err)
	}
	for _, e := range []struct {
		id               int
		trigger, outcome string
	}{{21, EngagementMention, EngagementAnswered}, {22, EngagementReply, EngagementAuthorOnly}, {21, EngagementMention, EngagementFailed},
		{23, EngagementMention, EngagementFailed}, {23, EngagementMention, EngagementAnswered}} {
		if noRow, err := s.RecordEngagement(ctx, key, e.id, e.trigger, e.outcome); err != nil || noRow {
			t.Fatalf("record %d: noRow %v, %v", e.id, noRow, err)
		}
	}
	other := key
	other.PRID++
	if noRow, err := s.RecordEngagement(ctx, other, 31, EngagementMention, EngagementAnswered); err != nil || !noRow {
		t.Fatalf("a PR without a row: noRow %v, %v; want true so the caller writes it", noRow, err)
	}
	if _, err := s.pool.Exec(ctx, `INSERT INTO activity_events (pull_request_id, kind, comment_id) VALUES ($1, 'engagement', 40)`, id); err == nil {
		t.Error("an engagement without trigger and outcome must be refused by the schema")
	}

	counts, err := s.ActivitySince(ctx, time.Now().Add(-time.Hour))
	if err != nil {
		t.Fatal(err)
	}
	byKind := map[string]int{}
	for _, c := range counts {
		if c.TeamSlug != "platform" {
			t.Errorf("row %+v: want the PR's team", c)
		}
		byKind[c.Kind] += c.Count
	}
	if len(byKind) != 2 || byKind[ActivityTask] != 2 || byKind[ActivityEngagement] != 3 {
		t.Errorf("counts = %+v, want 2 tasks and 3 engagements", byKind)
	}
	// An answer stands; a failure handled again takes the later outcome.
	for id, want := range map[int]string{21: EngagementAnswered, 23: EngagementAnswered} {
		var outcome string
		if err := s.pool.QueryRow(ctx, `SELECT outcome FROM activity_events WHERE comment_id = $1`, id).Scan(&outcome); err != nil ||
			outcome != want {
			t.Errorf("comment %d = %q, %v, want %q", id, outcome, err, want)
		}
	}
}
