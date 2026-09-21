package queue

import (
	"context"
	"log/slog"
	"sync"
	"testing"

	"github.com/trick77/noergler/internal/store"
	"github.com/trick77/noergler/internal/webhook"
)

func discardLog() *slog.Logger {
	return slog.New(slog.DiscardHandler)
}

// Depth deliberately excludes the in-flight item, so a snapshot that reported
// only a depth could not tell "nothing queued" from "nothing running". The
// two lists are separate for that reason.
func TestSnapshotSeparatesRunningFromWaiting(t *testing.T) {
	q := New(func(context.Context, string, *webhook.Payload, Scheduler) bool { return false }, 6, 2, discardLog())

	running := store.PRKey{Project: "PAY", Repo: "ledger", PRID: 1}
	hold(q, running)
	q.Submit(store.PRKey{Project: "PAY", Repo: "ledger", PRID: 2}, &webhook.Payload{}, "payments")

	s := q.Snapshot()

	if s.Capacity != 6 || s.PerTeam != 2 {
		t.Errorf("pool = %d/%d, want 6/2", s.Capacity, s.PerTeam)
	}
	if len(s.Running) != 1 || s.Running[0].Key != running {
		t.Fatalf("running = %+v, want just %v", s.Running, running)
	}
	if s.Running[0].Team != "payments" {
		t.Errorf("a running item must name its team, got %q", s.Running[0].Team)
	}
	if len(s.Waiting) != 1 {
		t.Fatalf("waiting = %d, want 1", len(s.Waiting))
	}
	if s.Waiting[0].Team != "payments" {
		t.Errorf("a waiting item must name its team, got %q", s.Waiting[0].Team)
	}
	if s.Depth != q.Depth() {
		t.Errorf("snapshot depth %d disagrees with Depth() %d", s.Depth, q.Depth())
	}
}

// The snapshot must not hand out the queue's own storage: a caller that
// ranged over it while the worker dequeued would race.
func TestSnapshotDoesNotAliasQueueState(t *testing.T) {
	q := New(func(context.Context, string, *webhook.Payload, Scheduler) bool { return false }, 2, 1, discardLog())
	key := store.PRKey{Project: "PAY", Repo: "ledger", PRID: 7}
	q.Submit(key, &webhook.Payload{}, "payments")

	s := q.Snapshot()
	s.Waiting[0].Team = "mutated"

	if got := q.Snapshot(); got.Waiting[0].Team != "payments" {
		t.Errorf("mutating a snapshot changed the queue: team = %q", got.Waiting[0].Team)
	}
}

// The dashboard polls Snapshot while the worker dequeues, holds and releases
// keys. It takes q.mu like every other reader, but the copy it makes is the
// part worth proving: a snapshot that handed out the queue's own slice or
// map would race here rather than at the lock.
func TestSnapshotRacesNothingWhileTheWorkerRuns(t *testing.T) {
	started := make(chan struct{}, 64)
	q := New(func(_ context.Context, _ string, _ *webhook.Payload, _ Scheduler) bool {
		select {
		case started <- struct{}{}:
		default:
		}
		return false
	}, 4, 2, discardLog())

	q.Start(context.Background())
	t.Cleanup(q.Stop)

	var wg sync.WaitGroup
	wg.Add(1)
	go func() {
		defer wg.Done()
		for i := range 200 {
			q.Submit(
				store.PRKey{Project: "PAY", Repo: "ledger", PRID: i % 17},
				&webhook.Payload{}, "payments",
			)
		}
	}()

	wg.Add(1)
	go func() {
		defer wg.Done()
		for range 400 {
			s := q.Snapshot()
			// Read every field the handler reads, so the race detector sees
			// the same accesses a real request makes.
			_ = s.Capacity + s.PerTeam + s.Staged + s.Depth
			for _, it := range s.Running {
				_ = it.Key.Tag() + it.Team + it.Kind
			}
			for _, it := range s.Waiting {
				_ = it.Key.Tag() + it.Team
			}
		}
	}()

	wg.Wait()
}
