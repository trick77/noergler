package queue

import (
	"context"
	"errors"
	"io"
	"log/slog"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/trick77/noergler/internal/logging"
	"github.com/trick77/noergler/internal/store"
	"github.com/trick77/noergler/internal/webhook"
)

func quietLogger() *slog.Logger {
	return slog.New(slog.NewTextHandler(io.Discard, nil))
}

func key(id int) store.PRKey {
	return store.PRKey{Project: "P", Repo: "r", PRID: id}
}

func payload(title string) *webhook.Payload {
	p := &webhook.Payload{EventKey: webhook.EventOpened}
	p.PullRequest.Title = title
	return p
}

// waitFor polls until cond holds or the deadline passes, so the tests do not
// depend on a fixed sleep.
func waitFor(t *testing.T, cond func() bool) {
	t.Helper()
	deadline := time.Now().Add(2 * time.Second)
	for time.Now().Before(deadline) {
		if cond() {
			return
		}
		time.Sleep(time.Millisecond)
	}
	t.Fatal("condition not met within the deadline")
}

func TestSingleReviewRunsAndCompletes(t *testing.T) {
	var mu sync.Mutex
	var seen []string

	q := New(func(_ context.Context, _ string, p *webhook.Payload) {
		mu.Lock()
		seen = append(seen, p.PullRequest.Title)
		mu.Unlock()
	}, quietLogger())
	q.Start(context.Background())
	defer q.Stop()

	q.Submit(key(1), payload("first"), "t1")
	waitFor(t, func() bool { mu.Lock(); defer mu.Unlock(); return len(seen) == 1 })

	if seen[0] != "first" {
		t.Errorf("reviewed %q", seen[0])
	}
}

// The worker is one long-lived goroutine, so the team must be bound per item
// rather than once at start.
func TestWorkerBindsTeamPerItem(t *testing.T) {
	var mu sync.Mutex
	var teams []string

	record := func(ctx context.Context) {
		mu.Lock()
		defer mu.Unlock()
		for _, a := range logging.Bound(ctx) {
			if a.Key == "team" {
				teams = append(teams, a.Value.String())
				return
			}
		}
		teams = append(teams, "<unbound>")
	}

	q := New(func(ctx context.Context, _ string, _ *webhook.Payload) { record(ctx) }, quietLogger())
	q.Start(context.Background())
	defer q.Stop()

	q.Submit(key(1), payload("a"), "payments")
	waitFor(t, func() bool { mu.Lock(); defer mu.Unlock(); return len(teams) == 1 })
	q.SubmitJob("P/r#2", "billing", record)
	waitFor(t, func() bool { mu.Lock(); defer mu.Unlock(); return len(teams) == 2 })

	mu.Lock()
	defer mu.Unlock()
	if teams[0] != "payments" || teams[1] != "billing" {
		t.Errorf("team bindings = %v, want [payments billing]", teams)
	}
}

// A burst on one PR collapses to the in-flight review plus the latest state.
func TestDedupeCollapsesPendingSubmits(t *testing.T) {
	release := make(chan struct{})
	var mu sync.Mutex
	var order []string

	q := New(func(_ context.Context, _ string, p *webhook.Payload) {
		mu.Lock()
		order = append(order, p.PullRequest.Title)
		first := len(order) == 1
		mu.Unlock()
		if first {
			<-release // hold the worker so the next submits queue up
		}
	}, quietLogger())
	q.Start(context.Background())
	defer q.Stop()

	q.Submit(key(1), payload("first"), "t1")
	waitFor(t, func() bool { mu.Lock(); defer mu.Unlock(); return len(order) == 1 })

	if got := q.Submit(key(1), payload("second"), "t1"); got != StatusQueued {
		t.Errorf("first queued submit = %q, want %q", got, StatusQueued)
	}
	if got := q.Submit(key(1), payload("third"), "t1"); got != StatusSuperseded {
		t.Errorf("second submit = %q, want %q", got, StatusSuperseded)
	}

	close(release)
	waitFor(t, func() bool { mu.Lock(); defer mu.Unlock(); return len(order) == 2 })

	mu.Lock()
	defer mu.Unlock()
	if order[1] != "third" {
		t.Errorf("deduped payload = %q, want the latest (third)", order[1])
	}
}

// One worker means one review at a time, whatever the PR.
func TestReviewsAreSerializedAcrossPRs(t *testing.T) {
	var inFlight, maxSeen atomic.Int32
	var done atomic.Int32

	q := New(func(_ context.Context, _ string, _ *webhook.Payload) {
		n := inFlight.Add(1)
		for {
			m := maxSeen.Load()
			if n <= m || maxSeen.CompareAndSwap(m, n) {
				break
			}
		}
		time.Sleep(5 * time.Millisecond)
		inFlight.Add(-1)
		done.Add(1)
	}, quietLogger())
	q.Start(context.Background())
	defer q.Stop()

	for i := 1; i <= 5; i++ {
		q.Submit(key(i), payload("p"), "t1")
	}
	waitFor(t, func() bool { return done.Load() == 5 })

	if got := maxSeen.Load(); got != 1 {
		t.Errorf("max concurrent reviews = %d, want 1", got)
	}
}

// A panic in the review path must not take the only worker down for every
// team. The Python cannot panic this way; this is the deliberate addition.
func TestPanicInReviewDoesNotKillTheWorker(t *testing.T) {
	var mu sync.Mutex
	var seen []int

	q := New(func(_ context.Context, _ string, _ *webhook.Payload) {
		mu.Lock()
		n := len(seen)
		mu.Unlock()
		if n == 0 {
			mu.Lock()
			seen = append(seen, 1)
			mu.Unlock()
			var s []int
			//nolint:govet // nilness: the panic this test exists to trigger
			_ = s[3] // the Phase 4 shape: a slice index out of range
			return
		}
		mu.Lock()
		seen = append(seen, 2)
		mu.Unlock()
	}, quietLogger())
	q.Start(context.Background())
	defer q.Stop()

	q.Submit(key(1), payload("boom"), "t1")
	waitFor(t, func() bool { mu.Lock(); defer mu.Unlock(); return len(seen) == 1 })
	q.Submit(key(2), payload("fine"), "t1")
	waitFor(t, func() bool { mu.Lock(); defer mu.Unlock(); return len(seen) == 2 })
}

func TestPanicInJobDoesNotKillTheWorker(t *testing.T) {
	var ran atomic.Int32

	q := New(func(context.Context, string, *webhook.Payload) {}, quietLogger())
	q.Start(context.Background())
	defer q.Stop()

	q.SubmitJob("P/r#1", "t1", func(context.Context) { panic(errors.New("job exploded")) })
	q.SubmitJob("P/r#2", "t1", func(context.Context) { ran.Add(1) })

	waitFor(t, func() bool { return ran.Load() == 1 })
}

// Jobs share the worker in arrival order and are never deduped.
func TestJobsShareTheWorkerInArrivalOrderWithoutDedupe(t *testing.T) {
	release := make(chan struct{})
	var mu sync.Mutex
	var order []string

	q := New(func(_ context.Context, _ string, _ *webhook.Payload) {
		mu.Lock()
		order = append(order, "review:1")
		mu.Unlock()
		<-release
	}, quietLogger())
	q.Start(context.Background())
	defer q.Stop()

	q.Submit(key(1), payload("p"), "t1")
	waitFor(t, func() bool { mu.Lock(); defer mu.Unlock(); return len(order) == 1 })

	add := func(label string) JobFunc {
		return func(context.Context) {
			mu.Lock()
			order = append(order, "job:"+label)
			mu.Unlock()
		}
	}
	// The same tag twice: jobs are never deduped.
	if got := q.SubmitJob("P/r#1", "t2", add("a")); got != StatusQueued {
		t.Errorf("SubmitJob = %q", got)
	}
	if got := q.SubmitJob("P/r#1", "t2", add("b")); got != StatusQueued {
		t.Errorf("second SubmitJob with the same tag = %q, want queued", got)
	}

	mu.Lock()
	if len(order) != 1 {
		t.Errorf("jobs must wait for the running review, order = %v", order)
	}
	mu.Unlock()

	close(release)
	waitFor(t, func() bool { mu.Lock(); defer mu.Unlock(); return len(order) == 3 })

	mu.Lock()
	defer mu.Unlock()
	if order[1] != "job:a" || order[2] != "job:b" {
		t.Errorf("jobs out of arrival order: %v", order)
	}
}

func TestSubmitReturnsOutcome(t *testing.T) {
	q := New(func(context.Context, string, *webhook.Payload) {}, quietLogger())
	// Not started: nothing drains, so the first stays pending.
	if got := q.Submit(key(1), payload("a"), "t1"); got != StatusQueued {
		t.Errorf("first = %q, want queued", got)
	}
	if got := q.Submit(key(1), payload("b"), "t1"); got != StatusSuperseded {
		t.Errorf("second = %q, want superseded", got)
	}
	if got := q.Submit(key(2), payload("c"), "t1"); got != StatusQueued {
		t.Errorf("other PR = %q, want queued", got)
	}
}

// Depth excludes the item in flight, as Python's qsize() does.
func TestDepthExcludesTheItemInFlight(t *testing.T) {
	release := make(chan struct{})
	started := make(chan struct{})

	q := New(func(context.Context, string, *webhook.Payload) {
		close(started)
		<-release
	}, quietLogger())
	q.Start(context.Background())
	defer q.Stop()

	q.Submit(key(1), payload("a"), "t1")
	<-started
	q.Submit(key(2), payload("b"), "t1")

	if got := q.Depth(); got != 1 {
		t.Errorf("Depth() = %d, want 1 (the in-flight item is not counted)", got)
	}
	close(release)
}

func TestStopIsIdempotentAndSafeBeforeStart(_ *testing.T) {
	q := New(func(context.Context, string, *webhook.Payload) {}, quietLogger())
	q.Stop() // never started

	q2 := New(func(context.Context, string, *webhook.Payload) {}, quietLogger())
	q2.Start(context.Background())
	q2.Stop()
	q2.Stop()
}

func TestStartIsIdempotent(t *testing.T) {
	var runs atomic.Int32
	q := New(func(context.Context, string, *webhook.Payload) { runs.Add(1) }, quietLogger())
	q.Start(context.Background())
	q.Start(context.Background())
	defer q.Stop()

	q.Submit(key(1), payload("a"), "t1")
	waitFor(t, func() bool { return runs.Load() == 1 })
	time.Sleep(20 * time.Millisecond)
	if got := runs.Load(); got != 1 {
		t.Errorf("review ran %d times, want 1: a second worker is draining the queue", got)
	}
}

// Submit must never block, whatever the backlog: in Phase 7 the caller is an
// inbound HTTP request goroutine.
func TestSubmitNeverBlocks(t *testing.T) {
	q := New(func(context.Context, string, *webhook.Payload) {}, quietLogger())
	// Deliberately not started, so nothing drains.
	done := make(chan struct{})
	go func() {
		for i := 1; i <= 500; i++ {
			q.Submit(key(i), payload("p"), "t1")
		}
		close(done)
	}()
	select {
	case <-done:
	case <-time.After(2 * time.Second):
		t.Fatal("Submit blocked on a backlog")
	}
	if got := q.Depth(); got != 500 {
		t.Errorf("Depth() = %d, want 500", got)
	}
}
