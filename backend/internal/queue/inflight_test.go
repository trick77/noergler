package queue

import (
	"context"
	"sync"
	"testing"

	"github.com/trick77/noergler/internal/store"
	"github.com/trick77/noergler/internal/webhook"
)

// hold marks a key in flight the way the staged pipeline will: the review
// returns while the work is still outstanding, and the hold is released
// later. Until the staged pipeline exists this is the only way to observe
// the set, because a synchronous review always releases before run loops.
func hold(q *Queue, k store.PRKey) {
	q.mu.Lock()
	q.inflight[k] = struct{}{}
	q.mu.Unlock()
}

func held(q *Queue, k store.PRKey) bool {
	q.mu.Lock()
	defer q.mu.Unlock()
	_, ok := q.inflight[k]
	return ok
}

// A held key is skipped, and the items behind it still run. Without the
// scan the worker would take the held item and start a second review of a
// PR whose inference is still outstanding.
func TestHeldKeyIsSkippedAndOthersProceed(t *testing.T) {
	var mu sync.Mutex
	var ran []int

	q := New(func(_ context.Context, _ string, p *webhook.Payload) {
		mu.Lock()
		ran = append(ran, p.PullRequest.ID)
		mu.Unlock()
	}, quietLogger())

	// Hold PR 1 before the worker starts, so the ordering is deterministic.
	hold(q, key(1))

	p1 := payload("one")
	p1.PullRequest.ID = 1
	p2 := payload("two")
	p2.PullRequest.ID = 2

	q.Submit(key(1), p1, "t")
	q.Submit(key(2), p2, "t")
	q.Start(context.Background())
	defer q.Stop()

	waitFor(t, func() bool {
		mu.Lock()
		defer mu.Unlock()
		return len(ran) == 1
	})

	mu.Lock()
	got := ran[0]
	mu.Unlock()
	if got != 2 {
		t.Fatalf("expected the unheld PR 2 to run, got PR %d", got)
	}

	// PR 1 is still queued, not dropped.
	if d := q.Depth(); d != 1 {
		t.Fatalf("held item should still be queued, depth=%d", d)
	}

	// Releasing it lets the worker pick it up without a new Submit.
	q.done1(key(1))
	waitFor(t, func() bool {
		mu.Lock()
		defer mu.Unlock()
		return len(ran) == 2
	})
	mu.Lock()
	defer mu.Unlock()
	if ran[1] != 1 {
		t.Fatalf("expected PR 1 after release, got %d", ran[1])
	}
}

// A job for an unheld PR runs even when a review ahead of it is held. The
// hold is per key, so it never stalls the whole worker; that is what keeps
// the staged post-stage from deadlocking behind an unrelated held review.
func TestJobRunsWhileAnotherPRIsHeld(t *testing.T) {
	done := make(chan struct{})
	q := New(func(context.Context, string, *webhook.Payload) {
		t.Error("no review should run")
	}, quietLogger())

	hold(q, key(1))
	q.Submit(key(1), payload("one"), "t")
	q.SubmitJob(key(2), "t", func(context.Context) { close(done) })

	q.Start(context.Background())
	defer q.Stop()

	<-done
}

// Depth counts held items: they are genuinely waiting. The in-flight item
// the worker already holds is still excluded, which queue_test.go pins.
func TestDepthCountsHeldItems(t *testing.T) {
	q := New(func(context.Context, string, *webhook.Payload) {}, quietLogger())
	hold(q, key(1))
	q.Submit(key(1), payload("one"), "t")

	if d := q.Depth(); d != 1 {
		t.Fatalf("held item must be counted, depth=%d", d)
	}
}

// Supersede still reaches a held item: its pending payload is replaced and
// no second slot is enqueued, so a burst during an outstanding inference
// still collapses.
func TestSupersedeReplacesHeldPayload(t *testing.T) {
	var mu sync.Mutex
	var titles []string

	q := New(func(_ context.Context, _ string, p *webhook.Payload) {
		mu.Lock()
		titles = append(titles, p.PullRequest.Title)
		mu.Unlock()
	}, quietLogger())

	hold(q, key(1))
	if got := q.Submit(key(1), payload("first"), "t"); got != StatusQueued {
		t.Fatalf("first submit: %s", got)
	}
	for _, title := range []string{"second", "third"} {
		if got := q.Submit(key(1), payload(title), "t"); got != StatusSuperseded {
			t.Fatalf("%s submit: expected superseded, got %s", title, got)
		}
	}
	if d := q.Depth(); d != 1 {
		t.Fatalf("supersede must not enqueue a second slot, depth=%d", d)
	}

	q.Start(context.Background())
	defer q.Stop()
	q.done1(key(1))

	waitFor(t, func() bool {
		mu.Lock()
		defer mu.Unlock()
		return len(titles) == 1
	})
	mu.Lock()
	defer mu.Unlock()
	if titles[0] != "third" {
		t.Fatalf("expected the newest payload, got %q", titles[0])
	}
}

// A synchronous review releases its own hold, so nothing leaks today and
// the set stays invisible until the staged pipeline lands.
func TestSynchronousReviewReleasesItsHold(t *testing.T) {
	ran := make(chan struct{})
	q := New(func(context.Context, string, *webhook.Payload) {
		close(ran)
	}, quietLogger())

	q.Start(context.Background())
	defer q.Stop()
	q.Submit(key(1), payload("one"), "t")

	<-ran
	waitFor(t, func() bool { return !held(q, key(1)) })
}

// A panicking review must not leave its key held: the worker recovers, and
// the next submit for that PR has to run.
func TestPanicReleasesTheHold(t *testing.T) {
	var calls int
	done := make(chan struct{}, 2)

	q := New(func(context.Context, string, *webhook.Payload) {
		calls++
		done <- struct{}{}
		if calls == 1 {
			panic("boom")
		}
	}, quietLogger())

	q.Start(context.Background())
	defer q.Stop()

	q.Submit(key(1), payload("one"), "t")
	<-done
	waitFor(t, func() bool { return !held(q, key(1)) })

	q.Submit(key(1), payload("two"), "t")
	<-done
}

// A lifecycle job for a held PR waits for that PR's review. Order matters
// per PR: ClaimRollup stamps the snapshot it reads and is never retried, so
// a merge rollup that overtook its review would miss that run's cost.
func TestJobForHeldPRWaitsForTheReview(t *testing.T) {
	var mu sync.Mutex
	var order []string

	q := New(func(context.Context, string, *webhook.Payload) {
		mu.Lock()
		order = append(order, "review")
		mu.Unlock()
	}, quietLogger())

	record := func(what string) func(context.Context) {
		return func(context.Context) {
			mu.Lock()
			order = append(order, what)
			mu.Unlock()
		}
	}

	hold(q, key(1))
	q.Submit(key(1), payload("one"), "t")
	q.SubmitJob(key(1), "t", record("merge-1"))
	// A job for an unheld PR behind them both. It must overtake, which is
	// what proves the worker really skipped the held pair instead of just
	// running everything in order.
	q.SubmitJob(key(2), "t", record("merge-2"))

	q.Start(context.Background())
	defer q.Stop()

	waitFor(t, func() bool {
		mu.Lock()
		defer mu.Unlock()
		return len(order) == 1
	})

	mu.Lock()
	got := append([]string(nil), order...)
	mu.Unlock()
	if got[0] != "merge-2" {
		t.Fatalf("the unheld PR's job must run first, got %v", got)
	}
	if d := q.Depth(); d != 2 {
		t.Fatalf("both held items must still be queued, depth=%d", d)
	}

	q.done1(key(1))
	waitFor(t, func() bool {
		mu.Lock()
		defer mu.Unlock()
		return len(order) == 3
	})

	mu.Lock()
	defer mu.Unlock()
	if order[1] != "review" || order[2] != "merge-1" {
		t.Fatalf("merge must not overtake its own review, got %v", order)
	}
}
