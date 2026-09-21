// Package queue is the single-worker review queue.
//
// Every review runs one at a time on one background worker, so only one
// diff/file/prompt set is ever resident: the pod's memory limit is sized for
// one. Webhooks may arrive at any rate, so the queue dedupes per PR. If a PR
// already has a pending entry the stored payload is replaced and no second
// slot is enqueued, which collapses a 50-commit push into at most two reviews
// (the one in flight plus the deduped latest state).
//
// SubmitJob puts other heavy work (mention answers, rollups) on the same
// worker in arrival order, without dedupe.
package queue

import (
	"context"
	"fmt"
	"log/slog"
	"runtime/debug"
	"sync"
	"time"

	"github.com/trick77/noergler/internal/logging"
	"github.com/trick77/noergler/internal/store"
	"github.com/trick77/noergler/internal/webhook"
)

// BacklogWarnThreshold is the depth at which the queue starts warning.
const BacklogWarnThreshold = 10

// Submit outcomes.
const (
	StatusQueued     = "queued"
	StatusSuperseded = "superseded"
)

// ReviewFunc reviews one PR. It is called with the slug of the team the PR
// belongs to; the team was authenticated by the webhook route, so the worker
// never has to look it up.
type ReviewFunc func(ctx context.Context, team string, payload *webhook.Payload)

// JobFunc is a queued unit of work that is not a PR review.
type JobFunc func(ctx context.Context)

// item is one unit of queued work. Exactly one of key or job is set.
type item struct {
	key store.PRKey
	job *job
}

type job struct {
	tag  string
	team string
	run  JobFunc
	at   time.Time
}

type entry struct {
	team    string
	payload *webhook.Payload
	at      time.Time
}

// Queue is the single-worker PR review queue.
type Queue struct {
	review ReviewFunc
	log    *slog.Logger

	mu      sync.Mutex
	cond    *sync.Cond
	items   []item
	pending map[store.PRKey]*entry
	stopped bool

	started bool
	done    chan struct{}
}

// New builds a queue. The worker does not run until Start.
func New(review ReviewFunc, log *slog.Logger) *Queue {
	q := &Queue{
		review:  review,
		log:     log,
		pending: make(map[store.PRKey]*entry),
		done:    make(chan struct{}),
	}
	q.cond = sync.NewCond(&q.mu)
	return q
}

// Start launches the worker goroutine. Calling it twice is a no-op.
func (q *Queue) Start(ctx context.Context) {
	q.mu.Lock()
	if q.started {
		q.mu.Unlock()
		return
	}
	q.started = true
	q.mu.Unlock()

	go q.run(ctx)
	q.log.Info("ReviewQueue worker started")
}

// Stop signals the worker to finish the item in flight and exit, then waits
// for it. Calling it before Start, or twice, is safe.
func (q *Queue) Stop() {
	q.mu.Lock()
	if !q.started || q.stopped {
		q.stopped = true
		q.mu.Unlock()
		return
	}
	q.stopped = true
	q.cond.Broadcast()
	q.mu.Unlock()

	<-q.done
	q.log.Info("ReviewQueue worker stopped")
}

// Submit enqueues a review, returning StatusQueued for a fresh entry or
// StatusSuperseded when the key was already pending (the payload is
// replaced).
//
// Never blocks: Python is an unbounded asyncio.Queue with put_nowait, and in
// Phase 7 the caller is an inbound HTTP request goroutine. A bounded channel
// would turn a backlog into Bitbucket-side webhook timeouts.
func (q *Queue) Submit(key store.PRKey, payload *webhook.Payload, team string) string {
	q.mu.Lock()
	defer q.mu.Unlock()

	if e, ok := q.pending[key]; ok {
		e.team = team
		e.payload = payload
		e.at = time.Now()
		q.log.Info(fmt.Sprintf("queue[%s]: superseded pending payload (depth=%d)", key.Tag(), len(q.items)))
		return StatusSuperseded
	}

	q.pending[key] = &entry{team: team, payload: payload, at: time.Now()}
	q.put(item{key: key}, key.Tag())
	return StatusQueued
}

// SubmitJob enqueues a non-review job. Never deduped.
func (q *Queue) SubmitJob(tag, team string, fn JobFunc) string {
	q.mu.Lock()
	defer q.mu.Unlock()

	q.put(item{job: &job{tag: tag, team: team, run: fn, at: time.Now()}}, tag)
	return StatusQueued
}

// put appends an item and logs the depth. Caller holds the lock.
func (q *Queue) put(it item, tag string) {
	q.items = append(q.items, it)
	depth := len(q.items)
	q.log.Info(fmt.Sprintf("queue[%s]: enqueued (depth=%d)", tag, depth))
	if depth >= BacklogWarnThreshold {
		q.log.Warn(fmt.Sprintf("ReviewQueue backlog: %d entries pending", depth))
	}
	q.cond.Signal()
}

// Depth is the number of items waiting, excluding the one in flight. Python's
// qsize() excludes the item already handed out by get(), so this must too.
func (q *Queue) Depth() int {
	q.mu.Lock()
	defer q.mu.Unlock()
	return len(q.items)
}

func (q *Queue) run(ctx context.Context) {
	defer close(q.done)
	for {
		q.mu.Lock()
		for len(q.items) == 0 && !q.stopped {
			q.cond.Wait()
		}
		if q.stopped {
			q.mu.Unlock()
			return
		}
		it := q.items[0]
		q.items = q.items[1:]

		var tag, team string
		var at time.Time
		var run func(context.Context)
		kind := "review"

		if it.job != nil {
			kind = "job"
			tag, team, at = it.job.tag, it.job.team, it.job.at
			run = it.job.run
		} else {
			tag = it.key.Tag()
			e, ok := q.pending[it.key]
			if !ok {
				q.mu.Unlock()
				q.log.Warn(fmt.Sprintf("queue[%s]: dequeued with no payload, skipping", tag))
				continue
			}
			delete(q.pending, it.key)
			team, at = e.team, e.at
			payload := e.payload
			run = func(ctx context.Context) { q.review(ctx, team, payload) }
		}
		depth := len(q.items)
		q.mu.Unlock()

		waited := time.Since(at)
		q.log.Info(fmt.Sprintf("queue[%s]: starting %s (waited %.1fs, depth=%d)",
			tag, kind, waited.Seconds(), depth))

		// The worker is one long-lived goroutine, so the team binding is
		// explicit per item: nothing else would ever clear it.
		jobCtx := logging.WithTeam(ctx, team)
		started := time.Now()
		q.runOne(jobCtx, run, tag, kind)
		q.log.Info(fmt.Sprintf("queue[%s]: completed in %.1fs", tag, time.Since(started).Seconds()))
	}
}

// runOne executes one item and contains its failures.
//
// The recover is the one deliberate addition over the Python, which cannot
// panic the way Go can. Without it a single bad PR takes the only worker down
// for every team: Phase 4 hit exactly that with a slice index in the scope
// search. Recovering per item means a panicking PR loses itself and nothing
// else, and the completed line below is still emitted.
func (q *Queue) runOne(ctx context.Context, run func(context.Context), tag, kind string) {
	defer func() {
		if r := recover(); r != nil {
			q.log.ErrorContext(ctx, fmt.Sprintf("queue[%s]: %s panicked: %v", tag, kind, r),
				slog.String("stack", string(debug.Stack())))
		}
	}()
	run(ctx)
}
