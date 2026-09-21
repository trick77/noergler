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
// worker in arrival order, without dedupe. Those jobs carry their PR's key
// too, so per-PR order survives once a review outlives its worker turn.
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
//
// It reports whether the review handed off work that outlives this call. A
// handoff keeps the PR's hold: the review is not finished, and whatever
// finishes it releases the key. Returning false releases it here.
type ReviewFunc func(ctx context.Context, team string, payload *webhook.Payload, sched Scheduler) (handedOff bool)

// Scheduler is the pool as a review uses it, so the review package never
// imports the queue back. Stage runs infer on the inference pool and then
// puts post back on the single worker.
type Scheduler interface {
	Stage(ctx context.Context, key store.PRKey, team string, infer, post func(context.Context))
}

// JobFunc is a queued unit of work that is not a PR review.
type JobFunc func(ctx context.Context)

// item is one unit of queued work. key is always set; job distinguishes a
// non-review job from a review, whose payload lives in pending.
type item struct {
	key store.PRKey
	job *job
}

type job struct {
	tag  string
	team string
	run  JobFunc
	at   time.Time
	// internal marks work submitted by the pipeline itself rather than a
	// webhook: the tail of a review that already holds this key. It runs
	// despite the hold, and releasing it on completion is what ends the
	// hold the handoff kept open.
	internal bool
}

type entry struct {
	team    string
	payload *webhook.Payload
	at      time.Time
}

// Queue is the single-worker PR review queue with a bounded inference pool.
type Queue struct {
	review ReviewFunc
	log    *slog.Logger

	// sem bounds inference process-wide; perTeam nests inside it so one
	// team's burst cannot take every slot. Acquire team first, then global,
	// and release in reverse: the other order lets a goroutine hold a global
	// slot while queuing for its team's, which is how a nested pair
	// deadlocks.
	sem     chan struct{}
	perTeam int
	teamSem map[string]chan struct{}
	infWG   sync.WaitGroup

	mu       sync.Mutex
	cond     *sync.Cond
	items    []item
	pending  map[store.PRKey]*entry
	stopped  bool
	draining bool
	// running is set while the worker is executing an item, so Stop can tell
	// an idle worker from one mid-item without racing on the item itself.
	running bool

	// inflight holds the keys of PRs whose work has been dequeued but is not
	// finished yet. A keyed item whose key is in flight is not runnable: the
	// worker skips over it and leaves it in items. Today a review finishes
	// before run loops, so nothing is ever held; the set exists for the
	// staged pipeline, where the inference call outlives the worker turn and
	// a second run of the same PR would race on the prior-commit pointer,
	// the summary and the inline comments.
	inflight map[store.PRKey]struct{}

	started bool
	done    chan struct{}
}

// New builds a queue. The worker does not run until Start.
//
// global and perTeam size the inference pool; both must be at least 1, which
// config enforces at load. perTeam above global never binds and config
// rejects it, but clamping here keeps a direct caller honest.
func New(review ReviewFunc, global, perTeam int, log *slog.Logger) *Queue {
	if global < 1 {
		global = 1
	}
	if perTeam < 1 || perTeam > global {
		perTeam = global
	}
	q := &Queue{
		review:   review,
		log:      log,
		sem:      make(chan struct{}, global),
		perTeam:  perTeam,
		teamSem:  make(map[string]chan struct{}),
		pending:  make(map[store.PRKey]*entry),
		inflight: make(map[store.PRKey]struct{}),
		done:     make(chan struct{}),
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

// Stop drains everything already accepted and then stops the worker.
// Calling it before Start, or twice, is safe.
//
// This is a fixpoint, not a sequence: a review spawns an inference call,
// which submits a post stage back onto the worker. Stopping as soon as the
// queue looks empty would return with an inference still in flight and its
// post stage never submitted, losing the run row, the findings and the cost
// on every shutdown.
//
// No timeout, deliberately: the drain now spans prepare, the pool wait, the
// gateway's own call timeout and posting, for every review in flight. The
// pod's termination grace period has to allow for it.
func (q *Queue) Stop() {
	q.mu.Lock()
	if !q.started || q.stopped {
		q.stopped = true
		q.mu.Unlock()
		return
	}
	q.draining = true
	q.cond.Broadcast()
	q.mu.Unlock()

	// Alternate between letting the worker empty the queue and waiting for
	// the inference goroutines, because each can create work for the other.
	// Settles when a pass adds nothing: the worker is idle, no inference is
	// in flight, and no key is still held.
	for {
		q.mu.Lock()
		// Wait while the worker can still make progress on its own. Items
		// that are only held are not progress: the goroutine that releases
		// them is an inference call, which infWG below waits for.
		for q.running || q.nextRunnable() >= 0 {
			q.cond.Wait()
		}
		q.mu.Unlock()

		q.infWG.Wait()

		// Settled when the worker is idle and nothing runnable is left.
		// Deliberately not "no key is held": a hold is released by an
		// inference goroutine, and infWG above has just waited for all of
		// them, so a key still held here is one nothing will ever release.
		// Blocking on it would hang shutdown forever.
		q.mu.Lock()
		settled := !q.running && q.nextRunnable() < 0
		q.mu.Unlock()
		if settled {
			break
		}
	}

	q.mu.Lock()
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
// Never blocks: the queue is unbounded, because the caller is an inbound HTTP
// request goroutine. A bounded channel would turn a backlog into
// Bitbucket-side webhook timeouts.
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

// SubmitJob enqueues a non-review job for a PR. Never deduped.
//
// The key is the PR the job belongs to, so the job is held while that PR's
// review is still outstanding. Order matters per PR: a merge rollup that
// overtook its review would claim the snapshot before the run row existed
// and miss that run's cost, and it is never retried. A decline that
// overtook one would reset the state the review then re-advances.
func (q *Queue) SubmitJob(key store.PRKey, team string, fn JobFunc) string {
	q.mu.Lock()
	defer q.mu.Unlock()

	tag := key.Tag()
	q.put(item{key: key, job: &job{tag: tag, team: team, run: fn, at: time.Now()}}, tag)
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

// Depth is the number of items waiting, excluding the one the worker already
// holds. The backlog warning is read off it, so counting the in-flight item
// would warn one entry early.
func (q *Queue) Depth() int {
	q.mu.Lock()
	defer q.mu.Unlock()
	return len(q.items)
}

// nextRunnable is the index of the first item the worker may start, or -1
// when every waiting item is held. Caller holds the lock.
//
// Held items stay in items rather than being parked somewhere else: Depth
// counts them, because they are genuinely waiting, and Submit still finds
// their pending entry and supersedes it.
func (q *Queue) nextRunnable() int {
	for i, it := range q.items {
		if it.job != nil && it.job.internal {
			return i
		}
		if _, held := q.inflight[it.key]; !held {
			return i
		}
	}
	return -1
}

// Stage runs infer on the inference pool and then puts post back on the
// single worker, so only the gateway call overlaps and every Bitbucket call
// stays one at a time.
//
// The caller must already hold key (it returned handedOff from a review), and
// Stage is what eventually releases it: post runs as an internal job, whose
// completion ends the hold. A panic in infer releases it here instead, so the
// PR is never stuck.
//
// ctx is the review's own, carrying pr_tag, team= and the httpstats scope. It
// must not be the worker's job context, which has none of them past the
// handoff.
func (q *Queue) Stage(ctx context.Context, key store.PRKey, team string, infer, post func(context.Context)) {
	q.infWG.Add(1)
	go func() {
		defer q.infWG.Done()

		posted := false
		defer func() {
			if r := recover(); r != nil {
				q.log.ErrorContext(ctx, fmt.Sprintf("queue[%s]: inference panicked: %v", key.Tag(), r),
					slog.String("stack", string(debug.Stack())))
			}
			// Whatever happened, the hold must end. Normally post ends it;
			// if it never got submitted, end it here.
			if !posted {
				q.done1(key)
			}
		}()

		waited := q.acquire(team)
		if waited > 50*time.Millisecond {
			q.log.InfoContext(ctx, fmt.Sprintf("queue[%s]: inference waited %.1fs for a pool slot",
				key.Tag(), waited.Seconds()))
		}
		func() {
			defer q.release(team)
			infer(ctx)
		}()

		// Submitted before the deferred release above runs, so the hold is
		// handed from this goroutine to the job without a gap.
		posted = true
		q.submitInternal(key, team, post)
	}()
}

// acquire takes the team slot, then the global one, and reports how long it
// waited. Order matters: holding a global slot while queuing for a team slot
// is how a nested pair deadlocks.
func (q *Queue) acquire(team string) time.Duration {
	started := time.Now()
	ts := q.teamSemFor(team)
	ts <- struct{}{}
	q.sem <- struct{}{}
	return time.Since(started)
}

// release reverses acquire.
func (q *Queue) release(team string) {
	<-q.sem
	<-q.teamSemFor(team)
}

// teamSemFor is the team's semaphore, created on first use. Teams are fixed
// at startup, so the map only ever grows to the configured team count.
func (q *Queue) teamSemFor(team string) chan struct{} {
	q.mu.Lock()
	defer q.mu.Unlock()
	ts, ok := q.teamSem[team]
	if !ok {
		ts = make(chan struct{}, q.perTeam)
		q.teamSem[team] = ts
	}
	return ts
}

// submitInternal queues the tail of a staged review. It runs despite that
// PR's hold, which the review itself is holding, and its completion is what
// releases the key.
func (q *Queue) submitInternal(key store.PRKey, team string, fn JobFunc) {
	q.mu.Lock()
	defer q.mu.Unlock()
	tag := key.Tag()
	q.put(item{key: key, job: &job{tag: tag, team: team, run: fn, at: time.Now(), internal: true}}, tag)
}

// done1 releases a key's hold and wakes the worker, which may have parked
// with only held items left.
func (q *Queue) done1(key store.PRKey) {
	q.mu.Lock()
	delete(q.inflight, key)
	q.mu.Unlock()
	q.cond.Broadcast()
}

func (q *Queue) run(ctx context.Context) {
	defer close(q.done)
	for {
		q.mu.Lock()
		idx := q.nextRunnable()
		for idx < 0 && !q.stopped {
			q.cond.Wait()
			idx = q.nextRunnable()
		}
		if q.stopped {
			q.mu.Unlock()
			return
		}
		it := q.items[idx]
		q.items = append(q.items[:idx], q.items[idx+1:]...)
		q.running = true

		var tag, team string
		var at time.Time
		var run func(context.Context)
		kind := "review"

		if it.job != nil {
			kind = "job"
			tag, team, at = it.job.tag, it.job.team, it.job.at
			q.inflight[it.key] = struct{}{}
			key := it.key
			fn := it.job.run
			run = func(ctx context.Context) {
				defer q.done1(key)
				fn(ctx)
			}
		} else {
			tag = it.key.Tag()
			e, ok := q.pending[it.key]
			if !ok {
				q.mu.Unlock()
				q.log.Warn(fmt.Sprintf("queue[%s]: dequeued with no payload, skipping", tag))
				continue
			}
			delete(q.pending, it.key)
			q.inflight[it.key] = struct{}{}
			key := it.key
			team, at = e.team, e.at
			payload := e.payload
			run = func(ctx context.Context) {
				// The defer, not a tail call: a panicking review must not
				// leave its key held forever. runOne recovers above, and
				// handedOff stays false, so the hold is released here.
				handedOff := false
				defer func() {
					if !handedOff {
						q.done1(key)
					}
				}()
				handedOff = q.review(ctx, team, payload, q)
			}
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

		// Wake Stop: it alternates between an empty queue and an idle
		// worker, and this is the only place the second becomes true.
		q.mu.Lock()
		q.running = false
		q.cond.Broadcast()
		q.mu.Unlock()
	}
}

// runOne executes one item and contains its failures.
//
// The recover is deliberate. Without it a single bad PR takes the only worker
// down for every team: Phase 4 hit exactly that with a slice index in the
// scope search. Recovering per item means a panicking PR loses itself and nothing
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
