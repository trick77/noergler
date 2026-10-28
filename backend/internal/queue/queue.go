// Package queue is the single-worker review queue.
//
// One background worker runs every review's Bitbucket work: the diff and
// file fetches, and the posting afterwards. Only the gateway call leaves it,
// onto a bounded pool, because a 200s+ inference would otherwise hold the
// queue behind it. The worker stops preparing once the pool is saturated, so
// the resident prompts stay bounded too. Webhooks may arrive at any rate, so the queue dedupes per PR. If a PR
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
	"sort"
	"sync"
	"time"

	"github.com/trick77/noergler/internal/store"
	"github.com/trick77/noergler/internal/webhook"
	"github.com/trick77/ship/kit/logging"
)

// BacklogWarnThreshold is the depth at which the queue starts warning.
const BacklogWarnThreshold = 10

// stagedSlack is how much pool work (prepared prompts and spawned calls) may
// wait beyond the pool's own capacity, so a slot is never idle while the
// worker fetches the next diff. A spawned call holds no prompt, but it holds
// a slot the same way, so it counts: preparing more prompts behind it would
// only park them.
const stagedSlack = 2

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

// Scheduler is the pool as a review uses it. Stage runs infer on the
// inference pool and then puts post back on the single worker. Spawn runs
// fn on the pool and nothing afterwards: no worker tail, no PR hold.
type Scheduler interface {
	Stage(ctx context.Context, key store.PRKey, team string, infer, post func(context.Context))
	Spawn(ctx context.Context, team, name string, fn func(context.Context))
}

// JobFunc is a queued unit of work that is not a PR review.
//
// It gets the scheduler for the same reason a review does: a mention can ask
// for a review, and that review belongs on the inference pool like any
// other. A job that needs nothing from it ignores the argument.
//
// It reports handoff on the same contract as ReviewFunc: true means work
// outlives this call and the PR's hold must survive with it, so whatever
// finishes that work releases the key. A job that finishes when it returns
// reports false. Releasing unconditionally here would leave a staged
// mention's review unprotected for its whole inference.
type JobFunc func(ctx context.Context, sched Scheduler) (handedOff bool)

// item is one unit of queued work. key is always set; job distinguishes a
// non-review job from a review, whose payload lives in pending.
type item struct {
	key store.PRKey
	job *job
}

type job struct {
	team string
	// name is what the queue logs for the job: the webhook event it serves
	// (merge-rollup, mention) or post for the tail of a staged review.
	name string
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

// heldEntry is what a held key carries for the dashboard. The set used to be
// keyed to nothing (map[PRKey]struct{}), which was enough for nextRunnable
// but left a RUNNING review anonymous: pending is deleted at dequeue, so
// after that moment neither the team nor the start time survived anywhere.
type heldEntry struct {
	team string
	// kind is "review" or "job" for the dashboard; the log names the job.
	kind  string
	since time.Time
	// waited is how long this item sat in the queue before it started.
	waited time.Duration
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
	// staged counts pool work (prepared prompts and spawned calls) handed
	// to the pool and not yet through it, running or still waiting for a
	// slot. Guarded by mu.
	staged int

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
	// worker skips over it and leaves it in items. A staged review's
	// inference call outlives the worker turn, and a second run of the same
	// PR would race it on the prior-commit pointer, the summary and the
	// inline comments.
	inflight map[store.PRKey]heldEntry

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
		inflight: make(map[store.PRKey]heldEntry),
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
// pod's termination grace period has to allow for it. Its SIGKILL, or a
// second shutdown signal (cmd/noergler), kills the process mid-drain.
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
		q.log.InfoContext(teamCtx(team), fmt.Sprintf("queue[%s]: superseded pending payload (depth=%d)", key.Tag(), len(q.items)))
		return StatusSuperseded
	}

	q.pending[key] = &entry{team: team, payload: payload, at: time.Now()}
	q.put(teamCtx(team), item{key: key}, key.Tag(), "enqueued")
	return StatusQueued
}

// teamCtx binds team= for a line logged off a webhook request, which hands
// the queue a slug and no context.
func teamCtx(team string) context.Context {
	return logging.With(context.Background(), "team", team)
}

// SubmitJob enqueues a non-review job for a PR. Never deduped.
//
// The key is the PR the job belongs to, so the job is held while that PR's
// review is still outstanding. Order matters per PR: a merge rollup that
// overtook its review would claim the snapshot before the run row existed
// and miss that run's cost, and it is never retried. A decline that
// overtook one would reset the state the review then re-advances.
func (q *Queue) SubmitJob(key store.PRKey, team, name string, fn JobFunc) string {
	q.mu.Lock()
	defer q.mu.Unlock()

	tag := key.Tag()
	q.put(teamCtx(team), item{key: key, job: &job{team: team, name: name, run: fn, at: time.Now()}}, tag, name+" enqueued")
	return StatusQueued
}

// put appends an item and logs what happened with the depth. Caller holds
// the lock. ctx carries team=, and for a staged tail the review's pr_tag.
func (q *Queue) put(ctx context.Context, it item, tag, what string) {
	q.items = append(q.items, it)
	depth := len(q.items)
	q.log.InfoContext(ctx, fmt.Sprintf("queue[%s]: %s (depth=%d)", tag, what, depth))
	if depth >= BacklogWarnThreshold {
		q.log.Warn(fmt.Sprintf("ReviewQueue backlog: %d entries pending", depth))
	}
	// Broadcast, not Signal: Stop waits on the same condition, so a Signal
	// can wake the drain instead of the worker and the new item sits there
	// until something else happens to wake it.
	q.cond.Broadcast()
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
	saturated := q.staged >= cap(q.sem)+stagedSlack
	// Keys whose review this scan has passed over. A job behind a skipped
	// review must not overtake it: the key is not in inflight, because the
	// review was never dequeued, so without this a merge rollup would claim
	// the snapshot before its own run row existed and never retry.
	var deferred map[store.PRKey]struct{}

	for i, it := range q.items {
		if it.job != nil && it.job.internal {
			return i
		}
		if _, skipped := deferred[it.key]; skipped {
			continue
		}
		// Draining: only the tails of reviews already in flight still run.
		// Starting queued work would make shutdown take the whole backlog,
		// not the one round of work Stop is there to finish.
		if q.draining {
			if it.job == nil {
				deferred = markDeferred(deferred, it.key)
			}
			continue
		}
		// A prepared prompt is resident until its inference finishes, so
		// preparing the whole queue ahead of a saturated pool would hold one
		// per queued PR. Stop taking new reviews and let the pool catch up;
		// jobs for OTHER PRs, including the post stages that free slots,
		// still run.
		if it.job == nil && saturated {
			deferred = markDeferred(deferred, it.key)
			continue
		}
		if _, held := q.inflight[it.key]; !held {
			return i
		}
	}
	return -1
}

func markDeferred(m map[store.PRKey]struct{}, k store.PRKey) map[store.PRKey]struct{} {
	if m == nil {
		m = make(map[store.PRKey]struct{}, 1)
	}
	m[k] = struct{}{}
	return m
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
	tag := key.Tag()
	posted := false
	q.onPool(ctx, team, tag, "inference",
		func(ctx context.Context) {
			inferStarted := time.Now()
			func() {
				defer q.release(team)
				infer(ctx)
			}()
			inferred := time.Since(inferStarted)

			// The pool slot is already released above; the PR's hold is
			// not. It ends when this job completes, or in onPool's panic
			// path if the submit never happened, so there is no window
			// where the key is free while its posting is still outstanding.
			//
			// post is called with the REVIEW's ctx, not the worker's: the
			// job context queue.run builds carries team= but no pr_tag and
			// no httpstats scope, so everything after the handoff would log
			// without them and record no HTTP counts.
			posted = true
			q.submitInternal(ctx, key, team, inferred, func(context.Context, Scheduler) bool {
				post(ctx)
				return false
			})
		},
		// Whatever happened, the hold must end. Normally post ends it; if
		// it never got submitted, end it here.
		func() {
			if !posted {
				q.done1(key)
			}
		})
}

// Spawn runs fn on the inference pool, under the team's slot and then the
// global one, with no worker tail and no PR hold. name is what the log
// calls it ("almanach note").
//
// For a gateway call that posts nothing to Bitbucket and whose result lands
// in the database only (the Almanach note that follows a review). Stage
// would be wrong for it: Stage's tail is what ends the PR's hold, so a
// second Stage issued from inside a post would lose the hold the moment
// that post returns and later release a key a newer review holds.
//
// Spawn touches inflight not at all, so the PR's other jobs (a purge, a
// decline, a merge) and its next review run beside fn. The caller's
// contract, pinned in INVARIANTS: fn re-asks ownership and the PR's state
// right before it writes, as post does; it opens its own httpstats scope
// and logs it (the review's counter was reported when post returned); and
// it carries only the few fields it needs, built before Spawn, never the
// plan, the result or a diff: a closure waiting for a slot keeps whatever
// it captured resident.
//
// Counted like staged work: Stop drains it through infWG and waits for it
// without a timeout, and staged keeps the worker from preparing prompts
// while spawned calls hold the slots. A panic is logged and the slot
// released; nothing else is affected.
func (q *Queue) Spawn(ctx context.Context, team, name string, fn func(context.Context)) {
	q.onPool(ctx, team, name, name, func(ctx context.Context) {
		defer q.release(team)
		started := time.Now()
		fn(ctx)
		q.log.InfoContext(ctx, fmt.Sprintf("queue[%s]: done in %.1fs", name, time.Since(started).Seconds()))
	}, nil)
}

// onPool is the pool goroutine Stage and Spawn share: counted in staged and
// infWG from the start, the team slot then the global one, a slow slot
// logged, a panic logged with its stack and the slot given back. body runs
// with the slot held and is responsible for releasing it (so Stage can
// release before its tail). onPanic, when set, runs after a panic was
// logged. tag names the work in the log, what names the wait.
func (q *Queue) onPool(ctx context.Context, team, tag, what string, body func(context.Context), onPanic func()) {
	q.infWG.Add(1)
	q.mu.Lock()
	q.staged++
	q.mu.Unlock()
	go func() {
		defer q.infWG.Done()

		acquired := false
		defer func() {
			if !acquired {
				// Never got a slot, so release never ran and the staged
				// count is still ours.
				q.mu.Lock()
				q.staged--
				q.mu.Unlock()
				q.cond.Broadcast()
			}
			if r := recover(); r != nil {
				q.log.ErrorContext(ctx, fmt.Sprintf("queue[%s]: %s panicked: %v", tag, what, r),
					slog.String("stack", string(debug.Stack())))
			}
			if onPanic != nil {
				onPanic()
			}
		}()

		waited := q.acquire(team)
		acquired = true
		if waited > 50*time.Millisecond {
			q.log.InfoContext(ctx, fmt.Sprintf("queue[%s]: %s waited %.1fs for a pool slot", tag, what, waited.Seconds()))
		}
		body(ctx)
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

// release reverses acquire and frees the prepared-prompt slot, waking a
// worker that stopped taking reviews because the pool was saturated.
func (q *Queue) release(team string) {
	<-q.sem
	<-q.teamSemFor(team)
	q.mu.Lock()
	q.staged--
	q.mu.Unlock()
	q.cond.Broadcast()
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
//
// inferred is how long the gateway call took; this is the only place that
// knows it, so the enqueue line reports it.
func (q *Queue) submitInternal(ctx context.Context, key store.PRKey, team string, inferred time.Duration, fn JobFunc) {
	q.mu.Lock()
	defer q.mu.Unlock()
	tag := key.Tag()
	q.put(ctx, item{key: key, job: &job{team: team, name: "post", run: fn, at: time.Now(), internal: true}}, tag,
		fmt.Sprintf("inference done in %.1fs, post enqueued", inferred.Seconds()))
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

		key, tag := it.key, it.key.Tag()
		var team string
		var at time.Time
		var fn JobFunc
		kind := "review"

		// what is the word the log uses for the item; kind is the
		// dashboard's coarser one.
		what := kind
		if it.job != nil {
			kind = "job"
			what = it.job.name
			team, at, fn = it.job.team, it.job.at, it.job.run
		} else {
			e, ok := q.pending[it.key]
			if !ok {
				// Reset running, or Stop waits on it forever: it has no
				// timeout, so a leak here hangs shutdown until the pod is
				// killed.
				q.running = false
				q.cond.Broadcast()
				q.mu.Unlock()
				q.log.Warn(fmt.Sprintf("queue[%s]: dequeued with no payload, skipping", tag))
				continue
			}
			delete(q.pending, it.key)
			team, at = e.team, e.at
			payload := e.payload
			fn = func(ctx context.Context, sched Scheduler) bool { return q.review(ctx, team, payload, sched) }
		}
		q.inflight[key] = heldEntry{team: team, kind: kind, since: time.Now(), waited: time.Since(at)}
		// Outside the closure: the end-of-turn line below reads it. A panic
		// leaves it false, which is the right answer there too.
		handedOff := false
		run := func(ctx context.Context) {
			// The defer, not a tail call: a panicking job or review must not
			// leave its key held forever. runOne recovers above, and
			// handedOff stays false, so the hold is released here; a staged
			// one keeps it.
			defer func() {
				if !handedOff {
					q.done1(key)
				}
			}()
			handedOff = fn(ctx, q)
		}
		depth := len(q.items)
		q.mu.Unlock()

		// The worker is one long-lived goroutine, so the team binding is
		// explicit per item: nothing else would ever clear it.
		jobCtx := logging.With(ctx, "team", team)
		waited := time.Since(at)
		q.log.InfoContext(jobCtx, fmt.Sprintf("queue[%s]: starting %s (waited %.1fs, depth=%d)",
			tag, what, waited.Seconds(), depth))

		started := time.Now()
		recovered := q.runOne(jobCtx, run, tag, what)
		// Never "completed" here: a staged review has only been prepared, and
		// its inference and posting follow under the same tag. The review
		// logs its own completion at the end of post; this line says only
		// what the worker turn did. A review that did not hand off exited
		// in prepare (production stages every review), so it ran no
		// inference, and the reason is on the review's own line above.
		secs := time.Since(started).Seconds()
		switch {
		case recovered:
			q.log.InfoContext(jobCtx, fmt.Sprintf("queue[%s]: %s failed in %.1fs (panic above)", tag, what, secs))
		case handedOff:
			q.log.InfoContext(jobCtx, fmt.Sprintf("queue[%s]: %s prepared in %.1fs, inference running", tag, what, secs))
		case it.job == nil:
			q.log.InfoContext(jobCtx, fmt.Sprintf("queue[%s]: review ended in %.1fs, no inference", tag, secs))
		default:
			q.log.InfoContext(jobCtx, fmt.Sprintf("queue[%s]: %s done in %.1fs", tag, what, secs))
		}

		// Wake Stop: it alternates between an empty queue and an idle
		// worker, and this is the only place the second becomes true.
		q.mu.Lock()
		q.running = false
		q.cond.Broadcast()
		q.mu.Unlock()
	}
}

// runOne executes one item and contains its failures. It reports whether it
// recovered a panic, so the end-of-turn line can say the item failed.
//
// The recover is deliberate. Without it a single bad PR takes the only worker
// down for every team: Phase 4 hit exactly that with a slice index in the
// scope search. Recovering per item means a panicking PR loses itself and
// nothing else, and the worker's end-of-turn line is still emitted.
func (q *Queue) runOne(ctx context.Context, run func(context.Context), tag, what string) (recovered bool) {
	defer func() {
		if r := recover(); r != nil {
			recovered = true
			q.log.ErrorContext(ctx, fmt.Sprintf("queue[%s]: %s panicked: %v", tag, what, r),
				slog.String("stack", string(debug.Stack())))
		}
	}()
	run(ctx)
	return false
}

// Snapshot is the queue as the dashboard reads it: one lock, one copy, no
// internal slice or map handed out.
//
// Depth is the only other accessor and it deliberately excludes the in-flight
// item (the backlog warning is read off it). This reports both, separately,
// because "nothing is queued" and "nothing is running" are different answers
// and a live panel needs to tell them apart.
type Snapshot struct {
	// Capacity and PerTeam size the inference pool.
	Capacity int
	PerTeam  int
	// Staged is pool work handed to the pool and not yet through it:
	// running, or still waiting for a slot. Prepared prompts and spawned
	// calls both; a spawned call holds no PR key, so it is in this count
	// and not in Running.
	Staged int
	// Depth matches Depth(): waiting items, in-flight excluded.
	Depth   int
	Running []RunningItem
	Waiting []WaitingItem
}

// RunningItem is one PR whose work has been dequeued and is not finished.
type RunningItem struct {
	Key    store.PRKey
	Team   string
	Kind   string
	Since  time.Time
	Waited time.Duration
}

// WaitingItem is one PR still in the queue.
type WaitingItem struct {
	Key   store.PRKey
	Team  string
	Since time.Time
}

// Snapshot copies the queue's live state.
func (q *Queue) Snapshot() Snapshot {
	q.mu.Lock()
	defer q.mu.Unlock()

	s := Snapshot{
		Capacity: cap(q.sem),
		PerTeam:  q.perTeam,
		Staged:   q.staged,
		Depth:    len(q.items),
	}
	for key, h := range q.inflight {
		s.Running = append(s.Running, RunningItem{
			Key: key, Team: h.team, Kind: h.kind, Since: h.since, Waited: h.waited,
		})
	}
	// items order is the queue's own, which is what the panel shows; a map
	// range is not, so only the running list needs sorting.
	sort.Slice(s.Running, func(i, j int) bool {
		return s.Running[i].Since.Before(s.Running[j].Since)
	})
	for _, it := range q.items {
		w := WaitingItem{Key: it.key}
		if it.job != nil {
			w.Team, w.Since = it.job.team, it.job.at
		} else if e, ok := q.pending[it.key]; ok {
			w.Team, w.Since = e.team, e.at
		}
		s.Waiting = append(s.Waiting, w)
	}
	return s
}
