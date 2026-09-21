package queue

import (
	"context"
	"fmt"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/trick77/noergler/internal/webhook"
)

// gate is a controllable fake inference call: it reports entry, then blocks
// until released. Concurrency is observed, never slept for.
type gate struct {
	mu       sync.Mutex
	in       int
	maxIn    int
	perTeam  map[string]int
	maxTeam  map[string]int
	entered  chan string
	release  chan struct{}
	released bool
}

// prFor is a payload whose PR id matches key(i), so a staged review can
// recover its own key the way the real one does.
func prFor(i int) *webhook.Payload {
	p := payload(fmt.Sprintf("pr-%d", i))
	p.PullRequest.ID = i
	return p
}

func newGate() *gate {
	return &gate{
		perTeam: map[string]int{},
		maxTeam: map[string]int{},
		entered: make(chan string, 64),
		release: make(chan struct{}),
	}
}

// enter blocks until releaseAll is called, recording peak concurrency.
func (g *gate) enter(team string) {
	g.mu.Lock()
	g.in++
	g.perTeam[team]++
	if g.in > g.maxIn {
		g.maxIn = g.in
	}
	if g.perTeam[team] > g.maxTeam[team] {
		g.maxTeam[team] = g.perTeam[team]
	}
	g.mu.Unlock()

	g.entered <- team
	<-g.release

	g.mu.Lock()
	g.in--
	g.perTeam[team]--
	g.mu.Unlock()
}

// releaseAll unblocks every gated call. Idempotent, so a test can release
// explicitly and still defer it as a cleanup.
func (g *gate) releaseAll() {
	g.mu.Lock()
	defer g.mu.Unlock()
	if !g.released {
		g.released = true
		close(g.release)
	}
}
func (g *gate) peak() int { g.mu.Lock(); defer g.mu.Unlock(); return g.maxIn }
func (g *gate) teamPeak(t string) int {
	g.mu.Lock()
	defer g.mu.Unlock()
	return g.maxTeam[t]
}

// waitEntered waits for n inference calls to have started, and fails rather
// than hanging if they never do.
func (g *gate) waitEntered(t *testing.T, n int) {
	t.Helper()
	for i := 0; i < n; i++ {
		select {
		case <-g.entered:
		case <-time.After(2 * time.Second):
			t.Fatalf("only %d of %d inference calls started", i, n)
		}
	}
}

// noMoreEntries fails if another call starts. The wait is a failure detector,
// not a synchronizer: the assertions above are all channel-based.
func (g *gate) noMoreEntries(t *testing.T) {
	t.Helper()
	select {
	case team := <-g.entered:
		t.Fatalf("an extra inference call started (team %s)", team)
	case <-time.After(100 * time.Millisecond):
	}
}

// stagedQueue builds a queue whose reviews all stage: each hands its
// inference to the pool and posts afterwards. posted counts completed tails.
func stagedQueue(g *gate, global, perTeam int, posted *sync.WaitGroup) *Queue {
	return New(func(ctx context.Context, team string, p *webhook.Payload, sched Scheduler) bool {
		sched.Stage(ctx, key(p.PullRequest.ID), team,
			func(context.Context) { g.enter(team) },
			func(context.Context) { posted.Done() },
		)
		return true
	}, global, perTeam, quietLogger())
}

// The pool overlaps inference across PRs: three reviews reach the gateway at
// once, which is the whole point of staging.
func TestInferenceOverlapsAcrossPRs(t *testing.T) {
	g := newGate()
	var posted sync.WaitGroup
	posted.Add(3)
	q := stagedQueue(g, 3, 3, &posted)

	q.Start(context.Background())
	defer q.Stop()
	// Registered after Stop, so it runs BEFORE it: a failed assertion must
	// unblock the gate or Stop waits on goroutines that never finish.
	defer g.releaseAll()

	for i := 1; i <= 3; i++ {
		q.Submit(key(i), prFor(i), "t")
	}

	g.waitEntered(t, 3)
	if got := g.peak(); got != 3 {
		t.Fatalf("peak concurrent inference = %d, want 3", got)
	}
	g.releaseAll()
	posted.Wait()
}

// The global cap binds: with 2 slots and 3 PRs, only 2 run.
func TestGlobalCapBinds(t *testing.T) {
	g := newGate()
	var posted sync.WaitGroup
	posted.Add(3)
	q := stagedQueue(g, 2, 2, &posted)

	q.Start(context.Background())
	defer q.Stop()
	// Registered after Stop, so it runs BEFORE it: a failed assertion must
	// unblock the gate or Stop waits on goroutines that never finish.
	defer g.releaseAll()

	// Three DISTINCT teams, so the per-team cap of 2 never binds and only
	// the global cap can hold the third call back.
	for i := 1; i <= 3; i++ {
		q.Submit(key(i), prFor(i), fmt.Sprintf("team-%d", i))
	}

	g.waitEntered(t, 2)
	g.noMoreEntries(t)
	if got := g.peak(); got != 2 {
		t.Fatalf("peak = %d, want 2", got)
	}

	g.releaseAll()
	posted.Wait()
}

// Cap 1 is exactly today's behavior: no overlap at all.
func TestConcurrencyOneIsSerial(t *testing.T) {
	g := newGate()
	var posted sync.WaitGroup
	posted.Add(2)
	q := stagedQueue(g, 1, 1, &posted)

	q.Start(context.Background())
	defer q.Stop()
	// Registered after Stop, so it runs BEFORE it: a failed assertion must
	// unblock the gate or Stop waits on goroutines that never finish.
	defer g.releaseAll()

	q.Submit(key(1), prFor(1), "t")
	q.Submit(key(2), prFor(2), "t")

	g.waitEntered(t, 1)
	g.noMoreEntries(t)
	if got := g.peak(); got != 1 {
		t.Fatalf("peak = %d, want 1", got)
	}

	g.releaseAll()
	posted.Wait()
}

// The per-team cap binds inside a larger global one: one team pushing 4 PRs
// uses 2 slots and leaves the rest of the pool free.
func TestPerTeamCapLimitsOneTeam(t *testing.T) {
	g := newGate()
	var posted sync.WaitGroup
	posted.Add(4)
	q := stagedQueue(g, 6, 2, &posted)

	q.Start(context.Background())
	defer q.Stop()
	// Registered after Stop, so it runs BEFORE it: a failed assertion must
	// unblock the gate or Stop waits on goroutines that never finish.
	defer g.releaseAll()

	for i := 1; i <= 4; i++ {
		q.Submit(key(i), prFor(i), "busy")
	}

	g.waitEntered(t, 2)
	g.noMoreEntries(t)
	if got := g.teamPeak("busy"); got != 2 {
		t.Fatalf("one team's peak = %d, want 2", got)
	}

	g.releaseAll()
	posted.Wait()
}

// ...and it does not over-bind: six teams with one PR each all run at once
// under a global cap of 6.
func TestPerTeamCapAllowsOtherTeams(t *testing.T) {
	g := newGate()
	var posted sync.WaitGroup
	posted.Add(6)
	q := stagedQueue(g, 6, 2, &posted)

	q.Start(context.Background())
	defer q.Stop()
	// Registered after Stop, so it runs BEFORE it: a failed assertion must
	// unblock the gate or Stop waits on goroutines that never finish.
	defer g.releaseAll()

	for i := 1; i <= 6; i++ {
		q.Submit(key(i), prFor(i), fmt.Sprintf("team-%d", i))
	}

	g.waitEntered(t, 6)
	if got := g.peak(); got != 6 {
		t.Fatalf("peak = %d, want 6 (one per team)", got)
	}

	g.releaseAll()
	posted.Wait()
}

// The global cap still dominates the per-team one: 5 teams x 2 PRs must not
// become teams * perTeam in flight.
func TestGlobalCapBoundsAllTeams(t *testing.T) {
	g := newGate()
	var posted sync.WaitGroup
	posted.Add(10)
	q := stagedQueue(g, 3, 2, &posted)

	q.Start(context.Background())
	defer q.Stop()
	// Registered after Stop, so it runs BEFORE it: a failed assertion must
	// unblock the gate or Stop waits on goroutines that never finish.
	defer g.releaseAll()

	n := 0
	for team := 1; team <= 5; team++ {
		for pr := 0; pr < 2; pr++ {
			n++
			q.Submit(key(n), prFor(n), fmt.Sprintf("team-%d", team))
		}
	}

	g.waitEntered(t, 3)
	g.noMoreEntries(t)
	if got := g.peak(); got != 3 {
		t.Fatalf("peak = %d, want 3; the global cap must bound the sum", got)
	}

	g.releaseAll()
	posted.Wait()
}

// Posting runs on the single worker, so no two post stages ever overlap even
// when their inference did. Bitbucket stays single-flight.
func TestPostStagesNeverOverlap(t *testing.T) {
	g := newGate()
	var mu sync.Mutex
	inPost, maxPost := 0, 0
	var posted sync.WaitGroup
	posted.Add(3)

	q := New(func(ctx context.Context, team string, p *webhook.Payload, sched Scheduler) bool {
		sched.Stage(ctx, key(p.PullRequest.ID), team,
			func(context.Context) { g.enter(team) },
			func(context.Context) {
				mu.Lock()
				inPost++
				if inPost > maxPost {
					maxPost = inPost
				}
				mu.Unlock()
				time.Sleep(5 * time.Millisecond)
				mu.Lock()
				inPost--
				mu.Unlock()
				posted.Done()
			},
		)
		return true
	}, 3, 3, quietLogger())

	q.Start(context.Background())
	defer q.Stop()
	// Registered after Stop, so it runs BEFORE it: a failed assertion must
	// unblock the gate or Stop waits on goroutines that never finish.
	defer g.releaseAll()

	for i := 1; i <= 3; i++ {
		q.Submit(key(i), prFor(i), "t")
	}
	g.waitEntered(t, 3)
	g.releaseAll()
	posted.Wait()

	mu.Lock()
	defer mu.Unlock()
	if maxPost != 1 {
		t.Fatalf("peak concurrent post stages = %d, want 1", maxPost)
	}
}

// A panic in the inference stage must not kill the process, must release
// both pool slots and must release the PR's hold, or that PR is stuck and a
// slot is leaked forever.
func TestPanicInInferenceReleasesEverything(t *testing.T) {
	var runs int
	ran := make(chan struct{}, 4)

	q := New(func(ctx context.Context, team string, p *webhook.Payload, sched Scheduler) bool {
		runs++
		first := runs == 1
		sched.Stage(ctx, key(p.PullRequest.ID), team,
			func(context.Context) {
				if first {
					panic("inference exploded")
				}
			},
			func(context.Context) { ran <- struct{}{} },
		)
		return true
	}, 2, 2, quietLogger())

	q.Start(context.Background())
	defer q.Stop()

	q.Submit(key(1), prFor(1), "t")

	// The hold must clear even though no post stage ran.
	waitFor(t, func() bool { return !held(q, key(1)) })

	// The pool slots must be free: a later review still completes.
	q.Submit(key(2), prFor(2), "t")
	select {
	case <-ran:
	case <-time.After(2 * time.Second):
		t.Fatal("a panic leaked a pool slot or the hold")
	}
}

// Stop drains the fixpoint: it must not return until the inference in flight
// AND the post stage it submits afterwards have both finished. Otherwise the
// run row and its cost are lost on every shutdown.
func TestStopWaitsForInferenceAndItsPost(t *testing.T) {
	g := newGate()
	postDone := make(chan struct{})

	q := New(func(ctx context.Context, team string, p *webhook.Payload, sched Scheduler) bool {
		sched.Stage(ctx, key(p.PullRequest.ID), team,
			func(context.Context) { g.enter(team) },
			func(context.Context) { close(postDone) },
		)
		return true
	}, 2, 2, quietLogger())

	q.Start(context.Background())
	q.Submit(key(1), prFor(1), "t")
	g.waitEntered(t, 1)

	stopped := make(chan struct{})
	go func() { q.Stop(); close(stopped) }()

	// Stop must still be waiting: the inference has not finished.
	select {
	case <-stopped:
		t.Fatal("Stop returned while an inference was in flight")
	case <-time.After(100 * time.Millisecond):
	}

	g.releaseAll()

	select {
	case <-stopped:
	case <-time.After(3 * time.Second):
		t.Fatal("Stop did not return after the work finished")
	}

	select {
	case <-postDone:
	default:
		t.Fatal("Stop returned before the post stage ran")
	}
}

// Nested acquire must not deadlock under contention from several teams.
func TestNestedSemaphoreNoDeadlock(t *testing.T) {
	var posted sync.WaitGroup
	total := 24
	posted.Add(total)

	q := New(func(ctx context.Context, team string, p *webhook.Payload, sched Scheduler) bool {
		sched.Stage(ctx, key(p.PullRequest.ID), team,
			func(context.Context) {},
			func(context.Context) { posted.Done() },
		)
		return true
	}, 3, 2, quietLogger())

	q.Start(context.Background())
	defer q.Stop()

	for i := 1; i <= total; i++ {
		q.Submit(key(i), prFor(i), fmt.Sprintf("team-%d", i%4))
	}

	done := make(chan struct{})
	go func() { posted.Wait(); close(done) }()
	select {
	case <-done:
	case <-time.After(10 * time.Second):
		t.Fatal("nested acquire deadlocked")
	}
}

// Stop finishes the round of work already in flight, and does NOT start
// queued reviews that never began. Draining the whole backlog would make
// shutdown take queue-depth worth of inference instead of one pool's.
func TestStopDoesNotStartQueuedReviews(t *testing.T) {
	g := newGate()
	var posted atomic.Int32
	var prepared atomic.Int32

	q := New(func(ctx context.Context, team string, p *webhook.Payload, sched Scheduler) bool {
		prepared.Add(1)
		sched.Stage(ctx, key(p.PullRequest.ID), team,
			func(context.Context) { g.enter(team) },
			func(context.Context) { posted.Add(1) },
		)
		return true
	}, 2, 2, quietLogger())

	q.Start(context.Background())
	defer g.releaseAll()

	// Saturate the pool so the worker stops preparing, leaving queued
	// reviews that have not started. Pool 2 plus stagedSlack is the ceiling.
	resident := 2 + stagedSlack
	for i := 1; i <= resident+3; i++ {
		q.Submit(key(i), prFor(i), "t")
	}
	g.waitEntered(t, 2)
	waitFor(t, func() bool { return prepared.Load() == int32(resident) })

	stopped := make(chan struct{})
	go func() { q.Stop(); close(stopped) }()

	g.releaseAll()
	select {
	case <-stopped:
	case <-time.After(3 * time.Second):
		t.Fatal("Stop did not return")
	}
	// Every review that reached the pool posted; none of the queued ones ran.
	if got := prepared.Load(); got != int32(resident) {
		t.Fatalf("prepared %d reviews, want %d: Stop must not start queued work", got, resident)
	}
	if got := posted.Load(); got != int32(resident) {
		t.Fatalf("posted %d, want %d: every staged review must finish its tail", got, resident)
	}
	if d := q.Depth(); d != 3 {
		t.Fatalf("depth after Stop = %d, want 3 queued reviews left unstarted", d)
	}
}

// The worker stops preparing new reviews once the pool is saturated,
// because every prepared prompt stays resident until its inference runs.
// Jobs still run, which is what lets post stages free the slots.
func TestSaturatedPoolStopsPreparingReviews(t *testing.T) {
	g := newGate()
	var prepared atomic.Int32
	jobRan := make(chan struct{})

	q := New(func(ctx context.Context, team string, p *webhook.Payload, sched Scheduler) bool {
		prepared.Add(1)
		sched.Stage(ctx, key(p.PullRequest.ID), team,
			func(context.Context) { g.enter(team) },
			func(context.Context) {},
		)
		return true
	}, 1, 1, quietLogger())

	q.Start(context.Background())
	defer q.Stop()
	defer g.releaseAll()

	// Pool of 1 plus stagedSlack: that many prompts may be resident, no more.
	for i := 1; i <= 8; i++ {
		q.Submit(key(i), prFor(i), "t")
	}
	g.waitEntered(t, 1)

	want := int32(1 + stagedSlack)
	waitFor(t, func() bool { return prepared.Load() == want })
	// It must stay there: nothing releases a slot while the gate holds.
	time.Sleep(50 * time.Millisecond)
	if got := prepared.Load(); got != want {
		t.Fatalf("prepared %d, want %d: the worker ran ahead of the pool", got, want)
	}

	// A job is still runnable despite the saturated pool.
	q.SubmitJob(key(99), "t", func(context.Context) { close(jobRan) })
	select {
	case <-jobRan:
	case <-time.After(2 * time.Second):
		t.Fatal("a job was blocked by the saturated pool")
	}
}

// The post stage runs on the review's own context, not the worker's job
// context: pr_tag and the httpstats scope live there, and everything after
// the handoff would otherwise log without them.
func TestPostStageRunsOnTheReviewContext(t *testing.T) {
	type ctxKey struct{}
	got := make(chan any, 1)

	q := New(func(ctx context.Context, team string, p *webhook.Payload, sched Scheduler) bool {
		reviewCtx := context.WithValue(ctx, ctxKey{}, "from-the-review")
		sched.Stage(reviewCtx, key(p.PullRequest.ID), team,
			func(context.Context) {},
			func(postCtx context.Context) { got <- postCtx.Value(ctxKey{}) },
		)
		return true
	}, 2, 2, quietLogger())

	q.Start(context.Background())
	defer q.Stop()
	q.Submit(key(1), prFor(1), "t")

	select {
	case v := <-got:
		if v != "from-the-review" {
			t.Fatalf("post ctx value = %v, want the review's", v)
		}
	case <-time.After(2 * time.Second):
		t.Fatal("post stage never ran")
	}
}
