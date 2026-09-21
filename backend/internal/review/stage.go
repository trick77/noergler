package review

import (
	"context"
	"time"

	"github.com/trick77/noergler/internal/httpstats"
	"github.com/trick77/noergler/internal/inference"
	"github.com/trick77/noergler/internal/jira"
	"github.com/trick77/noergler/internal/queue"
	"github.com/trick77/noergler/internal/render"
	"github.com/trick77/noergler/internal/store"
	"github.com/trick77/noergler/internal/webhook"
)

// Scheduler stages a review's inference off the review worker.
//
// infer runs on the inference pool; post runs back on the single worker, so
// every Bitbucket call stays one at a time. Both are called with the ctx
// given to Stage, which carries pr_tag and the httpstats scope.
//
// An alias of the queue's own type, not a second declaration: Go unifies
// interfaces structurally when assigning a value but NOT when matching a
// func type, and this one travels through queue.JobFunc. Two identical
// declarations would not interchange there. queue imports nothing from
// here, so the direction is safe.
type Scheduler = queue.Scheduler

// reviewPlan is what the inference and posting stages need from prepare.
//
// It carries no file bodies, no raw diff and nothing but the one prompt
// string: the summary needs only counts and names, so the heavy locals die
// with the prepare frame and a review waiting for a pool slot keeps roughly
// one prompt resident rather than a whole PR's content.
type reviewPlan struct {
	key          store.PRKey
	project      string
	repo         string
	prID         int
	prTag        string
	upsert       store.PRUpsert
	sourceCommit string
	// incrementalFrom is empty for a full review.
	incrementalFrom string
	// mention is the old skipAuthorCheck: this run answers a mention rather
	// than passing the author gate.
	mention bool
	// started is when the review began, so the elapsed time the summary and
	// the run row report is the author's wall clock, pool wait included.
	started time.Time
	// counter spans all three stages: the totals line is logged once, at the
	// end of post, and must include the gateway call and the posting.
	counter *httpstats.Counter

	prompt       string
	promptTokens int

	existing       []store.Finding
	contentSkipped []string
	ticket         *jira.Ticket
	parentTicket   *jira.Ticket
	breakdown      render.PromptBreakdown
	crossFileSyms  []string
	agentsMDFound  bool
	budget         int
	filesReviewed  int
	// totalFiles already includes the deleted and renamed paths.
	totalFiles  int
	diffAdded   int
	diffRemoved int

	// result is filled by the inference stage and read by the posting stage.
	// Only one goroutine touches it at a time: the pool goroutine writes it,
	// and the queue submits post only after that write has returned.
	result inference.ReviewResult
}

// abort logs a prepare exit's HTTP totals and reports "not ok", so each of
// prepare's early returns is one line.
//
// The totals used to be deferred in the entry function. With the inference
// call staged off the worker, that defer would fire before the gateway call
// and before posting, reporting inference=0 and none of the post stage's
// Bitbucket calls.
//
// why names the pre-flight exit. SkipNone means the exit was a hard failure
// rather than a decision (an unparseable payload, a diff that would not
// fetch), and writes no attempt row: the dashboard's skip breakdown counts
// choices the pipeline made, not faults it hit.
func (r *Reviewer) abort(ctx context.Context, key store.PRKey, kind store.RunKind, why SkipReason, prTag string, counter *httpstats.Counter) bool {
	r.logHTTPTotals(ctx, prTag, counter)
	if why != SkipNone {
		r.recordAttempt(ctx, store.Attempt{
			Key: key, TeamSlug: r.TeamSlug, Kind: kind,
			Outcome: "skipped", Reason: string(why),
		})
	}
	return false
}

// ReviewPullRequestStaged reviews a PR with the gateway call handed to sched
// instead of running inline, so the review worker is free during it.
//
// It reports whether it handed off. True means the PR's queue hold must
// outlive this call: the inference and the posting that follows are still
// outstanding, and a second run of the same PR would race on the
// prior-commit pointer, the summary and the inline comments. Every prepare
// exit returns false, having logged its own totals.
//
// skipAuthorCheck is true for a mention-triggered review: the person asking
// for it is the authorization, so the auto-review author list does not
// apply.
func (r *Reviewer) ReviewPullRequestStaged(ctx context.Context, payload *webhook.Payload, team string, skipAuthorCheck bool, sched Scheduler) bool {
	plan, ctx, ok := r.prepare(ctx, payload, skipAuthorCheck)
	if !ok {
		return false
	}

	sched.Stage(ctx, plan.key, team,
		func(inferCtx context.Context) { plan.result = r.infer(inferCtx, plan) },
		func(postCtx context.Context) { r.post(postCtx, plan, plan.result) },
	)
	return true
}
