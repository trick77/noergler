package review

import (
	"context"
	"strings"
	"testing"

	"github.com/trick77/noergler/internal/inference"
	"github.com/trick77/noergler/internal/store"
	"github.com/trick77/noergler/internal/webhook"
)

// inlineScheduler runs both stages immediately, so a staged review behaves
// exactly like a synchronous one. Every behavioural assertion below is then
// about the split itself, not about the pool.
type inlineScheduler struct{ staged int }

func (s *inlineScheduler) Stage(ctx context.Context, _ store.PRKey, _ string, infer, post func(context.Context)) {
	s.staged++
	infer(ctx)
	post(ctx)
}

// deferredScheduler holds the two stages so a test can observe what the
// review has, and has not, done by the time it returns from the worker.
type deferredScheduler struct {
	infer, post func(context.Context)
	ctx         context.Context
	key         store.PRKey
	team        string
}

func (s *deferredScheduler) Stage(ctx context.Context, key store.PRKey, team string, infer, post func(context.Context)) {
	s.ctx, s.key, s.team, s.infer, s.post = ctx, key, team, infer, post
}

func (s *deferredScheduler) finish() {
	s.infer(s.ctx)
	s.post(s.ctx)
}

// The staged entry must not have called the gateway, posted anything or
// written a row by the time it returns. That is the whole point: the worker
// is free while the inference runs.
func TestStagedReturnsBeforeInference(t *testing.T) {
	h := newHarness(t, nil)
	h.llm.review = inference.ReviewResult{
		Outcome: inference.OutcomeOK,
		Review:  inference.ParsedReview{Summary: inference.NewReviewSummary()},
	}

	sched := &deferredScheduler{}
	handedOff := h.r.ReviewPullRequestStaged(context.Background(),
		prPayload(webhook.EventOpened), "platform", sched)

	if !handedOff {
		t.Fatal("a staged review must report the handoff, or the queue drops its hold")
	}
	if len(h.llm.Reviews) != 0 {
		t.Errorf("the gateway was called on the worker: %d calls", len(h.llm.Reviews))
	}
	if len(h.bb.Inline) != 0 || len(h.bb.Posted) != 0 {
		t.Errorf("posted before the inference: %d inline, %d summary", len(h.bb.Inline), len(h.bb.Posted))
	}
	if len(h.st.Runs) != 0 {
		t.Errorf("wrote a run row before the inference: %d", len(h.st.Runs))
	}

	// Finishing the stages produces the complete review.
	sched.finish()
	if len(h.llm.Reviews) != 1 {
		t.Errorf("gateway calls after finishing = %d, want 1", len(h.llm.Reviews))
	}
	if len(h.st.Runs) != 1 {
		t.Errorf("run rows after finishing = %d, want 1", len(h.st.Runs))
	}
	if len(h.bb.Posted) == 0 {
		t.Error("no summary posted after finishing")
	}
}

// Stage is given the PR's own key and team, which is what the queue holds
// the review by and what the per-team pool cap is counted against.
func TestStagedPassesKeyAndTeam(t *testing.T) {
	h := newHarness(t, nil)
	h.llm.review = inference.ReviewResult{
		Outcome: inference.OutcomeOK,
		Review:  inference.ParsedReview{Summary: inference.NewReviewSummary()},
	}

	sched := &deferredScheduler{}
	h.r.ReviewPullRequestStaged(context.Background(), prPayload(webhook.EventOpened), "platform", sched)
	defer sched.finish()

	if sched.team != "platform" {
		t.Errorf("team = %q, want platform", sched.team)
	}
	if sched.key.PRID == 0 || sched.key.Project == "" || sched.key.Repo == "" {
		t.Errorf("incomplete key: %+v", sched.key)
	}
}

// A guard that stops the review before the gateway must not hand off: the
// queue would keep the PR held forever waiting for work nobody scheduled.
func TestStagedSkipDoesNotHandOff(t *testing.T) {
	h := newHarness(t, nil)
	// An author outside the auto-review list stops prepare at the gate.
	h.r.SetAuthorLists([]string{"someone-else"}, nil)

	sched := &deferredScheduler{}
	if h.r.ReviewPullRequestStaged(context.Background(),
		prPayload(webhook.EventOpened), "platform", sched) {
		t.Fatal("a skipped review must not report a handoff")
	}
	if sched.infer != nil {
		t.Error("a skipped review must not stage anything")
	}
}

// The outcome matrix is identical on the staged path. It is pinned for the
// synchronous one in TestTerminalOutcomes; the split must not move any of
// it, so the same table runs again through Stage.
func TestStagedTerminalOutcomesMatchSynchronous(t *testing.T) {
	cases := []struct {
		name        string
		outcome     inference.Outcome
		wantNotice  string
		wantPosting bool
	}{
		{"timed out", inference.OutcomeTimedOut, "no response from the model", true},
		{"unparseable", inference.OutcomeUnparseable, "could not be processed", true},
		{"too large", inference.OutcomeTooLarge, "too large to review", true},
		{"error", inference.OutcomeError, "", false},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			h := newHarness(t, nil)
			h.st.lastCommit, h.st.hasLast = "older", true
			h.llm.review = inference.ReviewResult{
				Outcome: c.outcome,
				Review:  inference.ParsedReview{Summary: inference.NewReviewSummary()},
			}

			sched := &inlineScheduler{}
			h.r.ReviewPullRequestStaged(context.Background(),
				prPayload(webhook.EventOpened), "platform", sched)

			if sched.staged != 1 {
				t.Fatalf("staged %d times, want 1", sched.staged)
			}
			if len(h.st.Runs) != 0 {
				t.Errorf("a non-ok outcome must write no run row, got %d", len(h.st.Runs))
			}
			if len(h.bb.Inline) != 0 {
				t.Errorf("a non-ok outcome must post no inline comments, got %d", len(h.bb.Inline))
			}

			if !c.wantPosting {
				// OutcomeError posts nothing and upserts nothing.
				if len(h.bb.Posted) != 0 || len(h.bb.Updates) != 0 {
					t.Errorf("OutcomeError must post nothing, got %d posts and %d updates",
						len(h.bb.Posted), len(h.bb.Updates))
				}
				if len(h.st.Upserts) != 0 {
					t.Errorf("OutcomeError must not upsert, got %d", len(h.st.Upserts))
				}
				return
			}

			if len(h.bb.Posted) != 1 {
				t.Fatalf("expected one notice, got %d", len(h.bb.Posted))
			}
			if !strings.Contains(h.bb.Posted[0].Text, c.wantNotice) {
				t.Errorf("notice = %q, want it to mention %q", h.bb.Posted[0].Text, c.wantNotice)
			}
			// The prior commit is preserved, never advanced by a failure.
			if len(h.st.Upserts) != 1 {
				t.Fatalf("expected one upsert, got %d", len(h.st.Upserts))
			}
			if got := h.st.Upserts[0].LastReviewedCommit; got == nil || *got != "older" {
				t.Errorf("pointer = %v, want the prior commit preserved", got)
			}
		})
	}
}
