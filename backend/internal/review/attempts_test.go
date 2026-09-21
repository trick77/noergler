package review

import (
	"context"
	"errors"
	"testing"

	"github.com/trick77/noergler/internal/inference"
	"github.com/trick77/noergler/internal/store"
	"github.com/trick77/noergler/internal/webhook"
)

// review_runs holds successes only, so before review_attempts existed a
// failure or a skip left no record anywhere but the log. These pin that every
// terminal outcome now writes one, WITHOUT disturbing what review_runs means:
// TestTerminalOutcomes still owns the run-row and upsert side.
func TestAttemptRecordedForEveryOutcome(t *testing.T) {
	cases := []struct {
		name    string
		outcome inference.Outcome
		want    string
	}{
		{"ok", inference.OutcomeOK, "ok"},
		{"timed out", inference.OutcomeTimedOut, "timed_out"},
		{"unparseable", inference.OutcomeUnparseable, "unparseable"},
		{"too large", inference.OutcomeTooLarge, "too_large"},
		// The one outcome that posts nothing and upserts nothing. It still
		// has to be visible: an operator whose gateway is down would
		// otherwise see an empty dashboard and no reason for it.
		{"error", inference.OutcomeError, "error"},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			h := newHarness(t, nil)
			h.llm.review = inference.ReviewResult{
				Outcome: c.outcome,
				Review:  inference.ParsedReview{Summary: inference.NewReviewSummary()},
			}

			h.r.ReviewPullRequest(context.Background(), prPayload(webhook.EventOpened), false)

			if len(h.st.Attempts) != 1 {
				t.Fatalf("expected one attempt row, got %d", len(h.st.Attempts))
			}
			got := h.st.Attempts[0]
			if got.Outcome != c.want {
				t.Errorf("outcome = %q, want %q", got.Outcome, c.want)
			}
			if got.Reason != "" {
				t.Errorf("a non-skip must carry no reason, got %q", got.Reason)
			}
			if got.TeamSlug == "" {
				t.Error("attempt must carry the authenticated team slug")
			}
			if got.Kind != store.RunAuto {
				t.Errorf("kind = %q, want %q", got.Kind, store.RunAuto)
			}
		})
	}
}

// A successful attempt links its run row, so the feed can show that run's
// findings and cost without guessing which run an attempt produced.
func TestSuccessfulAttemptLinksItsRun(t *testing.T) {
	h := newHarness(t, nil)

	h.r.ReviewPullRequest(context.Background(), prPayload(webhook.EventOpened), false)

	if len(h.st.Attempts) != 1 {
		t.Fatalf("expected one attempt, got %d", len(h.st.Attempts))
	}
	a := h.st.Attempts[0]
	if a.RunID == nil {
		t.Fatal("a successful attempt must carry its run id")
	}
	if len(h.st.Runs) != 1 {
		t.Fatalf("expected one run, got %d", len(h.st.Runs))
	}
	if a.ElapsedMS == nil {
		t.Error("a successful attempt must carry its elapsed time")
	}
}

// A pre-flight skip never reaches the gateway, so it has no outcome of its
// own: it is recorded as "skipped" plus the reason that decided it. Without
// the reason the dashboard can only say a PR was skipped, which is the
// question rather than the answer.
func TestSkipRecordsItsReason(t *testing.T) {
	h := newHarness(t, nil)
	h.r.SetAuthorLists(nil, []string{"renovate"})

	p := prPayload(webhook.EventOpened)
	p.PullRequest.Author.User.Name = "renovate"
	h.r.ReviewPullRequest(context.Background(), p, false)

	if len(h.st.Attempts) != 1 {
		t.Fatalf("expected one attempt, got %d", len(h.st.Attempts))
	}
	a := h.st.Attempts[0]
	if a.Outcome != "skipped" {
		t.Errorf("outcome = %q, want %q", a.Outcome, "skipped")
	}
	if a.Reason != string(SkipIgnoredAuthor) {
		t.Errorf("reason = %q, want %q", a.Reason, SkipIgnoredAuthor)
	}
	if len(h.st.Runs) != 0 {
		t.Errorf("a skip must write no run row, got %d", len(h.st.Runs))
	}
}

// resolveDiff used to answer with a bare false for four different stops, so
// the two commonest skips in the pipeline reached the abort as SkipNone and
// were recorded as nothing at all. Each of these was invisible before.
func TestDiffSkipsRecordTheirOwnReason(t *testing.T) {
	cases := []struct {
		name  string
		setup func(*harness)
		want  SkipReason
	}{
		{
			name: "an empty diff is a decision, not a fault",
			setup: func(h *harness) {
				h.bb.prDiff = "   \n"
			},
			want: SkipEmptyDiff,
		},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			h := newHarness(t, func(h *harness) { c.setup(h) })

			h.r.ReviewPullRequest(context.Background(), prPayload(webhook.EventOpened), false)

			if len(h.st.Attempts) != 1 {
				t.Fatalf("expected one attempt, got %d", len(h.st.Attempts))
			}
			if got := h.st.Attempts[0]; got.Reason != string(c.want) {
				t.Errorf("reason = %q, want %q", got.Reason, c.want)
			}
		})
	}
}

// A HEAD that has not moved is the single most common skip in a busy
// instance: every retrigger and no-op force push lands here.
func TestHeadUnchangedRecordsItsReason(t *testing.T) {
	h := newHarness(t, nil)
	h.st.lastCommit, h.st.hasLast = "src1", true

	p := prPayload(webhook.EventFromRefUpdated)
	h.r.ReviewPullRequest(context.Background(), p, false)

	if len(h.st.Attempts) != 1 {
		t.Fatalf("expected one attempt, got %d", len(h.st.Attempts))
	}
	if got := h.st.Attempts[0]; got.Reason != string(SkipHeadUnchanged) {
		t.Errorf("reason = %q, want %q", got.Reason, SkipHeadUnchanged)
	}
}

// A diff that would not fetch is a FAULT, not a decision: it writes no
// attempt row, so the skip breakdown counts only what the pipeline chose.
func TestDiffFetchFailureIsNotASkip(t *testing.T) {
	h := newHarness(t, func(h *harness) {
		h.bb.prDiffErr = errors.New("bitbucket is down")
	})

	h.r.ReviewPullRequest(context.Background(), prPayload(webhook.EventOpened), false)

	if len(h.st.Attempts) != 0 {
		t.Errorf("a fetch fault must write no attempt, got %+v", h.st.Attempts)
	}
}

// The attempt row is a record of what happened; it must never change what
// happened. A store that refuses every write still produces a full review.
func TestAttemptWriteFailsOpen(t *testing.T) {
	h := newHarness(t, nil)

	h.r.ReviewPullRequest(context.Background(), prPayload(webhook.EventOpened), false)
	posted := len(h.bb.Posted)
	inline := len(h.bb.Inline)
	if posted == 0 {
		t.Fatal("harness must post a summary for a healthy review")
	}

	h2 := newHarness(t, nil)
	h2.st.attemptErr = errors.New("attempts table is on fire")

	h2.r.ReviewPullRequest(context.Background(), prPayload(webhook.EventOpened), false)

	if len(h2.bb.Posted) != posted {
		t.Errorf("summary posts = %d, want %d: a failed attempt write must not change the review",
			len(h2.bb.Posted), posted)
	}
	if len(h2.bb.Inline) != inline {
		t.Errorf("inline comments = %d, want %d", len(h2.bb.Inline), inline)
	}
	if len(h2.st.Runs) != 1 {
		t.Errorf("run rows = %d, want 1: the run must still be recorded", len(h2.st.Runs))
	}
}

// A mention asking for a review IS a review, so it is recorded as one, with
// kind=mention rather than a separate outcome. The Q&A path writes nothing;
// that split is what keeps "what was reviewed" answerable.
func TestMentionReviewIsRecordedAsAMentionKind(t *testing.T) {
	h := newHarness(t, nil)

	// skipAuthorCheck=true is the mention entry: HandleMention delegates
	// here for an empty question or a review keyword.
	h.r.ReviewPullRequest(context.Background(), prPayload(webhook.EventOpened), true)

	if len(h.st.Attempts) != 1 {
		t.Fatalf("expected one attempt, got %d", len(h.st.Attempts))
	}
	if got := h.st.Attempts[0]; got.Kind != store.RunMention {
		t.Errorf("kind = %q, want %q", got.Kind, store.RunMention)
	}
	if got := h.st.Attempts[0]; got.Outcome != "ok" {
		t.Errorf("outcome = %q, want ok", got.Outcome)
	}
}
