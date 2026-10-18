package review

import (
	"context"
	"errors"
	"slices"
	"strings"
	"testing"

	"github.com/trick77/noergler/internal/inference"
	"github.com/trick77/noergler/internal/store"
	"github.com/trick77/noergler/internal/webhook"
)

// fixedHarness has one earlier finding (row 11, comment 501) on a.go, which
// sampleDiff changes, and a model claiming it fixed with the given evidence.
func fixedHarness(t *testing.T, evidence ...string) *harness {
	t.Helper()
	h := newHarness(t, nil)
	cid := 501
	h.st.existing = []store.Finding{{
		ID: 11, FilePath: "a.go", LineNumber: 2, Severity: "issue",
		CommentText: "old() leaks", BitbucketCommentID: &cid,
	}}
	res := okResultWith()
	res.Review.Fixed = []inference.FixedClaim{{ID: "F11", Evidence: evidence}}
	h.llm.review = res
	return h
}

func TestFixedFindingIsRepliedToThenResolved(t *testing.T) {
	h := fixedHarness(t, "func new() {}")

	h.r.ReviewPullRequest(context.Background(), prPayload(webhook.EventFromRefUpdated), false)

	if !strings.Contains(h.llm.Reviews[0].Prompt, "[F11] a.go:2 [issue]") {
		t.Errorf("the prompt must list the finding by its id:\n%s", h.llm.Reviews[0].Prompt)
	}
	// Bitbucket takes no reply on a resolved thread: the order is the point.
	if want := []string{"reply:501", "resolve:501"}; !slices.Equal(h.bb.Calls, want) {
		t.Errorf("calls = %v, want %v", h.bb.Calls, want)
	}
	if len(h.bb.Replies) != 1 || h.bb.Replies[0].Text != "Fixed in `src1`." {
		t.Errorf("replies = %+v", h.bb.Replies)
	}
	if len(h.st.Fixed) != 1 || h.st.Fixed[0] != [2]int64{11, 1} {
		t.Errorf("fixed = %v, want row 11 stamped by run 1", h.st.Fixed)
	}
}

// Context only: the diff did not change what the claim quotes. Nothing
// reaches Bitbucket and the row stays open.
func TestUnprovenFixedClaimTouchesNothing(t *testing.T) {
	h := fixedHarness(t, "package a")

	h.r.ReviewPullRequest(context.Background(), prPayload(webhook.EventFromRefUpdated), false)

	if len(h.bb.Calls) != 0 || len(h.st.Fixed) != 0 {
		t.Errorf("calls = %v, fixed = %v: an unproven claim must change nothing", h.bb.Calls, h.st.Fixed)
	}
}

// Neither Bitbucket call failing fails the run, and the fix is recorded
// either way: it is what the diff shows.
func TestFixedFindingSurvivesBitbucketFailures(t *testing.T) {
	h := fixedHarness(t, "func new() {}")
	h.bb.replyErr = errors.New("boom")
	h.bb.resolveErr = errors.New("boom")

	h.r.ReviewPullRequest(context.Background(), prPayload(webhook.EventFromRefUpdated), false)

	if want := []string{"reply:501", "resolve:501"}; !slices.Equal(h.bb.Calls, want) {
		t.Errorf("calls = %v, want %v: a failed reply still tries the resolve", h.bb.Calls, want)
	}
	if len(h.st.Fixed) != 1 {
		t.Errorf("fixed = %v, want the row stamped", h.st.Fixed)
	}
	if len(h.bb.Posted) != 1 {
		t.Errorf("summary posts = %d, want the run to finish", len(h.bb.Posted))
	}
}

// A run that did not succeed posts nothing, and retires nothing.
func TestFixedClaimOnAFailedRunIsIgnored(t *testing.T) {
	h := fixedHarness(t, "func new() {}")
	h.llm.review.Outcome = inference.OutcomeUnparseable

	h.r.ReviewPullRequest(context.Background(), prPayload(webhook.EventFromRefUpdated), false)

	if len(h.bb.Calls) != 0 || len(h.st.Fixed) != 0 {
		t.Errorf("calls = %v, fixed = %v", h.bb.Calls, h.st.Fixed)
	}
}
