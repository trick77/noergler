package review

import (
	"bytes"
	"context"
	"errors"
	"log/slog"
	"slices"
	"strings"
	"testing"

	"github.com/trick77/noergler/internal/bitbucket"
	"github.com/trick77/noergler/internal/inference"
	"github.com/trick77/noergler/internal/store"
	"github.com/trick77/noergler/internal/webhook"
)

// fixedHarness is an incremental review of sampleDiff with one earlier
// finding (row 11, comment 501) on a.go, and a model claiming it fixed with
// the given evidence.
func fixedHarness(t *testing.T, evidence ...string) *harness {
	t.Helper()
	h := newHarness(t, nil)
	h.st.lastCommit, h.st.hasLast = "older", true
	h.bb.commitDiff = sampleDiff
	cid := 501
	h.bb.comments[cid] = &bitbucket.Comment{ID: cid} // its thread, open on the PR
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

	h.r.reviewInline(context.Background(), prPayload(webhook.EventFromRefUpdated), false)

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

// A full review's diff is the whole PR, every line of it added, the buggy one
// too: no id is listed, and a claim retires nothing.
func TestFullReviewRetiresNothing(t *testing.T) {
	h := fixedHarness(t, "func new() {}")
	h.st.lastCommit, h.st.hasLast = "", false

	h.r.reviewInline(context.Background(), prPayload(webhook.EventFromRefUpdated), false)

	if p := h.llm.Reviews[0].Prompt; strings.Contains(p, "[F11]") || !strings.Contains(p, "a.go:2 [issue]") {
		t.Errorf("a full review lists the finding without its id:\n%s", p)
	}
	if len(h.bb.Calls) != 0 || len(h.st.Fixed) != 0 {
		t.Errorf("calls = %v, fixed = %v", h.bb.Calls, h.st.Fixed)
	}
}

// Context only: the diff did not change what the claim quotes. Nothing
// reaches Bitbucket and the row stays open.
func TestUnprovenFixedClaimTouchesNothing(t *testing.T) {
	h := fixedHarness(t, "package a")

	h.r.reviewInline(context.Background(), prPayload(webhook.EventFromRefUpdated), false)

	if len(h.bb.Calls) != 0 || len(h.st.Fixed) != 0 {
		t.Errorf("calls = %v, fixed = %v: an unproven claim must change nothing", h.bb.Calls, h.st.Fixed)
	}
}

// A push that half-fixes a finding: the model retires it and reports what is
// left on the same line. The new finding is not a repeat of the one retired.
func TestRetiredFindingDoesNotDedupANewOne(t *testing.T) {
	h := fixedHarness(t, "func new() {}")
	h.llm.review.Review.Findings = []inference.ReviewFinding{finding("a.go", 2, "issue", "still leaks on error")}

	h.r.reviewInline(context.Background(), prPayload(webhook.EventFromRefUpdated), false)

	if len(h.bb.Inline) != 1 {
		t.Errorf("inline = %+v, want the new finding posted", h.bb.Inline)
	}
	if len(h.st.Fixed) != 1 {
		t.Errorf("fixed = %v, want the old one retired", h.st.Fixed)
	}
}

// A failed reply leaves the thread open, never closed without saying why.
// The row is stamped all the same: no later diff holds the fix to retry on.
func TestFailedReplyLeavesTheThreadOpen(t *testing.T) {
	h := fixedHarness(t, "func new() {}")
	h.bb.replyErr = errors.New("boom")

	h.r.reviewInline(context.Background(), prPayload(webhook.EventFromRefUpdated), false)

	if want := []string{"reply:501"}; !slices.Equal(h.bb.Calls, want) {
		t.Errorf("calls = %v, want %v", h.bb.Calls, want)
	}
	if len(h.st.Fixed) != 1 {
		t.Errorf("fixed = %v, want the row stamped", h.st.Fixed)
	}
	if len(h.bb.Posted) != 1 {
		t.Errorf("summary posts = %d, want the run to finish", len(h.bb.Posted))
	}
}

// A reply Bitbucket refuses because the author resolved the thread first is
// not a failure: no resolve call, the row stamped, INFO not WARN.
func TestReplyOnResolvedThreadStampsQuietly(t *testing.T) {
	h := fixedHarness(t, "func new() {}")
	h.bb.replyErr = &bitbucket.StatusError{Status: 400,
		Body: `{"errors":[{"message":"Reply cannot be made to resolved thread."}]}`}
	var buf bytes.Buffer
	h.r.log = slog.New(slog.NewTextHandler(&buf, nil))

	h.r.reviewInline(context.Background(), prPayload(webhook.EventFromRefUpdated), false)

	if want := []string{"reply:501"}; !slices.Equal(h.bb.Calls, want) {
		t.Errorf("calls = %v, want %v", h.bb.Calls, want)
	}
	if len(h.st.Fixed) != 1 {
		t.Errorf("fixed = %v, want the row stamped", h.st.Fixed)
	}
	logs := buf.String()
	wantLogged(t, logs, "thread resolved already", "reply posted no, thread resolved no")
	if strings.Contains(logs, "level=WARN") {
		t.Errorf("logged as a warning:\n%s", logs)
	}
}

func TestFailedResolveDoesNotFailTheRun(t *testing.T) {
	h := fixedHarness(t, "func new() {}")
	h.bb.resolveErr = errors.New("boom")

	h.r.reviewInline(context.Background(), prPayload(webhook.EventFromRefUpdated), false)

	if len(h.st.Fixed) != 1 || len(h.bb.Posted) != 1 {
		t.Errorf("fixed = %v, summary posts = %d", h.st.Fixed, len(h.bb.Posted))
	}
}

// No run row: the thread is resolved, so the row is stamped with no run
// rather than left to be claimed again.
func TestFixedIsStampedWithoutARunRow(t *testing.T) {
	h := fixedHarness(t, "func new() {}")
	h.st.runErr = errors.New("db down")

	h.r.reviewInline(context.Background(), prPayload(webhook.EventFromRefUpdated), false)

	if len(h.st.Fixed) != 1 || h.st.Fixed[0] != [2]int64{11, 0} {
		t.Errorf("fixed = %v, want row 11 stamped with run 0", h.st.Fixed)
	}
}

// A run that did not succeed posts nothing, and retires nothing.
func TestFixedClaimOnAFailedRunIsIgnored(t *testing.T) {
	h := fixedHarness(t, "func new() {}")
	h.llm.review.Outcome = inference.OutcomeUnparseable

	h.r.reviewInline(context.Background(), prPayload(webhook.EventFromRefUpdated), false)

	if len(h.bb.Calls) != 0 || len(h.st.Fixed) != 0 {
		t.Errorf("calls = %v, fixed = %v", h.bb.Calls, h.st.Fixed)
	}
}
