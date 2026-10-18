package review

import (
	"context"
	"strings"
	"testing"

	"github.com/trick77/noergler/internal/inference"
	"github.com/trick77/noergler/internal/store"
	"github.com/trick77/noergler/internal/webhook"
)

// The summary says what its issue list leaves out: the earlier finding still
// open, and one the validator dropped. The repeat of that earlier finding is
// the open one, not a second item.
func TestSummaryAccountsForWhatItDoesNotList(t *testing.T) {
	h := newHarness(t, nil)
	h.st.lastCommit, h.st.hasLast = "older", true
	h.bb.commitDiff = sampleDiff
	h.st.existing = []store.Finding{{ID: 11, FilePath: "a.go", LineNumber: 2, Severity: "issue", CommentText: "old"}}
	invented := inference.ReviewFinding{File: "a.go", Line: 2, Severity: "suggestion", Comment: "x",
		Evidence: []string{"never in the diff"}}
	h.llm.review = okResultWith(finding("a.go", 2, "issue", "same again"), invented)

	h.r.ReviewPullRequest(context.Background(), prPayload(webhook.EventFromRefUpdated), false)

	if len(h.bb.Posted) != 1 {
		t.Fatalf("summaries = %d", len(h.bb.Posted))
	}
	want := "_Earlier findings: 1 still open. 1 finding dropped by validation._"
	if s := h.bb.Posted[0].Text; !strings.Contains(s, want) {
		t.Errorf("summary lacks %q:\n%s", want, s)
	}
}

// A retired finding is counted as fixed, not as open.
func TestSummaryCountsARetiredFinding(t *testing.T) {
	h := fixedHarness(t, "func new() {}")

	h.r.ReviewPullRequest(context.Background(), prPayload(webhook.EventFromRefUpdated), false)

	want := "_Earlier findings: 1 fixed by this push._"
	if s := h.bb.Posted[0].Text; !strings.Contains(s, want) {
		t.Errorf("summary lacks %q:\n%s", want, s)
	}
}

// A full review can retire nothing, so it says nothing about earlier
// findings, fixed or open.
func TestFullReviewSummarySaysNothingOfEarlierFindings(t *testing.T) {
	h := newHarness(t, nil)
	h.st.existing = []store.Finding{{ID: 11, FilePath: "a.go", LineNumber: 2, Severity: "issue", CommentText: "old"}}

	h.r.ReviewPullRequest(context.Background(), prPayload(webhook.EventOpened), false)

	if s := h.bb.Posted[0].Text; strings.Contains(s, "Earlier findings") {
		t.Errorf("a full review must not count earlier findings:\n%s", s)
	}
}

// A first review has nothing to account for: no line at all.
func TestFirstReviewSummaryHasNoAccounting(t *testing.T) {
	h := newHarness(t, nil)
	h.llm.review = okResultWith(finding("a.go", 2, "issue", "new"))

	h.r.ReviewPullRequest(context.Background(), prPayload(webhook.EventOpened), false)

	if s := h.bb.Posted[0].Text; strings.Contains(s, "Earlier findings") || strings.Contains(s, "dropped by validation") {
		t.Errorf("a first review must not carry the accounting line:\n%s", s)
	}
}
