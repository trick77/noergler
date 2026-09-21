package review

import (
	"context"
	"errors"
	"strings"
	"testing"

	"github.com/trick77/noergler/internal/bitbucket"
	"github.com/trick77/noergler/internal/webhook"
)

// countDiffLines ran on every review but nothing ever read its output, so the
// header and non-reviewable rules were executed and never checked.

func TestCountDiffLines(t *testing.T) {
	const binaryDiff = `diff --git a/logo.png b/logo.png
index 1111111..2222222 100644
Binary files a/logo.png and b/logo.png differ
`
	const lockDiff = `diff --git a/package-lock.json b/package-lock.json
index 1111111..2222222 100644
--- a/package-lock.json
+++ b/package-lock.json
@@ -1,3 +1,3 @@
-  "version": "1.0.0",
+  "version": "1.0.1",
`

	cases := []struct {
		name           string
		diff           string
		added, removed int
	}{
		// sampleDiff is one added and one removed line under a ---/+++ pair,
		// so a counter that mistook the headers for content would say 2/2.
		{"added and removed", sampleDiff, 1, 1},
		{"empty diff", "", 0, 0},
		{"binary marker is not counted", binaryDiff, 0, 0},
		{"non-reviewable file is skipped", lockDiff, 0, 0},
		{"reviewable and ignored mixed", sampleDiff + lockDiff, 1, 1},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			added, removed := countDiffLines(tc.diff)
			if added != tc.added || removed != tc.removed {
				t.Errorf("countDiffLines = +%d/-%d, want +%d/-%d", added, removed, tc.added, tc.removed)
			}
		})
	}
}

// The ---/+++ headers must not be read as content even though they start with
// the same characters as a real change.
func TestCountDiffLinesIgnoresFileHeaders(t *testing.T) {
	// Every line of this hunk except the two headers is context, so both
	// counts must be zero while the headers are present.
	const headersOnly = `diff --git a/a.go b/a.go
index 1111111..2222222 100644
--- a/a.go
+++ b/a.go
@@ -1,2 +1,2 @@
 package a
 // unchanged
`
	if added, removed := countDiffLines(headersOnly); added != 0 || removed != 0 {
		t.Errorf("headers counted as changes: +%d/-%d, want +0/-0", added, removed)
	}
}

// The counts reach the reader rather than being computed and discarded: the
// run row carries them on every review.
func TestDiffLineCountsReachTheRunRow(t *testing.T) {
	h := newHarness(t, nil)

	h.r.ReviewPullRequest(context.Background(), prPayload(webhook.EventOpened), false)

	if len(h.st.Runs) != 1 {
		t.Fatalf("expected one run row, got %d", len(h.st.Runs))
	}
	if got := h.st.Runs[0].LinesAdded; got != 1 {
		t.Errorf("run LinesAdded = %d, want 1", got)
	}
	if got := h.st.Runs[0].LinesRemoved; got != 1 {
		t.Errorf("run LinesRemoved = %d, want 1", got)
	}
}

// And the merge rollup carries them too, which is the figure that leaves the
// service.
func TestDiffLineCountsReachTheRollup(t *testing.T) {
	h := newHarness(t, nil)
	h.st.rollup = snapshot()

	h.r.HandlePRMerged(context.Background(), prPayload(webhook.EventMerged))

	if len(h.st.Claims) != 1 {
		t.Fatalf("expected one rollup claim, got %d", len(h.st.Claims))
	}
	added, removed := h.st.Claims[0].LinesAdded, h.st.Claims[0].LinesRemoved
	if added == nil || removed == nil {
		t.Fatalf("rollup carries no line counts: added=%v removed=%v", added, removed)
	}
	if *added != 1 {
		t.Errorf("rollup LinesAdded = %d, want 1", *added)
	}
	if *removed != 1 {
		t.Errorf("rollup LinesRemoved = %d, want 1", *removed)
	}
}

// prepareFiles draws a distinction the reader can see: a body dropped because
// it was too large or too long is reported as skipped, a body that failed to
// fetch for any other reason is not. Neither branch had a test.

// oversizeHarness drops a.go's body for the given reason and runs one review.
func contentHarness(t *testing.T, tweak func(*harness)) *harness {
	t.Helper()
	h := newHarness(t, tweak)
	h.r.ReviewPullRequest(context.Background(), prPayload(webhook.EventOpened), false)
	return h
}

func TestPerFileContentTooLargeIsSkippedAndReported(t *testing.T) {
	h := contentHarness(t, func(h *harness) {
		h.bb.fileErr["src1:a.go"] = &bitbucket.ContentTooLarge{Limit: 1024}
	})

	if len(h.llm.Reviews) != 1 {
		t.Fatalf("review should proceed from the diff alone, got %d LLM calls", len(h.llm.Reviews))
	}
	if len(h.bb.Posted) != 1 {
		t.Fatalf("expected one summary, got %d", len(h.bb.Posted))
	}
	if !strings.Contains(h.bb.Posted[0].Text, "a.go") {
		t.Errorf("summary should name the file whose content was skipped, got:\n%s", h.bb.Posted[0].Text)
	}
	if strings.Contains(h.llm.Reviews[0].Prompt, "package a\nfunc new() {}") {
		t.Error("prompt carries the full body of an oversized file")
	}
}

func TestFileOverMaxLinesIsSkippedAndReported(t *testing.T) {
	// MaxFileLines is 1000 in the harness config; 1001 lines is one over.
	h := contentHarness(t, func(h *harness) {
		h.bb.files["src1:a.go"] = strings.Repeat("x\n", 1001)
	})

	if len(h.bb.Posted) != 1 {
		t.Fatalf("expected one summary, got %d", len(h.bb.Posted))
	}
	if !strings.Contains(h.bb.Posted[0].Text, "a.go") {
		t.Errorf("summary should name the over-long file, got:\n%s", h.bb.Posted[0].Text)
	}
	if strings.Contains(h.llm.Reviews[0].Prompt, strings.Repeat("x\n", 50)) {
		t.Error("prompt carries the body of a file over the line limit")
	}
}

// A file just under the limit keeps its body, which is what makes the test
// above a limit test rather than a "content is never sent" test.
func TestFileAtMaxLinesKeepsItsContent(t *testing.T) {
	// strings.Count(content, "\n") + 1 is how the body is measured, so 999
	// newlines is exactly 1000 lines.
	h := contentHarness(t, func(h *harness) {
		h.bb.files["src1:a.go"] = strings.Repeat("x\n", 999)
	})

	if len(h.llm.Reviews) != 1 {
		t.Fatalf("expected one LLM call, got %d", len(h.llm.Reviews))
	}
	if !strings.Contains(h.llm.Reviews[0].Prompt, strings.Repeat("x\n", 50)) {
		t.Error("a file at the limit should keep its body in the prompt")
	}
}

// A generic fetch failure reviews from the diff alone and is deliberately NOT
// reported as skipped content: that line means "too large", not "unavailable".
func TestGenericFetchFailureIsNotReportedAsSkipped(t *testing.T) {
	h := contentHarness(t, func(h *harness) {
		h.bb.fileErr["src1:a.go"] = errors.New("bitbucket is sulking")
	})

	if len(h.llm.Reviews) != 1 {
		t.Fatalf("review should proceed from the diff alone, got %d LLM calls", len(h.llm.Reviews))
	}
	if strings.Contains(h.llm.Reviews[0].Prompt, "package a\nfunc new() {}") {
		t.Error("prompt carries a body that failed to fetch")
	}
	if len(h.bb.Posted) != 1 {
		t.Fatalf("expected one summary, got %d", len(h.bb.Posted))
	}
	if strings.Contains(h.bb.Posted[0].Text, "Reviewed without full file context") {
		t.Errorf("a failed fetch must not be reported as skipped content, got:\n%s", h.bb.Posted[0].Text)
	}
}
