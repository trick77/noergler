package review

import (
	"bytes"
	"context"
	"fmt"
	"log/slog"
	"strings"
	"testing"

	"github.com/trick77/noergler/internal/store"
	"github.com/trick77/noergler/internal/webhook"
)

// reviewLogs runs one review of an opened PR and returns its log.
func reviewLogs(t *testing.T, tweak func(*harness), payload *webhook.Payload) string {
	t.Helper()
	h := newHarness(t, tweak)
	var buf bytes.Buffer
	h.r.log = slog.New(slog.NewTextHandler(&buf, nil))
	if payload == nil {
		payload = prPayload(webhook.EventOpened)
	}
	h.r.ReviewPullRequest(context.Background(), payload, false)
	return buf.String()
}

func wantLogged(t *testing.T, logs string, wants ...string) {
	t.Helper()
	for _, want := range wants {
		if !strings.Contains(logs, want) {
			t.Errorf("log lacks %q:\n%s", want, logs)
		}
	}
}

// Each of these changes what a run reviews or posts. None left a log line:
// a finding the model raised could vanish, a file could go unreviewed and a
// verdict could change with nothing to say so.
func TestSilentDecisionsAreLogged(t *testing.T) {
	t.Run("a finding an earlier run already posted", func(t *testing.T) {
		logs := reviewLogs(t, func(h *harness) {
			h.st.existing = []store.Finding{{FilePath: "a.go", LineNumber: 2, Severity: "issue", CommentText: "old"}}
			h.llm.review = okResultWith(finding("a.go", 2, "issue", "again"))
		}, nil)
		wantLogged(t, logs, "finding on a.go:2 (issue) already posted by an earlier run, not posted again")
	})

	t.Run("a finding over the comment cap", func(t *testing.T) {
		h := newHarness(t, func(h *harness) {
			h.llm.review = okResultWith(finding("a.go", 2, "suggestion", "minor"), finding("a.go", 2, "issue", "major"))
		})
		h.r.cfg.MaxComments = 1
		var buf bytes.Buffer
		h.r.log = slog.New(slog.NewTextHandler(&buf, nil))
		h.r.ReviewPullRequest(context.Background(), prPayload(webhook.EventOpened), false)
		wantLogged(t, buf.String(), "finding on a.go:2 (suggestion) over the 1-comment cap, not posted")
		if strings.Contains(buf.String(), "(issue) over the") {
			t.Errorf("the kept finding was logged as cut:\n%s", buf.String())
		}
	})

	t.Run("a verdict lowered after validation", func(t *testing.T) {
		logs := reviewLogs(t, func(h *harness) {
			invented := finding("a.go", 2, "issue", "invented")
			invented.Evidence = []string{"func missing() {}"}
			res := okResultWith(invented)
			res.Review.Summary.VerdictDecision = "request_changes"
			h.llm.review = res
		}, nil)
		wantLogged(t, logs, "verdict lowered request_changes -> approve after 1 dropped finding(s)")
	})

	t.Run("previously posted findings trimmed from the prompt", func(t *testing.T) {
		logs := reviewLogs(t, func(h *harness) {
			for i := 0; i <= maxPreviouslyPostedFindings; i++ {
				h.st.existing = append(h.st.existing, store.Finding{
					FilePath: fmt.Sprintf("f%02d.go", i), LineNumber: i + 1, Severity: "issue", CommentText: "short",
				})
			}
		}, nil)
		wantLogged(t, logs, fmt.Sprintf("previously posted findings trimmed: %d of %d shown to the model",
			maxPreviouslyPostedFindings, maxPreviouslyPostedFindings+1))
	})

	t.Run("no Jira key on the branch", func(t *testing.T) {
		logs := reviewLogs(t, func(h *harness) { h.jr = &fakeJira{} }, nil)
		// The text handler escapes the quotes around the branch.
		wantLogged(t, logs, "no Jira key in branch ", "feature/thing", " or title, reviewing without ticket context")
	})

	t.Run("a Jira key that resolves to no ticket", func(t *testing.T) {
		p := prPayload(webhook.EventOpened)
		p.PullRequest.FromRef.DisplayID = "feature/ABC-123-thing"
		logs := reviewLogs(t, func(h *harness) { h.jr = &fakeJira{} }, p)
		wantLogged(t, logs, "Jira ticket ABC-123 not readable, reviewing without ticket context")
	})

	t.Run("files skipped as non-reviewable are named", func(t *testing.T) {
		logs := reviewLogs(t, func(h *harness) {
			h.bb.prDiff = sampleDiff + addedFileDiff("go.sum", "example.com/mod v1.0.0 h1:hash")
		}, nil)
		wantLogged(t, logs, "skipped as binary/non-reviewable: go.sum")
	})

	t.Run("files compression leaves unreviewed are named", func(t *testing.T) {
		logs := reviewLogs(t, func(h *harness) {
			h.llm.budget = 400
			h.bb.prDiff = sampleDiff + addedFileDiff("big.go", strings.Repeat("y", 4_000))
		}, nil)
		wantLogged(t, logs, "1 file(s) over the 400-token budget are NOT reviewed, named to the model only: big.go")
	})
}

// A list long enough to flood one log line is capped with a count.
func TestCapPaths(t *testing.T) {
	var paths []string
	for i := 0; i < maxLoggedPaths+3; i++ {
		paths = append(paths, fmt.Sprintf("f%d.go", i))
	}
	got := capPaths(paths)
	if !strings.HasSuffix(got, "f9.go, +3 more") {
		t.Errorf("got %q", got)
	}
	if got := capPaths(paths[:2]); got != "f0.go, f1.go" {
		t.Errorf("got %q", got)
	}
}
