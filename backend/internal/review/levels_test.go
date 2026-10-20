package review

import (
	"bytes"
	"context"
	"log/slog"
	"strings"
	"testing"

	"github.com/trick77/noergler/internal/bitbucket"
	"github.com/trick77/noergler/internal/webhook"
)

// An incremental diff over the cap is a decision, not a fault: INFO, and the
// full review still runs, since a push that deletes a big file the PR added
// nets out in the full diff.
func TestIncrementalDiffTooLargeFallsBackQuietly(t *testing.T) {
	h := newHarness(t, nil)
	h.st.lastCommit, h.st.hasLast = "older", true
	size := 99
	h.bb.commitErr = &bitbucket.ContentTooLarge{What: "push diff", Limit: 10, Size: &size}
	var buf bytes.Buffer
	h.r.log = slog.New(slog.NewTextHandler(&buf, nil))

	h.r.ReviewPullRequest(context.Background(), prPayload(webhook.EventFromRefUpdated), false)

	if len(h.llm.Reviews) != 1 || h.st.Runs[0].Incremental {
		t.Fatalf("reviews %d; want one full review", len(h.llm.Reviews))
	}
	logs := buf.String()
	wantLogged(t, logs, "level=INFO", "incremental diff too large")
	if strings.Contains(logs, "failed unexpectedly") {
		t.Errorf("logged as a fault:\n%s", logs)
	}
}

// The author resolving the thread before the answer lands is not an error:
// INFO, and nothing posted means nothing stamped.
func TestReplyOnResolvedThreadIsNotAnError(t *testing.T) {
	h, p := replyHarness(t)
	h.bb.replyErr = &bitbucket.StatusError{Method: "POST", Path: "/comments", Status: 400,
		Body: `{"errors":[{"message":"Reply cannot be made to resolved thread."}]}`}
	var buf bytes.Buffer
	h.r.log = slog.New(slog.NewTextHandler(&buf, nil))

	h.r.HandleFindingReply(context.Background(), p, "payments", nil)

	if len(h.st.Replied) != 0 {
		t.Errorf("marked %+v after a refused post", h.st.Replied)
	}
	logs := buf.String()
	wantLogged(t, logs, "thread resolved meanwhile")
	if strings.Contains(logs, "level=ERROR") {
		t.Errorf("logged as an error:\n%s", logs)
	}
}
