package queue

import (
	"bytes"
	"context"
	"errors"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/trick77/noergler/internal/webhook"
	"github.com/trick77/ship/kit/logging"
)

// syncBuf is a log sink the test may read while the worker still writes it.
type syncBuf struct {
	mu  sync.Mutex
	buf bytes.Buffer
}

func (b *syncBuf) Write(p []byte) (int, error) {
	b.mu.Lock()
	defer b.mu.Unlock()
	return b.buf.Write(p)
}

func (b *syncBuf) String() string {
	b.mu.Lock()
	defer b.mu.Unlock()
	return b.buf.String()
}

// await fails the test instead of hanging the package when ch never closes.
func await(t *testing.T, ch <-chan struct{}, what string) {
	t.Helper()
	select {
	case <-ch:
	case <-time.After(2 * time.Second):
		t.Fatalf("%s did not happen", what)
	}
}

// The worker's end-of-turn line says what the turn did. A staged review has
// only been prepared when the worker moves on, and its inference and posting
// follow under the same tag, so the queue never says "completed": that word
// was read as "review finished, no inference" in prod. The review's own
// completion line (review.post) is the one completion per run. A review
// that did not hand off exited in prepare and says so; a recovered panic
// says failed, not done. Every queue line carries team=.
func TestQueueLogSaysWhatEachTurnDid(t *testing.T) {
	var buf syncBuf
	log, err := logging.Setup(logging.Options{Level: "info", Format: logging.Text, Writer: &buf})
	if err != nil {
		t.Fatal(err)
	}

	posted := make(chan struct{})
	q := New(func(ctx context.Context, team string, p *webhook.Payload, sched Scheduler) bool {
		if p.PullRequest.ID == 3 {
			return false // a prepare exit: skipped, nothing staged
		}
		sched.Stage(ctx, key(p.PullRequest.ID), team,
			func(context.Context) {},
			func(context.Context) { close(posted) },
		)
		return true
	}, 1, 1, log)
	q.Start(context.Background())
	defer q.Stop()

	q.Submit(key(1), prFor(1), "t")
	await(t, posted, "post of PR 1")

	jobRan := make(chan struct{})
	q.SubmitJob(key(2), "t", "merge-rollup", func(context.Context, Scheduler) bool { close(jobRan); return false })
	await(t, jobRan, "merge-rollup job")

	q.Submit(key(3), prFor(3), "t")
	behind := make(chan struct{})
	q.SubmitJob(key(3), "t", "decline", func(context.Context, Scheduler) bool { close(behind); return false })
	await(t, behind, "job behind the skipped review")

	q.SubmitJob(key(4), "t", "purge", func(context.Context, Scheduler) bool { panic(errors.New("boom")) })
	waitFor(t, func() bool { return strings.Contains(buf.String(), "purge failed") })
	q.Stop()

	got := buf.String()
	for _, want := range []string{
		"queue[P/r#1]: enqueued (depth=1)",
		"queue[P/r#1]: starting review",
		"queue[P/r#1]: review prepared in ",
		", inference running",
		"queue[P/r#1]: inference done in ",
		", post enqueued (depth=",
		"queue[P/r#1]: starting post",
		"queue[P/r#1]: post done in ",
		"queue[P/r#2]: merge-rollup enqueued",
		"queue[P/r#2]: starting merge-rollup",
		"queue[P/r#2]: merge-rollup done in ",
		"queue[P/r#3]: review ended in ",
		", no inference",
		"queue[P/r#4]: purge panicked: boom",
		"queue[P/r#4]: purge failed in ",
	} {
		if !strings.Contains(got, want) {
			t.Errorf("log lacks %q:\n%s", want, got)
		}
	}
	if strings.Contains(got, "completed") {
		t.Errorf("queue log says completed; that line belongs to the review:\n%s", got)
	}
	for _, line := range strings.Split(strings.TrimSpace(got), "\n") {
		if strings.Contains(line, "queue[") && !strings.Contains(line, "team=t") {
			t.Errorf("queue line without team=: %s", line)
		}
	}
}
