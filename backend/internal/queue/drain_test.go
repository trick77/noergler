package queue

import (
	"context"
	"io"
	"log/slog"
	"testing"
	"time"
)

// A job in flight when Stop is called must run to completion with a LIVE
// context.
//
// Stop exists to drain the item in flight, but every job context is derived
// from the one Start was given. Hand Start a signal context and SIGTERM
// cancels the very review Stop is waiting for: its diff fetch, its LLM call
// and its store writes all fail with context canceled, and the review path
// swallows store failures through safeDB, so the run row and its cost are
// lost silently. cmd/noergler therefore starts the worker on a context
// cancelled only by Stop.
func TestStopDrainsTheItemInFlightWithALiveContext(t *testing.T) {
	root, cancel := context.WithCancel(context.Background())
	defer cancel()
	q := New(nil, slog.New(slog.NewTextHandler(io.Discard, nil)))
	q.Start(root)

	started := make(chan struct{})
	done := make(chan error, 1)
	q.SubmitJob("PROJ/repo#1", "platform", func(ctx context.Context) {
		close(started)
		time.Sleep(20 * time.Millisecond)
		done <- ctx.Err()
	})

	<-started
	q.Stop()

	if err := <-done; err != nil {
		t.Errorf("job context was %v during the drain, want a live one", err)
	}
}
