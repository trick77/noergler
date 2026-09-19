package inference

import (
	"context"
	"io"
	"log/slog"
	"testing"
)

// Before Startup the window is 0 and the curve floors at 2000 rather than
// failing, so a pre-flight check run too early would reject every prompt.
// Ready is the guard.
func TestReadyBeforeAndAfterStartup(t *testing.T) {
	const alias = "ai-gateway-gpt-5.5"
	f := &fakeGateway{}
	srv := f.start(t, alias)
	c := newTestClient(t, srv, alias)

	if c.Ready() {
		t.Error("a client must not report ready before Startup")
	}
	if got := c.InputTokenBudget(); got != budgetFloor {
		t.Errorf("unresolved budget = %d, want the floor %d", got, budgetFloor)
	}

	if err := c.Startup(context.Background()); err != nil {
		t.Fatalf("Startup: %v", err)
	}
	if !c.Ready() {
		t.Error("want ready after a successful Startup")
	}
	if got := c.InputTokenBudget(); got != 628_000 {
		t.Errorf("resolved budget = %d, want 628000", got)
	}
}

// An explicit window means the client is usable before Startup resolves one.
func TestReadyWithExplicitWindow(t *testing.T) {
	const alias = "ai-gateway-gpt-5.5"
	f := &fakeGateway{}
	srv := f.start(t, alias)
	c := newTestClient(t, srv, alias, func(o *Options) { o.ContextWindow = 1_000_000 })

	if !c.Ready() {
		t.Error("an explicit window makes the client ready immediately")
	}
}

// A pre-flight check on an unresolved window must not reject the prompt.
func TestPreflightSkippedWhenNotReady(t *testing.T) {
	const alias = "ai-gateway-gpt-5.5"
	f := &fakeGateway{chatBody: chatBody(`{"overview": "ok"}`)}
	srv := f.start(t, alias)
	c, err := New(Options{
		Model:           testProfile,
		ReasoningEffort: "medium",
		APIKey:          "team-key",
		HeadroomTokens:  defHeadroom,
		Threshold:       defThreshold,
		Tail:            defTail,
		Env:             env(srv.URL, alias, nil),
		Logger:          slog.New(slog.NewTextHandler(io.Discard, nil)),
	})
	if err != nil {
		t.Fatalf("New: %v", err)
	}

	// Far over the 2000 floor, but the window is unresolved, so the check is
	// skipped rather than rejecting everything.
	got := c.Review(context.Background(), ReviewRequest{Prompt: "x", PromptTokens: 500_000})
	if got.Outcome == OutcomeTooLarge {
		t.Error("an unresolved window must not reject the prompt as too large")
	}
}
