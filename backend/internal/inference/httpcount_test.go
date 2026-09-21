package inference

import (
	"context"
	"net/http"
	"strings"
	"testing"

	"github.com/trick77/llmwire"

	"github.com/trick77/noergler/internal/httpstats"
)

// The inference label has to reach the per-review HTTP totals. Before the
// client llmwire is built with wrapped its transport, the totals line reported
// inference=0 on every review while the call plainly happened, so the one
// upstream that costs money was the one the log said nothing about.
func TestInferenceRequestsAreCounted(t *testing.T) {
	const alias = "ai-gateway-gpt-5.5"
	f := &fakeGateway{}
	srv := f.start(t, alias)
	c := newTestClient(t, srv, alias)

	ctx, counter := httpstats.WithScope(context.Background())
	// Startup is one /models call plus the ping completion.
	if err := c.Startup(ctx); err != nil {
		t.Fatalf("Startup: %v", err)
	}

	totals := counter.Summarize()
	if totals["inference"] < 2 {
		t.Errorf("inference count = %d, want at least 2 (models + ping); totals: %v",
			totals["inference"], totals)
	}
	// The label must not leak into the two that name a different upstream.
	if totals["bitbucket"] != 0 || totals["jira"] != 0 {
		t.Errorf("inference requests counted under another label: %v", totals)
	}

	methods := counter.Methods()
	var sawInference bool
	for k := range methods {
		if strings.HasPrefix(k, "inference:") {
			sawInference = true
		}
	}
	if !sawInference {
		t.Errorf("no inference entry in the per-method detail: %v", methods)
	}
}

// Supplying an HTTPClient means llmwire skips building its own, so the two
// properties of its default are reproduced in countingClient and pinned here.
// Mirrors llmwire's TestNew_DefaultHTTPClientHasNoWholeRequestTimeout.
func TestCountingClientKeepsLLMWiresTimeoutShape(t *testing.T) {
	c := countingClient()

	// A whole-request timeout would cut a slow completion body mid-read, the
	// same reason Bitbucket's client must not carry one. The caller's ctx and
	// llmwire's CallTimeout bound the call instead.
	if c.Timeout != 0 {
		t.Errorf("Timeout = %v, want 0: a whole-request bound would cut a long completion", c.Timeout)
	}

	// Assert on the transport actually installed in the client, not a second
	// one built the same way: the point is that countingClient wires the tuned
	// transport in, which a fresh tunedTransport() call would not catch.
	wrapper, ok := c.Transport.(interface{ Unwrap() http.RoundTripper })
	if !ok {
		t.Fatalf("client transport %T does not expose the wrapped transport", c.Transport)
	}
	tr, ok := wrapper.Unwrap().(*http.Transport)
	if !ok {
		t.Fatalf("wrapped transport is %T, want *http.Transport", wrapper.Unwrap())
	}

	// The backstop must be strictly later than llmwire's own header bound, or
	// the two race and the transport's generic "timeout awaiting response
	// headers" replaces the guard's named bound.
	if tr.ResponseHeaderTimeout <= llmwire.DefaultHeaderTimeout {
		t.Errorf("ResponseHeaderTimeout = %v, want strictly more than the header bound %v, "+
			"so llmwire's guard reports the timeout under its own name",
			tr.ResponseHeaderTimeout, llmwire.DefaultHeaderTimeout)
	}
}

// A counter is per review. A client built once and used across reviews must
// count into whichever scope the call runs in, not the one it was built in.
func TestInferenceCountsIntoTheCallersScope(t *testing.T) {
	const alias = "ai-gateway-gpt-5.5"
	f := &fakeGateway{}
	srv := f.start(t, alias)
	c := newTestClient(t, srv, alias)

	if err := c.Startup(context.Background()); err != nil {
		t.Fatalf("Startup: %v", err)
	}

	ctx, counter := httpstats.WithScope(context.Background())
	if err := c.ping(ctx); err != nil {
		t.Fatalf("ping: %v", err)
	}
	if got := counter.Summarize()["inference"]; got != 1 {
		t.Errorf("inference count = %d, want 1 for a single call in this scope", got)
	}
}
