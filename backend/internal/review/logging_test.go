package review

import (
	"bytes"
	"context"
	"log/slog"
	"strings"
	"testing"

	"github.com/trick77/noergler/internal/httpstats"
	"github.com/trick77/noergler/internal/inference"
)

// capturingReviewer is a Reviewer whose log output can be read back.
func capturingReviewer(t *testing.T) (*Reviewer, *bytes.Buffer) {
	t.Helper()
	var buf bytes.Buffer
	r := New(Options{
		TeamSlug:  "payments",
		Bitbucket: newFakeBitbucket(),
		Log:       slog.New(slog.NewTextHandler(&buf, &slog.HandlerOptions{Level: slog.LevelDebug})),
	})
	return r, &buf
}

// The totals line is what one review spent upstream. Nothing is logged when no
// request was made: the author and actor gates return before any HTTP, and an
// empty line for every skipped PR is noise.
func TestLogHTTPTotals(t *testing.T) {
	t.Run("reports per-label totals and the method detail", func(t *testing.T) {
		r, buf := capturingReviewer(t)
		ctx, c := httpstats.WithScope(context.Background())
		c.Record("bitbucket", "GET")
		c.Record("bitbucket", "GET")
		c.Record("bitbucket", "POST")
		c.Record("jira", "GET")

		r.logHTTPTotals(ctx, "PROJ/repo#1", c)

		out := buf.String()
		for _, want := range []string{
			"Review HTTP totals", "bitbucket=3", "jira=1", "inference=0",
			"bitbucket:GET=2", "bitbucket:POST=1", "jira:GET=1",
		} {
			if !strings.Contains(out, want) {
				t.Errorf("log %q does not contain %q", out, want)
			}
		}
	})

	t.Run("says nothing when no request was made", func(t *testing.T) {
		r, buf := capturingReviewer(t)
		ctx, c := httpstats.WithScope(context.Background())

		r.logHTTPTotals(ctx, "PROJ/repo#1", c)

		if out := buf.String(); out != "" {
			t.Errorf("logged %q for a review that made no request, want nothing", out)
		}
	})

	t.Run("a label beyond the three named ones still shows in the detail", func(t *testing.T) {
		r, buf := capturingReviewer(t)
		ctx, c := httpstats.WithScope(context.Background())
		c.Record("riptide", "POST")

		r.logHTTPTotals(ctx, "PROJ/repo#1", c)

		if out := buf.String(); !strings.Contains(out, "riptide:POST=1") {
			t.Errorf("log %q drops a label outside the three named ones", out)
		}
	})
}

// logCost writes the per-call record, warning only where the gateway owed a
// price and did not give one.
func TestLogCost(t *testing.T) {
	n := func(v int64) *int64 { return &v }

	t.Run("priced call logs at info", func(t *testing.T) {
		r, buf := capturingReviewer(t)
		r.logCost(context.Background(), "PROJ/repo#1",
			inference.CallCost{NanoUSD: n(12_300_000), CallID: "abc"})

		out := buf.String()
		if !strings.Contains(out, "level=INFO") {
			t.Errorf("log %q is not INFO", out)
		}
		if !strings.Contains(out, "PROJ/repo#1") {
			t.Errorf("log %q does not carry the PR tag", out)
		}
	})

	t.Run("unpriced call from a LiteLLM proxy warns", func(t *testing.T) {
		r, buf := capturingReviewer(t)
		r.logCost(context.Background(), "PROJ/repo#1", inference.CallCost{CallID: "abc"})

		if out := buf.String(); !strings.Contains(out, "level=WARN") {
			t.Errorf("log %q is not a warning", out)
		}
	})
}
