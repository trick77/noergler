package inference

import (
	"bytes"
	"context"
	"encoding/json"
	"log/slog"
	"strings"
	"testing"
)

// The parser returns diagnostics as data; they only reach an operator if
// Review emits them. These pin the wiring, which a test of ParseReview alone
// cannot see.

// capturingClient is a client whose log output can be read back, answering
// every completion with body.
func capturingClient(t *testing.T, body string) (*Client, *bytes.Buffer) {
	t.Helper()
	const alias = "gateway-alias"
	f := &fakeGateway{chatBody: body}
	srv := f.start(t, alias)

	var buf bytes.Buffer
	c := newTestClient(t, srv, alias, func(o *Options) {
		o.Logger = slog.New(slog.NewTextHandler(&buf, &slog.HandlerOptions{Level: slog.LevelDebug}))
	})
	return c, &buf
}

// chatWith wraps content as a gateway completion response, JSON-quoting it so
// a body containing braces or quotes survives the envelope.
func chatWith(content string) string {
	blob, err := json.Marshal(content)
	if err != nil {
		panic(err)
	}
	return `{"choices":[{"message":{"role":"assistant","content":` + string(blob) + `}}],` +
		`"usage":{"prompt_tokens":5,"completion_tokens":1,"total_tokens":6}}`
}

func TestReviewEmitsParserDiagnostics(t *testing.T) {
	// An empty overview and a malformed finding: one WARN each, and both must
	// reach the log rather than being returned and dropped.
	c, buf := capturingClient(t, chatWith(`{"overview":"","findings":[42]}`))

	res := c.Review(context.Background(), ReviewRequest{Prompt: "p"})
	if res.Outcome != OutcomeOK {
		t.Fatalf("outcome = %v, want OK", res.Outcome)
	}
	if len(res.Review.Diagnostics) != 2 {
		t.Fatalf("diagnostics = %v, want 2", res.Review.Diagnostics)
	}

	out := buf.String()
	for _, want := range []string{"overview empty after parse", "Skipping malformed finding: 42"} {
		if !strings.Contains(out, want) {
			t.Errorf("log does not carry %q:\n%s", want, out)
		}
	}
	if !strings.Contains(out, "level=WARN") {
		t.Errorf("diagnostics did not reach the log at WARN:\n%s", out)
	}
}

// The level travels with the line: a dropped vacuous finding is INFO, and an
// operator alerting on ERROR must not be paged for it.
func TestReviewEmitsDiagnosticsAtTheirOwnLevel(t *testing.T) {
	content := `{"overview":"x","findings":[{"file":"a.py","line":1,"severity":"issue",` +
		`"comment":"c","suggestion":"No fix needed"}]}`
	c, buf := capturingClient(t, chatWith(content))

	c.Review(context.Background(), ReviewRequest{Prompt: "p"})

	out := buf.String()
	if !strings.Contains(out, "Dropping no-issue finding") {
		t.Fatalf("log does not carry the vacuous-finding line:\n%s", out)
	}
	if !strings.Contains(out, "level=INFO") {
		t.Errorf("vacuous finding was not logged at INFO:\n%s", out)
	}
	if strings.Contains(out, "level=ERROR") {
		t.Errorf("a dropped finding was logged at ERROR:\n%s", out)
	}
}

// A parse failure logs the prefix line AND still reports unparseable: it is
// logged twice, once in the parser and once in the review path.
func TestReviewEmitsTheParseFailureLine(t *testing.T) {
	c, buf := capturingClient(t, chatWith("not json at all"))

	res := c.Review(context.Background(), ReviewRequest{Prompt: "p"})

	if res.Outcome != OutcomeUnparseable {
		t.Fatalf("outcome = %v, want unparseable", res.Outcome)
	}
	out := buf.String()
	if !strings.Contains(out, "Failed to parse review response as JSON: not json at all") {
		t.Errorf("log does not carry the parse failure:\n%s", out)
	}
	if !strings.Contains(out, "level=ERROR") {
		t.Errorf("parse failure was not logged at ERROR:\n%s", out)
	}
}

// A clean response logs nothing from the parser: one line per good review
// would be noise.
func TestReviewIsSilentOnACleanResponse(t *testing.T) {
	content := `{"overview":"Adds a thing.","findings":[{"file":"a.py","line":1,` +
		`"severity":"issue","comment":"c"}]}`
	c, buf := capturingClient(t, chatWith(content))

	res := c.Review(context.Background(), ReviewRequest{Prompt: "p"})

	if len(res.Review.Diagnostics) != 0 {
		t.Errorf("clean response produced diagnostics: %v", res.Review.Diagnostics)
	}
	for _, unwanted := range []string{"overview empty", "Skipping malformed", "Dropping no-issue"} {
		if strings.Contains(buf.String(), unwanted) {
			t.Errorf("clean response logged %q:\n%s", unwanted, buf.String())
		}
	}
}

// A nil logger must not panic: New substitutes a discarding one.
func TestReviewWithoutALoggerDoesNotPanic(t *testing.T) {
	const alias = "gateway-alias"
	f := &fakeGateway{chatBody: chatWith(`{"overview":""}`)}
	srv := f.start(t, alias)
	c := newTestClient(t, srv, alias, func(o *Options) { o.Logger = nil })

	res := c.Review(context.Background(), ReviewRequest{Prompt: "p"})
	if len(res.Review.Diagnostics) != 1 {
		t.Fatalf("diagnostics = %v, want 1", res.Review.Diagnostics)
	}
}
