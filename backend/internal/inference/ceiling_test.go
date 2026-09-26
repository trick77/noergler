package inference

import (
	"context"
	"encoding/json"
	"errors"
	"testing"

	"github.com/trick77/llmwire"
)

// The fit ceiling is the window less the reply reserve (OutputTokenReserve),
// not the compression budget.
//
// Using the budget instead would skip PRs the model can hold: with a 1M window
// and the default knobs the budget is 628k and the ceiling 936k, and
// compression legitimately produces prompts between the two.
func TestFitCeilingIsNotTheCompressionBudget(t *testing.T) {
	const alias = "gateway-alias"
	f := &fakeGateway{}
	srv := f.start(t, alias)
	c := newTestClient(t, srv, alias, func(o *Options) { o.ContextWindow = 1_000_000 })

	if got, want := c.FitCeiling(), 1_000_000-OutputTokenReserve; got != want {
		t.Errorf("FitCeiling = %d, want %d", got, want)
	}
	if got := c.InputTokenBudget(); got != 628_000 {
		t.Errorf("InputTokenBudget = %d, want 628000", got)
	}
	if c.FitCeiling() <= c.InputTokenBudget() {
		t.Fatal("the ceiling must exceed the compression budget, or large PRs are skipped")
	}
}

// A prompt between the budget and the ceiling is reviewed, not skipped. This
// is the case the wrong ceiling silently lost.
func TestPromptBetweenBudgetAndCeilingIsReviewed(t *testing.T) {
	f := &fakeGateway{chatBody: chatBody(`{"overview": "ok"}`)}
	c := reviewClient(t, f)

	between := (c.InputTokenBudget() + c.FitCeiling()) / 2
	got := c.Review(context.Background(), ReviewRequest{Prompt: "x", PromptTokens: between})
	if got.Outcome != OutcomeOK {
		t.Errorf("Outcome = %v, want ok for a prompt the model can hold", got.Outcome)
	}
	if f.gotChat != 1 {
		t.Error("the model should have been called")
	}
}

// llmwire's own timeouts must read as timed out, not as a generic error: it
// applies CallTimeout to a derived context, so neither sentinel unwraps to
// context.DeadlineExceeded.
func TestLLMWireTimeoutSentinelsClassifyAsTimedOut(t *testing.T) {
	if errors.Is(llmwire.ErrCallCap, context.DeadlineExceeded) {
		t.Error("assumption broken: ErrCallCap now unwraps to DeadlineExceeded")
	}
	for _, err := range []error{llmwire.ErrCallCap, llmwire.ErrNoResponseHeaders} {
		if got := classifyCallError(context.Background(), err); got != OutcomeTimedOut {
			t.Errorf("classifyCallError(%v) = %v, want timed_out", err, got)
		}
	}
}

// A caller's own deadline still counts, and an unrelated error does not.
func TestClassifyCallError(t *testing.T) {
	if got := classifyCallError(context.Background(), context.DeadlineExceeded); got != OutcomeTimedOut {
		t.Errorf("a caller deadline = %v, want timed_out", got)
	}
	if got := classifyCallError(context.Background(), errors.New("boom")); got != OutcomeError {
		t.Errorf("an unrelated error = %v, want error", got)
	}
	overflow := &llmwire.APIError{StatusCode: 413}
	if got := classifyCallError(context.Background(), overflow); got != OutcomeTooLarge {
		t.Errorf("a 413 = %v, want too_large", got)
	}
}

// The wire contract accepts an integral float and refuses a fractional one.
func TestCoerceIntAcceptsIntegralFloat(t *testing.T) {
	cases := []struct {
		raw  string
		want int
		ok   bool
	}{
		{`5`, 5, true},
		{`"5"`, 5, true},
		{`5.0`, 5, true},
		{`5.5`, 0, false},
		{`"abc"`, 0, false},
		// null is refused: Go would unmarshal it into an int as 0, but a
		// required field holding null is invalid, not 0.
		{`null`, 0, false},
		// The wire contract coerces a bool too: line=true yields a finding
		// with line 1.
		{`true`, 1, true},
		{`false`, 0, true},
		{`-3.0`, -3, true},
	}
	for _, tc := range cases {
		got, ok := coerceInt(json.RawMessage(tc.raw))
		if ok != tc.ok || (ok && got != tc.want) {
			t.Errorf("coerceInt(%s) = (%d, %v), want (%d, %v)", tc.raw, got, ok, tc.want, tc.ok)
		}
	}
}

// confidence is a REQUIRED field in the response schema, so a model emitting
// it as 95.0 would otherwise drop every finding in the batch and report a
// silently empty review.
func TestIntegralFloatConfidenceKeepsTheFinding(t *testing.T) {
	got := ParseReview(`{"findings":[{"file":"a.py","line":1,"severity":"issue","comment":"c","confidence":95.0}]}`)
	if len(got.Findings) != 1 {
		t.Fatalf("got %d findings, want 1: an integral float confidence must not drop it", len(got.Findings))
	}
	if got.Findings[0].Confidence == nil || *got.Findings[0].Confidence != 95 {
		t.Errorf("Confidence = %v, want 95", got.Findings[0].Confidence)
	}
}

func TestIntegralFloatLineKeepsTheFinding(t *testing.T) {
	got := ParseReview(`{"findings":[{"file":"a.py","line":5.0,"severity":"issue","comment":"c"}]}`)
	if len(got.Findings) != 1 {
		t.Fatalf("got %d findings, want 1", len(got.Findings))
	}
	if got.Findings[0].Line != 5 {
		t.Errorf("Line = %d, want 5", got.Findings[0].Line)
	}
}

// A fractional line would move the finding, so it is refused.
func TestFractionalLineDropsTheFinding(t *testing.T) {
	got := ParseReview(`{"findings":[{"file":"a.py","line":5.5,"severity":"issue","comment":"c"}]}`)
	if len(got.Findings) != 0 {
		t.Errorf("got %+v, want the finding dropped", got.Findings)
	}
}
