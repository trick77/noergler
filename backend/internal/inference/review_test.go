package inference

import (
	"context"
	"errors"
	"io"
	"log/slog"
	"net/http"
	"strings"
	"testing"
	"time"
)

// reviewClient builds a client against the fake gateway with a resolved window,
// skipping Startup so a test can drive one call in isolation.
func reviewClient(t *testing.T, f *fakeGateway) *Client {
	t.Helper()
	const alias = "gateway-alias"
	srv := f.start(t, alias)
	c, err := New(Options{
		Model:          testProfile,
		Registry:       testRegistry,
		APIKey:         "team-key",
		ContextWindow:  1_000_000,
		HeadroomTokens: defHeadroom,
		Threshold:      defThreshold,
		Tail:           defTail,
		Env:            env(srv.URL, alias, nil),
		Logger:         slog.New(slog.NewTextHandler(io.Discard, nil)),
	})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	return c
}

func chatBody(content string) string {
	return `{"choices":[{"message":{"role":"assistant","content":` + quote(content) + `}}],` +
		`"usage":{"prompt_tokens":10,"completion_tokens":2,"total_tokens":12}}`
}

// quote is a minimal JSON string encoder for fixtures.
func quote(s string) string {
	var b strings.Builder
	b.WriteByte('"')
	for _, r := range s {
		switch r {
		case '"':
			b.WriteString(`\"`)
		case '\\':
			b.WriteString(`\\`)
		case '\n':
			b.WriteString(`\n`)
		default:
			b.WriteRune(r)
		}
	}
	b.WriteByte('"')
	return b.String()
}

func TestReviewOK(t *testing.T) {
	f := &fakeGateway{
		chatBody:   chatBody(`{"overview": "looks fine", "verdict": {"decision": "approve"}}`),
		costHeader: "0.0000925",
	}
	c := reviewClient(t, f)

	got := c.Review(context.Background(), ReviewRequest{Prompt: "review this"})
	if got.Outcome != OutcomeOK {
		t.Fatalf("Outcome = %v, want ok (err: %v)", got.Outcome, got.Err)
	}
	if got.Review.Summary.Overview != "looks fine" {
		t.Errorf("Overview = %q", got.Review.Summary.Overview)
	}
	if !got.Cost.Priced() || *got.Cost.NanoUSD != 92_500 {
		t.Errorf("Cost = %+v, want 92500 nano-USD", got.Cost)
	}
}

// The pre-flight check refuses before spending anything.
func TestReviewPreflightTooLarge(t *testing.T) {
	f := &fakeGateway{}
	c := reviewClient(t, f)

	got := c.Review(context.Background(), ReviewRequest{
		Prompt:       "huge",
		PromptTokens: c.FitCeiling() + 1,
	})
	if got.Outcome != OutcomeTooLarge {
		t.Fatalf("Outcome = %v, want too_large", got.Outcome)
	}
	if f.gotChat != 0 {
		t.Error("the pre-flight check must refuse before calling the model")
	}
	// The summary still carries the default verdict.
	if got.Review.Summary.VerdictDecision != DefaultVerdictDecision {
		t.Errorf("VerdictDecision = %q, want the default", got.Review.Summary.VerdictDecision)
	}
	var tooLarge *TooLargeError
	if !asErr(got.Err, &tooLarge) {
		t.Errorf("Err = %v, want a TooLargeError", got.Err)
	}
}

// Exactly at the ceiling fits: the check is strictly greater-than.
func TestReviewPreflightBoundary(t *testing.T) {
	f := &fakeGateway{chatBody: chatBody(`{"overview": "ok"}`)}
	c := reviewClient(t, f)

	got := c.Review(context.Background(), ReviewRequest{
		Prompt:       "exact",
		PromptTokens: c.FitCeiling(),
	})
	if got.Outcome != OutcomeOK {
		t.Errorf("Outcome = %v, want ok at exactly the ceiling", got.Outcome)
	}
}

func TestReviewOutcomes(t *testing.T) {
	cases := []struct {
		name   string
		gw     fakeGateway
		want   Outcome
		expect string
	}{
		{
			"413 is too large",
			fakeGateway{chatStatus: http.StatusRequestEntityTooLarge, chatBody: `{"error":{"message":"payload too large"}}`},
			OutcomeTooLarge, "",
		},
		{
			"400 naming context length is too large",
			fakeGateway{chatStatus: http.StatusBadRequest,
				chatBody: `{"error":{"message":"This model's maximum context length is 1000000 tokens"}}`},
			OutcomeTooLarge, "",
		},
		{
			"400 with the code marker is too large",
			fakeGateway{chatStatus: http.StatusBadRequest,
				chatBody: `{"error":{"message":"too long","code":"context_length_exceeded"}}`},
			OutcomeTooLarge, "",
		},
		{
			"unrelated 400 is an error",
			fakeGateway{chatStatus: http.StatusBadRequest, chatBody: `{"error":{"message":"bad model id"}}`},
			OutcomeError, "",
		},
		{
			"500 is an error",
			fakeGateway{chatStatus: http.StatusInternalServerError, chatBody: `{"error":{"message":"boom"}}`},
			OutcomeError, "",
		},
		{
			"empty content is unparseable",
			fakeGateway{chatBody: chatBody("")},
			OutcomeUnparseable, "",
		},
		{
			"refusal is unparseable",
			fakeGateway{chatBody: chatBody("I'm sorry, but I cannot assist with that request.")},
			OutcomeUnparseable, "",
		},
		{
			"non-object JSON is unparseable",
			fakeGateway{chatBody: chatBody(`[1, 2, 3]`)},
			OutcomeUnparseable, "",
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			gw := tc.gw
			c := reviewClient(t, &gw)
			got := c.Review(context.Background(), ReviewRequest{Prompt: "x"})
			if got.Outcome != tc.want {
				t.Errorf("Outcome = %v, want %v (err: %v)", got.Outcome, tc.want, got.Err)
			}
		})
	}
}

// A slow gateway plus an expired context is a timeout, not a generic error.
func TestReviewTimeout(t *testing.T) {
	f := &fakeGateway{delay: 200 * time.Millisecond}
	c := reviewClient(t, f)

	ctx, cancel := context.WithTimeout(context.Background(), 20*time.Millisecond)
	defer cancel()

	got := c.Review(ctx, ReviewRequest{Prompt: "x"})
	if got.Outcome != OutcomeTimedOut {
		t.Errorf("Outcome = %v, want timed_out (err: %v)", got.Outcome, got.Err)
	}
}

func TestMention(t *testing.T) {
	t.Run("envelope is rendered", func(t *testing.T) {
		f := &fakeGateway{chatBody: chatBody(`{"answer": "because of X", "refs": [{"file": "a.py", "line": 4}]}`)}
		c := reviewClient(t, f)
		got := c.Mention(context.Background(), MentionRequest{Prompt: "why?"})
		if got.Outcome != OutcomeOK {
			t.Fatalf("Outcome = %v (err: %v)", got.Outcome, got.Err)
		}
		if !strings.Contains(got.Answer, "because of X") || !strings.Contains(got.Answer, "`a.py`:4") {
			t.Errorf("Answer = %q", got.Answer)
		}
	})

	// No unparseable outcome: the fallback keeps a non-envelope reply usable.
	t.Run("raw text falls back rather than failing", func(t *testing.T) {
		f := &fakeGateway{chatBody: chatBody("just a plain answer")}
		c := reviewClient(t, f)
		got := c.Mention(context.Background(), MentionRequest{Prompt: "why?"})
		if got.Outcome != OutcomeOK {
			t.Fatalf("Outcome = %v, want ok", got.Outcome)
		}
		if got.Answer != "just a plain answer" {
			t.Errorf("Answer = %q", got.Answer)
		}
	})

	t.Run("pre-flight too large", func(t *testing.T) {
		f := &fakeGateway{}
		c := reviewClient(t, f)
		got := c.Mention(context.Background(), MentionRequest{
			Prompt:       "x",
			PromptTokens: c.FitCeiling() + 1,
		})
		if got.Outcome != OutcomeTooLarge {
			t.Errorf("Outcome = %v, want too_large", got.Outcome)
		}
		if f.gotChat != 0 {
			t.Error("must refuse before calling the model")
		}
	})
}

func TestOutcomeString(t *testing.T) {
	cases := map[Outcome]string{
		OutcomeOK:          "ok",
		OutcomeTimedOut:    "timed_out",
		OutcomeTooLarge:    "too_large",
		OutcomeUnparseable: "unparseable",
		OutcomeError:       "error",
	}
	for o, want := range cases {
		if got := o.String(); got != want {
			t.Errorf("Outcome(%d).String() = %q, want %q", o, got, want)
		}
	}
}

// asErr is errors.As with a bit less ceremony at the call site.
func asErr[T error](err error, target *T) bool {
	var t T
	if errors.As(err, &t) {
		*target = t
		return true
	}
	return false
}
