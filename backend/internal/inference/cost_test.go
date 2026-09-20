package inference

import (
	"context"
	"io"
	"log/slog"
	"strings"
	"testing"

	"github.com/trick77/llmwire"
)

// LogLine is the per-call cost record. It warns only where the gateway owed a
// price and did not give one: an endpoint that never prices must not warn on
// every call, or the warning stops meaning anything.
func TestCallCostLogLine(t *testing.T) {
	n := func(v int64) *int64 { return &v }

	cases := []struct {
		name     string
		cost     CallCost
		wantWarn bool
		contains []string
	}{
		{
			name:     "priced call is informational",
			cost:     CallCost{NanoUSD: n(12_300_000), KeySpendNanoUSD: n(5_000_000_000), CallID: "abc"},
			wantWarn: false,
			contains: []string{"response-cost=$0.012300000", "key-spend=$5.000000000", "call-id=abc"},
		},
		{
			name:     "no headers at all is not a LiteLLM proxy, so no warning",
			cost:     CallCost{},
			wantWarn: false,
			contains: []string{"response-cost=absent", "key-spend=absent", "call-id=absent"},
		},
		{
			name:     "a call id without a price is LiteLLM failing to price",
			cost:     CallCost{CallID: "abc"},
			wantWarn: true,
			contains: []string{"unpriced call", "call-id=abc"},
		},
		{
			name:     "zero cost after real work is suspicious",
			cost:     CallCost{NanoUSD: n(0), ZeroWithTokens: true, PromptTokens: 1200, CompletionTokens: 300},
			wantWarn: true,
			contains: []string{"zero cost", "1500 tokens"},
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			line, warn := tc.cost.LogLine()
			if warn != tc.wantWarn {
				t.Errorf("warn = %v, want %v (line: %s)", warn, tc.wantWarn, line)
			}
			for _, want := range tc.contains {
				if !strings.Contains(line, want) {
					t.Errorf("line %q does not contain %q", line, want)
				}
			}
		})
	}
}

func TestCostFrom(t *testing.T) {
	n := func(v int64) *int64 { return &v }

	t.Run("reported cost is priced", func(t *testing.T) {
		got := CostFrom(&llmwire.ChatResponse{
			Usage: llmwire.Usage{
				Cost: llmwire.Cost{NanoUSD: 92_500, Provenance: llmwire.Reported},
			},
		})
		if !got.Priced() {
			t.Fatal("want priced")
		}
		if *got.NanoUSD != 92_500 {
			t.Errorf("NanoUSD = %d, want 92500", *got.NanoUSD)
		}
		if got.ZeroWithTokens {
			t.Error("a non-zero cost is not suspicious")
		}
	})

	// Unpriced is the zero value, so a forgotten fill reads as unknown rather
	// than free. Cost fails open: the review proceeds with a NULL cost.
	t.Run("unpriced leaves the cost nil", func(t *testing.T) {
		got := CostFrom(&llmwire.ChatResponse{
			Usage: llmwire.Usage{Cost: llmwire.Cost{NanoUSD: 1234, Provenance: llmwire.Unpriced}},
		})
		if got.Priced() {
			t.Error("Unpriced must not report a cost, even with a non-zero figure")
		}
		if got.NanoUSD != nil {
			t.Errorf("NanoUSD = %v, want nil", got.NanoUSD)
		}
	})

	t.Run("zero cost with tokens is flagged", func(t *testing.T) {
		got := CostFrom(&llmwire.ChatResponse{
			Usage: llmwire.Usage{
				Input:  llmwire.InputTokens{Total: n(500)},
				Output: llmwire.OutputTokens{Total: n(20)},
				Cost:   llmwire.Cost{NanoUSD: 0, Provenance: llmwire.Reported},
			},
		})
		if !got.Priced() {
			t.Fatal("a reported zero is still priced")
		}
		if !got.ZeroWithTokens {
			t.Error("zero cost with tokens consumed should be flagged")
		}
	})

	t.Run("zero cost with no tokens is not flagged", func(t *testing.T) {
		got := CostFrom(&llmwire.ChatResponse{
			Usage: llmwire.Usage{Cost: llmwire.Cost{Provenance: llmwire.Reported}},
		})
		if got.ZeroWithTokens {
			t.Error("no tokens means nothing to be suspicious about")
		}
	})

	// Key spend is a gauge: carried through for display, never summed.
	t.Run("gateway fields are carried through", func(t *testing.T) {
		got := CostFrom(&llmwire.ChatResponse{
			Gateway: llmwire.Gateway{CallID: "call-123", KeySpendNanoUSD: n(42)},
		})
		if got.CallID != "call-123" {
			t.Errorf("CallID = %q, want call-123", got.CallID)
		}
		if got.KeySpendNanoUSD == nil || *got.KeySpendNanoUSD != 42 {
			t.Errorf("KeySpendNanoUSD = %v, want 42", got.KeySpendNanoUSD)
		}
	})

	t.Run("absent token lanes count as zero", func(t *testing.T) {
		if got := totalTokens(llmwire.Usage{}); got != 0 {
			t.Errorf("totalTokens = %d, want 0", got)
		}
	})
}

// The cost header the gateway sends must survive as a priced call end to end.
func TestCostFromLiveResponse(t *testing.T) {
	const alias = "ai-gateway-gpt-5.5"
	f := &fakeGateway{costHeader: "0.0000925"}
	srv := f.start(t, alias)

	c, err := New(Options{
		Model:           testProfile,
		ReasoningEffort: "medium",
		APIKey:          "team-key",
		Env:             env(srv.URL, alias, nil),
		Logger:          slog.New(slog.NewTextHandler(io.Discard, nil)),
	})
	if err != nil {
		t.Fatalf("New: %v", err)
	}

	resp, _, err := c.wire.Chat(context.Background(), llmwire.ChatRequest{
		Model:     testProfile,
		Reasoning: llmwire.ReasoningEffort("medium"),
		Messages:  []llmwire.Message{llmwire.User("hi")},
	})
	if err != nil {
		t.Fatalf("Chat: %v", err)
	}
	got := CostFrom(resp)
	if !got.Priced() {
		t.Fatalf("want a priced call, got %+v (provenance %v)", got, resp.Usage.Cost.Provenance)
	}
	// 0.0000925 USD is 92500 nano-USD.
	if *got.NanoUSD != 92_500 {
		t.Errorf("NanoUSD = %d, want 92500", *got.NanoUSD)
	}
}
