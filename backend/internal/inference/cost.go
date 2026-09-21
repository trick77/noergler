package inference

import (
	"fmt"

	"github.com/trick77/llmwire"
)

// CallCost is what one inference call cost, as noergler records it.
//
// NanoUSD is nil when the call went unpriced. Cost fails open (AGENTS.md): an
// unpriced run stores a NULL cost and the review proceeds; only the per-PR cap
// skips later auto-runs.
type CallCost struct {
	// NanoUSD is the per-call cost in billionths of a dollar, nil when the
	// gateway did not report one.
	NanoUSD *int64
	// KeySpendNanoUSD is the total already spent on the API key, as of this
	// call. A GAUGE, not a per-call amount: show it, never sum it, never store
	// it as a run cost, never feed it to the per-PR cap.
	KeySpendNanoUSD *int64
	// CallID is the id the gateway logged the call under: the handle for
	// matching an unpriced or failed review to the gateway's own log.
	CallID string
	// ZeroWithTokens records a priced call that reported zero cost despite
	// consuming tokens. Suspicious enough to warn about, not wrong enough to
	// fail a review.
	ZeroWithTokens bool

	// PromptTokens, CachedTokens and CompletionTokens are the endpoint's own
	// accounting, carried through for the run row and the summary footnote.
	// An endpoint that reports no accounting leaves them zero.
	//
	// CompletionTokens is Output.Total, which already includes reasoning
	// tokens; CachedTokens is the prompt tokens the endpoint served from its
	// own cache, a subset of PromptTokens rather than an addition to it.
	PromptTokens     int64
	CachedTokens     int64
	CompletionTokens int64
}

// CostFrom extracts the cost fields from a chat response.
//
// Reported is the only priced case: anything else, including a gateway that
// sends "None" for a deployment it cannot price, leaves NanoUSD nil.
func CostFrom(resp *llmwire.ChatResponse) CallCost {
	out := CallCost{
		CallID:           resp.Gateway.CallID,
		KeySpendNanoUSD:  resp.Gateway.KeySpendNanoUSD,
		PromptTokens:     deref(resp.Usage.Input.Total),
		CachedTokens:     deref(resp.Usage.Input.CacheRead),
		CompletionTokens: deref(resp.Usage.Output.Total),
	}
	if resp.Usage.Cost.Provenance != llmwire.Reported {
		return out
	}
	n := resp.Usage.Cost.NanoUSD
	out.NanoUSD = &n
	if n == 0 && totalTokens(resp.Usage) > 0 {
		out.ZeroWithTokens = true
	}
	return out
}

// Priced reports whether the call carried a usable cost.
func (c CallCost) Priced() bool { return c.NanoUSD != nil }

// LogLine is the per-call cost record, and whether it deserves a warning.
//
// Python writes one of these per call (llm_client.py:170). An unpriced call is
// only worth warning about when the endpoint IS a LiteLLM proxy: one that
// sends no x-litellm-* header at all never prices and would otherwise warn on
// every call forever. The call id is the handle for matching an unpriced or
// failed review to the gateway's own log.
func (c CallCost) LogLine() (line string, warn bool) {
	// Three decimals, not nano precision: the nine-decimal form buried the
	// figure a reader wants in trailing zeros. This is the log only; the DB
	// keeps BIGINT nano-USD and the riptide edge keeps its exact decimal
	// string. A sub-milli-dollar call therefore reads $0.000 here.
	cost := "absent"
	if c.NanoUSD != nil {
		cost = fmt.Sprintf("$%.3f", float64(*c.NanoUSD)/1e9)
	}
	spend := "absent"
	if c.KeySpendNanoUSD != nil {
		spend = fmt.Sprintf("$%.3f", float64(*c.KeySpendNanoUSD)/1e9)
	}
	callID := c.CallID
	if callID == "" {
		callID = "absent"
	}
	line = fmt.Sprintf("LLM cost headers: response-cost=%s key-spend=%s call-id=%s",
		cost, spend, callID)
	switch {
	case c.ZeroWithTokens:
		return line + "; gateway reported zero cost for a call that consumed " +
			fmt.Sprintf("%d tokens, recorded as $0.00", c.PromptTokens+c.CompletionTokens), true
	case c.NanoUSD == nil && c.CallID != "":
		// A call id means LiteLLM answered, so it should have priced this.
		return line + "; unpriced call", true
	}
	return line, false
}

// totalTokens is the prompt plus completion count, used only to tell a real
// zero cost from a zero that came with no work. Both lanes are nilable: an
// endpoint that reports no accounting leaves them unset, which counts as zero.
func totalTokens(u llmwire.Usage) int64 {
	return deref(u.Input.Total) + deref(u.Output.Total)
}

func deref(p *int64) int64 {
	if p == nil {
		return 0
	}
	return *p
}
