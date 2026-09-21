// Package inference wraps llmwire for per-team review and mention calls:
// startup checks, token budgets, prompt assembly, response parsing and cost.
//
// Behaviour is pinned by this package's tests; AGENTS.md lists the
// deliberate quirks.
package inference

// OutputTokenReserve is held back from the context window for the reply.
const OutputTokenReserve = 64_000

// MinContextWindow is the smallest window noergler will run on. A whole PR is
// reviewed in a single call, so a small-context model cannot hold a real PR
// coherently: refuse startup rather than review a partial view.
const MinContextWindow = 1_000_000

// budgetFloor is the smallest usable budget the curve will return.
const budgetFloor = 2000

// UsableContextBudget turns a model's advertised context window into a usable
// per-chunk token budget.
//
// A flat headroom is ~1.5% of a 1M window, which is useless, so the curve
// trusts the window fully up to a threshold and counts only a fraction beyond
// it. Large advertised windows are the least trustworthy: many endpoints
// enforce a lower server-side cap and 413 anything bigger.
//
// headroom, threshold and tail come from config (CONTEXT_WINDOW_HEADROOM_TOKENS,
// CONTEXT_TRUST_THRESHOLD, CONTEXT_TRUST_TAIL), which Phase 1 already reads.
//
// The curve is discontinuous at the threshold: with the defaults a window of
// 255_999 yields 239_999 while 256_001 yields 256_000. That jump is
// deliberate, and TestUsableContextBudgetIsDiscontinuous pins it.
func UsableContextBudget(window, headroom, threshold int, tail float64) int {
	var usable int
	if window <= threshold {
		usable = window - headroom
	} else {
		usable = threshold + int(float64(window-threshold)*tail)
	}
	if usable < budgetFloor {
		return budgetFloor
	}
	return usable
}

// MaxCumulativeContextTokens is the hard ceiling for the cumulative-diff
// context, regardless of model. Beyond it the model tends to drown the focused
// review in noise.
const MaxCumulativeContextTokens = 80_000

// MaxPreviouslyPostedFindingsTokens bounds the previously-posted block. Those
// findings are bounded by count as well: count guards prompt latency, tokens
// guard against a few very long comments inflating the prompt unboundedly.
const MaxPreviouslyPostedFindingsTokens = 4_000

// CumulativeDiffBudget is roughly a third of the input budget, so the focused
// review files still fit, capped at the hard ceiling.
//
// It belongs to the review pipeline of Phase 6, but it is a pure budget
// function with no pipeline state, so it sits beside the curve it derives
// from.
func CumulativeDiffBudget(inputBudget int) int {
	return clamp(inputBudget/3, 2_000, MaxCumulativeContextTokens)
}

// PreviouslyPostedBudget is roughly 5% of the input budget: the block is
// cross-context, not the focus of review, so the focused files dominate.
func PreviouslyPostedBudget(inputBudget int) int {
	return clamp(inputBudget/20, 500, MaxPreviouslyPostedFindingsTokens)
}

// clamp applies the lower bound first, so when the bounds conflict the upper
// bound wins. TestClampUpperBoundWins pins that order.
func clamp(v, lo, hi int) int {
	if v < lo {
		v = lo
	}
	if v > hi {
		v = hi
	}
	return v
}
