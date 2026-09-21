package inference

import "testing"

// Default knob values (CONTEXT_WINDOW_HEADROOM_TOKENS, CONTEXT_TRUST_THRESHOLD,
// CONTEXT_TRUST_TAIL) as Phase 1 reads them.
const (
	defHeadroom  = 16_000
	defThreshold = 256_000
	defTail      = 0.5
)

// Values generated from Python usable_context_budget. They are literals so a
// change to the curve fails here rather than silently repricing every review.
func TestUsableContextBudgetIsPinned(t *testing.T) {
	cases := []struct {
		window, want int
	}{
		// Floor: everything at or below headroom+floor clamps to 2000.
		{0, 2000},
		{1, 2000},
		{1000, 2000},
		{2000, 2000},
		{16_000, 2000},
		{16_001, 2000},
		{17_000, 2000},
		{18_000, 2000},
		// Below the trust threshold: window minus headroom.
		{128_000, 112_000},
		{200_000, 184_000},
		{255_999, 239_999},
		{256_000, 240_000},
		// Above it: threshold plus tail of the excess. Note the discontinuity,
		// 255_999 -> 239_999 but 256_001 -> 256_000, a jump up.
		{256_001, 256_000},
		{272_000, 264_000},
		{300_000, 278_000},
		{512_000, 384_000},
		{1_000_000, 628_000},
		{1_048_576, 652_288},
		{1_050_000, 653_000},
		{2_000_000, 1_128_000},
		{10_000_000, 5_128_000},
	}
	for _, tc := range cases {
		got := UsableContextBudget(tc.window, defHeadroom, defThreshold, defTail)
		if got != tc.want {
			t.Errorf("UsableContextBudget(%d) = %d, want %d (Python)", tc.window, got, tc.want)
		}
	}
}

// The curve steps up at the threshold rather than continuing smoothly. Pinned
// so a future "smoothing" is a decision, not an accident.
func TestUsableContextBudgetIsDiscontinuous(t *testing.T) {
	below := UsableContextBudget(defThreshold-1, defHeadroom, defThreshold, defTail)
	above := UsableContextBudget(defThreshold+1, defHeadroom, defThreshold, defTail)
	if above <= below {
		t.Errorf("expected a jump up at the threshold: %d -> %d", below, above)
	}
}

// The knobs are env-overridable, so the curve must honour non-default values.
func TestUsableContextBudgetHonoursKnobs(t *testing.T) {
	// No headroom, no tail: below the threshold the full window is usable,
	// above it nothing beyond the threshold counts.
	if got := UsableContextBudget(100_000, 0, 256_000, 0); got != 100_000 {
		t.Errorf("got %d, want 100000", got)
	}
	if got := UsableContextBudget(1_000_000, 0, 256_000, 0); got != 256_000 {
		t.Errorf("got %d, want 256000", got)
	}
	// Full trust: the whole window counts.
	if got := UsableContextBudget(1_000_000, 0, 256_000, 1.0); got != 1_000_000 {
		t.Errorf("got %d, want 1000000", got)
	}
}

func TestCumulativeDiffBudget(t *testing.T) {
	cases := []struct{ in, want int }{
		{0, 2_000},          // floor
		{3_000, 2_000},      // 1000 -> floor
		{6_000, 2_000},      // exactly the floor
		{6_003, 2_001},      // just above
		{112_000, 37_333},   // truncating division
		{240_000, 80_000},   // exactly the ceiling
		{628_000, 80_000},   // clamped
		{5_128_000, 80_000}, // clamped hard
	}
	for _, tc := range cases {
		if got := CumulativeDiffBudget(tc.in); got != tc.want {
			t.Errorf("CumulativeDiffBudget(%d) = %d, want %d", tc.in, got, tc.want)
		}
	}
}

func TestPreviouslyPostedBudget(t *testing.T) {
	cases := []struct{ in, want int }{
		{0, 500},         // floor
		{5_000, 500},     // 250 -> floor
		{10_000, 500},    // exactly the floor
		{10_020, 501},    // just above
		{112_000, 4_000}, // clamped
		{80_000, 4_000},  // exactly the ceiling
		{628_000, 4_000}, // clamped
	}
	for _, tc := range cases {
		if got := PreviouslyPostedBudget(tc.in); got != tc.want {
			t.Errorf("PreviouslyPostedBudget(%d) = %d, want %d", tc.in, got, tc.want)
		}
	}
}

// Both budgets use min(hi, max(lo, v)), so the upper bound wins when the
// bounds conflict.
func TestClampUpperBoundWins(t *testing.T) {
	if got := clamp(5, 100, 10); got != 10 {
		t.Errorf("clamp(5, 100, 10) = %d, want 10 (min applied last)", got)
	}
}
