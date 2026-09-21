package review

import "testing"

// Every reason the pipeline can write must have a label: the dashboard
// renders Label(), so a missing case shows an operator a bare enum value.
func TestEverySkipReasonHasALabel(t *testing.T) {
	all := []SkipReason{
		SkipNotAutoAuthor, SkipIgnoredAuthor, SkipIgnoredPR, SkipBranchOptOut,
		SkipNoAgentsMD, SkipAgentsMDTooLarge, SkipCostCap, SkipNoReviewable,
		SkipEmptyDiff, SkipHeadUnchanged, SkipDiffTooLarge,
	}
	seen := map[string]SkipReason{}
	for _, r := range all {
		label := r.Label()
		if label == "" {
			t.Errorf("%q has no label", r)
		}
		if label == string(r) {
			t.Errorf("%q falls through to its own value instead of a label", r)
		}
		if prev, dup := seen[label]; dup {
			t.Errorf("%q and %q share the label %q", prev, r, label)
		}
		seen[label] = r
	}
}

// A row written by a newer binary must never render blank: an unknown value
// prints itself, which is ugly but readable, where "" is a lie.
func TestUnknownSkipReasonPrintsItself(t *testing.T) {
	if got := SkipReason("invented_later").Label(); got != "invented_later" {
		t.Errorf("Label() = %q, want the raw value", got)
	}
	if got := SkipNone.Label(); got != "" {
		t.Errorf("SkipNone.Label() = %q, want empty", got)
	}
}
