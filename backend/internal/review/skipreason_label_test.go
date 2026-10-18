package review

import (
	"os"
	"regexp"
	"testing"
)

var allSkipReasons = []SkipReason{
	SkipNotAutoAuthor, SkipIgnoredAuthor, SkipIgnoredPR, SkipBranchOptOut,
	SkipNoAgentsMD, SkipAgentsMDTooLarge, SkipCostCap, SkipNoReviewable,
	SkipEmptyDiff, SkipHeadUnchanged, SkipDiffTooLarge, SkipRepoNotClaimed,
}

// Every reason the pipeline can write must have a label: the dashboard
// renders Label(), so a missing case shows an operator a bare enum value.
func TestEverySkipReasonHasALabel(t *testing.T) {
	seen := map[string]SkipReason{}
	for _, r := range allSkipReasons {
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

// The dashboard labels a known code from its own catalog and uses Label()
// only for codes it does not know, so the English catalog repeats Label()
// word for word. Reworded here and not there, the dashboard keeps the old
// text for every known code and nothing else notices.
func TestSkipReasonLabelsMatchTheUICatalog(t *testing.T) {
	src, err := os.ReadFile("../../../ui/src/i18n/en.ts")
	if err != nil {
		t.Fatal(err)
	}
	block := regexp.MustCompile(`(?s)\n  reason: \{\n(.*?)\n  \},`).FindSubmatch(src)
	if block == nil {
		t.Fatal("no reason block in ui/src/i18n/en.ts")
	}
	catalog := map[string]string{}
	for _, m := range regexp.MustCompile(`(?m)^\s+(\w+):\s*"([^"]*)",`).FindAllSubmatch(block[1], -1) {
		catalog[string(m[1])] = string(m[2])
	}
	for _, r := range allSkipReasons {
		if got, want := catalog[string(r)], r.Label(); got != want {
			t.Errorf("en.ts reason.%s = %q, want Label() %q", r, got, want)
		}
	}
	if len(catalog) != len(allSkipReasons) {
		t.Errorf("en.ts has %d reasons, the pipeline %d", len(catalog), len(allSkipReasons))
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
