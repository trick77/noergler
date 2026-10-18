package review

import (
	"errors"
	"go/ast"
	"go/parser"
	"go/token"
	"io/fs"
	"os"
	"regexp"
	"slices"
	"strconv"
	"testing"
)

var allSkipReasons = []SkipReason{
	SkipNotAutoAuthor, SkipIgnoredAuthor, SkipIgnoredPR, SkipBranchOptOut,
	SkipNoAgentsMD, SkipAgentsMDTooLarge, SkipCostCap, SkipNoReviewable,
	SkipEmptyDiff, SkipHeadUnchanged, SkipDiffTooLarge, SkipRepoNotClaimed,
}

// allSkipReasons is a hand-kept list the label tests below lean on. This
// reads the const block itself, so a reason added there and forgotten here
// fails instead of escaping every test that ranges over the list.
func TestAllSkipReasonsIsComplete(t *testing.T) {
	f, err := parser.ParseFile(token.NewFileSet(), "skipreason.go", nil, 0)
	if err != nil {
		t.Fatal(err)
	}
	var declared []SkipReason
	ast.Inspect(f, func(n ast.Node) bool {
		vs, ok := n.(*ast.ValueSpec)
		if !ok || len(vs.Values) != 1 {
			return true
		}
		if id, ok := vs.Type.(*ast.Ident); !ok || id.Name != "SkipReason" {
			return true
		}
		lit, ok := vs.Values[0].(*ast.BasicLit)
		if !ok {
			return true
		}
		v, err := strconv.Unquote(lit.Value)
		if err == nil && v != "" {
			declared = append(declared, SkipReason(v))
		}
		return true
	})
	got := slices.Clone(allSkipReasons)
	slices.Sort(got)
	slices.Sort(declared)
	if !slices.Equal(got, declared) {
		t.Errorf("allSkipReasons = %v, skipreason.go declares %v", got, declared)
	}
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
	if errors.Is(err, fs.ErrNotExist) {
		// A backend-only tree (the Containerfile's Go stage) has no UI to
		// compare against; the repo checkout ship check runs on always does.
		t.Skip("ui/ not present next to backend/")
	}
	if err != nil {
		t.Fatal(err)
	}
	block := regexp.MustCompile(`(?s)\n  reason: \{\n(.*?)\n  \},`).FindSubmatch(src)
	if block == nil {
		t.Fatal("no reason block in ui/src/i18n/en.ts")
	}
	// A double-quoted value, escapes allowed, a trailing comment tolerated.
	entry := regexp.MustCompile(`(?m)^\s+(\w+):\s*("(?:[^"\\]|\\.)*"),?\s*(?://.*)?$`)
	catalog := map[string]string{}
	for _, m := range entry.FindAllSubmatch(block[1], -1) {
		v, err := strconv.Unquote(string(m[2]))
		if err != nil {
			t.Fatalf("en.ts reason.%s: %v", m[1], err)
		}
		catalog[string(m[1])] = v
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
