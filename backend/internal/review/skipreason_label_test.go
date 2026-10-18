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
	"strings"
	"testing"
)

var allSkipReasons = []SkipReason{
	SkipNotAutoAuthor, SkipIgnoredAuthor, SkipIgnoredPR, SkipBranchOptOut,
	SkipNoAgentsMD, SkipAgentsMDTooLarge, SkipCostCap, SkipNoReviewable,
	SkipEmptyDiff, SkipHeadUnchanged, SkipDiffTooLarge, SkipRepoNotClaimed,
}

// allSkipReasons is a hand-kept list the label tests below lean on. This
// reads the package's own declarations, so a reason added there and
// forgotten here fails instead of escaping every test that ranges over the
// list. File-scope typed constants only (`X SkipReason = "x"`, grouped or
// not), which is how every reason is declared; a value two names share
// counts once.
func TestAllSkipReasonsIsComplete(t *testing.T) {
	entries, err := os.ReadDir(".")
	if err != nil {
		t.Fatal(err)
	}
	fset := token.NewFileSet()
	var declared []SkipReason
	for _, e := range entries {
		name := e.Name()
		if !strings.HasSuffix(name, ".go") || strings.HasSuffix(name, "_test.go") {
			continue
		}
		f, err := parser.ParseFile(fset, name, nil, 0)
		if err != nil {
			t.Fatal(err)
		}
		for _, decl := range f.Decls {
			gd, ok := decl.(*ast.GenDecl)
			if !ok || gd.Tok != token.CONST {
				continue
			}
			for _, spec := range gd.Specs {
				vs := spec.(*ast.ValueSpec)
				if id, ok := vs.Type.(*ast.Ident); !ok || id.Name != "SkipReason" {
					continue
				}
				for _, val := range vs.Values {
					lit, ok := val.(*ast.BasicLit)
					if !ok {
						continue
					}
					if v, err := strconv.Unquote(lit.Value); err == nil && v != "" {
						declared = append(declared, SkipReason(v))
					}
				}
			}
		}
	}
	got := slices.Clone(allSkipReasons)
	slices.Sort(got)
	slices.Sort(declared)
	declared = slices.Compact(declared)
	if !slices.Equal(got, declared) {
		t.Errorf("allSkipReasons = %v, the package declares %v", got, declared)
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
	// Skipped only when there is no ui/ at all (a backend-only copy of the
	// module). With ui/ present a missing catalog fails: a renamed or moved
	// en.ts must not turn this check into a silent skip.
	if _, err := os.Stat("../../../ui"); errors.Is(err, fs.ErrNotExist) {
		t.Skip("no ui/ next to backend/")
	}
	src, err := os.ReadFile("../../../ui/src/i18n/en.ts")
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
