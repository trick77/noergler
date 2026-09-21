package diff

import (
	"regexp"
	"strings"
	"testing"
)

func TestExtractChangedSymbols(t *testing.T) {
	cases := []struct {
		name string
		file FileReviewData
		want []string
	}{
		{
			"python def",
			FileReviewData{Path: "m.py", Diff: "@@ -1,1 +1,1 @@\n+def process_order(x):\n"},
			[]string{"process_order"},
		},
		{
			"python class",
			FileReviewData{Path: "m.py", Diff: "+class OrderService:\n"},
			[]string{"OrderService"},
		},
		{
			"typescript function",
			FileReviewData{Path: "a.ts", Diff: "+export function runOrder(a) {\n"},
			[]string{"runOrder"},
		},
		{
			// The jvm pattern has no method alternative, so a plain method
			// yields nothing. Intentional, pinned here.
			"jvm plain method yields nothing",
			FileReviewData{Path: "A.java", Diff: "+    public void processOrder(int a) {\n"},
			nil,
		},
		{
			"jvm class matches",
			FileReviewData{Path: "A.java", Diff: "+public class OrderService {\n"},
			[]string{"OrderService"},
		},
		{
			"short names below the minimum are dropped",
			FileReviewData{Path: "m.py", Diff: "+def ab():\n"},
			nil,
		},
		{
			"removed lines ignored",
			FileReviewData{Path: "m.py", Diff: "-def process_order(x):\n"},
			nil,
		},
		{
			"+++ header ignored",
			FileReviewData{Path: "m.py", Diff: "+++ b/m.py\n"},
			nil,
		},
		{
			"duplicates collapse, order preserved",
			FileReviewData{Path: "m.py", Diff: "+def alpha():\n+def beta():\n+def alpha():\n"},
			[]string{"alpha", "beta"},
		},
		{
			// Cross-file extraction ignores the no-dynamic set, so a docs file
			// still yields symbols.
			"markdown still extracted",
			FileReviewData{Path: "notes.md", Diff: "+def process_order(x):\n"},
			[]string{"process_order"},
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got := extractChangedSymbols(tc.file)
			if strings.Join(got, ",") != strings.Join(tc.want, ",") {
				t.Errorf("got %q, want %q", got, tc.want)
			}
		})
	}
}

func TestBuildRelationships(t *testing.T) {
	t.Run("fewer than two files yields nothing", func(t *testing.T) {
		got := BuildRelationships([]FileReviewData{
			{Path: "m.py", Diff: "+def process_order(x):\n"},
		})
		if got != nil {
			t.Errorf("got %+v, want nil", got)
		}
	})

	t.Run("finds a reference in another file", func(t *testing.T) {
		files := []FileReviewData{
			{Path: "svc.py", Diff: "+def process_order(x):\n"},
			{Path: "caller.py", Content: "import svc\n\nsvc.process_order(1)\n"},
		}
		got := BuildRelationships(files)
		if len(got) != 1 {
			t.Fatalf("got %d relationships, want 1", len(got))
		}
		if got[0].Symbol != "process_order" || got[0].DefinedIn != "svc.py" {
			t.Errorf("relationship = %+v", got[0])
		}
		if len(got[0].References) != 1 {
			t.Fatalf("got %d references, want 1", len(got[0].References))
		}
		ref := got[0].References[0]
		if ref.File != "caller.py" || ref.LineNumber != 3 {
			t.Errorf("reference = %+v", ref)
		}
		if ref.FromDiff {
			t.Error("reference came from content, should not be marked FromDiff")
		}
	})

	t.Run("word boundary prevents partial matches", func(t *testing.T) {
		files := []FileReviewData{
			{Path: "svc.py", Diff: "+def process_order(x):\n"},
			{Path: "caller.py", Content: "process_ordering(1)\n"},
		}
		if got := BuildRelationships(files); len(got) != 0 {
			t.Errorf("got %+v, want no relationships", got)
		}
	})

	t.Run("comment lines are skipped", func(t *testing.T) {
		files := []FileReviewData{
			{Path: "svc.py", Diff: "+def process_order(x):\n"},
			{Path: "caller.py", Content: "# process_order is nice\n// process_order\n * process_order\n"},
		}
		if got := BuildRelationships(files); len(got) != 0 {
			t.Errorf("got %+v, want no relationships", got)
		}
	})

	// Empty content falls through to the diff, and the reference is labelled
	// as a diff line.
	t.Run("empty content falls through to the diff", func(t *testing.T) {
		files := []FileReviewData{
			{Path: "svc.py", Diff: "+def process_order(x):\n"},
			{Path: "caller.py", Diff: "@@ -1,1 +1,1 @@\n+process_order(1)\n", Content: ""},
		}
		got := BuildRelationships(files)
		if len(got) != 1 || len(got[0].References) != 1 {
			t.Fatalf("got %+v", got)
		}
		if !got[0].References[0].FromDiff {
			t.Error("reference from a diff should be marked FromDiff")
		}
	})

	// The cap is per target file, so one symbol across several files can carry
	// more than maxRefsPerSymbol references in total.
	t.Run("reference cap is per target file", func(t *testing.T) {
		manyRefs := strings.Repeat("process_order(1)\n", 10)
		files := []FileReviewData{
			{Path: "svc.py", Diff: "+def process_order(x):\n"},
			{Path: "a.py", Content: manyRefs},
			{Path: "b.py", Content: manyRefs},
		}
		got := BuildRelationships(files)
		if len(got) != 1 {
			t.Fatalf("got %d relationships", len(got))
		}
		if n := len(got[0].References); n != 2*maxRefsPerSymbol {
			t.Errorf("got %d references, want %d (5 per target file)", n, 2*maxRefsPerSymbol)
		}
	})
}

func TestRenderRelationships(t *testing.T) {
	t.Run("empty renders empty", func(t *testing.T) {
		if got := RenderRelationships(nil); got != "" {
			t.Errorf("got %q, want empty", got)
		}
	})

	t.Run("renders a section", func(t *testing.T) {
		rels := []CrossFileRelationship{{
			Symbol:    "process_order",
			DefinedIn: "svc.py",
			References: []SymbolReference{
				{File: "caller.py", LineNumber: 3, LineText: "svc.process_order(1)"},
			},
		}}
		got := RenderRelationships(rels)
		for _, want := range []string{
			"## Cross-file relationships",
			"**`process_order`** (changed in `svc.py`) is referenced in:",
			"- `caller.py:3` — `svc.process_order(1)`",
		} {
			if !strings.Contains(got, want) {
				t.Errorf("missing %q in:\n%s", want, got)
			}
		}
		if strings.Contains(got, "truncated") {
			t.Error("should not be truncated")
		}
	})

	// A diff-sourced reference is labelled as a diff line rather than passed off
	// as a file line.
	t.Run("diff references are labelled", func(t *testing.T) {
		rels := []CrossFileRelationship{{
			Symbol:    "process_order",
			DefinedIn: "svc.py",
			References: []SymbolReference{
				{File: "caller.py", LineNumber: 2, LineText: "+process_order(1)", FromDiff: true},
			},
		}}
		got := RenderRelationships(rels)
		if !strings.Contains(got, "caller.py (diff line 2)") {
			t.Errorf("diff reference not labelled:\n%s", got)
		}
	})

	// The 30-line cap counts header and reference lines but not the blank
	// separators, and truncation ends with the exact marker and no newline.
	t.Run("truncates at the line cap", func(t *testing.T) {
		var rels []CrossFileRelationship
		for i := 0; i < 50; i++ {
			rels = append(rels, CrossFileRelationship{
				Symbol:    "sym" + itoa(i),
				DefinedIn: "svc.py",
				References: []SymbolReference{
					{File: "caller.py", LineNumber: i, LineText: "x"},
				},
			})
		}
		got := RenderRelationships(rels)
		if !strings.HasSuffix(got, "_(additional relationships truncated)_") {
			t.Errorf("want the exact truncation marker at the end, got:\n%s", got[maxInt(0, len(got)-120):])
		}
		if strings.HasSuffix(got, "\n") {
			t.Error("a truncated render must not end with a newline")
		}
		counted := 0
		for _, l := range strings.Split(got, "\n") {
			if strings.HasPrefix(l, "**`") || strings.HasPrefix(l, "- `") {
				counted++
			}
		}
		if counted != maxRelationshipLines {
			t.Errorf("counted %d capped lines, want %d", counted, maxRelationshipLines)
		}
	})
}

func maxInt(a, b int) int {
	if a > b {
		return a
	}
	return b
}

func refPattern(symbol string) *regexp.Regexp {
	return regexp.MustCompile(`\b` + regexp.QuoteMeta(symbol) + `\b`)
}

// Asserts exact counts and line numbers.
func TestFindReferencesCases(t *testing.T) {
	t.Run("finds references in content", func(t *testing.T) {
		target := FileReviewData{
			Path: "controller.py",
			Content: "from service import get_user\n\ndef handler():\n" +
				"    user = get_user(request.user_id)\n    return user\n",
		}
		refs := findReferences(refPattern("get_user"), target)
		// The import line counts as a reference.
		if len(refs) != 2 {
			t.Fatalf("got %d references, want 2", len(refs))
		}
		if refs[0].LineNumber != 1 {
			t.Errorf("refs[0].LineNumber = %d, want 1", refs[0].LineNumber)
		}
		if refs[1].LineNumber != 4 {
			t.Errorf("refs[1].LineNumber = %d, want 4", refs[1].LineNumber)
		}
	})

	t.Run("no partial match", func(t *testing.T) {
		target := FileReviewData{Path: "controller.py", Content: "def get_user_name():\n    pass\n"}
		if refs := findReferences(refPattern("get_user"), target); len(refs) != 0 {
			t.Errorf("got %d references, want 0", len(refs))
		}
	})

	t.Run("falls back to the diff", func(t *testing.T) {
		target := FileReviewData{Path: "controller.py", Diff: "+    result = get_user(1)\n", Content: ""}
		refs := findReferences(refPattern("get_user"), target)
		if len(refs) != 1 {
			t.Fatalf("got %d references, want 1", len(refs))
		}
		if !refs[0].FromDiff {
			t.Error("a reference found in the diff should be marked FromDiff")
		}
	})

	t.Run("skips comments", func(t *testing.T) {
		target := FileReviewData{
			Path:    "controller.py",
			Content: "# get_user is deprecated\ndef handler():\n    get_user(1)\n",
		}
		refs := findReferences(refPattern("get_user"), target)
		if len(refs) != 1 {
			t.Fatalf("got %d references, want 1", len(refs))
		}
		if refs[0].LineNumber != 3 {
			t.Errorf("LineNumber = %d, want 3", refs[0].LineNumber)
		}
	})

	t.Run("max refs limit", func(t *testing.T) {
		var lines []string
		for i := 0; i < 20; i++ {
			lines = append(lines, "    get_user("+itoa(i)+")")
		}
		target := FileReviewData{Path: "bulk.py", Content: strings.Join(lines, "\n")}
		if refs := findReferences(refPattern("get_user"), target); len(refs) != maxRefsPerSymbol {
			t.Errorf("got %d references, want %d", len(refs), maxRefsPerSymbol)
		}
	})
}

// The exact per-language cases.
func TestExtractChangedSymbolsCases(t *testing.T) {
	cases := []struct {
		name string
		path string
		diff string
		want string
	}{
		{"python function", "service.py", "+def get_user(id):\n", "get_user"},
		{"python async def", "service.py", "+async def fetch_user(id):\n", "fetch_user"},
		{"python class", "models.py", "+class UserModel:\n", "UserModel"},
		{"typescript function", "app.ts", "+export function getUser(id) {\n", "getUser"},
		{"typescript class", "app.ts", "+export class UserService {\n", "UserService"},
		{"java class", "User.java", "+public class UserService {\n", "UserService"},
		{"kotlin fun", "User.kt", "+fun fetchUser(id: Int) {\n", "fetchUser"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got := extractChangedSymbols(FileReviewData{Path: tc.path, Diff: tc.diff})
			if len(got) != 1 || got[0] != tc.want {
				t.Errorf("got %q, want [%s]", got, tc.want)
			}
		})
	}

	t.Run("ignores short names", func(t *testing.T) {
		if got := extractChangedSymbols(FileReviewData{Path: "m.py", Diff: "+def ab():\n"}); len(got) != 0 {
			t.Errorf("got %q, want none", got)
		}
	})

	t.Run("ignores context lines", func(t *testing.T) {
		if got := extractChangedSymbols(FileReviewData{Path: "m.py", Diff: " def get_user():\n"}); len(got) != 0 {
			t.Errorf("got %q, want none", got)
		}
	})

	t.Run("multiple symbols", func(t *testing.T) {
		got := extractChangedSymbols(FileReviewData{Path: "m.py", Diff: "+def alpha():\n+def beta():\n"})
		if len(got) != 2 || got[0] != "alpha" || got[1] != "beta" {
			t.Errorf("got %q, want [alpha beta]", got)
		}
	})

	t.Run("no duplicates", func(t *testing.T) {
		got := extractChangedSymbols(FileReviewData{Path: "m.py", Diff: "+def alpha():\n+def alpha():\n"})
		if len(got) != 1 || got[0] != "alpha" {
			t.Errorf("got %q, want [alpha]", got)
		}
	})
}

func TestBuildRelationshipsCases(t *testing.T) {
	t.Run("single file returns empty", func(t *testing.T) {
		got := BuildRelationships([]FileReviewData{{Path: "service.py", Diff: "+def get_user(id):\n"}})
		if len(got) != 0 {
			t.Errorf("got %+v, want none", got)
		}
	})

	t.Run("finds a cross-file relationship", func(t *testing.T) {
		files := []FileReviewData{
			{Path: "service.py", Diff: "+def get_user(id):\n"},
			{Path: "controller.py", Content: "user = get_user(1)\n"},
		}
		got := BuildRelationships(files)
		if len(got) != 1 {
			t.Fatalf("got %d relationships, want 1", len(got))
		}
		if got[0].Symbol != "get_user" || got[0].DefinedIn != "service.py" {
			t.Errorf("relationship = %+v", got[0])
		}
	})

	t.Run("no relationships when symbols are not referenced", func(t *testing.T) {
		files := []FileReviewData{
			{Path: "service.py", Diff: "+def get_user(id):\n"},
			{Path: "other.py", Content: "x = 1\n"},
		}
		if got := BuildRelationships(files); len(got) != 0 {
			t.Errorf("got %+v, want none", got)
		}
	})

	t.Run("no symbols extracted", func(t *testing.T) {
		files := []FileReviewData{
			{Path: "a.py", Content: "x = 1\n"},
			{Path: "b.py", Content: "y = 2\n"},
		}
		if got := BuildRelationships(files); len(got) != 0 {
			t.Errorf("got %+v, want none", got)
		}
	})
}

func TestRenderRelationshipsCases(t *testing.T) {
	t.Run("empty relationships", func(t *testing.T) {
		if got := RenderRelationships(nil); got != "" {
			t.Errorf("got %q, want empty", got)
		}
	})

	t.Run("renders relationships", func(t *testing.T) {
		rels := []CrossFileRelationship{{
			Symbol:    "get_user",
			DefinedIn: "service.py",
			References: []SymbolReference{
				{File: "controller.py", LineNumber: 4, LineText: "user = get_user(request.id)"},
			},
		}}
		got := RenderRelationships(rels)
		for _, want := range []string{
			"## Cross-file relationships", "`get_user`", "`service.py`", "`controller.py:4`",
		} {
			if !strings.Contains(got, want) {
				t.Errorf("missing %q in:\n%s", want, got)
			}
		}
	})

	t.Run("truncation", func(t *testing.T) {
		var refs []SymbolReference
		for i := 0; i < 50; i++ {
			refs = append(refs, SymbolReference{
				File: "file" + itoa(i) + ".py", LineNumber: i, LineText: "call_" + itoa(i) + "()",
			})
		}
		rels := []CrossFileRelationship{{Symbol: "big_func", DefinedIn: "source.py", References: refs}}
		if got := RenderRelationships(rels); !strings.Contains(got, "truncated") {
			t.Errorf("want a truncation marker in:\n%s", got)
		}
	})
}
