package diff

import (
	"regexp"
	"strings"
	"testing"
)

func refPattern(symbol string) *regexp.Regexp {
	return regexp.MustCompile(`\b` + regexp.QuoteMeta(symbol) + `\b`)
}

// Ported from Python TestFindReferences, which asserts exact counts and line
// numbers.
func TestFindReferencesPythonCases(t *testing.T) {
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

// Ported from Python TestExtractChangedSymbols, the exact per-language cases.
func TestExtractChangedSymbolsPythonCases(t *testing.T) {
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

// Ported from Python TestBuildCrossFileContext.
func TestBuildRelationshipsPythonCases(t *testing.T) {
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

// Ported from Python TestRenderCrossFileContext.
func TestRenderRelationshipsPythonCases(t *testing.T) {
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
