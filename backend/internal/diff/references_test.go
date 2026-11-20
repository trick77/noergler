package diff

import (
	"slices"
	"strings"
	"testing"
)

// removedFile turns a diff's added lines into removed ones: a removed
// definition is searched, an added-only one never.
func removedFile(path, d string) FileReviewData {
	return FileReviewData{Path: path, Diff: strings.ReplaceAll(d, "\n+", "\n-")}
}

// addedNames are the names a diff's added lines define, unfiltered.
func addedNames(f FileReviewData) []string {
	added, _ := definedNames(f)
	return added
}

// Searchable, long enough, first seen first, capped: a short or
// punctuated name never reaches the search.
func TestSearchSymbols(t *testing.T) {
	files := []FileReviewData{
		removedFile("a.go", "@@ -1,4 +1,0 @@\n+func Touch(ctx context.Context, ttlMillis int64) error {\n+func Get() {}\n+func (s *Store) Save(x int) {}\n+func Touch2() {}"),
		removedFile("b.go", "@@ -1,1 +1,0 @@\n+func Touch(x int) {}"),
	}
	if got, want := SearchSymbols(files), []string{"Touch", "Save", "Touch2"}; !slices.Equal(got, want) {
		t.Errorf("symbols = %v, want %v", got, want)
	}
	// JVM: a method needs a modifier, a call statement is no definition.
	jvm := FileReviewData{Path: "A.java", Diff: "@@ -1,0 +1,4 @@\n+    public Order process(Order o) {\n" +
		"+    order.process(x);\n+    @Override protected void handleEvent(Event e) {\n+public class Ledger {"}
	if got, want := addedNames(jvm), []string{"process", "handleEvent", "Ledger"}; !slices.Equal(got, want) {
		t.Errorf("jvm names = %v, want %v", got, want)
	}
	var many strings.Builder
	many.WriteString("@@ -1,9 +1,0 @@")
	for _, n := range []string{"Alpha", "Bravo", "Charlie", "Delta", "Echo", "Foxtrot", "Golf", "Hotel", "India"} {
		many.WriteString("\n+func " + n + "() {}")
	}
	if got := SearchSymbols([]FileReviewData{removedFile("c.go", many.String())}); len(got) != maxSearchSymbols {
		t.Errorf("symbols = %v, want the cap %d", got, maxSearchSymbols)
	}
}

// One query, scoped to the repo, within the search's length limit.
func TestSearchQuery(t *testing.T) {
	q, used := SearchQuery([]string{"Touch", "Save"}, "PROJ", "app")
	if q != "(Touch OR Save) project:PROJ repo:app" || len(used) != 2 {
		t.Errorf("query = %q, used %v", q, used)
	}
	if q, _ := SearchQuery([]string{"Touch"}, "PROJ", "app"); q != "Touch project:PROJ repo:app" {
		t.Errorf("one symbol needs no parentheses: %q", q)
	}
	long := strings.Repeat("x", 120)
	q, used = SearchQuery([]string{long + "a", long + "b", "Touch"}, "PROJ", "app")
	if len(q) > maxSearchQueryLen || len(used) != 1 {
		t.Errorf("over the limit: %d chars, used %v", len(q), used)
	}
}

// Whole word, case as written, outside the PR's files, no comment line,
// capped per symbol, ordered by symbol then path and line.
func TestMatchReferences(t *testing.T) {
	hits := []SearchHit{
		{File: "z/manager.go", Lines: []SearchLine{
			{41, "\ts.Touch(ctx, key, int64(d.Seconds()))"},
			{42, "\t// Touch keeps it alive"},
			{43, "\ts.touch(ctx)"},
			{44, "\ts.TouchAll(ctx)"},
		}},
		{File: "a/handler.go", Lines: []SearchLine{{7, "store.Touch(ctx, k, 30)"}, {9, "store.Save(ctx, x)"}}},
		{File: "pr/changed.go", Lines: []SearchLine{{1, "Touch(ctx)"}}},
	}
	got := MatchReferences([]string{"Touch", "Save"}, hits, map[string]bool{"pr/changed.go": true})
	want := []UnchangedRef{
		{"Touch", "a/handler.go", 7, "store.Touch(ctx, k, 30)"},
		{"Touch", "z/manager.go", 41, "s.Touch(ctx, key, int64(d.Seconds()))"},
		{"Save", "a/handler.go", 9, "store.Save(ctx, x)"},
	}
	if !slices.Equal(got, want) {
		t.Errorf("refs = %+v\nwant %+v", got, want)
	}

	var lines []SearchLine
	for i := range 9 {
		lines = append(lines, SearchLine{i + 1, "Touch()"})
	}
	if got := MatchReferences([]string{"Touch"}, []SearchHit{{File: "x.go", Lines: lines}}, nil); len(got) != maxRefsPerSearchSymbol {
		t.Errorf("refs = %d, want the per-symbol cap %d", len(got), maxRefsPerSearchSymbol)
	}
	block := RenderUnchangedReferences(want)
	if !strings.Contains(block, "## References in unchanged files") || !strings.Contains(block, "`z/manager.go:41`") ||
		RenderUnchangedReferences(nil) != "" {
		t.Errorf("block = %q", block)
	}
}

// A changed signature ranks first, then a removed or renamed name (its
// callers break); a new name is never searched (the default branch has no
// caller of it), nor is one every type defines.
func TestSearchSymbolsRanksWhatBreaksCallers(t *testing.T) {
	files := []FileReviewData{{Path: "s.go", Diff: "@@ -1,3 +1,3 @@\n" +
		"+func NewHelper() {}\n" +
		"-func (s *Store) Touch(key string, secs int64) {}\n" +
		"+func (s *Store) Touch(key string, millis int64) {}\n" +
		"-func Refresh(id string) {}\n" +
		"+func Renewed(id string) {}\n" +
		"-func (t *Token) String() string {}"}}
	if got, want := SearchSymbols(files), []string{"Touch", "Refresh"}; !slices.Equal(got, want) {
		t.Errorf("symbols = %v, want %v", got, want)
	}
}

// Kotlin's keyword prefixes and extension receivers yield the name, never
// the keyword or the receiver type; a TypeScript local is no definition, a
// class method with a modifier is, an anonymous default class names
// nothing. A no-break space separates like a space.
func TestDefinedNamesReadsTheName(t *testing.T) {
	kt := FileReviewData{Path: "A.kt", Diff: "@@ -0,0 +1,4 @@\n+enum class Status { A }\n" +
		"+fun interface Mapper { fun map(x: Int): Int }\n+fun String.toSlug(): String = this\n" +
		"+private fun <T> List<T>.second(): T = this[1]"}
	if got, want := addedNames(kt), []string{"Status", "Mapper", "toSlug", "second"}; !slices.Equal(got, want) {
		t.Errorf("kotlin names = %v, want %v", got, want)
	}
	ts := FileReviewData{Path: "s.ts", Diff: "@@ -0,0 +1,6 @@\n+export const enum Direction { Up }\n" +
		"+  const response = await fetch(url);\n+  public async saveOrder(order: Order) {\n" +
		"+  this.saveOrder(o);\n+export function fetchUser(id: string) {\n+export default class extends Base {"}
	if got, want := addedNames(ts), []string{"Direction", "saveOrder", "fetchUser"}; !slices.Equal(got, want) {
		t.Errorf("typescript names = %v, want %v", got, want)
	}
	nbsp := FileReviewData{Path: "n.go", Diff: "@@ -0,0 +1 @@\n+func Touch(x int) {}"}
	if got, want := addedNames(nbsp), []string{"Touch"}; !slices.Equal(got, want) {
		t.Errorf("no-break space names = %v, want %v", got, want)
	}
}

// Docs and config only mention a name; a production caller comes before a
// test, so the per-name cap never cuts it for tests that sort first.
func TestMatchReferencesPrefersCode(t *testing.T) {
	hits := []SearchHit{
		{File: "CHANGELOG.md", Lines: []SearchLine{{1, "Touch now takes millis"}}},
		{File: "a_test.go", Lines: []SearchLine{{1, "s.Touch(k, 1)"}, {2, "s.Touch(k, 2)"}, {3, "s.Touch(k, 3)"},
			{4, "s.Touch(k, 4)"}, {5, "s.Touch(k, 5)"}}},
		{File: "z/manager.go", Lines: []SearchLine{{9, "s.Touch(k, secs)"}}},
	}
	got := MatchReferences([]string{"Touch"}, hits, nil)
	if len(got) != maxRefsPerSearchSymbol || got[0].File != "z/manager.go" {
		t.Errorf("refs = %+v, want z/manager.go first, no CHANGELOG.md", got)
	}
}

// A renamed file's and a deleted file's old paths are the PR's; an edited
// file's own path is not repeated.
func TestSourcePaths(t *testing.T) {
	d := "diff --git a/store.go b/session/store.go\nsimilarity index 90%\nrename from store.go\nrename to session/store.go\n" +
		"--- a/store.go\n+++ b/session/store.go\n@@ -1 +1 @@\n-a\n+b\n" +
		"diff --git src://old.go dst://old.go\n--- src://old.go\n+++ /dev/null\n@@ -1 +0,0 @@\n-x\n" +
		"diff --git a/kept.go b/kept.go\n--- a/kept.go\n+++ b/kept.go\n@@ -1 +1 @@\n-c\n+d\n" +
		"diff --git a/new.go b/new.go\n--- /dev/null\n+++ b/new.go\n@@ -0,0 +1 @@\n+e\n"
	if got, want := SourcePaths(d), []string{"store.go"}; !slices.Equal(got, want) {
		t.Errorf("source paths = %v, want %v (a deleted file's path is its own)", got, want)
	}
}

// A pointer deref is code, a block comment is not; a backtick in the line
// never breaks the rendered span; an overlong line is left out, never cut.
func TestMatchReferencesLineRules(t *testing.T) {
	hits := []SearchHit{{File: "m.go", Lines: []SearchLine{
		{1, "\t*resp = s.Touch(ctx, k, 30)"},
		{2, "/* Touch is old */"},
		{3, "q := db.Query(`x`, Touch(y))"},
		{4, "Touch(" + strings.Repeat("a", 250) + ")"},
	}}}
	got := MatchReferences([]string{"Touch"}, hits, nil)
	if len(got) != 2 || got[0].Line != 1 || got[1].Line != 3 {
		t.Fatalf("refs = %+v, want lines 1 and 3", got)
	}
	if block := RenderUnchangedReferences(got); !strings.Contains(block, "``q := db.Query(`x`, Touch(y))``") {
		t.Errorf("backtick span = %q", block)
	}
}
