package diff

import "testing"

// The gutter is model-facing text: pinned byte for byte.
func TestNumberDiffIsPinned(t *testing.T) {
	cases := []struct{ name, in, want string }{
		{
			"added, context and removed lines",
			"@@ -8,3 +9,4 @@\n ctx\n-old\n+new\n+more\n tail",
			"@@ -8,3 +9,4 @@\n 9  ctx\n   -old\n10 +new\n11 +more\n12  tail",
		},
		{
			// A header before the first hunk passes through; so does the
			// trailing newline, which must not grow a gutter.
			"file header and trailing newline",
			"--- a/x.go\n+++ b/x.go\n@@ -1 +1 @@\n+x\n",
			"--- a/x.go\n+++ b/x.go\n@@ -1 +1 @@\n1 +x\n",
		},
		{
			// Not a new-file line: blank gutter, no advance.
			"no-newline marker",
			"@@ -1 +1,2 @@\n+a\n\\ No newline at end of file\n+b",
			"@@ -1 +1,2 @@\n1 +a\n  \\ No newline at end of file\n2 +b",
		},
		{
			// Width comes from the widest line any hunk reaches, so a
			// later hunk's numbers line up with an earlier one's.
			"two hunks share one width",
			"@@ -1 +1 @@\n+a\n@@ -99,1 +99,2 @@\n x\n+y",
			"@@ -1 +1 @@\n  1 +a\n@@ -99,1 +99,2 @@\n 99  x\n100 +y",
		},
		{
			// A context line whose single leading space was stripped.
			"empty context line",
			"@@ -1,3 +1,3 @@\n+a\n\n+b",
			"@@ -1,3 +1,3 @@\n1 +a\n2 \n3 +b",
		},
		{
			"no hunks",
			"+++ /dev/null\n-x",
			"+++ /dev/null\n-x",
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if got := NumberDiff(tc.in); got != tc.want {
				t.Errorf("NumberDiff() =\n%q\nwant\n%q", got, tc.want)
			}
		})
	}
}

func TestBuildAnchorIndex(t *testing.T) {
	idx := BuildAnchorIndex([]FileReviewData{{
		Path: "a.go",
		Diff: "@@ -8,3 +9,3 @@\n ctx\n-old line\n+new   line\n tail\n",
	}})
	fa, ok := idx["a.go"]
	if !ok {
		t.Fatal("a.go not indexed")
	}
	want := map[int]ShownLine{
		9:  {Hash: HashLine("ctx")},
		10: {Hash: HashLine("new line"), Added: true},
		11: {Hash: HashLine("tail")},
	}
	if len(fa.Shown) != len(want) {
		t.Fatalf("shown = %v, want %v", fa.Shown, want)
	}
	for n, w := range want {
		if fa.Shown[n] != w {
			t.Errorf("line %d = %+v, want %+v", n, fa.Shown[n], w)
		}
	}
	if !fa.Removed[HashLine("old line")] || len(fa.Removed) != 1 {
		t.Errorf("removed = %v, want only %q", fa.Removed, "old line")
	}
}

// Whitespace runs and the ends do not matter, NBSP included; everything else
// does.
func TestHashLine(t *testing.T) {
	if HashLine("  a\t b ") != HashLine("a b") {
		t.Error("whitespace runs must collapse")
	}
	if HashLine("a b") != HashLine("a b") {
		t.Error("NBSP is whitespace")
	}
	if HashLine("a b") == HashLine("ab") {
		t.Error("a word break is not whitespace to drop")
	}
}
