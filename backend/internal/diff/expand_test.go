package diff

import (
	"strings"
	"testing"
)

// numberedLines builds "line 1\nline 2\n..." with plain decimal numbering.
func numberedLines(n int) string {
	parts := make([]string, 0, n)
	for i := 1; i <= n; i++ {
		parts = append(parts, "line "+itoa(i))
	}
	return strings.Join(parts, "\n")
}

func itoa(n int) string {
	if n == 0 {
		return "0"
	}
	var digits []byte
	for n > 0 {
		digits = append([]byte{byte('0' + n%10)}, digits...)
		n /= 10
	}
	return string(digits)
}

func bodyOf(out string) []string {
	var body []string
	for _, l := range strings.Split(out, "\n") {
		if !strings.HasPrefix(l, "@@") {
			body = append(body, l)
		}
	}
	return body
}

func headersOf(out string) []string {
	var hs []string
	for _, l := range strings.Split(out, "\n") {
		if strings.HasPrefix(l, "@@") {
			hs = append(hs, l)
		}
	}
	return hs
}

// D0: adjacent hunks merge into one header AND keep every diff line. Python
// dropped -line 7 here and showed line 7 twice, once as context and once as an
// addition.
func TestAdjacentHunksMergeWithoutLosingLines(t *testing.T) {
	diff := "@@ -5,1 +5,1 @@\n-line 5\n+new1\n@@ -7,1 +7,1 @@\n-line 7\n+new2\n"
	out := ExpandContext(diff, numberedLines(20), "f.py", 2, 2, 10, false)

	if hs := headersOf(out); len(hs) != 1 {
		t.Errorf("got %d headers, want 1: %q", len(hs), hs)
	}

	body := bodyOf(out)
	want := []string{" line 3", " line 4", "-line 5", "+new1", " line 6", "-line 7", "+new2", " line 8", " line 9"}
	if strings.Join(body, "\n") != strings.Join(want, "\n") {
		t.Errorf("body =\n%q\nwant\n%q", body, want)
	}

	// The removal Python dropped must survive.
	if !strings.Contains(out, "-line 7") {
		t.Error("-line 7 was dropped, reintroducing the Python bug")
	}
	// No file line may appear both as context and as a change.
	seen := map[string]bool{}
	for _, l := range body {
		if l == "" {
			continue
		}
		key := strings.TrimSpace(l[1:])
		if strings.HasPrefix(l, " ") {
			if seen[key] {
				t.Errorf("line %q appears twice in the merged body", key)
			}
			seen[key] = true
		}
	}
	// No phantom blank line.
	for i, l := range body {
		if l == "" {
			t.Errorf("blank line at body index %d", i)
		}
	}
}

// A three-hunk chain drops two removals in Python.
func TestThreeHunkChainKeepsEveryRemoval(t *testing.T) {
	diff := "@@ -5,1 +5,1 @@\n-line 5\n+new5\n" +
		"@@ -7,1 +7,1 @@\n-line 7\n+new7\n" +
		"@@ -9,1 +9,1 @@\n-line 9\n+new9\n"
	out := ExpandContext(diff, numberedLines(20), "f.py", 2, 2, 10, false)

	for _, want := range []string{"-line 5", "-line 7", "-line 9", "+new5", "+new7", "+new9"} {
		if !strings.Contains(out, want) {
			t.Errorf("missing %q in:\n%s", want, out)
		}
	}
	if hs := headersOf(out); len(hs) != 1 {
		t.Errorf("got %d headers, want 1: %q", len(hs), hs)
	}
}

// Hunks far apart must not merge.
func TestDistantHunksDoNotMerge(t *testing.T) {
	diff := "@@ -3,1 +3,1 @@\n-line 3\n+new3\n@@ -15,1 +15,1 @@\n-line 15\n+new15\n"
	out := ExpandContext(diff, numberedLines(20), "f.py", 2, 2, 10, false)
	if hs := headersOf(out); len(hs) != 2 {
		t.Errorf("got %d headers, want 2: %q", len(hs), hs)
	}
}

// D2: no phantom blank line on the ordinary single-hunk path either.
func TestNoPhantomBlankLine(t *testing.T) {
	out := ExpandContext("@@ -5,1 +5,1 @@\n-line 5\n+new1\n", numberedLines(20), "f.py", 2, 2, 10, false)
	for i, l := range bodyOf(out) {
		if l == "" {
			t.Errorf("blank line at body index %d in:\n%s", i, out)
		}
	}
}

func TestExpandContextClamping(t *testing.T) {
	t.Run("start of file clamps ctx_start to 1", func(t *testing.T) {
		out := ExpandContext("@@ -1,1 +1,1 @@\n-line 1\n+new1\n", numberedLines(20), "f.py", 3, 2, 10, false)
		if hs := headersOf(out); len(hs) != 1 || !strings.HasPrefix(hs[0], "@@ -1,") {
			t.Errorf("header = %q, want old start 1", hs)
		}
	})

	t.Run("EOF clamps ctx_end", func(t *testing.T) {
		out := ExpandContext("@@ -20,1 +20,1 @@\n-line 20\n+new20\n", numberedLines(20), "f.py", 2, 3, 10, false)
		if strings.Contains(out, "line 21") {
			t.Error("read past end of file")
		}
	})

	// before_count is not clamped to the file length, so header counts can
	// exceed the body. Ported as is.
	t.Run("content shorter than the diff claims", func(t *testing.T) {
		out := ExpandContext("@@ -18,1 +18,1 @@\n-line 18\n+new18\n", "line 1\nline 2\nline 3", "f.py", 2, 2, 10, false)
		if len(headersOf(out)) != 1 {
			t.Errorf("want one header, got: %s", out)
		}
	})
}

func TestExpandContextPassthrough(t *testing.T) {
	t.Run("empty content returns the diff unchanged", func(t *testing.T) {
		diff := "@@ -5,1 +5,1 @@\n-line 5\n+new1\n"
		if got := ExpandContext(diff, "", "f.py", 2, 2, 10, false); got != diff {
			t.Errorf("got %q, want the input unchanged", got)
		}
	})

	t.Run("no hunks returns the diff unchanged", func(t *testing.T) {
		diff := "diff --git a/f.py b/f.py\nsimilarity index 100%\n"
		if got := ExpandContext(diff, numberedLines(20), "f.py", 2, 2, 10, false); got != diff {
			t.Errorf("got %q, want the input unchanged", got)
		}
	})
}

// The rebuilt header always emits both counts and discards function context.
func TestRebuiltHeaderShape(t *testing.T) {
	out := ExpandContext("@@ -5,1 +5,1 @@ def alpha():\n-line 5\n+new1\n", numberedLines(20), "f.py", 2, 2, 10, false)
	hs := headersOf(out)
	if len(hs) != 1 {
		t.Fatalf("got %d headers", len(hs))
	}
	if strings.Contains(hs[0], "def alpha") {
		t.Errorf("function context not discarded: %q", hs[0])
	}
	if strings.Count(hs[0], ",") != 2 {
		t.Errorf("header %q should carry both counts", hs[0])
	}
}

// old/new stay aligned when the two sides diverge.
func TestOldNewDelta(t *testing.T) {
	out := ExpandContext("@@ -12,1 +5,1 @@\n-line 5\n+new1\n", numberedLines(20), "f.py", 2, 2, 10, false)
	hs := headersOf(out)
	if len(hs) != 1 {
		t.Fatalf("got %d headers", len(hs))
	}
	// new start is 5-2=3; old start is 3 + (12-5) = 10.
	if !strings.HasPrefix(hs[0], "@@ -10,") || !strings.Contains(hs[0], "+3,") {
		t.Errorf("header = %q, want -10,... +3,...", hs[0])
	}
}

func TestFileHeaderLinesPreserved(t *testing.T) {
	diff := "diff --git a/f.py b/f.py\n--- a/f.py\n+++ b/f.py\n@@ -5,1 +5,1 @@\n-line 5\n+new1\n"
	out := ExpandContext(diff, numberedLines(20), "f.py", 2, 2, 10, false)
	for _, want := range []string{"diff --git a/f.py b/f.py", "--- a/f.py", "+++ b/f.py"} {
		if !strings.Contains(out, want) {
			t.Errorf("missing header line %q", want)
		}
	}
}

func TestDynamicContext(t *testing.T) {
	pySrc := strings.Join([]string{
		"import os", "", "", "def alpha():", "    a = 1", "    b = 2",
		"    c = 3", "    d = 4", "    return a",
	}, "\n")

	t.Run("pulls back to the enclosing def", func(t *testing.T) {
		out := ExpandContext("@@ -8,1 +8,1 @@\n-    d = 4\n+    d = 44\n", pySrc, "mod.py", 2, 2, 10, true)
		if !strings.Contains(out, "def alpha():") {
			t.Errorf("dynamic search did not reach the def:\n%s", out)
		}
	})

	t.Run("skipped for no-dynamic languages", func(t *testing.T) {
		out := ExpandContext("@@ -8,1 +8,1 @@\n-    d = 4\n+    d = 44\n", pySrc, "notes.md", 2, 2, 10, true)
		if strings.Contains(out, "def alpha():") {
			t.Errorf("docs should skip the dynamic search:\n%s", out)
		}
	})

	t.Run("disabled by flag", func(t *testing.T) {
		out := ExpandContext("@@ -8,1 +8,1 @@\n-    d = 4\n+    d = 44\n", pySrc, "mod.py", 2, 2, 10, false)
		if strings.Contains(out, "def alpha():") {
			t.Errorf("dynamic disabled but scope still pulled in:\n%s", out)
		}
	})
}

// RE2's \w is ASCII while Python's is Unicode, so the ported scope patterns use
// [\pL\pN_]. A non-ASCII identifier must still match.
func TestScopePatternUnicodeIdentifier(t *testing.T) {
	re := scopePatterns["typescript"]
	if !re.MatchString("const café = (x) => x") {
		t.Error("typescript scope pattern must match a non-ASCII identifier")
	}
	if !re.MatchString("const plain = (x) => x") {
		t.Error("typescript scope pattern must still match an ASCII identifier")
	}
}

// `\ No newline at end of file` is counted as a real new-file line, pushing the
// after-window one line further. Ported as is.
func TestNoNewlineMarkerCountedAsLine(t *testing.T) {
	withMarker := ExpandContext("@@ -5,1 +5,1 @@\n-line 5\n+new1\n\\ No newline at end of file\n",
		numberedLines(20), "f.py", 2, 2, 10, false)
	without := ExpandContext("@@ -5,1 +5,1 @@\n-line 5\n+new1\n",
		numberedLines(20), "f.py", 2, 2, 10, false)
	if withMarker == without {
		t.Error("the marker should shift the after-window, matching Python")
	}
}

func TestExpandAllFiles(t *testing.T) {
	in := []FileReviewData{
		{Path: "a.py", Diff: "@@ -5,1 +5,1 @@\n-line 5\n+new1\n", Content: numberedLines(20)},
		{Path: "b.py", Diff: "@@ -1,1 +1,1 @@\n-x\n+y\n", Content: ""},
	}
	out := ExpandAllFiles(in, 2, 2, 10, false)
	if len(out) != 2 {
		t.Fatalf("got %d files, want 2", len(out))
	}
	if out[0].Diff == in[0].Diff {
		t.Error("a.py should have been expanded")
	}
	if out[1].Diff != in[1].Diff {
		t.Error("b.py has no content and should pass through unchanged")
	}
	if out[0].Content != in[0].Content {
		t.Error("content must be carried through")
	}
}

// makeDiff mirrors the Python tests' _make_diff helper.
func makeDiff(oldStart, oldCount, newStart, newCount int, body string) string {
	return "diff --git a/file.py b/file.py\n--- a/file.py\n+++ b/file.py\n" +
		"@@ -" + itoa(oldStart) + "," + itoa(oldCount) +
		" +" + itoa(newStart) + "," + itoa(newCount) + " @@\n" + body
}

// Ported from Python TestExpandContext.
func TestExpandContextCases(t *testing.T) {
	t.Run("adds before context", func(t *testing.T) {
		diff := makeDiff(5, 1, 5, 1, "-old\n+new")
		got := ExpandContext(diff, numberedLines(10), "file.py", 3, 0, 8, false)
		for _, want := range []string{" line 2", " line 3", " line 4"} {
			if !strings.Contains(got, want) {
				t.Errorf("missing %q in:\n%s", want, got)
			}
		}
	})

	t.Run("adds after context", func(t *testing.T) {
		diff := makeDiff(5, 1, 5, 1, "-old\n+new")
		got := ExpandContext(diff, numberedLines(10), "file.py", 0, 2, 8, false)
		for _, want := range []string{" line 6", " line 7"} {
			if !strings.Contains(got, want) {
				t.Errorf("missing %q in:\n%s", want, got)
			}
		}
	})

	t.Run("asymmetric context", func(t *testing.T) {
		diff := makeDiff(8, 1, 8, 1, "-old\n+new")
		got := ExpandContext(diff, numberedLines(15), "file.py", 3, 1, 8, false)
		for _, want := range []string{" line 5", " line 6", " line 7", " line 9"} {
			if !strings.Contains(got, want) {
				t.Errorf("missing %q in:\n%s", want, got)
			}
		}
		if strings.Contains(got, " line 10") {
			t.Errorf("after=1 should not reach line 10:\n%s", got)
		}
	})

	t.Run("no content returns the original", func(t *testing.T) {
		diff := makeDiff(10, 1, 10, 2, "-old\n+new\n+added")
		if got := ExpandContext(diff, "", "file.py", 3, 1, 8, true); got != diff {
			t.Errorf("got %q, want the input unchanged", got)
		}
	})

	t.Run("no hunks returns the original", func(t *testing.T) {
		diff := "diff --git a/file.py b/file.py\n--- a/file.py\n+++ b/file.py"
		if got := ExpandContext(diff, numberedLines(20), "file.py", 3, 1, 8, true); got != diff {
			t.Errorf("got %q, want the input unchanged", got)
		}
	})

	pyContent := strings.Join([]string{
		"import os", "", "def process_data():", "    x = 1", "    y = 2",
		"    z = 3", "    result = x + y + z", "    return result",
	}, "\n")

	t.Run("dynamic context finds the function", func(t *testing.T) {
		diff := makeDiff(7, 1, 7, 1, "-old\n+new")
		got := ExpandContext(diff, pyContent, "app/utils.py", 1, 0, 8, true)
		if !strings.Contains(got, " def process_data():") {
			t.Errorf("dynamic search did not reach the def:\n%s", got)
		}
	})

	t.Run("dynamic context disabled", func(t *testing.T) {
		diff := makeDiff(7, 1, 7, 1, "-old\n+new")
		got := ExpandContext(diff, pyContent, "app/utils.py", 1, 0, 8, false)
		if strings.Contains(got, "def process_data():") {
			t.Errorf("dynamic disabled but the def was pulled in:\n%s", got)
		}
		if !strings.Contains(got, " z = 3") {
			t.Errorf("before=1 should include z = 3:\n%s", got)
		}
	})

	t.Run("clamps to start of file", func(t *testing.T) {
		diff := makeDiff(1, 1, 1, 1, "-old\n+new")
		got := ExpandContext(diff, numberedLines(5), "file.py", 5, 0, 8, false)
		if !strings.Contains(got, "+new") {
			t.Errorf("expected the addition to survive:\n%s", got)
		}
	})

	t.Run("clamps to end of file", func(t *testing.T) {
		diff := makeDiff(5, 1, 5, 1, "-old\n+new")
		got := ExpandContext(diff, numberedLines(5), "file.py", 0, 10, 8, false)
		if !strings.Contains(got, "+new") {
			t.Errorf("expected the addition to survive:\n%s", got)
		}
	})

	// New start is 10 - 3 = 7 on both sides.
	t.Run("hunk header updated", func(t *testing.T) {
		diff := makeDiff(10, 1, 10, 1, "-old\n+new")
		got := ExpandContext(diff, numberedLines(20), "file.py", 3, 1, 8, false)
		if !strings.Contains(got, "@@ -7,") {
			t.Errorf("missing '@@ -7,' in:\n%s", got)
		}
		if !strings.Contains(got, "+7,") {
			t.Errorf("missing '+7,' in:\n%s", got)
		}
	})
}

// Ported from Python TestExpandAllFiles.
func TestExpandAllFilesCases(t *testing.T) {
	content := numberedLines(20)

	t.Run("expands multiple files", func(t *testing.T) {
		files := []FileReviewData{
			{Path: "a.py", Diff: "diff --git a/a.py b/a.py\n@@ -5,1 +5,1 @@\n-old\n+new", Content: content},
			{Path: "b.py", Diff: "diff --git a/b.py b/b.py\n@@ -10,1 +10,1 @@\n-old\n+new", Content: content},
		}
		got := ExpandAllFiles(files, 2, 1, 8, false)
		if len(got) != 2 {
			t.Fatalf("got %d files, want 2", len(got))
		}
		if got[0].Path != "a.py" || got[1].Path != "b.py" {
			t.Errorf("paths = %q, %q", got[0].Path, got[1].Path)
		}
		if got[0].Content != content {
			t.Error("content not preserved")
		}
		for _, want := range []string{" line 3", " line 4"} {
			if !strings.Contains(got[0].Diff, want) {
				t.Errorf("missing %q in a.py:\n%s", want, got[0].Diff)
			}
		}
	})

	// Python passed content=None; Go's equivalent is the empty string.
	t.Run("preserves absent content", func(t *testing.T) {
		files := []FileReviewData{
			{Path: "a.py", Diff: "diff --git a/a.py b/a.py\n@@ -5,1 +5,1 @@\n-old\n+new", Content: ""},
		}
		got := ExpandAllFiles(files, 2, 1, 8, true)
		if len(got) != 1 {
			t.Fatalf("got %d files, want 1", len(got))
		}
		if got[0].Diff != files[0].Diff {
			t.Errorf("diff changed: %q", got[0].Diff)
		}
		if got[0].Content != "" {
			t.Errorf("content = %q, want empty", got[0].Content)
		}
	})
}

// Ported from Python TestMergeOverlappingHunks. The Python asserts only the
// header count; the stronger assertions live in
// TestAdjacentHunksMergeWithoutLosingLines.
func TestMergeHeaderCountsCases(t *testing.T) {
	t.Run("adjacent hunks merged", func(t *testing.T) {
		diff := "diff --git a/file.py b/file.py\n@@ -5,1 +5,1 @@\n-old1\n+new1\n@@ -7,1 +7,1 @@\n-old2\n+new2"
		got := ExpandContext(diff, numberedLines(20), "file.py", 2, 2, 8, false)
		if n := len(headersOf(got)); n != 1 {
			t.Errorf("got %d hunk headers, want 1:\n%s", n, got)
		}
		// Divergence: Python dropped -old2 here. It must survive.
		if !strings.Contains(got, "-old2") {
			t.Errorf("-old2 was dropped, reintroducing the Python bug:\n%s", got)
		}
	})

	t.Run("distant hunks not merged", func(t *testing.T) {
		diff := "diff --git a/file.py b/file.py\n@@ -5,1 +5,1 @@\n-old1\n+new1\n@@ -20,1 +20,1 @@\n-old2\n+new2"
		got := ExpandContext(diff, numberedLines(30), "file.py", 2, 1, 8, false)
		if n := len(headersOf(got)); n != 2 {
			t.Errorf("got %d hunk headers, want 2:\n%s", n, got)
		}
	})
}
