package diff

import (
	"strings"
	"testing"
)

// makeDiff mirrors the Python tests' _make_diff helper.
func makeDiff(oldStart, oldCount, newStart, newCount int, body string) string {
	return "diff --git a/file.py b/file.py\n--- a/file.py\n+++ b/file.py\n" +
		"@@ -" + itoa(oldStart) + "," + itoa(oldCount) +
		" +" + itoa(newStart) + "," + itoa(newCount) + " @@\n" + body
}

// Ported from Python TestExpandContext.
func TestExpandContextPythonCases(t *testing.T) {
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
func TestExpandAllFilesPythonCases(t *testing.T) {
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
func TestMergeHeaderCountsPythonCases(t *testing.T) {
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
