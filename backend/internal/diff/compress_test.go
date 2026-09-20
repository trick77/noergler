package diff

import (
	"reflect"
	"strings"
	"testing"
)

// charCount stands in for a tokenizer so the budget arithmetic is exact.
func charCount(s string) int { return len(s) }

func pathEntry(f FileReviewData) string { return f.Path }

func TestRemoveDeletionOnlyHunks(t *testing.T) {
	cases := []struct{ name, input, want string }{
		{
			"deletion-only hunk dropped, other kept",
			"hdr\n@@ -1,1 +1,0 @@\n-gone\n@@ -5,1 +5,1 @@\n-old\n+new\n",
			"hdr\n@@ -5,1 +5,1 @@\n-old\n+new\n",
		},
		{
			"all hunks deletion-only returns empty",
			"hdr\n@@ -1,1 +1,0 @@\n-gone\n",
			"",
		},
		{
			// A diff with no hunks at all also returns "". Compress then files
			// the path under deleted, which mislabels it. Ported as is.
			"no hunks returns empty",
			"diff --git a/f.go b/f.go\nold mode 100644\n",
			"",
		},
		{
			"+++ does not count as an addition",
			"@@ -1,1 +1,0 @@\n+++ b/f.go\n-gone\n",
			"",
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if got := RemoveDeletionOnlyHunks(tc.input); got != tc.want {
				t.Errorf("got %q, want %q", got, tc.want)
			}
		})
	}
}

func TestIsRenameOnly(t *testing.T) {
	cases := []struct {
		name  string
		input string
		want  bool
	}{
		{"rename from", "similarity index 100%\nrename from a.go\n", true},
		{"rename to", "similarity index 100%\nrename to b.go\n", true},
		{"similarity without rename", "similarity index 100%\n", false},
		{"rename without full similarity", "similarity index 95%\nrename from a.go\n", false},
		{"ordinary diff", "@@ -1,1 +1,1 @@\n", false},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if got := IsRenameOnly(tc.input); got != tc.want {
				t.Errorf("IsRenameOnly() = %v, want %v", got, tc.want)
			}
		})
	}
}

func TestCompressClassifies(t *testing.T) {
	files := []FileReviewData{
		{Path: "kept.py", Diff: "@@ -1,1 +1,1 @@\n-a\n+b\n"},
		{Path: "gone.py", Diff: "diff --git a/gone.py b/gone.py\n+++ /dev/null\n"},
		{Path: "moved.py", Diff: "similarity index 100%\nrename from old.py\n"},
	}
	got := Compress(files, 100000, "tpl {files}", charCount, pathEntry)

	if !reflect.DeepEqual(got.DeletedFilePaths, []string{"gone.py"}) {
		t.Errorf("deleted = %q", got.DeletedFilePaths)
	}
	if !reflect.DeepEqual(got.RenamedFilePaths, []string{"moved.py"}) {
		t.Errorf("renamed = %q", got.RenamedFilePaths)
	}
	if len(got.IncludedFiles) != 1 || got.IncludedFiles[0].Path != "kept.py" {
		t.Errorf("included = %+v", got.IncludedFiles)
	}
}

// A file whose hunks are all deletion-only is filed under deleted, mislabelling
// it. Ported as is.
func TestCompressMislabelsEmptiedFileAsDeleted(t *testing.T) {
	files := []FileReviewData{
		{Path: "emptied.py", Diff: "@@ -1,1 +1,0 @@\n-gone\n"},
	}
	got := Compress(files, 100000, "tpl {files}", charCount, pathEntry)
	if !reflect.DeepEqual(got.DeletedFilePaths, []string{"emptied.py"}) {
		t.Errorf("deleted = %q, want [emptied.py]", got.DeletedFilePaths)
	}
}

// First-fit does not stop at the first miss: a later, smaller file still fits.
func TestCompressFirstFitContinuesPastAMiss(t *testing.T) {
	files := []FileReviewData{
		{Path: "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaa.py", Diff: "@@ -1,1 +1,1 @@\n-a\n+b\n"},
		{Path: "b.py", Diff: "@@ -1,1 +1,1 @@\n-a\n+b\n"},
	}
	// Budget fits the short path but not the long one.
	got := Compress(files, 20, "", charCount, pathEntry)

	var included []string
	for _, f := range got.IncludedFiles {
		included = append(included, f.Path)
	}
	if !reflect.DeepEqual(included, []string{"b.py"}) {
		t.Errorf("included = %q, want [b.py]", included)
	}
	if !reflect.DeepEqual(got.OtherModifiedPaths, []string{"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaa.py"}) {
		t.Errorf("other = %q", got.OtherModifiedPaths)
	}
}

// The budget can go negative when the overhead exceeds the window.
func TestCompressNegativeBudgetIncludesNothing(t *testing.T) {
	files := []FileReviewData{{Path: "a.py", Diff: "@@ -1,1 +1,1 @@\n-a\n+b\n"}}
	got := Compress(files, 1, strings.Repeat("x", 500)+"{files}", charCount, pathEntry)
	if len(got.IncludedFiles) != 0 {
		t.Errorf("included %d files on a negative budget", len(got.IncludedFiles))
	}
	if !reflect.DeepEqual(got.OtherModifiedPaths, []string{"a.py"}) {
		t.Errorf("other = %q", got.OtherModifiedPaths)
	}
}

// The template's {files} placeholder is removed before measuring overhead.
func TestCompressOverheadDropsPlaceholder(t *testing.T) {
	files := []FileReviewData{{Path: "a.py", Diff: "@@ -1,1 +1,1 @@\n-a\n+b\n"}}
	got := Compress(files, 100, "abc{files}def", charCount, pathEntry)
	// overhead = len("abcdef") = 6; budget = (100-6)*9/10 = 84; "a.py" is 4.
	if len(got.IncludedFiles) != 1 {
		t.Errorf("included = %+v, want a.py to fit", got.IncludedFiles)
	}
}

func TestIsSmall(t *testing.T) {
	files := []FileReviewData{{Path: "a.py"}, {Path: "bb.py"}}
	// total = 4 + 5 = 9; available = 100 - 0 = 100; 9*1.5 = 13.5 <= 100.
	if !IsSmall(files, 100, "", charCount, pathEntry, 1.5) {
		t.Error("should be small")
	}
	// available = 10; 13.5 > 10.
	if IsSmall(files, 10, "", charCount, pathEntry, 1.5) {
		t.Error("should not be small")
	}
}

func TestCountDiffLines(t *testing.T) {
	cases := []struct {
		name           string
		diff           string
		added, removed int
	}{
		{
			"counts + and - only",
			"diff --git a/a.py b/a.py\n--- a/a.py\n+++ b/a.py\n@@ -1,2 +1,2 @@\n-old\n+new\n context\n",
			1, 1,
		},
		{
			// The +++ and --- headers are not changes.
			"file headers excluded",
			"diff --git a/a.py b/a.py\n--- a/a.py\n+++ b/a.py\n@@ -1,1 +1,1 @@\n+only\n",
			1, 0,
		},
		{
			"unreviewable file skipped",
			"diff --git a/logo.png b/logo.png\n--- a/logo.png\n+++ b/logo.png\n@@ -1,1 +1,1 @@\n+binary\n",
			0, 0,
		},
		{
			"reviewable and unreviewable mixed",
			"diff --git a/a.py b/a.py\n--- a/a.py\n+++ b/a.py\n@@ -1,1 +1,1 @@\n+kept\n" +
				"diff --git a/p.png b/p.png\n--- a/p.png\n+++ b/p.png\n@@ -1,1 +1,1 @@\n+skipped\n",
			1, 0,
		},
		{"empty diff", "", 0, 0},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			added, removed := CountDiffLines(tc.diff)
			if added != tc.added || removed != tc.removed {
				t.Errorf("got (+%d, -%d), want (+%d, -%d)", added, removed, tc.added, tc.removed)
			}
		})
	}
}
