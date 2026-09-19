package diff

import (
	"strings"
	"testing"
)

// fakeEntry mirrors the Python tests' _format_entry_fake, including its
// empty-string fallback for absent content.
func fakeEntry(f FileReviewData) string {
	return "## " + f.Path + "\n" + f.Diff + "\n" + f.Content
}

// Ported from Python TestIsSmallPr. The borderline cases pin the exact
// arithmetic, so an off-by-one in the ratio or the overhead fails here.
func TestIsSmallPythonCases(t *testing.T) {
	t.Run("small PR fits", func(t *testing.T) {
		files := []FileReviewData{{Path: "a.py", Diff: "short diff"}}
		if !IsSmall(files, 10000, "template {files}", charCount, fakeEntry, 1.5) {
			t.Error("want true")
		}
	})

	t.Run("large PR does not fit", func(t *testing.T) {
		files := []FileReviewData{{Path: "a.py", Diff: strings.Repeat("x", 10000)}}
		if IsSmall(files, 100, "template {files}", charCount, fakeEntry, 1.5) {
			t.Error("want false")
		}
	})

	// Fits without margin, not with the 1.5x expansion ratio:
	// entry = 90, available = 98, 90 <= 98 but 135 > 98.
	t.Run("expansion margin rejects borderline", func(t *testing.T) {
		files := []FileReviewData{{Path: "a.py", Diff: strings.Repeat("x", 80)}}
		if IsSmall(files, 100, "t {files}", charCount, fakeEntry, 1.5) {
			t.Error("want false at ratio 1.5")
		}
	})

	t.Run("no margin with ratio 1", func(t *testing.T) {
		files := []FileReviewData{{Path: "a.py", Diff: strings.Repeat("x", 80)}}
		if !IsSmall(files, 100, "t {files}", charCount, fakeEntry, 1.0) {
			t.Error("want true at ratio 1.0")
		}
	})
}

// Ported from Python TestCompressForLargePr.
func TestCompressPythonCases(t *testing.T) {
	t.Run("separates deleted and renamed", func(t *testing.T) {
		files := []FileReviewData{
			{Path: "active.py", Diff: "diff --git a/active.py b/active.py\n@@ -1 +1 @@\n+new"},
			{Path: "gone.py", Diff: "diff --git a/gone.py b/gone.py\n+++ /dev/null\n-old"},
			{Path: "moved.py", Diff: "diff --git a/old.py b/moved.py\nsimilarity index 100%\nrename from old.py\nrename to moved.py\n"},
		}
		got := Compress(files, 100000, "template {files}", charCount, fakeEntry)
		if len(got.IncludedFiles) != 1 || got.IncludedFiles[0].Path != "active.py" {
			t.Errorf("included = %+v, want [active.py]", got.IncludedFiles)
		}
		if !contains(got.DeletedFilePaths, "gone.py") {
			t.Errorf("deleted = %q", got.DeletedFilePaths)
		}
		if !contains(got.RenamedFilePaths, "moved.py") {
			t.Errorf("renamed = %q", got.RenamedFilePaths)
		}
	})

	t.Run("budget respected", func(t *testing.T) {
		files := []FileReviewData{
			{Path: "a.py", Diff: "@@ -1 +1 @@\n+" + strings.Repeat("x", 40), Content: "content"},
			{Path: "b.py", Diff: "@@ -1 +1 @@\n+" + strings.Repeat("y", 40), Content: "content"},
			{Path: "c.py", Diff: "@@ -1 +1 @@\n+" + strings.Repeat("z", 40), Content: "content"},
		}
		got := Compress(files, 150, "t {files}", charCount, fakeEntry)
		if n := len(got.IncludedFiles) + len(got.OtherModifiedPaths); n != 3 {
			t.Errorf("accounted for %d files, want 3", n)
		}
		if len(got.OtherModifiedPaths) == 0 {
			t.Error("want at least one file over budget")
		}
	})

	t.Run("deletion-only hunks removed", func(t *testing.T) {
		diff := "diff --git a/file.py b/file.py\n--- a/file.py\n+++ b/file.py\n@@ -1,3 +1,0 @@\n-deleted only"
		got := Compress([]FileReviewData{{Path: "file.py", Diff: diff}}, 100000, "template {files}", charCount, fakeEntry)
		if len(got.IncludedFiles) != 0 {
			t.Errorf("included = %+v, want none", got.IncludedFiles)
		}
		if !contains(got.DeletedFilePaths, "file.py") {
			t.Errorf("deleted = %q, want file.py", got.DeletedFilePaths)
		}
	})
}

// Ported from Python TestRemoveDeletionOnlyHunks and TestIsRenameOnly.
func TestRemoveDeletionOnlyHunksPythonCases(t *testing.T) {
	t.Run("preserves additions", func(t *testing.T) {
		diff := "diff --git a/f.py b/f.py\n@@ -1,1 +1,1 @@\n-old\n+new\n"
		if got := RemoveDeletionOnlyHunks(diff); !strings.Contains(got, "+new") {
			t.Errorf("got %q, want the addition preserved", got)
		}
	})

	t.Run("all deletion returns empty", func(t *testing.T) {
		diff := "diff --git a/f.py b/f.py\n@@ -1,1 +1,0 @@\n-old\n"
		if got := RemoveDeletionOnlyHunks(diff); got != "" {
			t.Errorf("got %q, want empty", got)
		}
	})
}

func TestIsDeletionOnlyHunkPythonCases(t *testing.T) {
	t.Run("deletion only", func(t *testing.T) {
		if !isDeletionOnlyHunk([]string{"@@ -1,1 +1,0 @@", "-gone"}) {
			t.Error("want true")
		}
	})
	t.Run("mixed hunk", func(t *testing.T) {
		if isDeletionOnlyHunk([]string{"@@ -1,1 +1,1 @@", "-old", "+new"}) {
			t.Error("want false")
		}
	})
}

func TestIsRenameOnlyPythonCases(t *testing.T) {
	cases := []struct {
		name string
		diff string
		want bool
	}{
		{"rename only", "diff --git a/old.py b/new.py\nsimilarity index 100%\nrename from old.py\nrename to new.py\n", true},
		{"rename with changes", "diff --git a/old.py b/new.py\nsimilarity index 95%\nrename from old.py\nrename to new.py\n@@ -1 +1 @@\n+x\n", false},
		{"not a rename", "diff --git a/f.py b/f.py\n@@ -1 +1 @@\n+x\n", false},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if got := IsRenameOnly(tc.diff); got != tc.want {
				t.Errorf("IsRenameOnly() = %v, want %v", got, tc.want)
			}
		})
	}
}

// Ported from Python TestDetectLanguage and TestIsTestFile, the exact pairs.
func TestDetectLanguagePythonPairs(t *testing.T) {
	cases := map[string]string{
		"src/main.py": "python", "stubs/types.pyi": "python",
		"src/Main.java": "jvm", "src/Main.kt": "jvm",
		"build.gradle.kts": "build-config",
		"src/app.ts":       "typescript", "src/App.tsx": "typescript", "utils.mjs": "typescript",
		"template.html": "html",
		"styles.scss":   "css", "app.css": "css",
		"config.yaml": "config", "settings.toml": "config",
		"README.md": "docs", "notes.rst": "docs",
		"pom.xml": "build-config", "angular.json": "build-config",
		"nx.json": "build-config", "package.json": "build-config",
		"tsconfig.json": "build-config", "tsconfig.app.json": "build-config",
		"Makefile": "other", "data.dat": "other",
	}
	for path, want := range cases {
		t.Run(path, func(t *testing.T) {
			if got := DetectLanguage(path); got != want {
				t.Errorf("DetectLanguage(%q) = %q, want %q", path, got, want)
			}
		})
	}
}

func TestIsTestFilePythonPairs(t *testing.T) {
	cases := map[string]bool{
		"tests/test_main.py": true, "src/main_test.py": true, "src/main.py": false,
		"src/test/MainTest.java": true, "src/test/MainTests.java": true,
		"src/main/Main.java": false,
		"app.spec.ts":        true, "app.test.ts": true, "app.ts": false,
		"src/test/Helper.java": true, "src/__tests__/App.tsx": true,
	}
	for path, want := range cases {
		t.Run(path, func(t *testing.T) {
			if got := IsTestFile(path); got != want {
				t.Errorf("IsTestFile(%q) = %v, want %v", path, got, want)
			}
		})
	}
}

// Ported from Python TestSortFilesByLanguagePriority, exact orderings.
func TestSortPythonCases(t *testing.T) {
	t.Run("sorts by language then path", func(t *testing.T) {
		in := []FileReviewData{
			{Path: "small.ts", Diff: "x"},
			{Path: "zzz.py", Diff: strings.Repeat("x", 100)},
			{Path: "aaa.py", Diff: strings.Repeat("x", 10)},
		}
		assertOrder(t, SortByLanguagePriority(in), "aaa.py", "zzz.py", "small.ts")
	})

	t.Run("test files after source", func(t *testing.T) {
		in := []FileReviewData{
			{Path: "tests/test_main.py", Diff: strings.Repeat("x", 50)},
			{Path: "src/main.py", Diff: strings.Repeat("x", 10)},
			{Path: "src/app.ts", Diff: strings.Repeat("x", 20)},
		}
		assertOrder(t, SortByLanguagePriority(in), "src/main.py", "src/app.ts", "tests/test_main.py")
	})

	t.Run("deprioritized groups last", func(t *testing.T) {
		in := []FileReviewData{
			{Path: "config.yaml", Diff: "x"},
			{Path: "src/main.py", Diff: "x"},
			{Path: "README.md", Diff: "x"},
		}
		assertOrder(t, SortByLanguagePriority(in), "src/main.py", "config.yaml", "README.md")
	})
}

func assertOrder(t *testing.T, got []FileReviewData, want ...string) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("got %d files, want %d", len(got), len(want))
	}
	for i, w := range want {
		if got[i].Path != w {
			var paths []string
			for _, f := range got {
				paths = append(paths, f.Path)
			}
			t.Fatalf("order = %q, want %q", paths, want)
		}
	}
}

func contains(haystack []string, needle string) bool {
	for _, s := range haystack {
		if s == needle {
			return true
		}
	}
	return false
}
