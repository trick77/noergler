package diff

import "testing"

// ContentFetched distinguishes a fetched empty file from one whose content was
// never fetched, which the prompt's file entry renders differently. Every
// function that rebuilds a FileReviewData must carry it, and dropping it is
// silent, so both paths are pinned here.
func TestContentFetchedSurvivesRebuilds(t *testing.T) {
	t.Run("ExpandAllFiles", func(t *testing.T) {
		in := []FileReviewData{
			{Path: "empty.py", Diff: "@@ -1,1 +1,1 @@\n-a\n+b\n", Content: "", ContentFetched: true},
			{Path: "unfetched.py", Diff: "@@ -1,1 +1,1 @@\n-a\n+b\n", Content: "", ContentFetched: false},
		}
		got := ExpandAllFiles(in, 2, 2, 10, false)
		if !got[0].ContentFetched {
			t.Error("a fetched empty file lost its flag")
		}
		if got[1].ContentFetched {
			t.Error("an unfetched file gained the flag")
		}
	})

	t.Run("Compress", func(t *testing.T) {
		in := []FileReviewData{
			{Path: "empty.py", Diff: "@@ -1,1 +1,1 @@\n-a\n+b\n", Content: "", ContentFetched: true},
		}
		got := Compress(in, 100000, "tpl {files}", func(s string) int { return len(s) },
			func(f FileReviewData) string { return f.Path })
		if len(got.IncludedFiles) != 1 {
			t.Fatalf("got %d included files, want 1", len(got.IncludedFiles))
		}
		if !got.IncludedFiles[0].ContentFetched {
			t.Error("Compress dropped the flag")
		}
	})
}

// HasContent stays truthiness-based, matching Python's `content or diff`: a
// fetched empty file still falls through to the diff for reference finding.
func TestHasContentIsTruthiness(t *testing.T) {
	f := FileReviewData{Content: "", ContentFetched: true}
	if f.HasContent() {
		t.Error("an empty string must report no usable content, whatever the flag")
	}
}
