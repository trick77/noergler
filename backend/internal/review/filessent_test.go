package review

import (
	"fmt"
	"strings"
	"testing"

	"github.com/trick77/noergler/internal/diff"
	"github.com/trick77/noergler/internal/inference"
)

// One token per line keeps the expected figures readable: a file's total is
// its formatted entry's line count, so the test can compute what it asserts.
func linesAsTokens(s string) int { return strings.Count(s, "\n") + 1 }

func sentFile(path, body string, diffLines int) diff.FileReviewData {
	var d strings.Builder
	fmt.Fprintf(&d, "diff --git a/%s b/%s\n--- a/%s\n+++ b/%s\n@@ -1,%d +1,%d @@\n", path, path, path, path, diffLines, diffLines)
	for i := 0; i < diffLines; i++ {
		fmt.Fprintf(&d, "+line %d\n", i)
	}
	return diff.FileReviewData{Path: path, Diff: strings.TrimSuffix(d.String(), "\n"), Content: body, ContentFetched: body != ""}
}

// The line names every file biggest first, splits a fetched file into diff
// and body, marks a diff-only file as such, and its total is the same sum
// the prompt line reports as "files N".
func TestFilesSentLineIsBiggestFirstAndSumsToThePromptFigure(t *testing.T) {
	files := []diff.FileReviewData{
		sentFile("small.go", "", 2),
		sentFile("big.go", strings.Repeat("x\n", 40), 5),
		sentFile("mid.go", strings.Repeat("y\n", 10), 3),
	}
	got := renderFilesSent(files, linesAsTokens)

	want := 0
	for _, f := range files {
		want += linesAsTokens(inference.FormatFileEntry(f))
	}
	if !strings.HasPrefix(got, fmt.Sprintf("files sent: 3, %d tokens - ", want)) {
		t.Fatalf("header: %s", got)
	}
	big, mid, small := strings.Index(got, "big.go "), strings.Index(got, "mid.go "), strings.Index(got, "small.go ")
	if big > mid || mid > small {
		t.Errorf("not biggest first: %s", got)
	}
	bigTotal := linesAsTokens(inference.FormatFileEntry(files[1]))
	bigDiff := linesAsTokens(diff.NumberDiff(files[1].Diff))
	if !strings.Contains(got, fmt.Sprintf("big.go %d (%d diff, 41 body)", bigTotal, bigDiff)) {
		t.Errorf("fetched file not split into diff and body: %s", got)
	}
	if !strings.Contains(got, "small.go ") || !strings.Contains(got, " (diff only)") {
		t.Errorf("diff-only file not marked: %s", got)
	}
}

// Past the path cap the tail is one count and one token sum, so the header
// total is still accounted for by what the line shows.
func TestFilesSentLineCapsNamesAndSumsTheTail(t *testing.T) {
	var files []diff.FileReviewData
	for i := 0; i < maxLoggedPaths+4; i++ {
		files = append(files, sentFile(fmt.Sprintf("f%02d.go", i), "", 1+i))
	}
	got := renderFilesSent(files, linesAsTokens)

	if strings.Count(got, ".go ") != maxLoggedPaths {
		t.Errorf("want %d names, got: %s", maxLoggedPaths, got)
	}
	// The four smallest are the tail: the first four files, 1..4 diff lines.
	tail := 0
	for _, f := range files[:4] {
		tail += linesAsTokens(inference.FormatFileEntry(f))
	}
	if !strings.HasSuffix(got, fmt.Sprintf(", +4 more, %d tokens", tail)) {
		t.Errorf("tail: %s", got)
	}
	for _, f := range files[:4] {
		if strings.Contains(got, f.Path) {
			t.Errorf("small file %s named although it belongs to the tail: %s", f.Path, got)
		}
	}
}

func TestFilesSentLineWithNoFiles(t *testing.T) {
	if got := renderFilesSent(nil, linesAsTokens); got != "files sent: 0, 0 tokens" {
		t.Errorf("got %q", got)
	}
}
