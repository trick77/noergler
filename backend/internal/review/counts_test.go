package review

import (
	"reflect"
	"testing"

	"github.com/trick77/noergler/internal/diff"
)

// "Reviewed without full file context" is read from what the prompt
// carries: a content-skipped file sent as an entry without its body. A file
// the prompt does not carry (compression left it out, or filed it as
// deleted) was not reviewed, and a rename-only one has no body to miss.
func TestNotFullContext(t *testing.T) {
	sent := []diff.FileReviewData{
		{Path: "big.go", Diff: "diff --git a/big.go b/big.go\n@@ -1 +1 @@\n-a\n+b\n"},
		{Path: "moved.go", Diff: "diff --git a/old.go b/moved.go\nsimilarity index 100%\nrename from old.go\nrename to moved.go\n"},
		{Path: "whole.go", Diff: "diff --git a/whole.go b/whole.go\n@@ -1 +1 @@\n-a\n+b\n", Content: "x", ContentFetched: true},
	}
	skipped := []string{"big.go", "moved.go", "left-out.go", "trimmed-to-deleted.go"}
	got := notFullContext(skipped, sent)
	if want := []string{"big.go"}; !reflect.DeepEqual(got, want) {
		t.Errorf("notFullContext = %v, want %v", got, want)
	}
}
