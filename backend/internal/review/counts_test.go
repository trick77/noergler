package review

import (
	"reflect"
	"testing"

	"github.com/trick77/noergler/internal/diff"
)

// "Reviewed without full file context" is decided per file from its own
// diff, so the small-PR path (no compression lists) and the compression
// path agree: a rename-only or deleted file has no body to miss, and a file
// left out of the prompt was not reviewed at all.
func TestNotFullContext(t *testing.T) {
	prepared := []diff.FileReviewData{
		{Path: "big.go", Diff: "diff --git a/big.go b/big.go\n--- a/big.go\n+++ b/big.go\n@@ -1 +1 @@\n-a\n+b\n"},
		{Path: "moved.go", Diff: "diff --git a/old.go b/moved.go\nsimilarity index 100%\nrename from old.go\nrename to moved.go\n"},
		{Path: "gone.go", Diff: "diff --git a/gone.go b/gone.go\n--- a/gone.go\n+++ /dev/null\n@@ -1 +0,0 @@\n-a\n"},
		{Path: "left-out.go", Diff: "diff --git a/left-out.go b/left-out.go\n@@ -1 +1 @@\n-a\n+b\n"},
	}
	skipped := []string{"big.go", "moved.go", "gone.go", "left-out.go"}
	for name, overBudget := range map[string][]string{
		"small PR (no compression)": nil,
		"compressed":                {"left-out.go"},
	} {
		got := notFullContext(skipped, prepared, overBudget)
		want := []string{"big.go"}
		if overBudget == nil {
			want = []string{"big.go", "left-out.go"}
		}
		if !reflect.DeepEqual(got, want) {
			t.Errorf("%s: notFullContext = %v, want %v", name, got, want)
		}
	}
}
