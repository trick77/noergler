package inference

import (
	"encoding/json"
	"testing"

	"github.com/trick77/noergler/internal/diff"
)

// A focus item stands only on a line the focused diff shows in a file the
// prompt carried; repeats and items past the cap go, each with its reason.
func TestValidateFocus(t *testing.T) {
	idx := diff.BuildAnchorIndex([]diff.FileReviewData{{Path: "a.go", Diff: "@@ -1,1 +1,3 @@\n x\n+y\n+z"}})
	items := []FocusItem{
		{File: "a.go", Line: 2, Why: "kept"},
		{File: "b/a.go", Line: 3, Why: "kept, prefix taken off"},
		{File: "a.go", Line: 9, Why: "not shown"},
		{File: "c.go", Line: 1, Why: "not in the prompt"},
		{File: "a.go", Line: 2, Why: "repeat"},
		{File: "a.go", Line: 1, Why: "kept, context line"},
		{File: "a.go", Line: 3, Why: "repeat of b/a.go:3"},
	}
	kept, dropped := ValidateFocus(items, idx, nil)
	if len(kept) != 3 || kept[1].File != "a.go" {
		t.Fatalf("kept = %+v", kept)
	}
	want := []FocusDropReason{FocusNotInDiff, FocusUnknownFile, FocusRepeat, FocusRepeat}
	if len(dropped) != len(want) {
		t.Fatalf("dropped = %+v", dropped)
	}
	for i, d := range dropped {
		if d.Reason != want[i] {
			t.Errorf("drop %d = %s, want %s", i, d.Reason, want[i])
		}
	}
	idx2 := diff.BuildAnchorIndex([]diff.FileReviewData{{Path: "a.go", Diff: "@@ -1,1 +1,4 @@\n x\n+y\n+z\n+w"}})
	if _, d := ValidateFocus([]FocusItem{{"a.go", 1, "a"}, {"a.go", 2, "b"}, {"a.go", 3, "c"}, {"a.go", 4, "d"}}, idx2, nil); len(d) != 1 || d[0].Reason != FocusOverCap {
		t.Errorf("a fourth distinct place must drop over the cap: %+v", d)
	}
}

// On an incremental review a place an earlier push changed stands: the
// focus is the summary's, and the summary is the whole PR's.
func TestValidateFocusReadsTheCumulativeDiff(t *testing.T) {
	push := diff.BuildAnchorIndex([]diff.FileReviewData{{Path: "README.md", Diff: "@@ -1,1 +1,1 @@\n-typo\n+fixed"}})
	history := diff.BuildAnchorIndex([]diff.FileReviewData{{Path: "auth.go", Diff: "@@ -38,2 +38,3 @@\n x\n+check()\n y"}})
	kept, dropped := ValidateFocus([]FocusItem{{"b/auth.go", 39, "the auth check"}, {"auth.go", 90, "not shown"}}, push, history)
	if len(kept) != 1 || kept[0].File != "auth.go" || len(dropped) != 1 || dropped[0].Reason != FocusNotInDiff {
		t.Errorf("kept %+v, dropped %+v", kept, dropped)
	}
}

// A malformed entry costs itself, never the review.
func TestParseFocus(t *testing.T) {
	for raw, ok := range map[string]bool{
		`{"file": "a.go", "line": 3, "why": "Read it."}`:   true,
		`{"file": "a.go", "line": 3.0, "why": "Read it."}`: true,
		`{"file": "a.go", "line": 3.5, "why": "Read it."}`: false,
		`{"file": "", "line": 3, "why": "Read it."}`:       false,
		`{"file": "a.go", "line": 3, "why": "  "}`:         false,
		`{"file": "a.go", "why": "Read it."}`:              false,
	} {
		if _, got := parseFocus(json.RawMessage(raw)); got != ok {
			t.Errorf("%s: ok = %v, want %v", raw, got, ok)
		}
	}
}
