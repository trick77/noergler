package inference

import (
	"strings"
	"testing"

	"github.com/trick77/noergler/internal/diff"
)

// pre_existing parses like fixed: null is none, a malformed entry is skipped,
// and the prose is capped.
func TestParsePreExisting(t *testing.T) {
	long := strings.Repeat("x", 2000)
	got := ParseReview(`{"overview": "o", "findings": [], "pre_existing": [
		{"file": "a.go", "line": "12", "evidence": ["x := 1"], "headline": "h", "explanation": "` + long + `"},
		{"file": "a.go", "line": 3, "headline": "", "explanation": "no headline"},
		{"line": 3, "headline": "h", "explanation": "no file"}
	]}`)
	if len(got.PreExisting) != 1 {
		t.Fatalf("pre_existing = %+v, want the one usable entry", got.PreExisting)
	}
	b := got.PreExisting[0]
	if b.Line != 12 || len(b.Evidence) != 1 || len([]rune(b.Explanation)) > maxPreExistingProse {
		t.Errorf("entry = %+v", b)
	}
	if got := ParseReview(`{"overview": "o", "pre_existing": null}`); got.PreExisting != nil {
		t.Errorf("null = %+v, want none", got.PreExisting)
	}
}

// Kept: evidence on an unchanged shown line (moved to it when the cited line
// is far off), or only in the full file (the cited line stands). Dropped: an
// unknown file, evidence not there, on a line the PR added (that is the PR's
// own bug, a finding), on a line the PR removed (gone), or none at all.
func TestValidatePreExisting(t *testing.T) {
	idx := validateFixture()
	bug := func(line int, evidence ...string) PreExistingBug {
		return PreExistingBug{File: "src/labels.ts", Line: line, Headline: "h", Explanation: "e", Evidence: evidence}
	}
	cases := []struct {
		name   string
		in     PreExistingBug
		reason string
		line   int
	}{
		{"context line, cited right", bug(11, "mu.Lock()"), "", 11},
		{"context line, cited far off", bug(80, "mu.Lock()"), "", 11},
		{"full file only", bug(3, "resp, err := client.Do(req)"), "", 3},
		{"unknown file", PreExistingBug{File: "x.go", Line: 1, Evidence: []string{"a"}}, string(DropUnknownFile), 0},
		{"not there", bug(11, "nothing like this"), string(DropEvidenceNotFound), 0},
		{"added line", bug(17, "a()"), "evidence_on_added_line", 0},
		{"removed line", bug(11, "defer mu.Unlock()"), "evidence_removed_by_pr", 0},
		{"no evidence", bug(11), string(DropNoEvidence), 0},
		{"cited on an added line", bug(12, "mu.Lock()"), "line_added_by_pr", 0},
		{"no line", bug(0, "resp, err := client.Do(req)"), "bad_line", 0},
		{"code in the explanation", PreExistingBug{File: "src/labels.ts", Line: 11, Headline: "h",
			Explanation: "calls `mu.Lock()` twice", Evidence: []string{"mu.Lock()"}}, "", 11},
		{"an identifier is fine", PreExistingBug{File: "src/labels.ts", Line: 11, Headline: "h",
			Explanation: "handleConnectionRecoveryTimeout never unlocks", Evidence: []string{"mu.Lock()"}}, "", 11},
		{"a path is fine", PreExistingBug{File: "src/labels.ts", Line: 11, Headline: "h",
			Explanation: "see backend/internal/review/cumulative", Evidence: []string{"mu.Lock()"}}, "", 11},
		{"a constant is fine", PreExistingBug{File: "src/labels.ts", Line: 11, Headline: "h",
			Explanation: "MAX_CONNECTION_RECOVERY_ATTEMPTS is never read", Evidence: []string{"mu.Lock()"}}, "", 11},
		{"hyphenated phrase is prose", PreExistingBug{File: "src/labels.ts", Line: 11, Headline: "h",
			Explanation: "a classic check-then-act-race-condition here", Evidence: []string{"mu.Lock()"}}, "", 11},
		{"german compound is prose", PreExistingBug{File: "src/labels.ts", Line: 11, Headline: "h",
			Explanation: "Die Verbindungswiederherstellung schlaegt fehl", Evidence: []string{"mu.Lock()"}}, "", 11},
		{"secret in the explanation", PreExistingBug{File: "src/labels.ts", Line: 11, Headline: "h",
			Explanation: "key 0123456789abcdef0123456789abcdef is committed", Evidence: []string{"mu.Lock()"}}, "secret_in_prose", 0},
		{"base64 in the headline", PreExistingBug{File: "src/labels.ts", Line: 11, Headline: "token c2VjcmV0LXRva2VuLXZhbHVlLTEyMw== leaks",
			Explanation: "e", Evidence: []string{"mu.Lock()"}}, "secret_in_prose", 0},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			kept, dropped := ValidatePreExisting([]PreExistingBug{c.in}, idx, nil)
			if c.reason != "" {
				if len(dropped) != 1 || dropped[0].Reason != c.reason {
					t.Errorf("dropped = %+v, want %q", dropped, c.reason)
				}
				return
			}
			if len(kept) != 1 || kept[0].Line != c.line {
				t.Errorf("kept = %+v, dropped = %+v; want line %d", kept, dropped, c.line)
			}
		})
	}
}

// At most five are kept, in the model's order; the rest are dropped with a
// reason, so the log says why.
func TestValidatePreExistingLimit(t *testing.T) {
	var in []PreExistingBug
	for i := 0; i < maxPreExisting+2; i++ {
		in = append(in, PreExistingBug{File: "src/labels.ts", Line: 11, Headline: "h", Explanation: "e",
			Evidence: []string{"mu.Lock()"}})
	}
	kept, dropped := ValidatePreExisting(in, validateFixture(), nil)
	if len(kept) != maxPreExisting || len(dropped) != 2 || dropped[0].Reason != "over_limit" {
		t.Errorf("kept %d, dropped %+v", len(kept), dropped)
	}
}

// On an incremental review a line an earlier push added reads as unchanged
// context; the whole PR's added lines catch it, and a file whose PR history
// is not known is dropped rather than trusted.
func TestValidatePreExistingIncremental(t *testing.T) {
	idx := validateFixture()
	bug := PreExistingBug{File: "src/labels.ts", Line: 11, Headline: "h", Explanation: "e",
		Evidence: []string{"mu.Lock()"}}
	earlier := diff.AddedLines("diff --git a/src/labels.ts b/src/labels.ts\n--- a/src/labels.ts\n+++ b/src/labels.ts\n" +
		"@@ -1,1 +1,2 @@\n ctx\n+    mu.Lock()\n")
	if _, dropped := ValidatePreExisting([]PreExistingBug{bug}, idx, earlier); len(dropped) != 1 ||
		dropped[0].Reason != "evidence_on_added_line" {
		t.Errorf("dropped = %+v, want an earlier push's line refused", dropped)
	}
	// A file the whole PR adds nothing to (a push reverted an earlier
	// one's edit): every line in it is old.
	if kept, _ := ValidatePreExisting([]PreExistingBug{bug}, idx, map[string]map[uint64]bool{}); len(kept) != 1 {
		t.Error("a file the PR adds nothing to must keep its old bug")
	}
	if kept, _ := ValidatePreExisting([]PreExistingBug{bug}, idx,
		map[string]map[uint64]bool{"src/labels.ts": {}}); len(kept) != 1 {
		t.Error("an old line in a known file must stand")
	}
	// Cited on line 10, which an earlier push added; the evidence (line 11)
	// is old and near enough that the line is not moved.
	cited := bug
	cited.Line = 10
	earlier10 := map[string]map[uint64]bool{"src/labels.ts": {diff.HashLine("func load() {"): true}}
	if _, dropped := ValidatePreExisting([]PreExistingBug{cited}, idx, earlier10); len(dropped) != 1 ||
		dropped[0].Reason != "line_added_by_pr" {
		t.Errorf("dropped = %+v, want the earlier push's line refused as the cited line", dropped)
	}
}

// A removed line that reads like one still in the file is not a removal of
// the bug; a real comment line is evidence, matched before any annotation
// skip.
func TestValidatePreExistingRemovedTwinAndComment(t *testing.T) {
	idx := diff.BuildAnchorIndex([]diff.FileReviewData{{
		Path:           "a.go",
		Diff:           "@@ -1,2 +1,1 @@\n ctx := 1\n-defer mu.Unlock()\n",
		Content:        "ctx := 1\nfunc b() {\n\tdefer mu.Unlock()\n}\n// timeout in seconds\n",
		ContentFetched: true,
	}})
	bugs := []PreExistingBug{
		{File: "a.go", Line: 3, Headline: "h", Explanation: "e", Evidence: []string{"defer mu.Unlock()"}},
		{File: "a.go", Line: 5, Headline: "h", Explanation: "e", Evidence: []string{"// timeout in seconds"}},
	}
	kept, dropped := ValidatePreExisting(bugs, idx, nil)
	if len(kept) != 2 {
		t.Errorf("kept = %+v, dropped = %+v; want both", kept, dropped)
	}
}
