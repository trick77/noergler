package inference

import (
	"strings"
	"testing"
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
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			kept, dropped := ValidatePreExisting([]PreExistingBug{c.in}, idx)
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
	kept, dropped := ValidatePreExisting(in, validateFixture())
	if len(kept) != maxPreExisting || len(dropped) != 2 || dropped[0].Reason != "over_limit" {
		t.Errorf("kept %d, dropped %+v", len(kept), dropped)
	}
}
