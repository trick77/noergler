package inference

import (
	"strings"
	"testing"

	"github.com/trick77/noergler/internal/diff"
)

// The push adds the missing unlock in svc.go, moves its nil check down a
// line, and leaves other.go's context alone. Earlier findings: 1 on svc.go (the leak), 2 on other.go, 3 on a file
// the push did not touch.
func fixedFixture() (diff.AnchorIndex, []PostedFinding) {
	idx := diff.BuildAnchorIndex([]diff.FileReviewData{
		{
			Path: "svc.go",
			Diff: "@@ -10,4 +10,5 @@\n" +
				" \tmu.Lock()\n" +
				"-\tif in == nil {\n" +
				"+\tdefer mu.Unlock()\n" +
				"-\treturn load()\n" +
				"+\tif in == nil {\n" +
				"+\treturn load(ctx)\n" +
				" }\n",
			Content:        "func f() {\n\tmu.Lock()\n\tdefer mu.Unlock()\n\treturn load(ctx)\n}\n",
			ContentFetched: true,
		},
		{Path: "other.go", Diff: "@@ -1,2 +1,2 @@\n x := 1\n+y := 2\n"},
	})
	idx.AddContextDiff("+\tdefer cancel()\n")
	line := 11
	return idx, []PostedFinding{
		{ID: 1, FilePath: "svc.go", LineNumber: &line, Severity: "issue", CommentText: "lock never released"},
		{ID: 2, FilePath: "b/other.go", LineNumber: &line, Severity: "suggestion", CommentText: "x unused"},
		{ID: 3, FilePath: "untouched.go", LineNumber: &line, Severity: "issue", CommentText: "nil map"},
	}
}

func TestValidateFixed(t *testing.T) {
	cases := []struct {
		name   string
		claim  FixedClaim
		reason FixedDropReason
	}{
		{"added line fixes it", FixedClaim{"F1", []string{"defer mu.Unlock()"}}, ""},
		{"removed line fixes it", FixedClaim{"F1", []string{"return load()"}}, ""},
		{"context beside a change", FixedClaim{"F1", []string{"mu.Lock()", "defer mu.Unlock()"}}, ""},
		{"copied marker is tolerated", FixedClaim{"F1", []string{"+\tdefer mu.Unlock()"}}, ""},
		{"model's note is skipped", FixedClaim{"F1", []string{"defer mu.Unlock()", "// released now"}}, ""},
		{"a/ b/ prefix on the stored path", FixedClaim{"F2", []string{"y := 2"}}, ""},
		{"id without the prefix", FixedClaim{"1", []string{"defer mu.Unlock()"}}, ""},
		{"id copied with its brackets", FixedClaim{"[F1]", []string{"defer mu.Unlock()"}}, ""},
		{"lowercase id", FixedClaim{"f1", []string{"defer mu.Unlock()"}}, ""},
		// Removed and added back: the code moved, its bug with it.
		{"moved line", FixedClaim{"F1", []string{"if in == nil {"}}, FixedEvidenceUnchanged},
		{"moved line copied as removed", FixedClaim{"F1", []string{"-\tif in == nil {"}}, FixedEvidenceUnchanged},
		{"id never listed", FixedClaim{"F9", []string{"defer mu.Unlock()"}}, FixedUnknownID},
		{"id not a number", FixedClaim{"Fx", []string{"defer mu.Unlock()"}}, FixedUnknownID},
		{"file the push did not touch", FixedClaim{"F3", []string{"defer mu.Unlock()"}}, FixedFileNotInDiff},
		{"no evidence", FixedClaim{"F1", nil}, FixedNoEvidence},
		{"notes alone", FixedClaim{"F1", []string{"// fixed", "..."}}, FixedNoEvidence},
		{"invented line", FixedClaim{"F1", []string{"defer mu.Unlock()", "mu = nil"}}, FixedEvidenceNotFound},
		// Another file's change proves nothing about this one.
		{"line of another file", FixedClaim{"F1", []string{"y := 2"}}, FixedEvidenceNotFound},
		{"context line only", FixedClaim{"F1", []string{"mu.Lock()"}}, FixedEvidenceUnchanged},
		{"full-file line only", FixedClaim{"F1", []string{"func f() {"}}, FixedEvidenceUnchanged},
		// The cumulative diff is context: an earlier push's change is not
		// this push's fix.
		{"cumulative diff line", FixedClaim{"F1", []string{"defer cancel()"}}, FixedEvidenceNotFound},
	}
	idx, sent := fixedFixture()
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			v := ValidateFixed([]FixedClaim{c.claim}, sent, idx)
			if c.reason == "" {
				if len(v.Kept) != 1 {
					t.Fatalf("dropped %+v, want kept", v.Dropped)
				}
				return
			}
			if len(v.Dropped) != 1 || v.Dropped[0].Reason != c.reason {
				t.Errorf("kept %+v dropped %+v, want %s", v.Kept, v.Dropped, c.reason)
			}
		})
	}
}

func TestValidateFixedKeepsTheFirstClaimOnly(t *testing.T) {
	idx, sent := fixedFixture()
	v := ValidateFixed([]FixedClaim{
		{"F1", []string{"defer mu.Unlock()"}},
		{"F1", []string{"return load(ctx)"}},
	}, sent, idx)
	if len(v.Kept) != 1 || v.Kept[0].Finding.ID != 1 {
		t.Fatalf("kept = %+v", v.Kept)
	}
	if len(v.Dropped) != 1 || v.Dropped[0].Reason != FixedDuplicate {
		t.Errorf("dropped = %+v, want one duplicate", v.Dropped)
	}
}

// A finding the prompt did not list (trimmed by budget) cannot be retired by
// its id: the model never saw it.
func TestValidateFixedOnlyKnowsWhatWasSent(t *testing.T) {
	idx, sent := fixedFixture()
	v := ValidateFixed([]FixedClaim{{"F1", []string{"defer mu.Unlock()"}}}, sent[1:], idx)
	if len(v.Kept) != 0 || len(v.Dropped) != 1 || v.Dropped[0].Reason != FixedUnknownID {
		t.Errorf("kept %+v dropped %+v, want unknown_id", v.Kept, v.Dropped)
	}
}

func TestParseReviewReadsFixed(t *testing.T) {
	got := ParseReview(`{"overview": "o", "findings": [], "fixed": [
		{"id": " F12 ", "evidence": ["a", "b"]},
		{"id": "F13", "evidence": "one\ntwo"},
		{"evidence": ["no id"]},
		{"id": 14, "evidence": ["numeric id"]},
		{"id": "F15"}
	]}`)
	if len(got.Fixed) != 3 {
		t.Fatalf("fixed = %+v, want 3 (two malformed skipped)", got.Fixed)
	}
	if got.Fixed[0].ID != "F12" || strings.Join(got.Fixed[0].Evidence, "|") != "a|b" {
		t.Errorf("first = %+v", got.Fixed[0])
	}
	if strings.Join(got.Fixed[1].Evidence, "|") != "one|two" {
		t.Errorf("a single string is split like a finding's evidence: %+v", got.Fixed[1])
	}
	if got.Fixed[2].ID != "F15" || got.Fixed[2].Evidence != nil {
		t.Errorf("absent evidence is left for ValidateFixed: %+v", got.Fixed[2])
	}
	skipped := 0
	for _, d := range got.Diagnostics {
		if strings.HasPrefix(d.Message, "Skipping malformed fixed entry") {
			skipped++
		}
	}
	if skipped != 2 {
		t.Errorf("diagnostics = %+v, want 2 skipped fixed entries", got.Diagnostics)
	}
}

// A `fixed` that is not an array is said, not dropped in silence: a run that
// claimed something malformed must not read like one that claimed nothing.
func TestParseReviewWarnsOnAMalformedFixedList(t *testing.T) {
	for _, v := range []string{`null`, `{"id": "F3"}`, `"F3"`} {
		got := ParseReview(`{"overview": "o", "findings": [], "fixed": ` + v + `}`)
		warned := false
		for _, d := range got.Diagnostics {
			warned = warned || strings.HasPrefix(d.Message, "Skipping malformed fixed list")
		}
		if len(got.Fixed) != 0 || !warned {
			t.Errorf("fixed %s: claims %+v, diagnostics %+v", v, got.Fixed, got.Diagnostics)
		}
	}
	if got := ParseReview(`{"overview": "o", "findings": [], "fixed": []}`); len(got.Diagnostics) != 0 {
		t.Errorf("an empty list is no fault: %+v", got.Diagnostics)
	}
}

func TestRenderPreviouslyPostedFindingsCarriesTheID(t *testing.T) {
	line := 3
	got := RenderPreviouslyPostedFindings([]PostedFinding{
		{ID: 12, FilePath: "a.go", LineNumber: &line, Severity: "issue", CommentText: "c"},
		{FilePath: "b.go", Severity: "suggestion", CommentText: "d"},
	})
	if !strings.Contains(got, "\n- [F12] a.go:3 [issue] c") {
		t.Errorf("id missing:\n%s", got)
	}
	if !strings.Contains(got, "\n- b.go [suggestion] d") {
		t.Errorf("a finding without an id renders none:\n%s", got)
	}
}
