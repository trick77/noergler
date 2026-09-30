package inference

import (
	"testing"

	"github.com/trick77/noergler/internal/diff"
)

// validateFixture is one file whose diff shows lines 10 to 20:
//
//	10  func load() {
//	11      mu.Lock()
//	12 +    labels := map[string]string{
//	13 +        de: 'Rechnungsadresse',
//	14 +        it: 'Indirizzo di fatturazione',
//	15 +        fr: 'Adresse de facturation',
//	16 +    }
//	17 +    a()
//	18 +    b()
//	19      return
//	20      return
//
// plus one removed line, `defer mu.Unlock()`. Line 19 and 20 read the same,
// so quoting `return` cannot tell them apart.
func validateFixture() diff.AnchorIndex {
	return diff.BuildAnchorIndex([]diff.FileReviewData{{
		Path: "src/labels.ts",
		Diff: "@@ -10,5 +10,11 @@\n" +
			" func load() {\n" +
			"     mu.Lock()\n" +
			"-    defer mu.Unlock()\n" +
			"+    labels := map[string]string{\n" +
			"+        de: 'Rechnungsadresse',\n" +
			"+        it: 'Indirizzo di fatturazione',\n" +
			"+        fr: 'Adresse de facturation',\n" +
			"+    }\n" +
			"+    a()\n" +
			"+    b()\n" +
			"     return\n" +
			"     return\n",
		// Only the first line matters here: it is in the full file but
		// outside the diff.
		Content:        "resp, err := client.Do(req)\n",
		ContentFetched: true,
	}})
}

func sptr(s string) *string { return &s }

func vf(file string, line int, evidence ...string) ReviewFinding {
	return ReviewFinding{File: file, Line: line, Severity: "issue", Comment: "c", Evidence: evidence}
}

func TestValidateFindings(t *testing.T) {
	withSuggestion := func(f ReviewFinding, s string) ReviewFinding {
		f.Suggestion = sptr(s)
		return f
	}
	cases := []struct {
		name     string
		in       ReviewFinding
		reason   DropReason
		wantLine int
	}{
		{"unknown file", vf("src/other.ts", 12, "mu.Lock()"), DropUnknownFile, 0},
		{"a/ prefix still finds the file", vf("a/src/labels.ts", 11, "mu.Lock()"), "", 11},
		{"no evidence", vf("src/labels.ts", 11), DropNoEvidence, 0},
		{"blank evidence only", vf("src/labels.ts", 11, "  ", ""), DropNoEvidence, 0},
		{"quoted code that is not there", vf("src/labels.ts", 11, "mu.Lock()", "db.Close()"), DropEvidenceNotFound, 0},
		{"re-indented quote matches", vf("src/labels.ts", 13, "de:   'Rechnungsadresse',"), "", 13},
		{"copied gutter and marker are tolerated", vf("src/labels.ts", 17, "17 +    a()"), "", 17},
		{"copied diff marker is tolerated", vf("src/labels.ts", 17, "+    a()"), "", 17},
		{"line within slack stays", vf("src/labels.ts", 16, "a()"), "", 16},
		{"line far off moves to unique evidence", vf("src/labels.ts", 40, "b()"), "", 18},
		// Line 21 is past the hunk: Bitbucket would refuse it. Within slack
		// of the quoted line 20, so the finding moves there instead.
		{"near evidence but past the hunk moves onto it", vf("src/labels.ts", 21, "b()", "return"), "", 20},
		{"line far off, evidence ambiguous", vf("src/labels.ts", 40, "return"), DropAnchorMismatch, 0},
		{"removed line is evidence, anchor shown", vf("src/labels.ts", 11, "defer mu.Unlock()"), "", 11},
		{"removed line quoted with its marker", vf("src/labels.ts", 11, "-    defer mu.Unlock()"), "", 11},
		{"removed line only, anchor not shown", vf("src/labels.ts", 3, "defer mu.Unlock()"), DropAnchorMismatch, 0},
		// The resource-leak shape from the evals: the call the bug hangs on
		// is quoted from the full file, the anchor from the diff.
		{"full-file line plus a diff line", vf("src/labels.ts", 11, "resp, err := client.Do(req)", "mu.Lock()"), "", 11},
		// Full-file lines alone tie the finding to nothing the diff changed:
		// without this, line 11 would pass on any unrelated shown line.
		{"full-file line only", vf("src/labels.ts", 11, "resp, err := client.Do(req)"), DropEvidenceOutsideDiff, 0},
		{"full-file line plus a removed line", vf("src/labels.ts", 11, "resp, err := client.Do(req)", "defer mu.Unlock()"), "", 11},
		{
			"suggestion identical to the code there",
			withSuggestion(vf("src/labels.ts", 17, "a()"), "    a()"),
			DropNoopSuggestion, 0,
		},
		{
			// The production false positive this check exists for, with
			// invented content: every key already holds its own language,
			// only the order is unusual, and the "fix" reorders entries.
			"suggestion only reorders key: value entries",
			withSuggestion(vf("src/labels.ts", 12,
				"it: 'Indirizzo di fatturazione',", "fr: 'Adresse de facturation',"),
				"    labels := map[string]string{\n"+
					"        de: 'Rechnungsadresse',\n"+
					"        fr: 'Adresse de facturation',\n"+
					"        it: 'Indirizzo di fatturazione',\n"+
					"    }"),
			DropNoopSuggestion, 0,
		},
		{
			// Statement order can be the bug, so reordering statements is
			// a real suggestion.
			"suggestion reorders statements",
			withSuggestion(vf("src/labels.ts", 17, "a()", "b()"), "    b()\n    a()"),
			"", 17,
		},
		{
			// In Python or YAML the indent is the code. A fix that only
			// dedents a line changes what the program does.
			"suggestion that only re-indents is a real fix",
			withSuggestion(vf("src/labels.ts", 17, "a()"), "a()"),
			"", 17,
		},
		{
			"suggestion that changes the code",
			withSuggestion(vf("src/labels.ts", 17, "a()"), "    a(ctx)"),
			"", 17,
		},
		{
			// The suggestion keeps a() and stops before b(), which the
			// finding quoted: applied, it deletes b(). Not a no-op.
			"suggestion that stops short of a quoted line deletes it",
			withSuggestion(vf("src/labels.ts", 17, "a()", "b()"), "    a()"),
			"", 17,
		},
		{
			// Line 21 was never shown, so nothing can be said about it.
			"suggestion reaching past the shown lines is not judged",
			withSuggestion(vf("src/labels.ts", 20, "return"), "    return\n    x()"),
			"", 20,
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			v := ValidateFindings([]ReviewFinding{tc.in}, validateFixture())
			if tc.reason != "" {
				if len(v.Dropped) != 1 || v.Dropped[0].Reason != tc.reason {
					t.Fatalf("dropped = %+v, want one %s", v.Dropped, tc.reason)
				}
				return
			}
			if len(v.Kept) != 1 {
				t.Fatalf("kept none, dropped %+v", v.Dropped)
			}
			if v.Kept[0].Line != tc.wantLine {
				t.Errorf("line = %d, want %d", v.Kept[0].Line, tc.wantLine)
			}
			if moved := len(v.Reanchored) == 1; moved != (tc.in.Line != tc.wantLine) {
				t.Errorf("reanchored = %+v for %d -> %d", v.Reanchored, tc.in.Line, tc.wantLine)
			}
		})
	}
}

// Order-sensitive lines that look like `key: value` keep a reorder finding:
// swapping two annotated assignments changes which value the second reads.
func TestValidateFindings_AnnotatedAssignmentReorderIsARealFix(t *testing.T) {
	idx := diff.BuildAnchorIndex([]diff.FileReviewData{{
		Path: "calc.py",
		Diff: "@@ -1,2 +1,2 @@\n" +
			"+total: int = base + tax\n" +
			"+tax: int = compute()\n",
	}})
	f := vf("calc.py", 1, "total: int = base + tax")
	f.Suggestion = sptr("tax: int = compute()\ntotal: int = base + tax")
	if v := ValidateFindings([]ReviewFinding{f}, idx); len(v.Kept) != 1 {
		t.Errorf("dropped %+v: reordering statements is a real fix", v.Dropped)
	}
}

// The quote as written is tried everywhere before a stripped form is tried
// anywhere. A removed YAML list item must not match an unrelated shown line
// once its leading `-` is stripped, and so must not move the finding there.
func TestValidateFindings_VerbatimQuoteBeatsStrippedForm(t *testing.T) {
	idx := diff.BuildAnchorIndex([]diff.FileReviewData{{
		Path: "ci.yaml",
		Diff: "@@ -1,4 +1,13 @@\n" +
			" steps:\n" +
			"-- run: make test\n" +
			"+- run: make lint\n" +
			"+  a: 1\n+  b: 2\n+  c: 3\n+  d: 4\n+  e: 5\n+  f: 6\n+  g: 7\n+  h: 8\n+  i: 9\n" +
			"+  run: make test\n",
	}})
	v := ValidateFindings([]ReviewFinding{vf("ci.yaml", 2, "- run: make test")}, idx)
	if len(v.Kept) != 1 || v.Kept[0].Line != 2 {
		t.Errorf("kept = %+v, dropped = %+v: the removed line is the evidence, line 2 stays", v.Kept, v.Dropped)
	}
}

func TestAdjustVerdict(t *testing.T) {
	issue := ReviewFinding{Severity: "issue"}
	sugg := ReviewFinding{Severity: "suggestion"}
	strict := ReviewSummary{VerdictDecision: "request_changes", VerdictRationale: "the dropped bug"}
	cases := []struct {
		name     string
		in       ReviewSummary
		kept     []ReviewFinding
		dropped  int
		decision string
		lowered  bool
	}{
		{"nothing dropped leaves it alone", strict, nil, 0, "request_changes", false},
		{"every finding dropped", strict, nil, 2, "approve", true},
		{"only a suggestion survives", strict, []ReviewFinding{sugg}, 1, "approve_with_followups", true},
		{"an issue survives", strict, []ReviewFinding{issue, sugg}, 1, "request_changes", false},
		{
			// A drop never makes a review stricter.
			"never raises",
			ReviewSummary{VerdictDecision: "approve", VerdictRationale: "fine"},
			[]ReviewFinding{issue}, 1, "approve", false,
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got := AdjustVerdict(tc.in, tc.kept, tc.dropped)
			if got.VerdictDecision != tc.decision {
				t.Errorf("decision = %q, want %q", got.VerdictDecision, tc.decision)
			}
			if lowered := got.VerdictRationale != tc.in.VerdictRationale; lowered != tc.lowered {
				t.Errorf("rationale = %q, lowered = %v, want %v", got.VerdictRationale, lowered, tc.lowered)
			}
		})
	}
}

// Kept findings keep their input order, and one bad finding does not take
// the others down with it.
func TestValidateFindings_KeepsOrderAndIsPerFinding(t *testing.T) {
	v := ValidateFindings([]ReviewFinding{
		vf("src/labels.ts", 18, "b()"),
		vf("src/labels.ts", 11, "nope()"),
		vf("src/labels.ts", 17, "a()"),
	}, validateFixture())
	if len(v.Kept) != 2 || v.Kept[0].Line != 18 || v.Kept[1].Line != 17 {
		t.Errorf("kept = %+v, want lines 18 then 17", v.Kept)
	}
	if len(v.Dropped) != 1 || v.Dropped[0].Finding.Line != 11 {
		t.Errorf("dropped = %+v, want the line-11 finding", v.Dropped)
	}
}

// Only a comma-terminated literal entry reorders freely. Annotated
// assignments, dataclass fields and CSS declarations read `key: value` too,
// and their order matters.
func TestKeyValueRE(t *testing.T) {
	for s, want := range map[string]bool{
		"de: 'x',":                  true,
		`"retries": 3,`:             true,
		"Timeout: 5 * time.Second,": true,
		"'a.b.c': {":                false,
		"total: int = base + tax":   false,
		"name: str":                 false,
		"color: red;":               false,
		"x := 1":                    false,
		"else:":                     false,
		"loop:":                     false,
		"a = 1":                     false,
		"return x":                  false,
	} {
		if got := keyValueRE.MatchString(s); got != want {
			t.Errorf("keyValueRE(%q) = %v, want %v", s, got, want)
		}
	}
}

// The schema asks for an array; a single string is tolerated and split, and
// anything else reads as absent so ValidateFindings drops it with a reason.
func TestParseReview_Evidence(t *testing.T) {
	for name, tc := range map[string]struct {
		raw  string
		want []string
	}{
		"array":           {`["a()", "b()"]`, []string{"a()", "b()"}},
		"multi-line item": {`["a()\nb()", "c()"]`, []string{"a()", "b()", "c()"}},
		"single string":   {`"a()\nb()"`, []string{"a()", "b()"}},
		"absent":          {``, nil},
		"null":            {`null`, nil},
		"wrong type":      {`42`, nil},
	} {
		t.Run(name, func(t *testing.T) {
			ev := ""
			if tc.raw != "" {
				ev = `,"evidence":` + tc.raw
			}
			p := ParseReview(`{"findings":[{"file":"f","line":1,"severity":"issue","comment":"c"` + ev + `}]}`)
			if len(p.Findings) != 1 {
				t.Fatalf("findings = %+v: evidence must never drop a finding at parse time", p.Findings)
			}
			got := p.Findings[0].Evidence
			if len(got) != len(tc.want) {
				t.Fatalf("evidence = %q, want %q", got, tc.want)
			}
			for i := range got {
				if got[i] != tc.want[i] {
					t.Errorf("evidence = %q, want %q", got, tc.want)
				}
			}
		})
	}
}
