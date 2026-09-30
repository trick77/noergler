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
		{"line far off, evidence ambiguous", vf("src/labels.ts", 40, "return"), DropAnchorMismatch, 0},
		{"removed line is evidence, anchor shown", vf("src/labels.ts", 11, "defer mu.Unlock()"), "", 11},
		{"removed line quoted with its marker", vf("src/labels.ts", 11, "-    defer mu.Unlock()"), "", 11},
		{"removed line only, anchor not shown", vf("src/labels.ts", 3, "defer mu.Unlock()"), DropAnchorMismatch, 0},
		// The resource-leak shape from the evals: the call the bug hangs on
		// is quoted from the full file, the anchor from the diff.
		{"full-file line plus a diff line", vf("src/labels.ts", 11, "resp, err := client.Do(req)", "mu.Lock()"), "", 11},
		{"full-file line only, anchor shown", vf("src/labels.ts", 11, "resp, err := client.Do(req)"), "", 11},
		{"full-file line only, anchor not shown", vf("src/labels.ts", 3, "resp, err := client.Do(req)"), DropAnchorMismatch, 0},
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
			"suggestion that changes the code",
			withSuggestion(vf("src/labels.ts", 17, "a()"), "    a(ctx)"),
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

// `:=`, `else:` and a label are not `key: value` entries.
func TestKeyValueRE(t *testing.T) {
	for s, want := range map[string]bool{
		"de: 'x',":                  true,
		`"retries": 3,`:             true,
		"Timeout: 5 * time.Second,": true,
		"'a.b.c': {":                true,
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
		"array":         {`["a()", "b()"]`, []string{"a()", "b()"}},
		"single string": {`"a()\nb()"`, []string{"a()", "b()"}},
		"absent":        {``, nil},
		"null":          {`null`, nil},
		"wrong type":    {`42`, nil},
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
