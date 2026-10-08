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

// The a/ or b/ fallback rewrites File to the shown path, so the poster can
// send File verbatim. A real top-level b/ directory keeps its name.
func TestValidateFindings_SidePrefixResolvesToTheShownPath(t *testing.T) {
	idx := validateFixture()
	if got := ValidateFindings([]ReviewFinding{vf("a/src/labels.ts", 11, "mu.Lock()")}, idx).Kept; len(got) != 1 || got[0].File != "src/labels.ts" {
		t.Errorf("kept = %+v, want File src/labels.ts", got)
	}

	idx = diff.BuildAnchorIndex([]diff.FileReviewData{{Path: "b/main.go", Diff: "@@ -1,1 +1,1 @@\n+x := 1\n"}})
	if got := ValidateFindings([]ReviewFinding{vf("b/main.go", 1, "x := 1")}, idx).Kept; len(got) != 1 || got[0].File != "b/main.go" {
		t.Errorf("kept = %+v, want File b/main.go", got)
	}
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
		// From the evals: a correct resource-leak finding added its own note
		// between quoted lines. A note or an elision is not code; skipped.
		{"model's own comment line is skipped", vf("src/labels.ts", 11, "mu.Lock()", "// defer is missing here"), "", 11},
		{"elision marker is skipped", vf("src/labels.ts", 17, "a()", "...", "b()"), "", 17},
		{"annotation alone is no evidence", vf("src/labels.ts", 11, "// note", "…"), DropNoEvidence, 0},
		// Code that merely starts like a comment or an elision is still
		// checked: an invented one drops the finding.
		{"invented pointer write is not a note", vf("src/labels.ts", 11, "mu.Lock()", "*cfg = Config{}"), DropEvidenceNotFound, 0},
		{"invented decrement is not a note", vf("src/labels.ts", 11, "mu.Lock()", "--count;"), DropEvidenceNotFound, 0},
		{"invented attribute is not a note", vf("src/labels.ts", 11, "mu.Lock()", "#[derive(Debug)]"), DropEvidenceNotFound, 0},
		{"invented spread is not a note", vf("src/labels.ts", 11, "mu.Lock()", "...defaults,"), DropEvidenceNotFound, 0},
		{"code after a block comment is not a note", vf("src/labels.ts", 11, "mu.Lock()", "/* fallthrough */ return nil"), DropEvidenceNotFound, 0},
		{"block comment alone is a note", vf("src/labels.ts", 11, "mu.Lock()", "/* body never closed */"), "", 11},
		{"spaced predecrement is not a note", vf("src/labels.ts", 11, "mu.Lock()", "-- count;"), DropEvidenceNotFound, 0},
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

// The cross-file shape from the evals: store.go changes Touch's unit, and the
// finding on manager.go quotes the changed signature beside the stale call.
// A line of another shown file is evidence, in whichever of its three places
// it sits, but it never stands in for a diff line of the finding's own file.
func TestValidateFindings_EvidenceFromAnotherShownFile(t *testing.T) {
	idx := diff.BuildAnchorIndex([]diff.FileReviewData{
		{
			Path: "session/store.go",
			Diff: "@@ -1,3 +1,3 @@\n" +
				"-func (s *Store) Touch(id string, ttlSeconds int64) error {\n" +
				"+func (s *Store) Touch(id string, ttlMillis int64) error {\n" +
				" \treturn s.rdb.PExpire(id, ttlMillis)\n" +
				" }\n",
			Content:        "var ErrNotFound = errors.New(\"session: not found\")\n",
			ContentFetched: true,
		},
		{
			Path: "session/manager.go",
			Diff: "@@ -7,2 +7,3 @@\n" +
				" func (m *Manager) Hold(id string, d time.Duration) error {\n" +
				"+\treturn m.store.Touch(id, int64(d.Seconds()))\n" +
				" }\n",
		},
	})
	call := "return m.store.Touch(id, int64(d.Seconds()))"
	cases := []struct {
		name     string
		evidence []string
		reason   DropReason
	}{
		{"added line of the other file", []string{call, "func (s *Store) Touch(id string, ttlMillis int64) error {"}, ""},
		{"removed line of the other file", []string{call, "func (s *Store) Touch(id string, ttlSeconds int64) error {"}, ""},
		{"full-content line of the other file", []string{call, "var ErrNotFound = errors.New(\"session: not found\")"}, ""},
		{"a line in no shown file still drops", []string{call, "if err := s.rdb.PExpire(id, ttlMillis); err != nil {"}, DropEvidenceNotFound},
		{"other file's lines alone tie to nothing here", []string{"func (s *Store) Touch(id string, ttlMillis int64) error {"}, DropEvidenceOutsideDiff},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			v := ValidateFindings([]ReviewFinding{vf("session/manager.go", 8, tc.evidence...)}, idx)
			if tc.reason != "" {
				if len(v.Dropped) != 1 || v.Dropped[0].Reason != tc.reason {
					t.Fatalf("dropped = %+v, want one %s", v.Dropped, tc.reason)
				}
				return
			}
			if len(v.Kept) != 1 || v.Kept[0].Line != 8 || len(v.Reanchored) != 0 {
				t.Fatalf("kept = %+v, dropped = %+v, reanchored = %+v: want line 8 kept unmoved", v.Kept, v.Dropped, v.Reanchored)
			}
		})
	}
}

// From the evals: a correct finding quoted a doc comment line without its
// `//` and was dropped whole. A quote that is a shown comment line minus its
// marker is that line. Code that merely starts with a marker character is
// not a comment, so its tail alone is still an invented line.
func TestValidateFindings_CommentLineQuotedWithoutItsMarker(t *testing.T) {
	idx := diff.BuildAnchorIndex([]diff.FileReviewData{
		{
			Path: "notify/dispatcher.go",
			Diff: "@@ -1,4 +1,13 @@\n" +
				" // Deliver posts the event. A transient failure is retried with\n" +
				"-// backoff, a permanent one is dropped.\n" +
				"+// backoff, a permanent one is returned at once.\n" +
				"+if retryable(lastErr) {\n" +
				"+\t*cfg = Config{}\n" +
				"+\t--count;\n" +
				"+\t#include <stdio.h>\n" +
				"+\t# retried by the caller\n" +
				"+\t * @param attempts upper bound\n" +
				"+\t-- newest first\n" +
				"+\t//nolint:errcheck\n" +
				"+\t// return nil\n" +
				" }\n",
			Content:        "// ErrPermanent marks a rejection.\n",
			ContentFetched: true,
		},
		{
			Path: "notify/policy.go",
			Diff: "@@ -1,1 +1,2 @@\n" +
				"+// Policy decides what is retried.\n" +
				" type Policy struct{}\n",
		},
	})
	const code = "if retryable(lastErr) {"
	cases := []struct {
		name     string
		line     int
		evidence []string
		reason   DropReason
	}{
		{"doc comment line beside a code line", 3, []string{code, "backoff, a permanent one is returned at once."}, ""},
		{"comment line alone, anchored on it", 2, []string{"backoff, a permanent one is returned at once."}, ""},
		{"removed comment line", 3, []string{code, "backoff, a permanent one is dropped."}, ""},
		{"full-file comment line", 3, []string{code, "ErrPermanent marks a rejection."}, ""},
		{"comment line of another shown file", 3, []string{code, "Policy decides what is retried."}, ""},
		{"hash comment", 7, []string{"retried by the caller"}, ""},
		{"block comment continuation", 8, []string{"@param attempts upper bound"}, ""},
		{"dash comment", 9, []string{"newest first"}, ""},
		{"comment with no space after the slashes", 10, []string{"nolint:errcheck"}, ""},
		// Known limit, pinned: the marker rule cannot tell a comment from
		// commented-out code. The text is in the file either way.
		{"commented-out code quoted as code passes", 11, []string{"return nil"}, ""},
		{"comment text that is in no file", 3, []string{code, "backoff, a permanent one is retried."}, DropEvidenceNotFound},
		{"tail of a pointer write is not a comment", 3, []string{code, "cfg = Config{}"}, DropEvidenceNotFound},
		{"tail of a decrement is not a comment", 3, []string{code, "count;"}, DropEvidenceNotFound},
		{"tail of a directive is not a comment", 3, []string{code, "include <stdio.h>"}, DropEvidenceNotFound},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			v := ValidateFindings([]ReviewFinding{vf("notify/dispatcher.go", tc.line, tc.evidence...)}, idx)
			if tc.reason != "" {
				if len(v.Dropped) != 1 || v.Dropped[0].Reason != tc.reason {
					t.Fatalf("dropped = %+v, want one %s", v.Dropped, tc.reason)
				}
				return
			}
			if len(v.Kept) != 1 || v.Kept[0].Line != tc.line || len(v.Reanchored) != 0 {
				t.Fatalf("kept = %+v, dropped = %+v, reanchored = %+v: want line %d kept unmoved", v.Kept, v.Dropped, v.Reanchored, tc.line)
			}
		})
	}
}

// An incremental review: an earlier push changed Touch, this push only adds
// the stale call. The signature is shown in the cumulative PR diff alone.
// Quotable, but a file shown only there is still not a file to post on.
func TestValidateFindings_EvidenceFromTheContextDiff(t *testing.T) {
	idx := diff.BuildAnchorIndex([]diff.FileReviewData{{
		Path: "session/manager.go",
		Diff: "@@ -7,2 +7,3 @@\n" +
			" func (m *Manager) Hold(id string, d time.Duration) error {\n" +
			"+\treturn m.store.Touch(id, int64(d.Seconds()))\n" +
			" }\n",
	}})
	idx.AddContextDiff("diff --git a/session/store.go b/session/store.go\n" +
		"@@ -1,3 +1,3 @@\n" +
		"-func (s *Store) Touch(id string, ttlSeconds int64) error {\n" +
		"+func (s *Store) Touch(id string, ttlMillis int64) error {\n" +
		" \treturn s.rdb.PExpire(id, ttlMillis)\n")
	call := "return m.store.Touch(id, int64(d.Seconds()))"
	sig := "func (s *Store) Touch(id string, ttlMillis int64) error {"

	v := ValidateFindings([]ReviewFinding{vf("session/manager.go", 8, call, sig)}, idx)
	if len(v.Kept) != 1 || v.Kept[0].Line != 8 {
		t.Errorf("kept = %+v, dropped = %+v: the context diff's line is evidence", v.Kept, v.Dropped)
	}
	v = ValidateFindings([]ReviewFinding{vf("session/manager.go", 8, sig)}, idx)
	if len(v.Dropped) != 1 || v.Dropped[0].Reason != DropEvidenceOutsideDiff {
		t.Errorf("dropped = %+v, want evidence_outside_diff: no diff line of the file quoted", v.Dropped)
	}
	v = ValidateFindings([]ReviewFinding{vf("session/store.go", 1, sig)}, idx)
	if len(v.Dropped) != 1 || v.Dropped[0].Reason != DropUnknownFile {
		t.Errorf("dropped = %+v, want unknown_file: store.go is context, not a reviewed file", v.Dropped)
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

// A moved finding loses its suggestion: Bitbucket's Apply replaces the line
// the comment sits on, and the fix was written for the line the model cited.
// An unmoved one keeps it.
func TestValidateFindings_MovedFindingDropsItsSuggestion(t *testing.T) {
	moved := vf("src/labels.ts", 40, "b()")
	moved.Suggestion = sptr("    b(ctx)")
	stays := vf("src/labels.ts", 18, "b()")
	stays.Suggestion = sptr("    b(ctx)")
	v := ValidateFindings([]ReviewFinding{moved, stays}, validateFixture())
	if len(v.Kept) != 2 {
		t.Fatalf("kept = %+v, dropped = %+v", v.Kept, v.Dropped)
	}
	if v.Kept[0].Line != 18 || v.Kept[0].Suggestion != nil {
		t.Errorf("moved finding: line %d, suggestion %v; want 18 and none", v.Kept[0].Line, v.Kept[0].Suggestion)
	}
	if v.Kept[1].Suggestion == nil {
		t.Error("an unmoved finding keeps its suggestion")
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
