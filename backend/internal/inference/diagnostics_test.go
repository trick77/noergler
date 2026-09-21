package inference

import (
	"log/slog"
	"strings"
	"testing"
)

// The review parser produces six diagnostics. ParseReview is a pure
// function, so it returns them and the caller emits them; these pin which
// line fires for which input, and at what level.
//
// The fixed strings are pinned byte for byte. The two that interpolate the
// offending item carry its raw JSON bytes (`{"requirement":null}`): what
// they pin is that the item warns at all, and that its own bytes reach the
// line.

// diagnosticsOf renders one parse's diagnostics as "LEVEL|message" lines.
func diagnosticsOf(content string) []string {
	var out []string
	for _, d := range ParseReview(content).Diagnostics {
		out = append(out, d.Level.String()+"|"+d.Message)
	}
	return out
}

func TestParseDiagnosticsIsPinned(t *testing.T) {
	cases := []struct {
		name    string
		content string
		want    []string
	}{
		// A decode failure and a well-formed non-object are split: only
		// malformed JSON reports the content prefix.
		{
			name:    "malformed json reports the prefix",
			content: "{bad",
			want:    []string{"ERROR|Failed to parse review response as JSON: {bad"},
		},
		{
			name:    "array is not an object",
			content: "[1]",
			want:    []string{"ERROR|Review response is not a JSON object"},
		},
		{
			name:    "scalar is not an object",
			content: `"x"`,
			want:    []string{"ERROR|Review response is not a JSON object"},
		},
		// null decodes without error into a nil map, so only the nil check
		// catches it: a bare null is not an object.
		{
			name:    "null is not an object",
			content: "null",
			want:    []string{"ERROR|Review response is not a JSON object"},
		},

		{
			name:    "missing overview warns",
			content: `{}`,
			want:    []string{"WARN|overview empty after parse"},
		},
		{
			name:    "blank overview warns",
			content: `{"overview": "   "}`,
			want:    []string{"WARN|overview empty after parse"},
		},
		{
			name:    "non-string overview warns",
			content: `{"overview": 42}`,
			want:    []string{"WARN|overview empty after parse"},
		},
		{
			name:    "present overview is silent",
			content: `{"overview": "Adds a thing."}`,
			want:    nil,
		},

		{
			name:    "malformed compliance requirement warns per item",
			content: `{"overview":"x","compliance_requirements":[{"requirement":"r"},42]}`,
			want: []string{
				`WARN|Skipping malformed compliance requirement: {"requirement":"r"}`,
				"WARN|Skipping malformed compliance requirement: 42",
			},
		},
		// A present key holding null fails the type check. Go's
		// json.Unmarshal decodes null as a silent no-op, so a presence probe
		// would accept these and turn {"met":null} into a real "not met".
		{
			name:    "null requirement is malformed",
			content: `{"overview":"x","compliance_requirements":[{"requirement":null,"met":true}]}`,
			want:    []string{`WARN|Skipping malformed compliance requirement: {"requirement":null,"met":true}`},
		},
		{
			name:    "null met is malformed",
			content: `{"overview":"x","compliance_requirements":[{"requirement":"r","met":null}]}`,
			want:    []string{`WARN|Skipping malformed compliance requirement: {"requirement":"r","met":null}`},
		},

		{
			name:    "malformed finding warns",
			content: `{"overview":"x","findings":[42]}`,
			want:    []string{"WARN|Skipping malformed finding: 42"},
		},
		{
			name:    "vacuous suggestion is info, not warn",
			content: `{"overview":"x","findings":[{"file":"a.py","line":1,"severity":"issue","comment":"c","suggestion":"No fix needed"}]}`,
			want: []string{
				`INFO|Dropping no-issue finding (vacuous suggestion): {"file":"a.py","line":1,"severity":"issue","comment":"c","suggestion":"No fix needed"}`,
			},
		},

		// A good response says nothing at all: a diagnostic per review would
		// be noise in the operator's log.
		{
			name:    "a clean response is silent",
			content: `{"overview":"x","findings":[{"file":"a.py","line":1,"severity":"issue","comment":"c"}]}`,
			want:    nil,
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got := diagnosticsOf(tc.content)
			if len(got) != len(tc.want) {
				t.Fatalf("diagnostics = %q, want %q", got, tc.want)
			}
			for i := range got {
				if got[i] != tc.want[i] {
					t.Errorf("diagnostic %d = %q, want %q", i, got[i], tc.want[i])
				}
			}
		})
	}
}

// The warning is only half of it: a nulled field must not reach the summary
// either. {"met":null} accepted as false renders a real ❌ against a
// requirement the model never judged, and {"requirement":null} renders the
// "???" placeholder.
func TestNulledComplianceFieldsAreDropped(t *testing.T) {
	cases := []struct {
		name string
		item string
	}{
		{"null requirement", `{"requirement":null,"met":true}`},
		{"null met", `{"requirement":"r","met":null}`},
		{"missing met", `{"requirement":"r"}`},
		{"missing requirement", `{"met":true}`},
		{"wrong requirement type", `{"requirement":42,"met":true}`},
		{"wrong met type", `{"requirement":"r","met":"yes"}`},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got := ParseReview(`{"overview":"x","compliance_requirements":[` + tc.item + `]}`)
			if len(got.ComplianceRequirements) != 0 {
				t.Errorf("kept %+v, want it skipped", got.ComplianceRequirements)
			}
		})
	}
}

// A well-formed item is still kept, including met:false, which must not be
// confused with an absent or nulled value.
func TestWellFormedComplianceRequirementsSurvive(t *testing.T) {
	got := ParseReview(`{"overview":"x","compliance_requirements":[` +
		`{"requirement":"does X","met":true},{"requirement":"does Y","met":false}]}`)

	if len(got.ComplianceRequirements) != 2 {
		t.Fatalf("kept %+v, want 2", got.ComplianceRequirements)
	}
	if got.ComplianceRequirements[0] != (ComplianceRequirement{Requirement: "does X", Met: true}) {
		t.Errorf("first = %+v", got.ComplianceRequirements[0])
	}
	if got.ComplianceRequirements[1] != (ComplianceRequirement{Requirement: "does Y", Met: false}) {
		t.Errorf("second = %+v", got.ComplianceRequirements[1])
	}
	if len(got.Diagnostics) != 0 {
		t.Errorf("well-formed items warned: %v", got.Diagnostics)
	}
}

// A parse failure returns before the overview check, so it reports the failure
// alone rather than also complaining that the overview is empty.
func TestParseFailureDoesNotAlsoWarnAboutTheOverview(t *testing.T) {
	got := diagnosticsOf("not json")
	if len(got) != 1 {
		t.Fatalf("diagnostics = %q, want exactly one", got)
	}
	if strings.Contains(got[0], "overview") {
		t.Errorf("a parse failure warned about the overview: %q", got[0])
	}
}

// The prefix is capped at 200 characters. A cut at 200 bytes would split a
// multi-byte rune and corrupt the log line.
func TestParseFailurePrefixIsCappedInRunes(t *testing.T) {
	content := strings.Repeat("ü", 300)
	got := diagnosticsOf(content)
	if len(got) != 1 {
		t.Fatalf("diagnostics = %q, want exactly one", got)
	}

	const prefix = "ERROR|Failed to parse review response as JSON: "
	body := strings.TrimPrefix(got[0], prefix)
	if body == got[0] {
		t.Fatalf("unexpected message shape: %q", got[0])
	}
	if n := len([]rune(body)); n != 200 {
		t.Errorf("prefix is %d runes, want 200", n)
	}
	if strings.ContainsRune(body, '�') {
		t.Error("prefix split a multi-byte rune")
	}
}

// Short content is not padded or truncated.
func TestParseFailurePrefixKeepsShortContentWhole(t *testing.T) {
	got := diagnosticsOf("{oops")
	if len(got) != 1 || !strings.HasSuffix(got[0], "{oops") {
		t.Fatalf("diagnostics = %q, want the whole content", got)
	}
}

func TestParseDiagnosticLevels(t *testing.T) {
	// The levels matter: an operator alerting on ERROR must not be paged for
	// a dropped vacuous finding.
	if d := ParseReview("{bad").Diagnostics; d[0].Level != slog.LevelError {
		t.Errorf("parse failure level = %v, want ERROR", d[0].Level)
	}
	if d := ParseReview(`{}`).Diagnostics; d[0].Level != slog.LevelWarn {
		t.Errorf("empty overview level = %v, want WARN", d[0].Level)
	}
	content := `{"overview":"x","findings":[{"file":"a.py","line":1,"severity":"issue","comment":"c","suggestion":"No fix needed"}]}`
	if d := ParseReview(content).Diagnostics; d[0].Level != slog.LevelInfo {
		t.Errorf("vacuous finding level = %v, want INFO", d[0].Level)
	}
}
