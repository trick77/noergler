package inference

import (
	"encoding/json"
	"testing"
)

// The wire contract for a finding: file, line, severity and comment are
// required, severity is one of "issue" or "suggestion", and a validation
// failure skips that finding rather than failing the batch.
func TestParseFindingIsPinned(t *testing.T) {
	cases := []struct {
		name string
		item string
		keep bool
	}{
		{"valid issue", `{"file":"a.py","line":1,"severity":"issue","comment":"c"}`, true},
		{"valid suggestion", `{"file":"a.py","line":1,"severity":"suggestion","comment":"c"}`, true},
		// Belt-and-braces past the JSON-schema enum.
		{"unknown severity", `{"file":"a.py","line":1,"severity":"critical","comment":"c"}`, false},
		{"missing file", `{"line":1,"severity":"issue","comment":"c"}`, false},
		{"missing line", `{"file":"a.py","severity":"issue","comment":"c"}`, false},
		{"missing severity", `{"file":"a.py","line":1,"comment":"c"}`, false},
		{"missing comment", `{"file":"a.py","line":1,"severity":"issue"}`, false},
		// The wire contract coerces a numeric string.
		{"line as string", `{"file":"a.py","line":"1","severity":"issue","comment":"c"}`, true},
		// But not a float with a fractional part, which would move the finding.
		{"line as float", `{"file":"a.py","line":1.5,"severity":"issue","comment":"c"}`, false},
		{"extra field ignored", `{"file":"a.py","line":1,"severity":"issue","comment":"c","bogus":1}`, true},
		{"not an object", `42`, false},
		{"string item", `"nope"`, false},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			_, ok := parseFinding(json.RawMessage(tc.item))
			if ok != tc.keep {
				t.Errorf("kept = %v, want %v (pinned)", ok, tc.keep)
			}
		})
	}
}

func TestParseFindingCoercedLine(t *testing.T) {
	got, ok := parseFinding(json.RawMessage(`{"file":"a.py","line":"42","severity":"issue","comment":"c"}`))
	if !ok {
		t.Fatal("a numeric string line must coerce")
	}
	if got.Line != 42 {
		t.Errorf("Line = %d, want 42", got.Line)
	}
}

// The optional fields are absent rather than empty when the model omits them.
func TestParseFindingOptionals(t *testing.T) {
	t.Run("absent", func(t *testing.T) {
		got, ok := parseFinding(json.RawMessage(`{"file":"a.py","line":1,"severity":"issue","comment":"c"}`))
		if !ok {
			t.Fatal("want kept")
		}
		if got.Confidence != nil || got.Headline != nil || got.Suggestion != nil {
			t.Errorf("optionals should be nil: %+v", got)
		}
	})

	t.Run("present", func(t *testing.T) {
		got, ok := parseFinding(json.RawMessage(
			`{"file":"a.py","line":1,"severity":"issue","comment":"c","confidence":80,"headline":"h","suggestion":"s"}`))
		if !ok {
			t.Fatal("want kept")
		}
		if got.Confidence == nil || *got.Confidence != 80 {
			t.Errorf("Confidence = %v, want 80", got.Confidence)
		}
		if got.Headline == nil || *got.Headline != "h" {
			t.Errorf("Headline = %v, want h", got.Headline)
		}
		if got.Suggestion == nil || *got.Suggestion != "s" {
			t.Errorf("Suggestion = %v, want s", got.Suggestion)
		}
	})
}

// One malformed finding must not drop the valid ones alongside it.
func TestMalformedFindingSkippedNotFatal(t *testing.T) {
	content := `{"findings":[` +
		`{"file":"a.py","line":1,"severity":"critical","comment":"stale"},` +
		`{"file":"b.py","line":2,"severity":"issue","comment":"ok"}` +
		`]}`
	got := ParseReview(content)
	if got.ParseFailed {
		t.Fatal("a malformed finding must not fail the parse")
	}
	if len(got.Findings) != 1 {
		t.Fatalf("got %d findings, want 1", len(got.Findings))
	}
	if got.Findings[0].File != "b.py" {
		t.Errorf("kept %q, want b.py", got.Findings[0].File)
	}
}

// A vacuous suggestion drops the finding; a real one keeps it.
func TestVacuousSuggestionDropsFinding(t *testing.T) {
	vacuous := ParseReview(`{"findings":[{"file":"a.py","line":1,"severity":"issue","comment":"c","suggestion":"No fix needed"}]}`)
	if len(vacuous.Findings) != 0 {
		t.Errorf("want the finding dropped, got %+v", vacuous.Findings)
	}

	real := ParseReview(`{"findings":[{"file":"a.py","line":1,"severity":"issue","comment":"c","suggestion":"Add a nil check"}]}`)
	if len(real.Findings) != 1 {
		t.Errorf("want the finding kept, got %+v", real.Findings)
	}

	// An absent suggestion is not vacuous.
	none := ParseReview(`{"findings":[{"file":"a.py","line":1,"severity":"issue","comment":"c"}]}`)
	if len(none.Findings) != 1 {
		t.Errorf("want the finding kept, got %+v", none.Findings)
	}
}
