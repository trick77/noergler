package diff

import "testing"

// Expectations generated from Python re with \b, which is Unicode-aware. RE2's
// \b is ASCII-only, so symbolBoundaryRE spells the boundary out; these pairs
// pin it to Python's behaviour, including the cases that must NOT match.
func TestSymbolBoundaryMatchesPython(t *testing.T) {
	lines := []string{
		"new Ölservice();",
		"this.café()",
		"x = plain(1)",
		"xÖlservice y",
		"get_user(1)",
		"def get_user_name():",
		"Ölservice",
		"café",
		"(café)",
		"prefixcafé",
		"caféSuffix",
	}
	// want[symbol] lists, per line above, what Python's \b reports.
	want := map[string][]bool{
		"Ölservice": {true, false, false, false, false, false, true, false, false, false, false},
		"café":      {false, true, false, false, false, false, false, true, true, false, false},
		"plain":     {false, false, true, false, false, false, false, false, false, false, false},
		"get_user":  {false, false, false, false, true, false, false, false, false, false, false},
	}

	for symbol, expected := range want {
		re := symbolBoundaryRE(symbol)
		for i, line := range lines {
			if got := re.MatchString(line); got != expected[i] {
				t.Errorf("symbol %q vs %q: got %v, want %v (Python \\b)", symbol, line, got, expected[i])
			}
		}
	}
}

// The bug this replaced: a non-ASCII symbol was extracted but its callers were
// never reported, so the cross-file section silently dropped the relationship.
func TestNonASCIISymbolFindsItsReferences(t *testing.T) {
	files := []FileReviewData{
		{Path: "A.java", Diff: "+public class Ölservice {\n"},
		{Path: "B.java", Content: "void run() {\n  new Ölservice();\n}\n"},
	}
	got := BuildRelationships(files)
	if len(got) != 1 {
		t.Fatalf("got %d relationships, want 1", len(got))
	}
	if got[0].Symbol != "Ölservice" {
		t.Errorf("symbol = %q, want Ölservice", got[0].Symbol)
	}
	if len(got[0].References) != 1 {
		t.Fatalf("got %d references, want 1", len(got[0].References))
	}
	if got[0].References[0].LineNumber != 2 {
		t.Errorf("LineNumber = %d, want 2", got[0].References[0].LineNumber)
	}
}

// A symbol must not match inside a larger identifier.
func TestSymbolBoundaryRejectsSubstrings(t *testing.T) {
	files := []FileReviewData{
		{Path: "svc.py", Diff: "+def get_user(id):\n"},
		{Path: "caller.py", Content: "def get_user_name():\n    pass\n"},
	}
	if got := BuildRelationships(files); len(got) != 0 {
		t.Errorf("got %+v, want no relationships", got)
	}
}
