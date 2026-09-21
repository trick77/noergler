package render

import (
	"encoding/json"
	"os"
	"testing"
)

// Expected values generated from the Python _SECURITY_KEYWORDS with the venv.
// Both directions of the RE2 divergence are covered: "üinsecure" and
// "insecureü" are false in Python (no word boundary next to a letter) where a
// naive \b port says true, and "1insecure" is false for the same reason.
func TestSecurityKeywordsIsPinned(t *testing.T) {
	blob, err := os.ReadFile("testdata/security_golden.json")
	if err != nil {
		t.Fatalf("read security golden: %v", err)
	}
	var cases map[string]bool
	if err := json.Unmarshal(blob, &cases); err != nil {
		t.Fatalf("decode security golden: %v", err)
	}
	if len(cases) < 50 {
		t.Fatalf("expected at least 50 cases, got %d", len(cases))
	}
	for in, want := range cases {
		if got := IsSecurityFinding(in); got != want {
			t.Errorf("IsSecurityFinding(%q) = %v, want %v", in, got, want)
		}
	}
}

// Called out separately because they are the whole reason the pattern is not
// a transcription of the Python source.
func TestSecurityKeywordsUnicodeBoundary(t *testing.T) {
	for _, in := range []string{"üinsecure", "insecureü", "1insecure"} {
		if IsSecurityFinding(in) {
			t.Errorf("%q must not match: Python's \\b is Unicode and finds no boundary there", in)
		}
	}
	for _, in := range []string{"a-insecure", "(insecure)", "insecure!", "The code is insecure."} {
		if !IsSecurityFinding(in) {
			t.Errorf("%q must match", in)
		}
	}
}

// Parity quirks worth keeping visible: the optional-character class means
// "secretleak" matches while "secrets leak" does not.
func TestSecurityKeywordsQuirks(t *testing.T) {
	if !IsSecurityFinding("secretleak") {
		t.Error(`"secretleak" should match: the class is secret[s ]?leak`)
	}
	if IsSecurityFinding("secrets leak") {
		t.Error(`"secrets leak" should not match: [s ]? is ONE optional character`)
	}
	if IsSecurityFinding("vulnerabilities") {
		t.Error(`"vulnerabilities" should not match: the alternation has no plural`)
	}
}
