package render

import (
	"encoding/json"
	"os"
	"testing"
)

// Expected values are the golden corpus in testdata, compared case by case.
// Both directions of the Unicode-boundary rule are covered: "üinsecure" and
// "insecureü" are false (no word boundary next to a letter) where a naive
// RE2 \b says true, and "1insecure" is false for the same reason.
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

// Called out separately because they are the whole reason the pattern spells
// the word boundary out instead of using \b.
func TestSecurityKeywordsUnicodeBoundary(t *testing.T) {
	for _, in := range []string{"üinsecure", "insecureü", "1insecure"} {
		if IsSecurityFinding(in) {
			t.Errorf("%q must not match: a Unicode word character leaves no boundary there", in)
		}
	}
	for _, in := range []string{"a-insecure", "(insecure)", "insecure!", "The code is insecure."} {
		if !IsSecurityFinding(in) {
			t.Errorf("%q must match", in)
		}
	}
}

// Quirks worth keeping visible: the optional-character class means
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
