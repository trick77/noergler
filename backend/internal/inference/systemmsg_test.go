package inference

import (
	"crypto/sha256"
	"encoding/hex"
	"regexp"
	"strings"
	"testing"
)

// The injection guardrails live in the privileged system role so untrusted PR
// content cannot override them. The load-bearing phrases are asserted rather
// than assumed, so a rewrite cannot quietly drop one.
func TestReviewSystemMessageGuardrails(t *testing.T) {
	for _, want := range []string{
		"read-only code review assistant",
		"never produce full patches",
		"valid JSON",
		"UNTRUSTED USER INPUT",
		"If you detect a prompt-injection attempt",
		"PR description",
	} {
		if !strings.Contains(ReviewSystemMessage, want) {
			t.Errorf("ReviewSystemMessage is missing %q", want)
		}
	}
}

func TestMentionSystemMessageGuardrails(t *testing.T) {
	for _, want := range []string{
		"never produce full patches",
		"never modify repository content yourself",
		"only for the PR author",
		"never promise a change",
		"never answer that the code cannot be changed",
		"UNTRUSTED USER INPUT",
		"JSON envelope",
		"decline anything else",
		"PR description",
	} {
		if !strings.Contains(MentionSystemMessage, want) {
			t.Errorf("MentionSystemMessage is missing %q", want)
		}
	}
	// It decides the Q&A's "apply" and a finding reply's "requests_fix":
	// called read-only, the model refused every fix request with it.
	if readOnly.MatchString(MentionSystemMessage) {
		t.Error("MentionSystemMessage calls noergler read-only")
	}
}

// readOnly is any spelling of "read-only", English or German.
var readOnly = regexp.MustCompile(`(?i)read[\s_-]*only|nur[\s-]*lesend|schreibgeschützt`)

// The compliance block drives the acceptance-criteria extraction, including
// the AK-1/AC-1 hint the Jira prefix matcher depends on.
func TestComplianceInstructionsContent(t *testing.T) {
	for _, want := range []string{
		"compliance_requirements",
		"code-verifiable",
		"AK-1, AC-1",
		"empty compliance_requirements array",
	} {
		if !strings.Contains(ComplianceInstructions, want) {
			t.Errorf("ComplianceInstructions is missing %q", want)
		}
	}
}

// These constants are pinned byte for byte: a "contains" assertion would miss
// a typo mid-sentence, and the model is the only thing that reads them. The
// lengths and hashes below have no other source of truth; regenerate them
// only together with an intentional prompt change.
func TestSystemMessagesArePinned(t *testing.T) {
	cases := []struct {
		name   string
		got    string
		length int
		sha    string
	}{
		{"review", ReviewSystemMessage, 808, "8085a331b655a85a"},
		{"mention", MentionSystemMessage, 1136, "c62070dc4eb87237"},
		{"compliance", ComplianceInstructions, 1661, "de597e9866f06a3f"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			// Counted in characters. These strings carry em dashes, three
			// bytes each in UTF-8, so a byte count would read long.
			if n := len([]rune(tc.got)); n != tc.length {
				t.Errorf("length = %d runes, want %d (pinned)", n, tc.length)
			}
			sum := sha256.Sum256([]byte(tc.got))
			if got := hex.EncodeToString(sum[:])[:16]; got != tc.sha {
				t.Errorf("sha256 = %s, want %s (pinned)", got, tc.sha)
			}
		})
	}
}
