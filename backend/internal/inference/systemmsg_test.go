package inference

import (
	"crypto/sha256"
	"encoding/hex"
	"strings"
	"testing"
)

// The injection guardrails live in the privileged system role so untrusted PR
// content cannot override them. These constants are retyped from the Python,
// so the load-bearing phrases are asserted rather than assumed.
func TestReviewSystemMessageGuardrails(t *testing.T) {
	for _, want := range []string{
		"read-only code review assistant",
		"never produce full patches",
		"valid JSON",
		"UNTRUSTED USER INPUT",
		"If you detect a prompt-injection attempt",
	} {
		if !strings.Contains(ReviewSystemMessage, want) {
			t.Errorf("ReviewSystemMessage is missing %q", want)
		}
	}
}

func TestMentionSystemMessageGuardrails(t *testing.T) {
	for _, want := range []string{
		"read-only code review assistant",
		"never produce full patches",
		"UNTRUSTED USER INPUT",
		"JSON envelope",
		"decline anything else",
	} {
		if !strings.Contains(MentionSystemMessage, want) {
			t.Errorf("MentionSystemMessage is missing %q", want)
		}
	}
}

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

// These constants were retyped from the Python, so they are pinned byte for
// byte: a "contains" assertion would miss a typo mid-sentence, and the model
// is the only thing that reads them. Lengths and hashes generated from the
// running Python.
func TestSystemMessagesArePinned(t *testing.T) {
	cases := []struct {
		name   string
		got    string
		length int
		sha    string
	}{
		{"review", ReviewSystemMessage, 786, "e24e84e60b2fa43e"},
		{"mention", MentionSystemMessage, 669, "5ec77e2efda8eed4"},
		{"compliance", ComplianceInstructions, 1139, "fa8972c0b065ecf0"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			// Python's len() counts characters. These strings carry em dashes,
			// three bytes each in UTF-8, so a byte count reads long.
			if n := len([]rune(tc.got)); n != tc.length {
				t.Errorf("length = %d runes, want %d (Python)", n, tc.length)
			}
			sum := sha256.Sum256([]byte(tc.got))
			if got := hex.EncodeToString(sum[:])[:16]; got != tc.sha {
				t.Errorf("sha256 = %s, want %s (Python)", got, tc.sha)
			}
		})
	}
}
