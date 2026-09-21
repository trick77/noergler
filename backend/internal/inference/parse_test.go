package inference

import (
	"strings"
	"testing"
)

// The review-response parse is pinned: these are the shapes the model emits.
func TestParseReviewIsPinned(t *testing.T) {
	cases := []struct {
		name        string
		content     string
		wantFailed  bool
		wantReqsNil bool
		wantReqs    int
		overview    string
		strengths   []string
		decision    string
		rationale   string
	}{
		{name: "plain object", content: `{"overview": "ok"}`, overview: "ok", decision: "approve"},
		{name: "fenced json", content: "```json\n{\"overview\": \"ok\"}\n```", overview: "ok", decision: "approve"},
		{name: "fenced no lang", content: "```\n{\"overview\": \"ok\"}\n```", overview: "ok", decision: "approve"},
		// An unclosed fence still parses: only the opening line is dropped.
		{name: "fence unclosed", content: "```json\n{\"overview\": \"ok\"}", overview: "ok", decision: "approve"},

		// Unparseable: requirements are nil, not empty, and the verdict keeps
		// its default.
		{name: "empty", content: "", wantFailed: true, wantReqsNil: true, decision: "approve"},
		{name: "refusal", content: "I'm sorry, but I cannot assist with that request.", wantFailed: true, wantReqsNil: true, decision: "approve"},
		{name: "json array", content: `[1, 2, 3]`, wantFailed: true, wantReqsNil: true, decision: "approve"},
		{name: "json scalar", content: `"hello"`, wantFailed: true, wantReqsNil: true, decision: "approve"},
		{name: "json null", content: `null`, wantFailed: true, wantReqsNil: true, decision: "approve"},

		{name: "overview not string", content: `{"overview": 123}`, overview: "", decision: "approve"},
		{name: "overview padded", content: `{"overview": "  ok  "}`, overview: "ok", decision: "approve"},

		// Non-strings and blanks are dropped; surviving entries keep their
		// original spacing.
		{
			name:      "strengths mixed",
			content:   `{"strengths": ["good", 42, "", "  ", " keep "]}`,
			strengths: []string{"good", " keep "},
			decision:  "approve",
		},
		{name: "strengths not list", content: `{"strengths": "nope"}`, decision: "approve"},

		{name: "compliance good", content: `{"compliance_requirements": [{"requirement": "r", "met": true}]}`, wantReqs: 1, decision: "approve"},
		{name: "compliance missing met", content: `{"compliance_requirements": [{"requirement": "r"}]}`, wantReqs: 0, decision: "approve"},
		{name: "compliance met not bool", content: `{"compliance_requirements": [{"requirement": "r", "met": "yes"}]}`, wantReqs: 0, decision: "approve"},
		{name: "compliance not list", content: `{"compliance_requirements": {"requirement": "r"}}`, wantReqs: 0, decision: "approve"},

		{name: "verdict good", content: `{"verdict": {"decision": "approve", "rationale": " why "}}`, decision: "approve", rationale: "why"},
		// An unrecognised decision keeps the default and the rationale.
		{name: "verdict bad decision", content: `{"verdict": {"decision": "nope", "rationale": "why"}}`, decision: "approve", rationale: "why"},
		{name: "verdict not dict", content: `{"verdict": "approve"}`, decision: "approve"},
		{name: "verdict request_changes", content: `{"verdict": {"decision": "request_changes"}}`, decision: "request_changes"},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got := ParseReview(tc.content)

			if got.ParseFailed != tc.wantFailed {
				t.Errorf("ParseFailed = %v, want %v", got.ParseFailed, tc.wantFailed)
			}
			if tc.wantReqsNil {
				if got.ComplianceRequirements != nil {
					t.Errorf("ComplianceRequirements = %v, want nil sentinel", got.ComplianceRequirements)
				}
			} else {
				if got.ComplianceRequirements == nil {
					t.Fatal("ComplianceRequirements is nil, want non-nil")
				}
				if len(got.ComplianceRequirements) != tc.wantReqs {
					t.Errorf("got %d requirements, want %d", len(got.ComplianceRequirements), tc.wantReqs)
				}
			}
			if got.Summary.Overview != tc.overview {
				t.Errorf("Overview = %q, want %q", got.Summary.Overview, tc.overview)
			}
			if got.Summary.VerdictDecision != tc.decision {
				t.Errorf("VerdictDecision = %q, want %q", got.Summary.VerdictDecision, tc.decision)
			}
			if got.Summary.VerdictRationale != tc.rationale {
				t.Errorf("VerdictRationale = %q, want %q", got.Summary.VerdictRationale, tc.rationale)
			}
			if strings.Join(got.Summary.Strengths, "|") != strings.Join(tc.strengths, "|") {
				t.Errorf("Strengths = %q, want %q", got.Summary.Strengths, tc.strengths)
			}
		})
	}
}

// The nil sentinel and ParseFailed carry different information: timeout and
// 413 paths also emit the sentinel, so only the flag identifies a refusal.
func TestNilRequirementsDistinctFromEmpty(t *testing.T) {
	empty := ParseReview(`{"compliance_requirements": []}`)
	if empty.ComplianceRequirements == nil {
		t.Error("an explicit empty list must not become the nil sentinel")
	}
	if empty.ParseFailed {
		t.Error("an empty list is a successful parse")
	}

	failed := ParseReview("not json")
	if failed.ComplianceRequirements != nil {
		t.Error("an unparseable response must yield the nil sentinel")
	}
	if !failed.ParseFailed {
		t.Error("an unparseable response must set ParseFailed")
	}
}

func TestParseReviewFindings(t *testing.T) {
	// Non-object entries alongside a valid finding are skipped, not fatal. The
	// full validation table lives in finding_test.go.
	t.Run("non-object entries are skipped", func(t *testing.T) {
		got := ParseReview(`{"findings": [{"file": "a.py", "line": 1, "severity": "issue", "comment": "real"}, 42, "nope"]}`)
		if got.ParseFailed {
			t.Error("a malformed finding must not fail the whole parse")
		}
		if len(got.Findings) != 1 || got.Findings[0].Comment != "real" {
			t.Errorf("Findings = %+v, want the one valid finding", got.Findings)
		}
	})

	t.Run("findings not a list", func(t *testing.T) {
		got := ParseReview(`{"findings": "nope"}`)
		if got.ParseFailed || len(got.Findings) != 0 {
			t.Errorf("got %+v, want a clean parse with no findings", got)
		}
	})
}

// The vacuous-suggestion rule is pinned: a change here silently changes
// which findings are dropped.
func TestIsVacuousSuggestionIsPinned(t *testing.T) {
	cases := []struct {
		in   string
		want bool
	}{
		{"No fix needed", true},
		{"no fixes required", true},
		{"No changes needed.", true},
		{"nothing to change", true},
		{"The code is correct", true},
		{"code is actually correct", true},
		{"This is correct", true},
		{"n/a", true},
		{"N/A", true},
		{"na", true},
		{"n a", false},
		{"", false},
		{"   ", false},
		{"Add a nil check", false},
		// Non-breaking spaces must match, which RE2's ASCII \s would not,
		// hence [\s\p{Zs}].
		{"no fix needed", true},
		// ü is a word character, so there is no boundary here and this must
		// NOT match. RE2's ASCII \b would have matched, hence the explicit
		// boundary.
		{"üno fix needed", false},
		// Over 120 characters is a real suggestion regardless of content.
		{strings.Repeat("x", 121) + " no fix needed", false},
		{"no fix needed " + strings.Repeat("x", 100), true},
	}
	for _, tc := range cases {
		if got := IsVacuousSuggestion(tc.in); got != tc.want {
			t.Errorf("IsVacuousSuggestion(%q) = %v, want %v (pinned)", tc.in, got, tc.want)
		}
	}
}

// The length bound counts characters, not bytes: a suggestion of 100 two-byte
// runes is 200 bytes, so a byte-based bound would wrongly reject it as long.
func TestVacuousLengthBoundIsRunes(t *testing.T) {
	// 100 umlauts, then the vacuous phrase: 114 runes (under the bound) but
	// 214 bytes (over it).
	s := strings.Repeat("ü", 100) + " no fix needed"
	if n := len([]rune(s)); n > maxVacuousSuggestionLen {
		t.Fatalf("fixture is %d runes, want under %d", n, maxVacuousSuggestionLen)
	}
	if len(s) <= maxVacuousSuggestionLen {
		t.Fatalf("fixture is %d bytes, want over %d", len(s), maxVacuousSuggestionLen)
	}
	// The phrase is preceded by a space, so the Unicode boundary holds and
	// this matches. A byte-based length bound would have rejected it first.
	if !IsVacuousSuggestion(s) {
		t.Error("a multi-byte suggestion under 120 runes must still be checked")
	}
}

// The mention-response parse is pinned: these are the shapes the model emits.
func TestParseMentionIsPinned(t *testing.T) {
	cases := []struct {
		name, content, want string
	}{
		{"plain envelope", `{"answer": "hi"}`, "hi"},
		{"with refs", `{"answer": "hi", "refs": [{"file": "a.py", "line": 3}]}`, "hi\n\n**References:**\n- `a.py`:3"},
		{"ref no line", `{"answer": "hi", "refs": [{"file": "a.py"}]}`, "hi\n\n**References:**\n- `a.py`"},
		// A non-integer line renders the file alone.
		{"ref line string", `{"answer": "hi", "refs": [{"file": "a.py", "line": "3"}]}`, "hi\n\n**References:**\n- `a.py`"},
		{"ref line float", `{"answer": "hi", "refs": [{"file": "a.py", "line": 3.5}]}`, "hi\n\n**References:**\n- `a.py`"},
		{"ref not dict", `{"answer": "hi", "refs": ["a.py"]}`, "hi"},
		{"refs empty", `{"answer": "hi", "refs": []}`, "hi"},
		{"refs not list", `{"answer": "hi", "refs": "a.py"}`, "hi"},
		// A non-string answer falls back to the raw text.
		{"answer not string", `{"answer": 42}`, `{"answer": 42}`},
		{"no answer key", `{"other": "x"}`, `{"other": "x"}`},
		{"not json", "just text", "just text"},
		// null unmarshals into a nil map in Go rather than failing, so the
		// explicit nil check is what makes it fall back to the raw text.
		{"json null", "null", "null"},
		{"empty", "", ""},
		{"whitespace", "   ", ""},
		{"fenced", "```json\n{\"answer\": \"hi\"}\n```", "hi"},
		{"answer padded", `{"answer": "  hi  "}`, "hi"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if got := ParseMention(tc.content); got != tc.want {
				t.Errorf("ParseMention(%q) = %q, want %q", tc.content, got, tc.want)
			}
		})
	}
}

func TestStripFence(t *testing.T) {
	cases := []struct{ name, in, want string }{
		{"no fence", `{"a": 1}`, `{"a": 1}`},
		{"fenced with lang", "```json\n{\"a\": 1}\n```", `{"a": 1}`},
		{"fenced bare", "```\n{\"a\": 1}\n```", `{"a": 1}`},
		{"unclosed", "```json\n{\"a\": 1}", `{"a": 1}`},
		// The closing fence is dropped only when the line is exactly a fence.
		{"trailing fence with text", "```json\n{\"a\": 1}\n``` trailing", "{\"a\": 1}\n``` trailing"},
		{"fence only", "```", ""},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if got := stripFence(tc.in); got != tc.want {
				t.Errorf("stripFence(%q) = %q, want %q", tc.in, got, tc.want)
			}
		})
	}
}
