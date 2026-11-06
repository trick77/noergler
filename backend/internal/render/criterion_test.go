package render

import (
	"slices"
	"testing"
)

func TestLabelRequirement(t *testing.T) {
	labels := criterionLabels("AK-1: Login rejects expired tokens\nAK-2 Refresh issues a new token\n" +
		"AK3 Logout clears the session\nAK-1.2 Sub")
	for _, tc := range []struct{ name, in, want string }{
		{"colon", "AK-1: Login rejects expired tokens", "`AK-1` Login rejects expired tokens"},
		{"dash", "AK-2 - Refresh issues a new token", "`AK-2` Refresh issues a new token"},
		{"em dash", "AK-2 — Refresh issues a new token", "`AK-2` Refresh issues a new token"},
		{"bare", "AK-1 Login rejects expired tokens", "`AK-1` Login rejects expired tokens"},
		{"no-break space", "AK-1 : Login", "`AK-1` Login"},
		{"leading list", "AK-1, AK-2: Tokens rotate", "`AK-1` `AK-2` Tokens rotate"},
		{"parens first", "(AK-1) Login rejects expired tokens", "`AK-1` Login rejects expired tokens"},
		{"brackets first", "[AK-1] Login rejects expired tokens", "`AK-1` Login rejects expired tokens"},
		{"two in parens first", "(AK-1 and AK-2) tokens rotate", "`AK-1` `AK-2` tokens rotate"},
		{"already spanned", "`AK-1` Login rejects expired tokens", "`AK-1` Login rejects expired tokens"},
		{"trailing parens", "Logout clears the session (AK-3)", "`AK-3` Logout clears the session"},
		{"trailing full stop", "Logout clears the session (AK-3).", "`AK-3` Logout clears the session."},
		{"trailing dash", "Logout clears the session - AK-3", "`AK-3` Logout clears the session"},
		{"two trailing", "Tokens rotate (AK-1, AK-2)", "`AK-1` `AK-2` Tokens rotate"},
		{"two trailing joined", "Tokens rotate (AK-1 und AK-2)", "`AK-1` `AK-2` Tokens rotate"},
		{"both ends, once", "AK-1: Login rejects expired tokens (AK-1)", "`AK-1` Login rejects expired tokens"},
		{"sub-numbered", "AK-1.2: Sub works", "`AK-1.2` Sub works"},
		{"sub-numbered inside", "Covers AK-1.2 too", "Covers `AK-1.2` too"},
		{"sub-number of a plain label", "Covers AK-3.4 edge case", "Covers `AK-3.4` edge case"},
		{"range", "AK-1–AK-3: all done", "`AK-1–AK-3` all done"},
		{"short range", "AK-1-2: x", "`AK-1-2` x"},
		{"trailing exclamation", "Login works (AK-1)!", "`AK-1` Login works!"},
		{"url kept", "See https://jira/browse/AK-1 for details", "See https://jira/browse/AK-1 for details"},
		{"markdown link kept", "[AK-1](https://jira/browse/AK-1) works", "[`AK-1`](https://jira/browse/AK-1) works"},
		{"label keeps the dot after it", "AK-1: .env files are ignored", "`AK-1` .env files are ignored"},
		{"possessive", "AK-1's check passes", "`AK-1`'s check passes"},
		// Spanned in place: the label is part of the sentence.
		{"mid sentence", "Covers AK-1 and the refresh path", "Covers `AK-1` and the refresh path"},
		{"sentence ends on it", "Login behaves per AK-1", "Login behaves per `AK-1`"},
		{"german sentence ends on it", "Umgesetzt gemäss AK-1", "Umgesetzt gemäss `AK-1`"},
		{"parens with words", "Logout clears the session (see AK-3)", "Logout clears the session (see `AK-3`)"},
		{"inside a code span", "Implemented in `cfg AK-1 x`", "Implemented in `cfg AK-1 x`"},
		{"label only", "AK-1", "`AK-1`"},
		{"ticket label without hyphen", "AK-3: Logout clears the session", "`AK-3` Logout clears the session"},
		// No label: the line comes back as it came.
		{"leading dot", ".env files are ignored", ".env files are ignored"},
		{"leading flag", "--force overwrites files", "--force overwrites files"},
		{"leading dash", "- leading dash", "- leading dash"},
		{"trailing colon", "Login works:", "Login works:"},
		{"leading parens", "(optional) Login works", "(optional) Login works"},
		{"real key kept", "PROJ-123: Login rejects expired tokens", "PROJ-123: Login rejects expired tokens"},
		{"longer key kept", "XAK-1 and AK-12 are other issues", "XAK-1 and AK-12 are other issues"},
		{"no label", "Login rejects expired tokens", "Login rejects expired tokens"},
		{"umlaut after", "AK-1ä stays", "AK-1ä stays"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			if got := labelRequirement(tc.in, labels); got != tc.want {
				t.Errorf("labelRequirement(%q) = %q, want %q", tc.in, got, tc.want)
			}
		})
	}
}

// The labels come from the ticket: with no labelled criteria nothing is
// touched, and a parent's criteria count, where a subtask's often live.
func TestCriterionLabels(t *testing.T) {
	for _, tc := range []struct {
		name string
		ac   []string
		want []string
	}{
		{"none", []string{"Login rejects expired tokens"}, nil},
		{"parent", []string{"", "AK-1 Login rejects expired tokens"}, []string{"AK-1"}},
		{"word prefix", []string{"AKzeptanz: Login rejects expired tokens"}, nil},
		{"spellings", []string{"AK1 a\nAK 2 b\nak-3 c\nAK-1 again"}, []string{"AK-1", "AK-2", "AK-3"}},
		{"sub-numbered", []string{"AK-1.2 Sub\nAK-2. Main"}, []string{"AK-1", "AK-2"}},
		{"no digit in a bare prefix", []string{"AK12 a"}, []string{"AK-12"}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			if got := criterionLabels(tc.ac...); !slices.Equal(got, tc.want) {
				t.Errorf("criterionLabels(%q) = %q, want %q", tc.ac, got, tc.want)
			}
		})
	}
}
