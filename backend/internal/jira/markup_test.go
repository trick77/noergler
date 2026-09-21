package jira

import "testing"

// These are golden values. Stripping feeds prompt text, so a drift shows up
// here as a failure rather than as differently-worded prose in a prompt.
func TestStripMarkup(t *testing.T) {
	tests := []struct {
		name string
		in   string
		want string
	}{
		{"headings", "h1. Title\nh2. Subtitle", "Title\nSubtitle"},
		{"code fence keeps body", "{code:java}\nSystem.out.println();\n{code}", "System.out.println();"},
		{"noformat", "{noformat}\nplain\n{noformat}", "plain"},
		{"quote", "{quote}\nquoted\n{quote}", "quoted"},
		{"panel with attributes", "{panel:title=Info}\ncontent here\n{panel}", "content here"},
		{"color keeps body", "{color:red}warning{color}", "warning"},
		// The image goes, the spaces around it stay: no whitespace collapsing.
		{"image removed", "text !image.png! more", "text  more"},
		{"image with attributes", "!screenshot.png|width=500!", ""},
		{"link", "[Google|https://google.com]", "Google (https://google.com)"},
		{"table header", "||Name||Age||", "| Name | Age |"},
		{"emphasis", "*bold* _italic_ -strike-", "bold italic strike"},
		{"blank lines collapse", "line1\n\n\n\n\nline2", "line1\n\nline2"},
		{
			"combined",
			"h1. Overview\n{noformat}\ncode here\n{noformat}\n*important*\n[link|http://x.com]",
			"Overview\n\ncode here\n\nimportant\nlink (http://x.com)",
		},

		// Emphasis edge cases. These are where a plausible-looking
		// lookaround-free rewrite quietly diverges.

		// One pass only: the non-greedy body eats the inner marker, so a single
		// orphan is left. Stripping until stable would wrongly yield "bold".
		{"double marker leaves one", "**bold**", "*bold*"},
		// The trailing boundary must not be eaten, or the space between the
		// words is lost.
		{"two spans keep their separator", "*a* *b*", "a b"},
		// The boundary class covers Unicode letters, so a letter with an
		// umlaut is a word character and suppresses the match.
		{"unicode letter suppresses the match", "ü*fett*", "ü*fett*"},
		{"unicode letter elsewhere still matches", "Müller *fett*", "Müller fett"},
		{"underscores inside a word survive", "snake_case_name", "snake_case_name"},
		{"marker inside a word is not emphasis", "a*b*c", "a*b*c"},
		{"marker at both ends", "*bold*", "bold"},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			if got := stripMarkup(tc.in); got != tc.want {
				t.Errorf("stripMarkup(%q)\n got %q\nwant %q", tc.in, got, tc.want)
			}
		})
	}
}

func TestStripMarkupEmpty(t *testing.T) {
	if got := stripMarkup(""); got != "" {
		t.Errorf("stripMarkup(%q) = %q", "", got)
	}
}

var defaultPrefixes = []string{
	"AK", "AC", "Acceptance Criteria", "Acceptance Criterion",
	"Akzeptanzkriterium", "Akzeptanzkriterien", "DoD", "Req",
}

func TestAcceptanceCriteria(t *testing.T) {
	tests := []struct {
		name     string
		in       string
		prefixes []string
		want     string
	}{
		{
			"basic AK", "Intro\nAK-1: Must handle auth\nAK-2: Must log errors\nOutro",
			[]string{"AK"}, "AK-1: Must handle auth\nAK-2: Must log errors",
		},
		{"case insensitive", "ak-1: lower\nAK-2: upper", []string{"AK"}, "ak-1: lower\nAK-2: upper"},
		{
			"separators: space, dot, none",
			"AK 1: Space separator\nAK-2. Dot separator\nAK3 No separator",
			[]string{"AK"},
			"AK 1: Space separator\nAK-2. Dot separator\nAK3 No separator",
		},
		{"no number", "DoD: no number\nDoD: another", []string{"DoD"}, "DoD: no number\nDoD: another"},
		{"german multi-word prefix", "Akzeptanzkriterium 1: Muss gehen", defaultPrefixes, "Akzeptanzkriterium 1: Muss gehen"},
		{"english full prefix", "Acceptance Criteria: works", defaultPrefixes, "Acceptance Criteria: works"},
		{"only at line start", "This is not an AK-1 criterion\nAK-2: Real criterion", []string{"AK"}, "AK-2: Real criterion"},
		{"duplicates dropped", "AK-1: dup\nAK-1: dup", []string{"AK"}, "AK-1: dup"},
		{"no matches", "nothing here", defaultPrefixes, ""},
		{"no prefixes configured", "AK-1: ignored", nil, ""},
		{"empty description", "", defaultPrefixes, ""},
		// Prefix order wins over document order: every AK before every AC.
		{
			"prefix order beats document order",
			"AC-1: second\nAK-1: first",
			[]string{"AK", "AC"},
			"AK-1: first\nAC-1: second",
		},
		{"multiple prefixes", "AK-1: a\nDoD-2: b", []string{"AK", "DoD"}, "AK-1: a\nDoD-2: b"},

		// The prefix must end on a word boundary. Matching any line merely
		// starting with the letters files ordinary prose as an acceptance
		// criterion.
		{"AC does not match Actual", "Actual behaviour is wrong\nAC-1: real", []string{"AC"}, "AC-1: real"},
		{"Req does not match Request", "Request: please\nReq-2: real", []string{"Req"}, "Req-2: real"},
		{"AK does not match AKuebung", "AKübung foo\nAK-9: real", []string{"AK"}, "AK-9: real"},
		{"AK does not match Aktuell", "Aktuell: nope\nAK-1: real", []string{"AK"}, "AK-1: real"},
		// ...while a digit still counts as part of the prefix, so this keeps working.
		{"numbering still matches without a separator", "AK3 No separator", []string{"AK"}, "AK3 No separator"},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			if got := acceptanceCriteria(tc.in, tc.prefixes); got != tc.want {
				t.Errorf("acceptanceCriteria(%q, %v)\n got %q\nwant %q", tc.in, tc.prefixes, got, tc.want)
			}
		})
	}
}

// An empty string in the prefix list would otherwise build a pattern that
// matches every line.
func TestAcceptanceCriteriaIgnoresEmptyPrefix(t *testing.T) {
	if got := acceptanceCriteria("some prose\nAK-1: real", []string{"", "AK"}); got != "AK-1: real" {
		t.Errorf("got %q, want only the AK line", got)
	}
}
