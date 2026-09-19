package jira

import (
	"regexp"
	"strings"
)

// Jira wiki markup, stripped so the model reads prose instead of syntax. The
// order below is load-bearing: block tags go before emphasis, so a {code:java}
// fence does not leave a stray brace for the emphasis pass to trip over.
var (
	headingRE    = regexp.MustCompile(`(?m)^h[1-6]\.[ \t]*`)
	blockTagRE   = regexp.MustCompile(`(?i)\{(noformat|code(?::[^}]*)?|quote|panel(?::[^}]*)?)\}`)
	colorRE      = regexp.MustCompile(`(?s)\{color(?::[^}]*)?\}(.*?)\{color\}`)
	imageRE      = regexp.MustCompile(`!(?:[^|!\n]+\|)?[^!\n]+!`)
	linkRE       = regexp.MustCompile(`\[([^|]+)\|([^\]]+)\]`)
	tableHeadRE  = regexp.MustCompile(`\|\|`)
	blankLinesRE = regexp.MustCompile(`\n{3,}`)
)

// emphasisMarks are stripped in this order, each in one pass.
var emphasisMarks = []string{"*", "_", "-"}

// emphasisRE holds one compiled pattern per mark, built once.
var emphasisRE = func() map[string]*regexp.Regexp {
	m := make(map[string]*regexp.Regexp, len(emphasisMarks))
	for _, mark := range emphasisMarks {
		q := regexp.QuoteMeta(mark)
		// Python used lookaround: (?<!\w)MARK(.+?)MARK(?!\w). RE2 has none, so
		// the boundary characters are captured and written back instead.
		m[mark] = regexp.MustCompile(`([^\pL\pN_])` + q + `([^\n]+?)` + q + `([^\pL\pN_])`)
	}
	return m
}()

// stripMarkup turns Jira wiki markup into plain text.
func stripMarkup(text string) string {
	text = headingRE.ReplaceAllString(text, "")
	text = blockTagRE.ReplaceAllString(text, "")
	text = colorRE.ReplaceAllString(text, "$1")
	text = imageRE.ReplaceAllString(text, "")
	text = linkRE.ReplaceAllString(text, "$1 ($2)")
	text = tableHeadRE.ReplaceAllString(text, " | ")
	for _, mark := range emphasisMarks {
		text = stripEmphasis(text, mark)
	}
	text = blankLinesRE.ReplaceAllString(text, "\n\n")
	return strings.TrimSpace(text)
}

// stripEmphasis removes one emphasis marker in a single left-to-right pass.
//
// Two details decide whether this matches Python. It must be one pass: on
// "**bold**" the non-greedy body swallows the inner marker, so Python leaves
// "*bold*" behind, and a second pass would wrongly strip that too. And the
// trailing boundary must be given back rather than consumed, because Python's
// (?!\w) is a lookahead that matches without eating a character; otherwise
// "*a* *b*" loses the space between the words.
//
// Sentinels stand in for the string ends so a marker at position zero still has
// a preceding character to match.
func stripEmphasis(text, mark string) string {
	re := emphasisRE[mark]
	if re == nil {
		return text
	}
	const sentinel = "\x00"
	s := sentinel + text + sentinel

	var out strings.Builder
	pos := 0
	for {
		loc := re.FindStringSubmatchIndex(s[pos:])
		if loc == nil {
			break
		}
		out.WriteString(s[pos : pos+loc[3]])        // everything up to and including the leading boundary
		out.WriteString(s[pos+loc[4] : pos+loc[5]]) // the emphasised text, markers dropped
		pos += loc[6]                               // resume at the trailing boundary, not past it
	}
	out.WriteString(s[pos:])
	return strings.ReplaceAll(out.String(), sentinel, "")
}

// acceptanceCriteria pulls the acceptance-criteria lines out of a description.
//
// One pattern per configured prefix, matched over every line. Prefixes are
// tried in config order and matches kept in document order, so all "AK" hits
// precede all "AC" hits regardless of where they sit in the text. Duplicate
// lines are dropped.
//
// The prefix must end on a word boundary, which is a deliberate divergence from
// the Python: there "AC" also matched "Actual behaviour..." and "Req" matched
// "Request: ...". Optional numbering still counts as part of the prefix, so
// "AK3 No separator" matches while "AKzeptanz" and "AKübung" do not. The
// boundary class is Unicode because Python's \b is.
func acceptanceCriteria(description string, prefixes []string) string {
	if len(prefixes) == 0 || description == "" {
		return ""
	}
	var lines []string
	seen := make(map[string]bool)
	for _, prefix := range prefixes {
		if prefix == "" {
			continue
		}
		re := regexp.MustCompile(`(?im)^[ \t]*` + regexp.QuoteMeta(prefix) + `(?:[- ]?\d+)?(?:[^\pL\pN\n].*)?$`)
		for _, m := range re.FindAllString(description, -1) {
			line := strings.TrimSpace(m)
			if line == "" || seen[line] {
				continue
			}
			seen[line] = true
			lines = append(lines, line)
		}
	}
	return strings.Join(lines, "\n")
}
