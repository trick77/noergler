package render

import (
	"regexp"
	"slices"
	"strings"
	"unicode"
	"unicode/utf8"
)

// Bitbucket's Jira integration links every issue-key-shaped token in a
// comment. A ticket's criterion label (AK-1, AC-2) has that shape, so a
// requirement line that carries it bare links to <jira>/browse/AK-1, an
// issue that does not exist. Inside a code span it is not linked.
//
// Digits are \p{Nd}: RE2's classes are ASCII (INVARIANTS: adapters).
var (
	criterionKeyRE = regexp.MustCompile(`[A-Z][A-Z0-9_]+-\p{Nd}+(?:\.\p{Nd}+)*`)
	// A ticket writes its label "AK-1", "AK 1" or "AK1"; without a separator
	// the prefix holds no digit, so "AK12" stays AK-12 rather than AK1-2.
	criterionLineRE = regexp.MustCompile(
		`(?i)^(?:([A-Z][A-Z0-9_]+)[-\p{Zs}]|([A-Z][A-Z_]+))(\p{Nd}+)(?:\.\p{Nd}+)*(?:[^\pL\pN_]|$)`)
	// labelSepRE ends the labels at the front of a line: "AK-1: x", "AK-1 - x".
	// One mark only, so "AK-1: .env" keeps its dot.
	labelSepRE = regexp.MustCompile(`^[\s\p{Zs}]*[:.,;\-–—/]?[\s\p{Zs}]*`)
	// labelJoinRE sits between two labels of one list: "AK-1, AK-2", "AK-1 und AK-2".
	labelJoinRE = regexp.MustCompile(`^[\s\p{Zs}]*(?:[,/&;+]|and|und)?[\s\p{Zs}]*`)
	// labelRangeRE joins a range into one label: "AK-1–AK-3", "AK-1-2".
	labelRangeRE = regexp.MustCompile(`^[\s\p{Zs}]*[\-–—][\s\p{Zs}]*`)
	// labelDashRE sets trailing labels off the sentence: "x - AK-1".
	labelDashRE = regexp.MustCompile(`[\s\p{Zs}]+[\-–—][\s\p{Zs}]+`)
)

// bracketPairs are the wrappers a label may come in: "(AK-1)", "[AK-1]", "`AK-1`".
const bracketPairs = "()[]``"

// criterionLabels is the label that starts each line of the acceptance
// criteria, as the key it is linked as: "AK1", "AK 1" and "ak-1" are AK-1,
// "AK-1.2" counts as AK-1. Every such line already starts with one of the
// team's configured prefixes (jira.acceptanceCriteria), so a cited issue
// key keeps its link unless a prefix is also a Jira project key.
func criterionLabels(acceptanceCriteria ...string) []string {
	var labels []string
	for _, ac := range acceptanceCriteria {
		for line := range strings.Lines(ac) {
			m := criterionLineRE.FindStringSubmatch(strings.TrimSpace(line))
			if m == nil {
				continue
			}
			if l := strings.ToUpper(m[1]+m[2]) + "-" + m[3]; !slices.Contains(labels, l) {
				labels = append(labels, l)
			}
		}
	}
	return labels
}

// labelRequirement leads a requirement line with its criterion labels, each
// in a code span, when the model set them apart: "AK-1: x", "(AK-1) x",
// "x (AK-1)" and "x - AK-1" all read "`AK-1` x". A label inside the
// sentence ("as AK-1 asks") is spanned in place. A line without a label is
// returned as it came.
func labelRequirement(requirement string, labels []string) string {
	if len(labels) == 0 {
		return requirement
	}
	rest := requirement
	var front []string
	if got, after, ok := cutLeadingLabels(rest, labels); ok {
		front, rest = got, after
	}
	if got, before, ok := cutTrailingLabels(rest, labels); ok {
		front, rest = append(front, got...), before
	}
	if strings.TrimSpace(rest) == "" {
		// Nothing but labels: no sentence to lead.
		return spanLabels(requirement, labels)
	}
	rest = spanLabels(rest, labels)
	if len(front) == 0 {
		return rest
	}
	var spans []string
	for _, l := range front {
		if s := "`" + l + "`"; !slices.Contains(spans, s) {
			spans = append(spans, s)
		}
	}
	return strings.Join(spans, " ") + " " + rest
}

// cutLeadingLabels takes the labels off the front of s: bare ("AK-1, AK-2:
// x") or as the whole of a bracket or code span ("(AK-1 and AK-2) x"). A
// markdown link's text ("[AK-1](url)") is not a bracket.
func cutLeadingLabels(s string, labels []string) (got []string, after string, ok bool) {
	t := strings.TrimLeftFunc(s, unicode.IsSpace)
	if t == "" {
		return nil, "", false
	}
	if c := closing(t[0]); c != 0 {
		end := strings.IndexByte(t[1:], c)
		if end < 0 {
			return nil, "", false
		}
		after = t[2+end:]
		if t[0] == '[' && strings.HasPrefix(after, "(") {
			return nil, "", false
		}
		if got, ok = labelGroup(t[1:1+end], labels); !ok {
			return nil, "", false
		}
		return got, after[len(labelSepRE.FindString(after)):], true
	}
	got, after = takeLabels(t, labels)
	sep := labelSepRE.FindString(after)
	if len(got) == 0 || (sep == "" && after != "") {
		return nil, "", false
	}
	return got, after[len(sep):], true
}

// cutTrailingLabels takes the labels off the end of s when they stand apart
// from the sentence: a closing bracket or code span holding only labels
// ("x (AK-1)."), or labels after a spaced dash ("x - AK-1"). The sentence's
// closing mark stays.
func cutTrailingLabels(s string, labels []string) (got []string, before string, ok bool) {
	t := strings.TrimRightFunc(s, unicode.IsSpace)
	stop := ""
	if n := len(t); n > 0 && strings.ContainsRune(".!?", rune(t[n-1])) {
		t, stop = strings.TrimRightFunc(t[:n-1], unicode.IsSpace), t[n-1:]
	}
	if t == "" {
		return nil, "", false
	}
	if open := opening(t[len(t)-1]); open != 0 {
		start := strings.LastIndexByte(t[:len(t)-1], open)
		if start < 0 {
			return nil, "", false
		}
		if got, ok = labelGroup(t[start+1:len(t)-1], labels); !ok {
			return nil, "", false
		}
		return got, strings.TrimRightFunc(t[:start], unicode.IsSpace) + stop, true
	}
	dashes := labelDashRE.FindAllStringIndex(t, -1)
	if len(dashes) == 0 {
		return nil, "", false
	}
	dash := dashes[len(dashes)-1]
	if got, ok = labelGroup(t[dash[1]:], labels); !ok {
		return nil, "", false
	}
	return got, t[:dash[0]] + stop, true
}

// labelGroup: s is labels and the words that join them, nothing else.
func labelGroup(s string, labels []string) ([]string, bool) {
	got, rest := takeLabels(strings.TrimSpace(s), labels)
	return got, len(got) > 0 && strings.TrimSpace(rest) == ""
}

// takeLabels reads the list of labels at the start of s. A range stays one
// label as written: "AK-1–AK-3" names AK-2 too, "AK-1-2" is not AK-1.
func takeLabels(s string, labels []string) (got []string, rest string) {
	rest = s
	for {
		next := rest
		if len(got) > 0 {
			next = rest[len(labelJoinRE.FindString(rest)):]
		}
		n := leadingLabel(next, labels)
		if n == 0 {
			return got, rest
		}
		label := next[:n]
		next = next[n:]
		if dash := labelRangeRE.FindString(next); dash != "" {
			end := next[len(dash):]
			if m := leadingLabel(end, labels); m > 0 {
				label, next = label+dash+end[:m], end[m:]
			} else if d := leadingDigits(end); d > 0 {
				label, next = label+dash+end[:d], end[d:]
			}
		}
		got = append(got, label)
		rest = next
	}
}

// leadingLabel is the length of the label s starts with, or 0.
func leadingLabel(s string, labels []string) int {
	loc := criterionKeyRE.FindStringIndex(s)
	if loc == nil || loc[0] != 0 || !keyBoundary(s, loc) || !isLabel(s[:loc[1]], labels) {
		return 0
	}
	return loc[1]
}

// leadingDigits is the length of the number s starts with when nothing
// word-like follows it, or 0.
func leadingDigits(s string) int {
	n := len(s) - len(strings.TrimLeftFunc(s, unicode.IsDigit))
	if r, _ := utf8.DecodeRuneInString(s[n:]); n == 0 || (n < len(s) && isWordRune(r)) {
		return 0
	}
	return n
}

// spanLabels wraps each label in s in a code span, unless it already sits
// in one or in a URL.
func spanLabels(s string, labels []string) string {
	var b strings.Builder
	last := 0
	for _, loc := range criterionKeyRE.FindAllStringIndex(s, -1) {
		if !keyBoundary(s, loc) || !isLabel(s[loc[0]:loc[1]], labels) ||
			strings.Count(s[:loc[0]], "`")%2 == 1 || inURL(s, loc[0]) {
			continue
		}
		b.WriteString(s[last:loc[0]])
		b.WriteString("`" + s[loc[0]:loc[1]] + "`")
		last = loc[1]
	}
	b.WriteString(s[last:])
	return b.String()
}

// isLabel: key is one of the labels, sub-numbers aside ("AK-1.2" is AK-1).
func isLabel(key string, labels []string) bool {
	if i := strings.IndexByte(key, '.'); i >= 0 {
		key = key[:i]
	}
	return slices.Contains(labels, key)
}

// inURL: the word holding position i is a link target ("https://jira/browse/AK-1").
func inURL(s string, i int) bool {
	start := strings.LastIndexFunc(s[:i], unicode.IsSpace) + 1
	word := s[start:i]
	return strings.Contains(word, "://") || strings.Contains(word, "](")
}

// keyBoundary: the match is a whole token, not part of a longer word or key
// ("XAK-1", "AK-12a"). Spelled out because RE2's \b is ASCII.
func keyBoundary(s string, loc []int) bool {
	if r, _ := utf8.DecodeLastRuneInString(s[:loc[0]]); loc[0] > 0 && (isWordRune(r) || r == '-') {
		return false
	}
	if r, _ := utf8.DecodeRuneInString(s[loc[1]:]); loc[1] < len(s) && isWordRune(r) {
		return false
	}
	return true
}

func isWordRune(r rune) bool { return unicode.IsLetter(r) || unicode.IsDigit(r) || r == '_' }

// closing is the bracket that closes open, or 0 when open is none.
func closing(open byte) byte {
	if i := strings.IndexByte(bracketPairs, open); i >= 0 && i%2 == 0 {
		return bracketPairs[i+1]
	}
	return 0
}

// opening is the bracket that close closes, or 0 when close is none.
func opening(close byte) byte {
	if i := strings.LastIndexByte(bracketPairs, close); i >= 0 && i%2 == 1 {
		return bracketPairs[i-1]
	}
	return 0
}
