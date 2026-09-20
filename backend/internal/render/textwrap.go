package render

import (
	"strings"
	"unicode"
	"unicode/utf8"
)

// This file is a port of the subset of Python's textwrap that wrap_prose
// uses: break_long_words=False, break_on_hyphens=False, drop_whitespace=True,
// expand_tabs=True, replace_whitespace=True, max_lines=None.
//
// It is a port, not an equivalent. Three details decide parity and none of
// them are what a Go author would reach for:
//
//  1. textwrap's whitespace set is textwrap._whitespace, exactly
//     "\t\n\v\f\r " — ASCII only. A non-breaking space is NOT a word
//     separator, so "non breaking" is one indivisible token.
//     strings.Fields would split it and diverge.
//  2. Tabs are expanded to 8-column tab stops BEFORE anything else, so a
//     tab-separated line comes back space-padded at those stops.
//  3. Widths are counted in characters (Python len() on str), so runes.
//
// With break_on_hyphens=False the chunker is textwrap's wordsep_simple_re,
// a plain split on runs of those six characters keeping the separators,
// which is what splitChunks does.

// pyWhitespace is textwrap._whitespace. Deliberately not unicode.IsSpace.
const pyWhitespace = "\t\n\v\f\r "

const tabSize = 8

// isPyWhitespace reports whether r is one of the six characters textwrap
// treats as whitespace.
func isPyWhitespace(r rune) bool {
	return strings.ContainsRune(pyWhitespace, r)
}

// expandTabs reproduces str.expandtabs(8): each tab advances to the next
// multiple of tabSize, counted from the last line start.
func expandTabs(s string) string {
	if !strings.ContainsRune(s, '\t') {
		return s
	}
	var b strings.Builder
	col := 0
	for _, r := range s {
		switch r {
		case '\t':
			n := tabSize - col%tabSize
			b.WriteString(strings.Repeat(" ", n))
			col += n
		case '\n', '\r':
			// Only these reset the column. CPython's str.expandtabs treats
			// \v and \f as ordinary width-1 characters:
			// "a\vb\tc".expandtabs(8) is "a\vb     c", not "a\vb       c".
			b.WriteRune(r)
			col = 0
		default:
			b.WriteRune(r)
			col++
		}
	}
	return b.String()
}

// replaceWhitespace maps each of the six whitespace characters to a plain
// space, as textwrap does after tab expansion.
func replaceWhitespace(s string) string {
	return strings.Map(func(r rune) rune {
		if isPyWhitespace(r) {
			return ' '
		}
		return r
	}, s)
}

// splitChunks is textwrap's wordsep_simple_re split: runs of whitespace and
// runs of non-whitespace, alternating, separators kept. Empty strings are
// dropped, as textwrap does.
func splitChunks(s string) []string {
	var out []string
	var cur strings.Builder
	inWS := false
	for i, r := range s {
		ws := isPyWhitespace(r)
		if i > 0 && ws != inWS {
			out = append(out, cur.String())
			cur.Reset()
		}
		inWS = ws
		cur.WriteRune(r)
	}
	if cur.Len() > 0 {
		out = append(out, cur.String())
	}
	return out
}

// isBlankChunk mirrors chunk.strip() == "" in _wrap_chunks.
//
// str.strip() is Unicode, and replaceWhitespace only normalises the six ASCII
// characters, so a chunk of non-breaking spaces reaches here intact and is
// still blank to Python: textwrap.wrap(" ") is [] and
// textwrap.wrap("hello  ") is ["hello "]. Trimming ASCII spaces alone
// would keep both, so this needs unicode.IsSpace even though the splitter
// deliberately does not.
func isBlankChunk(s string) bool {
	return strings.TrimFunc(s, unicode.IsSpace) == ""
}

// wrapChunks is textwrap.TextWrapper._wrap_chunks with break_long_words and
// max_lines removed: a chunk wider than the line is emitted on a line of its
// own and allowed to overflow.
func wrapChunks(chunks []string, width int, initialIndent, subsequentIndent string) []string {
	var lines []string
	if width <= 0 {
		return nil
	}
	i := 0
	for i < len(chunks) {
		indent := initialIndent
		if len(lines) > 0 {
			indent = subsequentIndent
		}
		avail := width - utf8.RuneCountInString(indent)

		// A leading whitespace chunk is dropped, except at the very start
		// of the text.
		if isBlankChunk(chunks[i]) && len(lines) > 0 {
			i++
			continue
		}

		var cur []string
		curLen := 0
		for i < len(chunks) {
			l := utf8.RuneCountInString(chunks[i])
			if curLen+l > avail {
				break
			}
			cur = append(cur, chunks[i])
			curLen += l
			i++
		}

		// break_long_words=False: a chunk too big for any line goes on this
		// line alone rather than being chopped.
		if len(cur) == 0 && i < len(chunks) {
			cur = append(cur, chunks[i])
			i++
		}

		// A trailing whitespace chunk is dropped.
		if len(cur) > 0 && isBlankChunk(cur[len(cur)-1]) {
			cur = cur[:len(cur)-1]
		}

		if len(cur) > 0 {
			lines = append(lines, indent+strings.Join(cur, ""))
		}
	}
	return lines
}

// textwrapWrap is textwrap.wrap for the option set wrap_prose uses.
func textwrapWrap(text string, width int, initialIndent, subsequentIndent string) []string {
	s := replaceWhitespace(expandTabs(text))
	return wrapChunks(splitChunks(s), width, initialIndent, subsequentIndent)
}
