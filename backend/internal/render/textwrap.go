package render

import (
	"strings"
	"unicode"
	"unicode/utf8"
)

// This file is the line wrapper WrapProse uses: long words are never
// broken, hyphens are never break points, leading and trailing whitespace is
// dropped, tabs are expanded, whitespace is normalised, no line cap.
//
// Three rules decide the output and none of them are what a Go author would
// reach for:
//
//  1. The whitespace set is exactly "\t\n\v\f\r " — ASCII only. A
//     non-breaking space is NOT a word separator, so "non breaking" is
//     one indivisible token. strings.Fields would split it
//     (TestNBSPIsNotABreakPoint).
//  2. Tabs are expanded to 8-column tab stops BEFORE anything else, so a
//     tab-separated line comes back space-padded at those stops
//     (TestTabsExpandToEightColumnStops).
//  3. Widths are counted in characters, so runes and not bytes.
//
// Because hyphens are never break points, the chunker is a plain split on
// runs of those six characters keeping the separators, which is what
// splitChunks does.

// pyWhitespace is the wrapper's six-character whitespace set. Deliberately not
// unicode.IsSpace: a non-breaking space must not become a break point.
const pyWhitespace = "\t\n\v\f\r "

const tabSize = 8

// isPyWhitespace reports whether r is one of the six characters the wrapper
// treats as whitespace.
func isPyWhitespace(r rune) bool {
	return strings.ContainsRune(pyWhitespace, r)
}

// expandTabs expands to 8-column tab stops: each tab advances to the next
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
			// Only these reset the column. \v and \f are ordinary width-1
			// characters here: expanding "a\vb\tc" gives "a\vb     c", not
			// "a\vb       c" (TestExpandTabsOnlyResetsOnNewlineAndCarriageReturn).
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
// space. Runs after tab expansion.
func replaceWhitespace(s string) string {
	return strings.Map(func(r rune) rune {
		if isPyWhitespace(r) {
			return ' '
		}
		return r
	}, s)
}

// splitChunks splits into runs of whitespace and runs of non-whitespace,
// alternating, separators kept. Empty strings are dropped.
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

// isBlankChunk reports whether a chunk is whitespace-only, for the
// drop-whitespace step in wrapChunks.
//
// Blank detection is Unicode (unicode.IsSpace) even though the splitter is
// deliberately ASCII-only: replaceWhitespace normalises only the six ASCII
// characters, so a chunk of non-breaking spaces arrives here intact and must
// still count as blank. Wrapping " " yields no lines, and
// "hello  " yields ["hello "]. Trimming ASCII spaces alone would keep
// both as content (TestNBSPChunkCountsAsBlank).
func isBlankChunk(s string) bool {
	return strings.TrimFunc(s, unicode.IsSpace) == ""
}

// wrapChunks greedily fills lines from the chunk list. Long words are never
// broken and there is no line cap: a chunk wider than the line is emitted on
// a line of its own and allowed to overflow.
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

		// Long words are never broken: a chunk too big for any line goes on
		// this line alone rather than being chopped.
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

// textwrapWrap wraps text to width: expand tabs, normalise whitespace,
// chunk, then fill lines.
func textwrapWrap(text string, width int, initialIndent, subsequentIndent string) []string {
	s := replaceWhitespace(expandTabs(text))
	return wrapChunks(splitChunks(s), width, initialIndent, subsequentIndent)
}
