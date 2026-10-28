package render

import "strings"

// emDash is what NoEmDash replaces.
const emDash = "—"

// NoEmDash replaces every em dash in a comment's prose with a hyphen (user,
// 2026-10-28): the model writes them and so do the templates. Code is left
// alone: a fenced block or an inline code span is code, and a changed
// character there changes a suggestion's code or a quoted line.
//
// Fences are read as WrapProse reads them (fences). Inline code follows
// CommonMark: a run of N backticks opens a span that only a run of exactly N
// closes, so a span opened by two backticks may hold a single one; an
// unmatched run is literal text.
func NoEmDash(s string) string {
	if !strings.Contains(s, emDash) {
		return s
	}
	var b strings.Builder
	b.Grow(len(s))
	var f fences
	for _, line := range strings.SplitAfter(s, "\n") {
		if f.code(line) {
			b.WriteString(line)
			continue
		}
		b.WriteString(dashOutsideCode(line))
	}
	return b.String()
}

// dashOutsideCode is one prose line with its em dashes replaced outside
// inline code spans.
func dashOutsideCode(line string) string {
	var b strings.Builder
	for i := 0; i < len(line); {
		if line[i] != '`' {
			j := strings.IndexByte(line[i:], '`')
			if j < 0 {
				j = len(line) - i
			}
			b.WriteString(strings.ReplaceAll(line[i:i+j], emDash, "-"))
			i += j
			continue
		}
		n := runLen(line, i)
		if end := closingRun(line, i+n, n); end >= 0 {
			b.WriteString(line[i : end+n])
			i = end + n
			continue
		}
		// No closing run: the backticks are literal, prose goes on.
		b.WriteString(line[i : i+n])
		i += n
	}
	return b.String()
}

// runLen is the length of the backtick run at i.
func runLen(s string, i int) int {
	n := 0
	for i+n < len(s) && s[i+n] == '`' {
		n++
	}
	return n
}

// closingRun is the index of the next run of exactly n backticks from i,
// -1 when there is none.
func closingRun(s string, i, n int) int {
	for i < len(s) {
		j := strings.IndexByte(s[i:], '`')
		if j < 0 {
			return -1
		}
		i += j
		m := runLen(s, i)
		if m == n {
			return i
		}
		i += m
	}
	return -1
}
