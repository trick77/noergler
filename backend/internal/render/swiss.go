package render

import "strings"

// SwissSS writes the model's German prose with Swiss spelling: ß becomes ss
// (ẞ becomes SS). The prompt asks for it, and a model does not always obey;
// this is the blunt rule that makes it hold.
//
// Code is left alone: inline `code` spans and fenced blocks may quote an
// identifier or a string that really contains ß.
func SwissSS(s string) string {
	if !strings.ContainsAny(s, "ßẞ") {
		return s
	}
	var b strings.Builder
	b.Grow(len(s) + 8)
	inFence, inSpan := false, false
	lines := strings.SplitAfter(s, "\n")
	for _, line := range lines {
		if strings.HasPrefix(strings.TrimLeft(line, " \t"), "```") {
			inFence = !inFence
			inSpan = false
			b.WriteString(line)
			continue
		}
		if inFence {
			b.WriteString(line)
			continue
		}
		for _, r := range line {
			switch {
			case r == '`':
				inSpan = !inSpan
				b.WriteRune(r)
			case inSpan:
				b.WriteRune(r)
			case r == 'ß':
				b.WriteString("ss")
			case r == 'ẞ':
				b.WriteString("SS")
			default:
				b.WriteRune(r)
			}
		}
		// A span never crosses a line in Bitbucket's markdown: an unclosed
		// backtick must not shield the rest of the text.
		inSpan = false
	}
	return b.String()
}
