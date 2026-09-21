package inference

import "strconv"

// splitLines splits on a boundary set wider than newline: \v, \f, \x1c-\x1e,
// \x85, U+2028 and U+2029 all break a line. Terminators are not kept.
//
// internal/diff has the same helper, unexported. Duplicated rather than
// exported because the two are pinned independently and must stay free to
// diverge.
func splitLines(s string) []string {
	if s == "" {
		return nil
	}
	var out []string
	start := 0
	runes := []rune(s)
	for i := 0; i < len(runes); i++ {
		if !isPythonLineBreak(runes[i]) {
			continue
		}
		end := i
		next := i + 1
		// \r\n counts as one break.
		if runes[i] == '\r' && next < len(runes) && runes[next] == '\n' {
			next++
		}
		out = append(out, string(runes[start:end]))
		start = next
		i = next - 1
	}
	if start < len(runes) {
		out = append(out, string(runes[start:]))
	}
	return out
}

func isPythonLineBreak(r rune) bool {
	switch r {
	case '\n', '\r', '\v', '\f', 0x1c, 0x1d, 0x1e, 0x85, 0x2028, 0x2029:
		return true
	}
	return false
}

func itoa(n int) string { return strconv.Itoa(n) }
