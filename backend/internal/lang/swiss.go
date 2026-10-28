package lang

import "strings"

// SwissSS writes the model's German prose with Swiss spelling: ß becomes ss
// (ẞ becomes SS). The prompt asks for it, and a model does not always obey;
// this is the blunt rule that makes it hold.
//
// Code is left alone: fenced blocks (``` and ~~~), indented code blocks
// (four spaces or a tab) and inline spans of any backtick run may quote an
// identifier or a string that really contains ß.
//
// It lives here, not in render, so every package that stores or posts
// prose (render for Bitbucket, inference for an Almanach note) applies the
// one rule without a dependency on the other.
func SwissSS(s string) string {
	if !strings.ContainsAny(s, "ßẞ") {
		return s
	}
	var b strings.Builder
	b.Grow(len(s) + 8)
	fence := ""
	for _, line := range strings.SplitAfter(s, "\n") {
		trimmed := strings.TrimLeft(line, " \t")
		if fence != "" {
			if strings.HasPrefix(trimmed, fence) {
				fence = ""
			}
			b.WriteString(line)
			continue
		}
		if f := fenceOf(trimmed); f != "" {
			fence = f
			b.WriteString(line)
			continue
		}
		if strings.HasPrefix(line, "    ") || strings.HasPrefix(line, "\t") {
			b.WriteString(line)
			continue
		}
		b.WriteString(swissLine(line))
	}
	return b.String()
}

// OpenFence is the fence of a code block s opens and leaves unclosed, or "".
// A closing fence is the opener's character, at least as long, alone on its
// line; a backtick opener whose rest holds a backtick is an inline span.
func OpenFence(s string) string {
	open := ""
	for _, line := range strings.Split(s, "\n") {
		trimmed := strings.TrimLeft(line, " \t")
		f := fenceOf(trimmed)
		switch {
		case f == "":
		case open == "":
			if f[0] == '`' && strings.Contains(trimmed[len(f):], "`") {
				continue
			}
			open = f
		case f[0] == open[0] && len(f) >= len(open) && strings.TrimSpace(trimmed[len(f):]) == "":
			open = ""
		}
	}
	return open
}

// fenceOf is the opening fence of a line ("```", "~~~~", ...), or "".
func fenceOf(trimmed string) string {
	for _, c := range []byte{'`', '~'} {
		n := 0
		for n < len(trimmed) && trimmed[n] == c {
			n++
		}
		if n >= 3 {
			return trimmed[:n]
		}
	}
	return ""
}

// swissLine rewrites one prose line outside its code spans. A span opens
// with a run of backticks and closes with a run of the same length; an
// unclosed one shields the rest of the line, as Bitbucket renders it.
func swissLine(line string) string {
	var b strings.Builder
	for i := 0; i < len(line); {
		if line[i] == '`' {
			j := i
			for j < len(line) && line[j] == '`' {
				j++
			}
			run := line[i:j]
			end := strings.Index(line[j:], run)
			if end < 0 {
				b.WriteString(line[i:])
				return b.String()
			}
			b.WriteString(line[i : j+end+len(run)])
			i = j + end + len(run)
			continue
		}
		next := strings.IndexByte(line[i:], '`')
		if next < 0 {
			next = len(line) - i
		}
		b.WriteString(strings.NewReplacer("ß", "ss", "ẞ", "SS").Replace(line[i : i+next]))
		i += next
	}
	return b.String()
}
