package config

// FnMatch is case-sensitive shell-style globbing: `*` any run, `?` one
// character, `[seq]` a set with ranges and `[!seq]` its negation. No path
// semantics: a `*` crosses `/`, unlike path.Match. A pattern that ends inside
// an unclosed bracket treats the `[` literally.
func FnMatch(pattern, name string) bool {
	p, n := []rune(pattern), []rune(name)
	return match(p, n)
}

func match(p, n []rune) bool {
	for len(p) > 0 {
		switch p[0] {
		case '*':
			// Collapse a run of stars, then try every split.
			for len(p) > 0 && p[0] == '*' {
				p = p[1:]
			}
			if len(p) == 0 {
				return true
			}
			for i := 0; i <= len(n); i++ {
				if match(p, n[i:]) {
					return true
				}
			}
			return false
		case '?':
			if len(n) == 0 {
				return false
			}
			p, n = p[1:], n[1:]
		case '[':
			set, rest, ok := bracket(p)
			if !ok {
				// Literal '['.
				if len(n) == 0 || n[0] != '[' {
					return false
				}
				p, n = p[1:], n[1:]
				continue
			}
			if len(n) == 0 || !set(n[0]) {
				return false
			}
			p, n = rest, n[1:]
		default:
			if len(n) == 0 || n[0] != p[0] {
				return false
			}
			p, n = p[1:], n[1:]
		}
	}
	return len(n) == 0
}

// bracket parses a `[...]` class at the head of p. ok is false when there is
// no closing bracket.
func bracket(p []rune) (func(rune) bool, []rune, bool) {
	i := 1
	negate := false
	if i < len(p) && p[i] == '!' {
		negate = true
		i++
	}
	// A ']' right after the opening (or the '!') is a literal member.
	j := i
	if j < len(p) && p[j] == ']' {
		j++
	}
	for j < len(p) && p[j] != ']' {
		j++
	}
	if j >= len(p) {
		return nil, nil, false
	}
	members := p[i:j]
	f := func(r rune) bool {
		for k := 0; k < len(members); k++ {
			if k+2 < len(members) && members[k+1] == '-' {
				if members[k] <= r && r <= members[k+2] {
					return !negate
				}
				k += 2
				continue
			}
			if members[k] == r {
				return !negate
			}
		}
		return negate
	}
	return f, p[j+1:], true
}
