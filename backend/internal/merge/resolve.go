package merge

import (
	"errors"
	"fmt"
	"strconv"
	"strings"
)

// Ref names one line of a conflict by side and 1-based number: "T3" is
// the target side's third line, "S1" the source side's first. The model
// answers in refs, never in text: it cannot write a line neither side has,
// and the bytes (line ends, a BOM) stay as they were.
type Ref struct {
	Target bool
	N      int
}

func (r Ref) String() string {
	if r.Target {
		return "T" + strconv.Itoa(r.N)
	}
	return "S" + strconv.Itoa(r.N)
}

// ParseRef reads "T3" or "S1". A base ref ("B2") is no ref: the base is
// what both sides changed, taking it back undoes both.
func ParseRef(s string) (Ref, bool) {
	s = strings.TrimSpace(s)
	if len(s) < 2 || (s[0] != 'T' && s[0] != 'S') {
		return Ref{}, false
	}
	n, err := strconv.Atoi(s[1:])
	if err != nil || n < 1 {
		return Ref{}, false
	}
	return Ref{Target: s[0] == 'T', N: n}, true
}

// TrivialResolution is a trivial conflict's resolution: target's lines,
// then source's.
func TrivialResolution(c *Conflict) []Ref {
	out := make([]Ref, 0, len(c.Target)+len(c.Source))
	for i := range c.Target {
		out = append(out, Ref{Target: true, N: i + 1})
	}
	for i := range c.Source {
		out = append(out, Ref{N: i + 1})
	}
	return out
}

// ValidateResolution is the blunt check on a resolution, no judge: every
// ref names a line of the conflict, none twice, each side's lines keep
// their order, and it is empty only when both sides are. reason is log
// vocabulary, empty when it stands.
func ValidateResolution(c *Conflict, refs []Ref) (reason string) {
	if len(refs) == 0 {
		if len(c.Target) == 0 && len(c.Source) == 0 {
			return ""
		}
		return "empty"
	}
	seen := map[Ref]bool{}
	lastT, lastS := 0, 0
	for _, r := range refs {
		side := c.Source
		if r.Target {
			side = c.Target
		}
		switch {
		case r.N < 1 || r.N > len(side):
			return "unknown_ref"
		case seen[r]:
			return "duplicate_ref"
		}
		seen[r] = true
		if r.Target {
			if r.N < lastT {
				return "order"
			}
			lastT = r.N
		} else {
			if r.N < lastS {
				return "order"
			}
			lastS = r.N
		}
	}
	return ""
}

// Pick is the lines refs name.
func Pick(c *Conflict, refs []Ref) []string {
	out := make([]string, 0, len(refs))
	for _, r := range refs {
		if r.Target {
			out = append(out, c.Target[r.N-1])
		} else {
			out = append(out, c.Source[r.N-1])
		}
	}
	return out
}

// Dropped is the lines of either side a resolution leaves out, target's
// first, without their terminators. Counted per text: a side's line is
// dropped as often as the side has its text more often than the merged
// region does, so a line both sides have, kept once, is no loss, and one
// copy of a line a side repeats on purpose is one.
func Dropped(c *Conflict, refs []Ref) (target, source []string) {
	out := map[string]int{}
	for _, l := range Pick(c, refs) {
		out[TrimEOL(l)]++
	}
	side := func(lines []string, isTarget bool) []string {
		have := map[string]int{}
		for _, l := range lines {
			have[TrimEOL(l)]++
		}
		used := map[int]bool{}
		for _, r := range refs {
			if r.Target == isTarget {
				used[r.N] = true
			}
		}
		var dropped []string
		for i, l := range lines {
			t := TrimEOL(l)
			if !used[i+1] && have[t] > out[t] {
				dropped = append(dropped, t)
				have[t]--
			}
		}
		return dropped
	}
	return side(c.Target, true), side(c.Source, false)
}

// TrimEOL is a line without its terminator, "\n" or "\r\n".
func TrimEOL(l string) string {
	return strings.TrimSuffix(strings.TrimSuffix(l, "\n"), "\r")
}

// ErrNoEOL is a line without a newline that would not end the file: joined
// with the next one it changes code nobody chose.
var ErrNoEOL = errors.New("a line without a newline inside the file")

// Assemble writes the merged file: settled lines as they are, each conflict
// as its resolution (one per conflict, in order).
func Assemble(chunks []Chunk, resolutions [][]Ref) ([]byte, error) {
	var lines []string
	k := 0
	for _, c := range chunks {
		if c.Conflict == nil {
			lines = append(lines, c.Lines...)
			continue
		}
		if k >= len(resolutions) {
			return nil, fmt.Errorf("conflict %d has no resolution", k+1)
		}
		lines = append(lines, Pick(c.Conflict, resolutions[k])...)
		k++
	}
	if k != len(resolutions) {
		return nil, fmt.Errorf("%d resolutions for %d conflicts", len(resolutions), k)
	}
	var b strings.Builder
	for i, l := range lines {
		if i < len(lines)-1 && !strings.HasSuffix(l, "\n") {
			return nil, ErrNoEOL
		}
		b.WriteString(l)
	}
	return []byte(b.String()), nil
}

// LineOf is the 1-based line in the assembled file where conflict k
// (0-based) starts, given the resolutions before it.
func LineOf(chunks []Chunk, resolutions [][]Ref, k int) int {
	line, seen := 1, 0
	for _, c := range chunks {
		if c.Conflict == nil {
			line += len(c.Lines)
			continue
		}
		if seen == k {
			return line
		}
		if seen < len(resolutions) {
			line += len(resolutions[seen])
		}
		seen++
	}
	return line
}
