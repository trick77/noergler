package merge

import (
	"errors"
	"slices"
	"strings"
	"unicode"
)

// The bounds of a simple merge. Constants, never settings: a conflict past
// them goes to the author, whatever a team would like (user, 2026-11-13).
const (
	// MaxConflicts is the most conflict regions a request may resolve, over
	// all files.
	MaxConflicts = 5
	// MaxConflictFiles is the most files with a conflict.
	MaxConflictFiles = 3
	// MaxBothChanged is the most paths both sides changed, conflicting or
	// not: each one runs through this package's merge, which no build
	// checks.
	MaxBothChanged = 10
	// MaxRegionLines is the most lines a conflict's base, target or source
	// side may have.
	MaxRegionLines = 15
	// MaxEdits bounds the line diff of one side against the base.
	MaxEdits = 2000
)

// ErrTooDifferent is a side that changed a file past MaxEdits.
var ErrTooDifferent = errors.New("a side changed the file too much")

// ErrAmbiguous is a file whose merge depends on the line diff: Myers and
// patience align its lines differently, so git's histogram merge may
// settle or conflict elsewhere than this one.
var ErrAmbiguous = errors.New("the merge depends on how lines are aligned")

// Chunk is one piece of a merged file: settled lines, or a conflict.
type Chunk struct {
	Lines    []string
	Conflict *Conflict
}

// Conflict is a region both sides changed differently. Target is the
// branch merged in (the PR's target), Source the branch merged into (the
// PR's source); Base is the region in their merge base. Lines keep their
// terminators.
type Conflict struct {
	Base, Target, Source []string
	// Trimmed is a conflict whose edges lost lines both sides have there,
	// settled once as git's zealous merge does.
	Trimmed bool
	// Head and Tail are those lines, settled before and after the
	// conflict; Base still holds the whole region. Set by Merge3 only: a
	// diff3 text cannot carry them.
	Head, Tail []string
}

// Trivial reports a conflict code resolves without the model: both sides
// only inserted (the base region is empty), nothing was trimmed from its
// edges, they share no line but blank ones and set no key both ("lodash":
// "^4.17.20" against "^4.17.21", const X = 1 against 2), so target's lines
// then source's lose nothing and repeat nothing. A shared key is the
// model's to judge.
//
// Lines and keys are compared without any whitespace, and a trimmed edge
// rules code out: git/git 8f0d663e, both sides adding the same function,
// left a conflict of two headers one space apart (`f () {` against
// `f() {`) once the shared body was settled, and target's then source's
// defined it twice and broke the file. A trimmed edge means both added
// one block; whatever is left of it is the model's.
func (c *Conflict) Trivial() bool {
	if len(c.Base) > 0 || c.Trimmed {
		return false
	}
	have, keys := map[string]bool{}, map[string]bool{}
	for _, l := range c.Target {
		if t := noSpace(l); t != "" {
			have[t] = true
		}
		if k := lineKey(l); k != "" {
			keys[k] = true
		}
	}
	for _, l := range c.Source {
		if k := lineKey(l); have[noSpace(l)] || k != "" && keys[k] {
			return false
		}
	}
	return true
}

// noSpace is a line without any whitespace.
func noSpace(l string) string {
	return strings.Map(func(r rune) rune {
		if unicode.IsSpace(r) {
			return -1
		}
		return r
	}, l)
}

// lineKey is what a line sets: its text before the first ':' or '=',
// without whitespace, "" for a line that sets nothing that way. Blunt on
// purpose: a key both sides set is a reason to ask, never to drop.
func lineKey(l string) string {
	i := strings.IndexAny(l, ":=")
	if i <= 0 {
		return ""
	}
	return noSpace(l[:i])
}

// Oversized reports a region past MaxRegionLines on any side.
func (c *Conflict) Oversized() bool {
	return len(c.Base) > MaxRegionLines || len(c.Target) > MaxRegionLines || len(c.Source) > MaxRegionLines
}

// Merge3 merges target and source, both changed from base, as git's
// three-way merge does: a change of one side is taken, the same change on
// both is taken once, and different changes to the same or adjacent base
// lines are a conflict. Lines common to both sides at a conflict's edges
// are settled, as git's zealous merge does.
//
// The merge is made twice, aligned by Myers (git merge-file's default) and
// by patience (the histogram family, a server-side merge's), and stands
// only when both agree: where they differ, which lines conflict depends on
// the algorithm, and the merge is ErrAmbiguous rather than a guess.
func Merge3(base, target, source []string) ([]Chunk, error) {
	m, err := merge3(myers, base, target, source)
	if err != nil {
		return nil, err
	}
	p, err := merge3(patience, base, target, source)
	if err != nil {
		return nil, err
	}
	if render3(m) != render3(p) {
		return nil, ErrAmbiguous
	}
	return m, nil
}

// render3 is chunks as one comparable string.
func render3(chunks []Chunk) string {
	var b strings.Builder
	for _, c := range chunks {
		if c.Conflict == nil {
			b.WriteString(strings.Join(c.Lines, ""))
			continue
		}
		b.WriteString("\x00<" + strings.Join(c.Conflict.Base, "") + "\x00|" + strings.Join(c.Conflict.Target, "") +
			"\x00|" + strings.Join(c.Conflict.Source, "") + "\x00>")
	}
	return b.String()
}

func merge3(align aligner, base, target, source []string) ([]Chunk, error) {
	ht, ok := diffWith(align, base, target, MaxEdits)
	if !ok {
		return nil, ErrTooDifferent
	}
	hs, ok := diffWith(align, base, source, MaxEdits)
	if !ok {
		return nil, ErrTooDifferent
	}
	type tagged struct {
		hunk
		target bool
	}
	all := make([]tagged, 0, len(ht)+len(hs))
	i, j := 0, 0
	for i < len(ht) || j < len(hs) {
		if j >= len(hs) || (i < len(ht) && ht[i].b0 <= hs[j].b0) {
			all = append(all, tagged{ht[i], true})
			i++
		} else {
			all = append(all, tagged{hs[j], false})
			j++
		}
	}
	var out []Chunk
	settle := func(ls []string) {
		if len(ls) == 0 {
			return
		}
		if n := len(out); n > 0 && out[n-1].Conflict == nil {
			out[n-1].Lines = append(out[n-1].Lines, ls...)
			return
		}
		out = append(out, Chunk{Lines: append([]string(nil), ls...)})
	}
	pos := 0
	for k := 0; k < len(all); {
		start, end := all[k].b0, all[k].b1
		var firstT, lastT, firstS, lastS *hunk
		n := k
		// Adjacent changes conflict, as in git: a hunk starting where the
		// group ends joins it.
		for n < len(all) && (n == k || all[n].b0 <= end) {
			h := all[n].hunk
			end = max(end, h.b1)
			if all[n].target {
				if firstT == nil {
					firstT = &all[n].hunk
				}
				lastT = &all[n].hunk
			} else {
				if firstS == nil {
					firstS = &all[n].hunk
				}
				lastS = &all[n].hunk
			}
			n++
		}
		settle(base[pos:start])
		region := func(side []string, first, last *hunk) []string {
			if first == nil {
				return base[start:end]
			}
			return side[first.s0-(first.b0-start) : last.s1+(end-last.b1)]
		}
		t, s := region(target, firstT, lastT), region(source, firstS, lastS)
		switch {
		case firstS == nil:
			settle(t)
		case firstT == nil:
			settle(s)
		case slices.Equal(t, s):
			settle(t)
		default:
			p := 0
			for p < len(t) && p < len(s) && t[p] == s[p] {
				p++
			}
			q := 0
			for q < len(t)-p && q < len(s)-p && t[len(t)-1-q] == s[len(s)-1-q] {
				q++
			}
			settle(t[:p])
			out = append(out, Chunk{Conflict: &Conflict{
				Base:    append([]string(nil), base[start:end]...),
				Target:  append([]string(nil), t[p:len(t)-q]...),
				Source:  append([]string(nil), s[p:len(s)-q]...),
				Trimmed: p > 0 || q > 0,
				Head:    append([]string(nil), t[:p]...),
				Tail:    append([]string(nil), t[len(t)-q:]...),
			}})
			settle(t[len(t)-q:])
		}
		pos = end
		k = n
	}
	settle(base[pos:])
	return out, nil
}

// Conflicts is the chunks' conflicts, in order.
func Conflicts(chunks []Chunk) []*Conflict {
	var out []*Conflict
	for _, c := range chunks {
		if c.Conflict != nil {
			out = append(out, c.Conflict)
		}
	}
	return out
}
