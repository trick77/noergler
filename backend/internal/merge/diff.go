// Package merge is noergler's three-way merge of one file: the line diff,
// the diff3 regions, which conflicts count as simple, the resolution of the
// trivial ones and the blunt check of the model's.
//
// It mirrors git's merge closely enough to agree with it on what conflicts
// (TestAgreesWithGit), never exactly: a merge whose alignment it doubts is
// refused, never guessed.
package merge

import "bytes"

// Lines splits b into lines, each keeping its terminator ("\n" or "\r\n");
// the last one has none when b does not end in a newline. Byte for byte:
// strings.Join(Lines(b), "") is b.
func Lines(b []byte) []string {
	var out []string
	for len(b) > 0 {
		i := bytes.IndexByte(b, '\n')
		if i < 0 {
			out = append(out, string(b))
			break
		}
		out = append(out, string(b[:i+1]))
		b = b[i+1:]
	}
	return out
}

// hunk is one change of a side against the base: base lines [b0,b1) became
// side lines [s0,s1). An insertion has b0 == b1, a deletion s0 == s1.
type hunk struct{ b0, b1, s0, s1 int }

// aligner is a line diff: the matched index pairs of a and b, ascending,
// false past maxEdits.
type aligner func(a, b []int, maxEdits int) ([][2]int, bool)

// diff is the hunks turning base into side by Myers' diff as found, nil and
// false when they need more than maxEdits line edits: a file changed that
// much is no simple merge, and the search's memory grows with the square of
// the edits.
func diff(base, side []string, maxEdits int) ([]hunk, bool) {
	hs, err := diffWith(myers, false, base, side, maxEdits)
	return hs, err == nil
}

// diffWith is the hunks turning base into side by align; slid, with every
// change placed as git places it (compact). ErrTooDifferent past maxEdits;
// ErrAmbiguous when compact loses its footing, which a correct alignment
// never makes it do: refused, never looped on.
func diffWith(align aligner, slid bool, base, side []string, maxEdits int) ([]hunk, error) {
	ids := map[string]int{}
	intern := func(ls []string) []int {
		out := make([]int, len(ls))
		for i, l := range ls {
			id, ok := ids[l]
			if !ok {
				id = len(ids)
				ids[l] = id
			}
			out[i] = id
		}
		return out
	}
	a, b := intern(base), intern(side)
	// The common head and tail cost the search nothing.
	pre := 0
	for pre < len(a) && pre < len(b) && a[pre] == b[pre] {
		pre++
	}
	suf := 0
	for suf < len(a)-pre && suf < len(b)-pre && a[len(a)-1-suf] == b[len(b)-1-suf] {
		suf++
	}
	matches, ok := align(a[pre:len(a)-suf], b[pre:len(b)-suf], maxEdits)
	if !ok {
		return nil, ErrTooDifferent
	}
	// Changed lines of each side, the alignment's gaps.
	ca, cb := make([]bool, len(a)), make([]bool, len(b))
	x, y := pre, pre
	for _, m := range append(matches, [2]int{len(a) - pre - suf, len(b) - pre - suf}) {
		for ; x < m[0]+pre; x++ {
			ca[x] = true
		}
		for ; y < m[1]+pre; y++ {
			cb[y] = true
		}
		x, y = m[0]+pre+1, m[1]+pre+1
	}
	if slid && (!compact(a, ca, cb) || !compact(b, cb, ca)) {
		return nil, ErrAmbiguous
	}
	var hs []hunk
	i, j := 0, 0
	for i < len(a) || j < len(b) {
		if (i < len(a) && ca[i]) || (j < len(b) && cb[j]) {
			h := hunk{b0: i, s0: j}
			for i < len(a) && ca[i] {
				i++
			}
			for j < len(b) && cb[j] {
				j++
			}
			h.b1, h.s1 = i, j
			hs = append(hs, h)
			continue
		}
		i, j = i+1, j+1
	}
	return hs, nil
}

// compact places each group of changed lines in rec (changed marks it,
// other marks the other side's) as git's xdl_change_compact does, without
// the indent heuristic, which git merge-file leaves off: a group that can
// slide (an inserted line next to an equal one, a blank beside a blank)
// goes as far down as it can, unless on the way it lined up with a change
// of the other side, where it goes back to the last such place. Where a
// group sits decides whether it touches the other branch's change.
//
// False where git calls BUG(): the two sides' groups out of step. A
// correct alignment never gets there; a wrong one is refused, never looped
// on.
func compact(rec []int, changed, other []bool) bool {
	n := len(changed)
	at := func(c []bool, i int) bool { return i >= 0 && i < len(c) && c[i] }
	// A group is [start, end) of changed lines, possibly empty; the k-th
	// group of one side faces the k-th of the other.
	type group struct{ start, end int }
	next := func(c []bool, g *group) bool {
		if g.end >= len(c) {
			return false
		}
		g.start = g.end + 1
		for g.end = g.start; at(c, g.end); g.end++ {
		}
		return true
	}
	prev := func(c []bool, g *group) bool {
		if g.start == 0 {
			return false
		}
		g.end = g.start - 1
		for g.start = g.end; at(c, g.start-1); g.start-- {
		}
		return true
	}
	slideDown := func(g *group) bool {
		if g.end < n && rec[g.start] == rec[g.end] {
			changed[g.start], changed[g.end] = false, true
			g.start++
			g.end++
			for at(changed, g.end) {
				g.end++
			}
			return true
		}
		return false
	}
	slideUp := func(g *group) bool {
		if g.start > 0 && rec[g.start-1] == rec[g.end-1] {
			g.start--
			g.end--
			changed[g.start], changed[g.end] = true, false
			for at(changed, g.start-1) {
				g.start--
			}
			return true
		}
		return false
	}
	// The first groups: next from before the start.
	g, og := group{-1, -1}, group{-1, -1}
	next(changed, &g)
	next(other, &og)
	for {
		if g.end != g.start {
			var earliestEnd, endMatchingOther int
			for {
				size := g.end - g.start
				endMatchingOther = -1
				for slideUp(&g) {
					if !prev(other, &og) {
						return false
					}
				}
				earliestEnd = g.end
				if og.end > og.start {
					endMatchingOther = g.end
				}
				for slideDown(&g) {
					if !next(other, &og) {
						return false
					}
					if og.end > og.start {
						endMatchingOther = g.end
					}
				}
				if size == g.end-g.start {
					break
				}
			}
			if g.end != earliestEnd && endMatchingOther != -1 {
				for og.end == og.start {
					if !slideUp(&g) || !prev(other, &og) {
						return false
					}
				}
			}
		}
		if !next(changed, &g) {
			return true
		}
		if !next(other, &og) {
			return false
		}
	}
}

// myers is the matched index pairs of a and b, ascending, by Myers' greedy
// O(ND) search; false past maxEdits edits.
func myers(a, b []int, maxEdits int) ([][2]int, bool) {
	n, m := len(a), len(b)
	if n == 0 || m == 0 {
		return nil, n+m <= maxEdits
	}
	limit := min(maxEdits, n+m)
	off := limit + 1
	v := make([]int, 2*limit+3)
	// trace[d] is v over k in [-d,d] after step d, indexed k+d.
	var trace [][]int
	for d := 0; d <= limit; d++ {
		for k := -d; k <= d; k += 2 {
			var x int
			if k == -d || (k != d && v[off+k-1] < v[off+k+1]) {
				x = v[off+k+1]
			} else {
				x = v[off+k-1] + 1
			}
			y := x - k
			for x < n && y < m && a[x] == b[y] {
				x++
				y++
			}
			v[off+k] = x
			if x >= n && y >= m {
				return backtrack(trace, d, n, m), true
			}
		}
		trace = append(trace, append([]int(nil), v[off-d:off+d+1]...))
	}
	return nil, false
}

// patience is the patience diff, the family git's histogram diff (the
// merge's default since ort) belongs to: lines unique on both sides anchor
// the alignment, their longest common sequence first, Myers only between
// anchors. It aligns repeated lines ("}", blank) by their unique
// neighbours, where Myers may pick another copy.
func patience(a, b []int, maxEdits int) ([][2]int, bool) {
	var out [][2]int
	var rec func(a0, a1, b0, b1 int) bool
	rec = func(a0, a1, b0, b1 int) bool {
		for a0 < a1 && b0 < b1 && a[a0] == b[b0] {
			out = append(out, [2]int{a0, b0})
			a0++
			b0++
		}
		var tail [][2]int
		for a0 < a1 && b0 < b1 && a[a1-1] == b[b1-1] {
			a1--
			b1--
			tail = append(tail, [2]int{a1, b1})
		}
		defer func() {
			for i := len(tail) - 1; i >= 0; i-- {
				out = append(out, tail[i])
			}
		}()
		if a0 == a1 || b0 == b1 {
			return (a1-a0)+(b1-b0) <= maxEdits
		}
		// Lines once in each range, in a's order, with their place in b.
		count := map[int][2]int{}
		for i := a0; i < a1; i++ {
			c := count[a[i]]
			c[0]++
			count[a[i]] = c
		}
		at := map[int]int{}
		for j := b0; j < b1; j++ {
			c := count[b[j]]
			c[1]++
			count[b[j]] = c
			at[b[j]] = j
		}
		var cand [][2]int
		for i := a0; i < a1; i++ {
			if c := count[a[i]]; c[0] == 1 && c[1] == 1 {
				cand = append(cand, [2]int{i, at[a[i]]})
			}
		}
		anchors := lis(cand)
		if len(anchors) == 0 {
			m, ok := myers(a[a0:a1], b[b0:b1], maxEdits)
			for _, p := range m {
				out = append(out, [2]int{p[0] + a0, p[1] + b0})
			}
			return ok
		}
		pa, pb := a0, b0
		for _, an := range anchors {
			if !rec(pa, an[0], pb, an[1]) {
				return false
			}
			out = append(out, an)
			pa, pb = an[0]+1, an[1]+1
		}
		return rec(pa, a1, pb, b1)
	}
	if !rec(0, len(a), 0, len(b)) {
		return nil, false
	}
	return out, true
}

// lis is the longest run of pairs increasing in both components, pairs
// given in ascending first component (patience sorting).
func lis(pairs [][2]int) [][2]int {
	if len(pairs) == 0 {
		return nil
	}
	var tops []int // index into pairs of each pile's top
	prev := make([]int, len(pairs))
	for i, p := range pairs {
		lo, hi := 0, len(tops)
		for lo < hi {
			mid := (lo + hi) / 2
			if pairs[tops[mid]][1] < p[1] {
				lo = mid + 1
			} else {
				hi = mid
			}
		}
		prev[i] = -1
		if lo > 0 {
			prev[i] = tops[lo-1]
		}
		if lo == len(tops) {
			tops = append(tops, i)
		} else {
			tops[lo] = i
		}
	}
	out := make([][2]int, len(tops))
	for k, i := len(tops)-1, tops[len(tops)-1]; k >= 0; k, i = k-1, prev[i] {
		out[k] = pairs[i]
	}
	return out
}

// backtrack walks the trace from (n,m) at step d back to the origin and
// returns the diagonal moves, the matches.
func backtrack(trace [][]int, d, x, y int) [][2]int {
	var rev [][2]int
	for ; d > 0; d-- {
		prev := trace[d-1]
		at := func(k int) int { return prev[k+d-1] }
		k := x - y
		var pk int
		if k == -d || (k != d && at(k-1) < at(k+1)) {
			pk = k + 1
		} else {
			pk = k - 1
		}
		px := at(pk)
		py := px - pk
		// The edit lands on (sx,sy); the snake runs from there to (x,y).
		sx := px
		if pk == k-1 {
			sx = px + 1
		}
		sy := sx - k
		for x > sx && y > sy {
			x--
			y--
			rev = append(rev, [2]int{x, y})
		}
		x, y = px, py
	}
	for x > 0 && y > 0 {
		x--
		y--
		rev = append(rev, [2]int{x, y})
	}
	out := make([][2]int, len(rev))
	for i, p := range rev {
		out[len(rev)-1-i] = p
	}
	return out
}
