package snapshot

import (
	"strings"

	"github.com/trick77/noergler/internal/diff"
)

// Own is a build file's lines that name the project's own version, by
// line number, that only other files tell (a POM read whole, the build
// script naming gradle.properties' version key): Old in the file before
// the PR, New after. OldUnknown: the file before could not be told
// (unread, unparsed); a block's removed lines then exempt only as many
// added dependencies as they outnumber its added own lines (an own bump
// pairs with an own line): the user's 1:1 bump of a dependency still
// brings none, an own bump never exempts a dependency. The zero value
// knows none.
type Own struct {
	Old, New   map[int]bool
	OldUnknown bool
}

// bodyLine is one line of a hunk body: op '+', '-' or ' ' (context), its
// number on its side (the new file's for context); snap when it names an
// unreleased build (Line), cand when it is changed, snap and no own
// version by the line alone (OwnLine).
type bodyLine struct {
	op         byte
	text       string
	n          int
	snap, cand bool
}

// FileDiff is one build file's diff, parsed once.
type FileDiff struct {
	path  string
	hunks [][]bodyLine
}

// before is the last line on one side that is no blank or comment, and
// whether it is known: at a file's start nothing is before; at a hunk's
// start mid-file it is not known.
type before struct {
	text  string
	known bool
}

func (b before) next(l string) before {
	t := strings.TrimSpace(l)
	if t == "" || strings.HasPrefix(t, "//") || strings.HasPrefix(t, "/*") || strings.HasPrefix(t, "*") ||
		strings.HasPrefix(t, "#") {
		return b
	}
	return before{text: t, known: true}
}

// ParseDiff parses one build file's diff at path p. A deleted file has no
// lines. "\ No newline at end of file" is dropped: it sits inside a block,
// never ends one.
func ParseDiff(p, fileDiff string) *FileDiff {
	d := &FileDiff{path: p}
	if diff.IsDeleted(fileDiff) {
		return d
	}
	line := func(op byte, text string, n int, b before) bodyLine {
		bl := bodyLine{op: op, text: text, n: n, snap: Line(text)}
		bl.cand = bl.snap && op != ' ' && !OwnLine(p, text, b.text, b.known)
		return bl
	}
	_, hunks := diff.ParseHunks(fileDiff)
	for _, h := range hunks {
		var bls []bodyLine
		n, o := h.NewStart, h.OldStart
		bNew, bOld := before{known: n <= 1}, before{known: o <= 1}
		for _, l := range h.BodyLines {
			l = strings.TrimSuffix(l, "\r")
			switch {
			case strings.HasPrefix(l, `\`):
			case strings.HasPrefix(l, "-"):
				bls = append(bls, line('-', l[1:], o, bOld))
				bOld = bOld.next(l[1:])
				o++
			case strings.HasPrefix(l, "+"):
				bls = append(bls, line('+', l[1:], n, bNew))
				bNew = bNew.next(l[1:])
				n++
			default:
				t := strings.TrimPrefix(l, " ")
				bls = append(bls, line(' ', t, n, bNew))
				bNew, bOld = bNew.next(t), bOld.next(t)
				n++
				o++
			}
		}
		d.hunks = append(d.hunks, bls)
	}
	return d
}

// candidates is the candidate lines of one side by number ('+': new,
// '-': old).
func (d *FileDiff) candidates(op byte) map[int]bool {
	out := map[int]bool{}
	for _, h := range d.hunks {
		for _, bl := range h {
			if bl.op == op && bl.cand {
				out[bl.n] = true
			}
		}
	}
	return out
}

// Added is the added candidate lines by number: only they can bring a
// hit, and only then is another file worth reading.
func (d *FileDiff) Added() map[int]bool { return d.candidates('+') }

// Removed is the removed candidate lines by their number in the file
// before.
func (d *FileDiff) Removed() map[int]bool { return d.candidates('-') }

// OldMatters reports whether the file before the PR can change the hits,
// the new side's own lines known: only a change block that adds a
// dependency line and removes a candidate, which may exempt it. A reactor
// version bump (own lines only) never needs the old file.
func (d *FileDiff) OldMatters(newOwn map[int]bool) bool {
	for _, h := range d.hunks {
		deps, removed := 0, 0
		for _, bl := range append(h, bodyLine{op: ' '}) {
			switch {
			case bl.op == ' ':
				if deps > 0 && removed > 0 {
					return true
				}
				deps, removed = 0, 0
			case !bl.cand:
			case bl.op == '+' && !newOwn[bl.n]:
				deps++
			case bl.op == '-':
				removed++
			}
		}
	}
	return false
}

// Consistent reports whether content is the file the diff shows on one
// side (old: before the PR): every candidate line of that side stands at
// its number. A push between the diff's read and the file's misaligns
// the own lines. Only the candidates are compared, the lines that matter,
// and a line the raw read could not decode (U+FFFD: a charset the diff
// converted) is skipped.
func (d *FileDiff) Consistent(content string, old bool) bool {
	lines := strings.Split(content, "\n")
	op := byte('+')
	if old {
		op = '-'
	}
	for _, h := range d.hunks {
		for _, bl := range h {
			if bl.op != op || !bl.cand {
				continue
			}
			if bl.n < 1 || bl.n > len(lines) {
				return false
			}
			got := lines[bl.n-1]
			if !strings.ContainsRune(got, '�') && strings.TrimSpace(got) != strings.TrimSpace(bl.text) {
				return false
			}
		}
	}
	return true
}

// AddedNaming is the added candidate lines that name one of the versions
// as an element's whole text (`>1.0-SNAPSHOT<`): a POM's own lines told
// by the version strings alone, when it is not read whole.
func (d *FileDiff) AddedNaming(versions map[string]bool) map[int]bool {
	out := map[int]bool{}
	for _, h := range d.hunks {
		for _, bl := range h {
			if bl.op != '+' || !bl.cand {
				continue
			}
			for v := range versions {
				if strings.Contains(bl.text, ">"+v+"<") {
					out[bl.n] = true
					break
				}
			}
		}
	}
	return out
}

// PropertyOwn is a gradle.properties diff's own lines on both sides: the
// candidates whose key is one the build script sets the project's version
// from (GradleVersionKeys).
func (d *FileDiff) PropertyOwn(keys map[string]bool) Own {
	own := Own{Old: map[int]bool{}, New: map[int]bool{}}
	for _, h := range d.hunks {
		for _, bl := range h {
			if !bl.cand || !keys[propertyKey(bl.text)] {
				continue
			}
			if bl.op == '+' {
				own.New[bl.n] = true
			} else {
				own.Old[bl.n] = true
			}
		}
	}
	return own
}

// TakenAsOwn is the added lines that name an unreleased build but count
// as the project's own version, for the log line that says so.
func (d *FileDiff) TakenAsOwn(own Own) []int {
	var out []int
	for _, h := range d.hunks {
		for _, bl := range h {
			if bl.op == '+' && bl.snap && (!bl.cand || own.New[bl.n]) {
				out = append(out, bl.n)
			}
		}
	}
	return out
}

// Hits is the PR delta's hits in the file: the lines it adds that name an
// unreleased dependency, never a context line, a removed one or the
// project's own version (OwnLine, own). A change that replaces such a
// line with another (1.0-SNAPSHOT to 1.1-SNAPSHOT) brings none in (user,
// 2026-11-15): per change block, a run of removed and added lines, as many
// added hits are exempt as the block removes; only the rest count. Own
// lines count on neither side, so the own version's bump never exempts a
// dependency's line. A hit moved to another block counts there, the safe
// way.
func (d *FileDiff) Hits(own Own) []Hit {
	var hits []Hit
	for _, h := range d.hunks {
		removed, addedOwn := 0, 0
		var added []Hit
		flush := func() {
			if own.OldUnknown {
				removed = max(0, removed-addedOwn)
			}
			if len(added) > removed {
				hits = append(hits, added[removed:]...)
			}
			removed, addedOwn, added = 0, 0, nil
		}
		for _, bl := range h {
			switch {
			case bl.op == ' ':
				flush()
			case !bl.cand:
			case bl.op == '-':
				if !own.Old[bl.n] {
					removed++
				}
			case own.New[bl.n]:
				addedOwn++
			default:
				added = append(added, Hit{Path: d.path, Line: bl.n, Text: bl.text})
			}
		}
		flush()
	}
	seq := map[string]int{}
	for i := range hits {
		t := noSpace(hits[i].Text)
		seq[t]++
		hits[i].Seq = seq[t]
	}
	return hits
}
