package diff

import (
	"hash/fnv"
	"slices"
	"strconv"
	"strings"
)

// NumberDiff prefixes every hunk body line with its new-file line number, so
// the model reads a finding's line instead of counting it from the `@@`
// header. Counting is what put findings past the end of a file and one hunk
// line off in the evals.
//
// Added and context lines carry their number; a removed line and a
// `\ No newline` marker carry a blank gutter of the same width, since neither
// exists in the new file. Header lines (diff --git, ---, +++) and `@@` lines
// pass through unchanged. The gutter width is the widest number actually
// written, so columns line up.
//
// The width is NOT taken from the `@@` counts: a body can hold more new-side
// lines than its header claims (context expansion does not count
// space-stripped blank lines), and a width sized from the header would be one
// digit short the moment the real count crosses 9, 99, 999.
//
// Splits on "\n" only, like ParseHunks: this is the diff the prompt shows,
// and its own terminators are the only ones that may split it.
func NumberDiff(fileDiff string) string {
	lines := strings.Split(fileDiff, "\n")

	// Pass one: the number each line carries, 0 for none.
	nums := make([]int, len(lines))
	highest, next, inHunk := 0, 0, false
	for i, line := range lines {
		if m := hunkHeaderRE.FindStringSubmatch(line); m != nil {
			next, inHunk = atoiOr(m[3], 0), true
			continue
		}
		// The trailing "" of a diff ending in a newline stays as it was, so
		// the rejoin reproduces that newline and nothing more.
		if !inHunk || (i == len(lines)-1 && line == "") || !newSideLine(line) {
			continue
		}
		nums[i] = next
		highest = max(highest, next)
		next++
	}

	// Pass two: render with the width the numbers need.
	width := len(strconv.Itoa(highest))
	blank := strings.Repeat(" ", width)
	out := make([]string, 0, len(lines))
	inHunk = false
	for i, line := range lines {
		switch {
		case hunkHeaderRE.MatchString(line):
			inHunk = true
			out = append(out, line)
		case !inHunk || (i == len(lines)-1 && line == ""):
			out = append(out, line)
		case newSideLine(line):
			out = append(out, pad(nums[i], width)+" "+line)
		default:
			out = append(out, blank+" "+line)
		}
	}
	return strings.Join(out, "\n")
}

// newSideLine reports whether a hunk body line exists in the new file: an
// added line, a context line, or an empty line (a context line whose single
// leading space was stripped in transit).
func newSideLine(line string) bool {
	return line == "" || line[0] == '+' || line[0] == ' '
}

func pad(n, width int) string {
	s := strconv.Itoa(n)
	return strings.Repeat(" ", width-len(s)) + s
}

// ShownLine is one new-side line the prompt showed for a file: its text
// reduced to two hashes, and whether the diff added it.
//
// Hash is whitespace-loose, for matching a quote. Exact keeps the leading
// whitespace, for judging a suggestion: in Python, YAML or a Makefile a
// re-indented line is a different line, and a fix that only changes the
// indent is a real fix.
type ShownLine struct {
	Hash  uint64
	Exact uint64
	Added bool
}

// FileAnchors is what one file's diff showed: its new-side lines by line
// number, and the hashes of the lines it removed.
//
// Removed lines are evidence but never an anchor. A finding about deleted
// code (a dropped `defer`, a lost check) quotes the removed line, yet can
// only be posted on a line that exists in the new file.
//
// Content is every line of the full file the prompt showed, hashed, sorted
// and deduplicated for binary search. Evidence may quote it (the prompt shows
// the whole file, and a model quoting the call a bug hangs on quotes it from
// there), but like a removed line it anchors nothing: only diff lines can be
// posted on. A slice, not a map: eight bytes a line, for every file.
type FileAnchors struct {
	Shown   map[int]ShownLine
	Removed map[uint64]bool
	Content []uint64
}

// InContent reports whether the full file has a line hashing to h.
func (fa FileAnchors) InContent(h uint64) bool {
	_, found := slices.BinarySearch(fa.Content, h)
	return found
}

// AnchorIndex maps each reviewed path to what its diff showed.
//
// Hashes rather than text: it rides on the review plan across the inference
// stage, and the plan carries no file bodies and no diff. Eight bytes and a
// flag per shown line is what a finding's evidence is checked against.
type AnchorIndex map[string]FileAnchors

// BuildAnchorIndex indexes the diffs as the prompt renders them. Build it
// from the final file list, after compression and context expansion, so it
// holds exactly the lines the model saw.
func BuildAnchorIndex(files []FileReviewData) AnchorIndex {
	idx := make(AnchorIndex, len(files))
	for _, f := range files {
		fa := FileAnchors{Shown: map[int]ShownLine{}, Removed: map[uint64]bool{}}
		_, hunks := ParseHunks(f.Diff)
		for _, h := range hunks {
			n := h.NewStart
			for _, line := range h.BodyLines {
				switch {
				case newSideLine(line):
					text := line
					if text != "" {
						text = text[1:]
					}
					fa.Shown[n] = ShownLine{Hash: HashLine(text), Exact: HashExact(text), Added: strings.HasPrefix(line, "+")}
					n++
				case line[0] == '-':
					fa.Removed[HashLine(line[1:])] = true
				}
			}
		}
		if f.ContentFetched {
			for _, line := range strings.Split(f.Content, "\n") {
				fa.Content = append(fa.Content, HashLine(line))
			}
			slices.Sort(fa.Content)
			fa.Content = slices.Compact(fa.Content)
		}
		idx[f.Path] = fa
	}
	return idx
}

// HashLine is the hash a shown line and a quoted evidence line are compared
// by. Whitespace runs collapse to one space and the ends are trimmed, so a
// quote that re-indents or re-spaces the code still matches it. strings.Fields
// splits on Unicode space, NBSP included, which the diff text can carry.
func HashLine(text string) uint64 {
	h := fnv.New64a()
	_, _ = h.Write([]byte(strings.Join(strings.Fields(text), " ")))
	return h.Sum64()
}

// HashExact hashes a line as written, trailing whitespace and a CR aside:
// the indent counts. See ShownLine.
func HashExact(text string) uint64 {
	h := fnv.New64a()
	_, _ = h.Write([]byte(strings.TrimRight(text, " \t\r")))
	return h.Sum64()
}
