// Package diff parses unified diffs, decides which files are worth reviewing,
// expands hunk context and fits files into a token budget.
//
// Ported from the Python service's diff_compression.py, context_expansion.py,
// cross_file_context.py and the diff half of llm_client.py. Behaviour is pinned
// to the Python except where AGENTS.md records a divergence.
package diff

import (
	"regexp"
	"strconv"
	"strings"
)

// FileReviewData is one file's diff plus, when fetched, its full new-side
// content.
type FileReviewData struct {
	Path string
	Diff string
	// Content is the full new-side file, empty when it was not fetched OR when
	// the file itself is empty. ContentFetched tells the two apart.
	Content string
	// ContentFetched records that content was fetched, even if the file turned
	// out to be empty.
	//
	// Python's content is Optional[str] and the two cases differ where it
	// matters: the prompt's file entry renders an empty code block for a
	// fetched empty file and "content omitted" for an unfetched one. A newly
	// created empty file in a PR reaches that path, since the reviewer never
	// coerces "" to None.
	ContentFetched bool
}

// HasContent reports whether content is usable as text, which is Python's
// `content or diff` truthiness: an empty string falls through to the diff
// exactly like None did. Use ContentFetched, not this, when the distinction
// between unfetched and fetched-but-empty matters.
func (f FileReviewData) HasContent() bool { return f.Content != "" }

var diffPathRE = regexp.MustCompile(`(?m)^diff --git (?:a/.+ b/|src://.+ dst://)(.+)$`)

var plusPlusPlusRE = regexp.MustCompile(`(?m)^\+\+\+ (?:b/|dst://)(.+)$`)

// hunkHeaderRE matches `@@ -a,b +c,d @@`. It is deliberately not anchored at the
// end: `@@ -1,2 +1,3 @@ def foo():` parses and the trailing function context is
// discarded, as in Python.
var hunkHeaderRE = regexp.MustCompile(`^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@`)

// Hunk is one parsed `@@` section. BodyLines excludes the header line.
type Hunk struct {
	OldStart  int
	OldCount  int
	NewStart  int
	NewCount  int
	BodyLines []string
}

// SplitByFile splits a combined diff into per-file diffs.
//
// Lossless: concatenating the parts reproduces the input. Any preamble before
// the first `diff --git ` becomes its own leading part rather than being glued
// onto the first file, matching Python.
//
// Mirrors Python's splitlines(keepends=True), which also breaks on \v, \f,
// \x1c-\x1e, \x85, U+2028 and U+2029, not just \n.
func SplitByFile(diffText string) []string {
	var parts []string
	var current []string

	for _, line := range splitLinesKeepEnds(diffText) {
		if strings.HasPrefix(line, "diff --git ") && len(current) > 0 {
			parts = append(parts, strings.Join(current, ""))
			current = nil
		}
		current = append(current, line)
	}
	if len(current) > 0 {
		parts = append(parts, strings.Join(current, ""))
	}
	return parts
}

// ExtractPath returns the b/ side path from a per-file diff header, or "" when
// no path can be parsed. The path is not lowercased.
func ExtractPath(fileDiff string) string {
	if m := diffPathRE.FindStringSubmatch(fileDiff); m != nil {
		return strings.TrimRight(m[1], "\r")
	}
	// Fallback: the +++ header, in standard b/ or Bitbucket dst:// form.
	if m := plusPlusPlusRE.FindStringSubmatch(fileDiff); m != nil {
		return strings.TrimRight(m[1], "\r")
	}
	return ""
}

// IsDeleted reports whether a per-file diff represents a deletion.
func IsDeleted(fileDiff string) bool {
	return strings.Contains(fileDiff, "\n+++ /dev/null") ||
		strings.HasPrefix(fileDiff, "+++ /dev/null")
}

// ParseHunks splits a per-file diff into the leading header lines and its hunks.
//
// Splits on "\n" rather than by Unicode line boundaries: a diff's own line
// terminators are the only ones that may split a hunk.
//
// Pinned (AGENTS.md): a trailing empty body line, the artifact of splitting a
// diff that ends in a newline, is dropped here. Keeping it leaves a phantom
// blank line mid-body whenever after-context follows.
func ParseHunks(fileDiff string) (headerLines []string, hunks []*Hunk) {
	var current *Hunk

	for _, line := range strings.Split(fileDiff, "\n") {
		if m := hunkHeaderRE.FindStringSubmatch(line); m != nil {
			if current != nil {
				hunks = append(hunks, current)
			}
			current = &Hunk{
				OldStart: atoiOr(m[1], 0),
				OldCount: atoiOr(m[2], 1),
				NewStart: atoiOr(m[3], 0),
				NewCount: atoiOr(m[4], 1),
			}
			continue
		}
		if current != nil {
			current.BodyLines = append(current.BodyLines, line)
		} else {
			headerLines = append(headerLines, line)
		}
	}
	if current != nil {
		hunks = append(hunks, current)
	}

	for _, h := range hunks {
		h.stripTrailingEmpty()
	}
	return headerLines, hunks
}

// stripTrailingEmpty drops the split("\n") artifact from a diff ending in a
// newline. Only the last hunk of a diff can carry it.
func (h *Hunk) stripTrailingEmpty() {
	if n := len(h.BodyLines); n > 0 && h.BodyLines[n-1] == "" {
		h.BodyLines = h.BodyLines[:n-1]
	}
}

// atoiOr parses s, returning def when s is empty (an absent `,count` means 1).
func atoiOr(s string, def int) int {
	if s == "" {
		return def
	}
	n, err := strconv.Atoi(s)
	if err != nil {
		return def
	}
	return n
}

// splitLinesKeepEnds splits on every boundary Python's str.splitlines treats as
// a line break, keeping the terminator on each line. Used only where the Python
// used splitlines; code mirroring split("\n") uses strings.Split directly.
func splitLinesKeepEnds(s string) []string {
	if s == "" {
		return nil
	}
	var out []string
	start := 0
	runes := []rune(s)
	for i := 0; i < len(runes); i++ {
		r := runes[i]
		if !isPythonLineBreak(r) {
			continue
		}
		end := i + 1
		// \r\n counts as one break.
		if r == '\r' && end < len(runes) && runes[end] == '\n' {
			end++
		}
		out = append(out, string(runes[start:end]))
		start = end
		i = end - 1
	}
	if start < len(runes) {
		out = append(out, string(runes[start:]))
	}
	return out
}

// isPythonLineBreak reports whether r is a boundary for str.splitlines.
func isPythonLineBreak(r rune) bool {
	switch r {
	case '\n', '\r', '\v', '\f', 0x1c, 0x1d, 0x1e, 0x85, 0x2028, 0x2029:
		return true
	}
	return false
}
