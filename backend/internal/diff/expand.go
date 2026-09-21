package diff

import (
	"fmt"
	"regexp"
	"strings"
)

// noDynamicLanguages skip the enclosing-scope search. Cross-file extraction
// deliberately does not honour this set.
var noDynamicLanguages = map[string]bool{
	"config": true, "docs": true, "css": true, "html": true, "build-config": true,
}

// scopePatterns find an enclosing declaration to pull context back to.
//
// Every \w ported from a Python pattern becomes [\pL\pN_]: RE2's \w is ASCII
// while Python's is Unicode, so `const café = (x) => x` matches in Python and
// would not match here with \w.
var scopePatterns = map[string]*regexp.Regexp{
	"python": regexp.MustCompile(`^\s*(?:def |class |async def )`),
	"jvm": regexp.MustCompile(
		`^\s*(?:public |private |protected |static |final |abstract |default |fun |class |interface |enum |override |@)` +
			`|.*\{\s*$`),
	"typescript": regexp.MustCompile(
		`^\s*(?:function |class |interface |enum |export |const [\pL\pN_]+\s*=\s*(?:\(|async))` +
			`|.*\{\s*$`),
	"other": regexp.MustCompile(`^\s*(?:def |func |fn )` + `|.*\{\s*$`),
}

// expandedHunk carries the rebuilt header plus the before/after context counts,
// which the merge needs and cannot recompute from the body alone.
type expandedHunk struct {
	oldStart, oldCount int
	newStart, newCount int
	body               []string
	beforeCount        int
	afterCount         int
}

// findEnclosingScopeLine searches strictly above fromLine for a declaration,
// returning its 1-based line number, or 0 when none is found.
//
// The caller passes ctx_start, not the hunk start, so total before-context caps
// at before+maxDynamicBefore. The line at fromLine itself is never tested and
// the nearest match wins.
func findEnclosingScopeLine(fileLines []string, fromLine, maxLines int, path string) int {
	lang := DetectLanguage(path)
	if noDynamicLanguages[lang] {
		return 0
	}
	pattern, ok := scopePatterns[lang]
	if !ok {
		pattern = scopePatterns["other"]
	}

	startIdx := fromLine - 1
	searchLimit := maxLines
	if startIdx < searchLimit {
		searchLimit = startIdx
	}
	for offset := 1; offset <= searchLimit; offset++ {
		idx := startIdx - offset
		if idx < 0 {
			break
		}
		// Pinned (AGENTS.md): skip lines past the end of content instead of
		// indexing past it. Content is byte-capped at the socket while the diff
		// is not, so a truncated large file plus a late hunk reaches here.
		// Indexing past the end would panic and take down the single queue
		// worker for every team.
		if idx >= len(fileLines) {
			continue
		}
		if pattern.MatchString(fileLines[idx]) {
			return idx + 1
		}
	}
	return 0
}

// ExpandContext widens each hunk with surrounding file content and merges hunks
// whose windows overlap.
//
// content is the full new-side file; when empty the diff is returned unchanged,
// when empty the diff is returned unchanged.
//
// Pinned (AGENTS.md): adjacent hunks merge without losing diff lines. The
// overlap comes off the first hunk's added context, never off the second
// hunk's body; trimming the body silently drops removal lines whenever the
// overlap exceeds that hunk's before-context.
func ExpandContext(fileDiff, content, path string, before, after, maxDynamicBefore int, dynamicContext bool) string {
	if content == "" {
		return fileDiff
	}
	headerLines, hunks := ParseHunks(fileDiff)
	if len(hunks) == 0 {
		return fileDiff
	}

	// Mirrors Python split("\n"), not splitlines().
	fileLines := strings.Split(content, "\n")
	totalFileLines := len(fileLines)

	expanded := make([]*expandedHunk, 0, len(hunks))
	for _, hunk := range hunks {
		ctxStart := hunk.NewStart - before
		if ctxStart < 1 {
			ctxStart = 1
		}
		if dynamicContext {
			if scope := findEnclosingScopeLine(fileLines, ctxStart, maxDynamicBefore, path); scope > 0 && scope < ctxStart {
				ctxStart = scope
			}
		}

		// Not clamped to the file length; only ctxStart >= 1 is enforced. A
		// content that disagrees with the diff yields header counts exceeding
		// the body, as in Python.
		beforeCount := hunk.NewStart - ctxStart

		// Counts body lines that are non-empty and do not start with "-". This
		// counts `\ No newline at end of file` as a real line, pushing the
		// after-window one line too far. Ported as is.
		hunkNewLineCount := 0
		for _, line := range hunk.BodyLines {
			if line != "" && !strings.HasPrefix(line, "-") {
				hunkNewLineCount++
			}
		}
		hunkEnd := hunk.NewStart + hunkNewLineCount
		ctxEnd := hunkEnd + after - 1
		if ctxEnd > totalFileLines {
			ctxEnd = totalFileLines
		}
		afterCount := ctxEnd - hunkEnd + 1
		if afterCount < 0 {
			afterCount = 0
		}

		var beforeLines []string
		for i := ctxStart - 1; i < ctxStart-1+beforeCount; i++ {
			if i >= 0 && i < totalFileLines {
				beforeLines = append(beforeLines, " "+fileLines[i])
			}
		}
		var afterLines []string
		for i := hunkEnd - 1; i < hunkEnd-1+afterCount; i++ {
			if i >= 0 && i < totalFileLines {
				afterLines = append(afterLines, " "+fileLines[i])
			}
		}

		body := make([]string, 0, len(beforeLines)+len(hunk.BodyLines)+len(afterLines))
		body = append(body, beforeLines...)
		body = append(body, hunk.BodyLines...)
		body = append(body, afterLines...)

		// The trailing-empty strip fires only when the after-context is empty.
		// ParseHunks already removed the split artifact, so this catches a body
		// that genuinely ends blank. afterCount is decremented so the merge
		// arithmetic does not work from a stale count.
		if n := len(body); n > 0 && body[n-1] == "" {
			body = body[:n-1]
			if afterCount > 0 {
				afterCount--
			}
		}

		// Per-hunk counting rule: "non-empty and not +/-". The merged rule below
		// differs; they disagree on `\ No newline`. Do not unify them.
		var oldRemoved, oldUnchanged, newAdded int
		for _, ln := range hunk.BodyLines {
			switch {
			case strings.HasPrefix(ln, "-"):
				oldRemoved++
			case strings.HasPrefix(ln, "+"):
				newAdded++
			case ln != "":
				oldUnchanged++
			}
		}

		// old/new stay aligned through the delta, since beforeCount is computed
		// in new-file coordinates and the two sides can diverge.
		oldNewDelta := hunk.OldStart - hunk.NewStart
		newNewStart := hunk.NewStart - beforeCount
		newOldStart := newNewStart + oldNewDelta
		if newOldStart < 1 {
			newOldStart = 1
		}

		expanded = append(expanded, &expandedHunk{
			oldStart:    newOldStart,
			oldCount:    beforeCount + oldRemoved + oldUnchanged + afterCount,
			newStart:    newNewStart,
			newCount:    beforeCount + newAdded + oldUnchanged + afterCount,
			body:        body,
			beforeCount: beforeCount,
			afterCount:  afterCount,
		})
	}

	merged := mergeExpandedHunks(expanded)

	out := make([]string, 0, len(headerLines)+len(merged)*4)
	out = append(out, headerLines...)
	for _, m := range merged {
		// Always emits both counts, even when the input was `-10,1`. Any
		// function-context text after the original @@ is discarded.
		out = append(out, fmt.Sprintf("@@ -%d,%d +%d,%d @@", m.oldStart, m.oldCount, m.newStart, m.newCount))
		out = append(out, m.body...)
	}
	return strings.Join(out, "\n")
}

// mergeExpandedHunks joins hunks whose windows overlap or touch.
//
// The overlap comes off hunk 1's after-context, which expansion invented, not
// off hunk 2's body, which is real diff. Trimming hunk 2 instead drops its
// removal lines and leaves hunk 1's context asserting a line is unchanged that
// hunk 2 deletes.
func mergeExpandedHunks(expanded []*expandedHunk) []*expandedHunk {
	if len(expanded) == 0 {
		return nil
	}
	merged := []*expandedHunk{expanded[0]}

	for _, h := range expanded[1:] {
		prev := merged[len(merged)-1]
		prevNewEnd := prev.newStart + prev.newCount

		if h.newStart > prevNewEnd {
			merged = append(merged, h)
			continue
		}

		excess := prevNewEnd - h.newStart
		if excess < 0 {
			excess = 0
		}
		// Drop hunk 1's trailing context that hunk 2's region covers, never
		// more than is actually context.
		drop := excess
		if drop > prev.afterCount {
			drop = prev.afterCount
		}
		combined := prev.body[:len(prev.body)-drop]
		// Then take hunk 2 whole, minus only its own leading context that still
		// overlaps.
		trim := excess - drop
		if trim < 0 {
			trim = 0
		}
		if trim > h.beforeCount {
			trim = h.beforeCount
		}
		if trim > len(h.body) {
			trim = len(h.body)
		}
		combined = append(append([]string{}, combined...), h.body[trim:]...)

		// Merged counting rule: "starts with - or space" and "starts with + or
		// space". Differs from the per-hunk rule above; both are required.
		var oldCount, newCount int
		for _, ln := range combined {
			if ln == "" {
				continue
			}
			if strings.HasPrefix(ln, "-") || strings.HasPrefix(ln, " ") {
				oldCount++
			}
			if strings.HasPrefix(ln, "+") || strings.HasPrefix(ln, " ") {
				newCount++
			}
		}

		prev.body = combined
		prev.oldCount = oldCount
		prev.newCount = newCount
		prev.afterCount = h.afterCount
	}
	return merged
}

// ExpandAllFiles expands every file's diff in place.
func ExpandAllFiles(files []FileReviewData, before, after, maxDynamicBefore int, dynamicContext bool) []FileReviewData {
	out := make([]FileReviewData, 0, len(files))
	for _, f := range files {
		out = append(out, FileReviewData{
			Path:           f.Path,
			Diff:           ExpandContext(f.Diff, f.Content, f.Path, before, after, maxDynamicBefore, dynamicContext),
			Content:        f.Content,
			ContentFetched: f.ContentFetched,
		})
	}
	return out
}
