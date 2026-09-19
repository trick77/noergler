package diff

import "strings"

// CountTokensFunc and FormatEntryFunc are supplied by the caller, exactly as the
// Python took callbacks. They keep this package free of any tokenizer or prompt
// dependency.
type CountTokensFunc func(string) int

// FormatEntryFunc renders one file as it will appear in the prompt.
type FormatEntryFunc func(FileReviewData) string

// CompressionResult splits a PR's files into what fits the budget and what is
// only named. Both file lists keep sorted order.
type CompressionResult struct {
	IncludedFiles      []FileReviewData
	OtherModifiedPaths []string
	DeletedFilePaths   []string
	RenamedFilePaths   []string
}

// isDeletionOnlyHunk reports whether a hunk adds nothing.
func isDeletionOnlyHunk(hunkLines []string) bool {
	for _, line := range hunkLines {
		if strings.HasPrefix(line, "+") && !strings.HasPrefix(line, "+++") {
			return false
		}
	}
	return true
}

// RemoveDeletionOnlyHunks drops hunks that only remove code.
//
// Returns "" when no hunk survives, including a diff with no hunks at all (a
// pure mode change, or a rename whose hunks were not included). Compress then
// files that path under deleted, which mislabels it. Ported as is.
//
// Mirrors Python's split("\n"), not splitlines().
func RemoveDeletionOnlyHunks(fileDiff string) string {
	var headerLines []string
	var hunks [][]string
	var current []string

	for _, line := range strings.Split(fileDiff, "\n") {
		switch {
		case strings.HasPrefix(line, "@@"):
			if len(current) > 0 {
				hunks = append(hunks, current)
			}
			current = []string{line}
		case len(current) > 0:
			current = append(current, line)
		default:
			headerLines = append(headerLines, line)
		}
	}
	if len(current) > 0 {
		hunks = append(hunks, current)
	}

	var kept []string
	for _, h := range hunks {
		if !isDeletionOnlyHunk(h) {
			kept = append(kept, h...)
		}
	}
	if len(kept) == 0 {
		return ""
	}
	return strings.Join(append(headerLines, kept...), "\n")
}

// IsRenameOnly reports whether a diff is a pure rename.
func IsRenameOnly(fileDiff string) bool {
	return strings.Contains(fileDiff, "similarity index 100%") &&
		(strings.Contains(fileDiff, "rename from") || strings.Contains(fileDiff, "rename to"))
}

// Compress fits as many files as the token budget allows, naming the rest.
//
// The budget is 90% of what remains after the prompt overhead, truncated toward
// zero, and can go negative when the overhead exceeds the window. First-fit does
// not stop at the first miss: a later, smaller file still fits.
func Compress(
	files []FileReviewData,
	maxTokens int,
	promptTemplate string,
	countTokens CountTokensFunc,
	formatEntry FormatEntryFunc,
) CompressionResult {
	var result CompressionResult
	var active []FileReviewData

	for _, f := range files {
		switch {
		case IsDeleted(f.Diff):
			result.DeletedFilePaths = append(result.DeletedFilePaths, f.Path)
		case IsRenameOnly(f.Diff):
			result.RenamedFilePaths = append(result.RenamedFilePaths, f.Path)
		default:
			active = append(active, f)
		}
	}

	var compressedActive []FileReviewData
	for _, f := range active {
		cleaned := RemoveDeletionOnlyHunks(f.Diff)
		if cleaned == "" {
			result.DeletedFilePaths = append(result.DeletedFilePaths, f.Path)
			continue
		}
		compressedActive = append(compressedActive, FileReviewData{
			Path: f.Path, Diff: cleaned, Content: f.Content, ContentFetched: f.ContentFetched,
		})
	}

	sorted := SortByLanguagePriority(compressedActive)

	promptOverhead := countTokens(strings.ReplaceAll(promptTemplate, "{files}", ""))
	budget := (maxTokens - promptOverhead) * 9 / 10

	used := 0
	for _, f := range sorted {
		entryTokens := countTokens(formatEntry(f))
		if used+entryTokens <= budget {
			result.IncludedFiles = append(result.IncludedFiles, f)
			used += entryTokens
			continue
		}
		result.OtherModifiedPaths = append(result.OtherModifiedPaths, f.Path)
	}
	return result
}

// IsSmall reports whether every file fits with room for context expansion.
func IsSmall(
	files []FileReviewData,
	maxTokens int,
	promptTemplate string,
	countTokens CountTokensFunc,
	formatEntry FormatEntryFunc,
	contextExpansionRatio float64,
) bool {
	promptOverhead := countTokens(strings.ReplaceAll(promptTemplate, "{files}", ""))
	available := maxTokens - promptOverhead
	total := 0
	for _, f := range files {
		total += countTokens(formatEntry(f))
	}
	// Python multiplies in floating point and compares <= against an int.
	return float64(total)*contextExpansionRatio <= float64(available)
}

// CountDiffLines counts added and removed lines in a whole unified diff.
//
// Files the reviewer ignores (lockfiles, generated or minified files, binaries,
// build output) are skipped, so the count reflects real code changes rather
// than reformatted JSON or vendored bundles.
//
// Mirrors Python splitlines(), not split("\n").
func CountDiffLines(diff string) (added, removed int) {
	for _, fileDiff := range SplitByFile(diff) {
		if !IsReviewable(fileDiff) {
			continue
		}
		for _, line := range splitLines(fileDiff) {
			switch {
			case strings.HasPrefix(line, "+") && !strings.HasPrefix(line, "+++"):
				added++
			case strings.HasPrefix(line, "-") && !strings.HasPrefix(line, "---"):
				removed++
			}
		}
	}
	return added, removed
}
