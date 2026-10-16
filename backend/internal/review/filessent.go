package review

import (
	"fmt"
	"sort"
	"strings"

	"github.com/trick77/noergler/internal/diff"
	"github.com/trick77/noergler/internal/inference"
)

// renderFilesSent is the log line naming every file the prompt carries and
// what each costs: the whole entry as the model sees it, split into the
// numbered diff and the full body when one was fetched.
//
// Biggest first, because the line exists to show where the tokens went, and
// capped at maxLoggedPaths with the tail summed so the total stays honest on
// a 40-file PR. The per-file total is count(FormatFileEntry), the same figure
// AssembleReviewPrompt sums into the prompt line's "files N", so the two
// agree to the token.
func renderFilesSent(files []diff.FileReviewData, count func(string) int) string {
	type entry struct {
		path        string
		total, body int
		diff        int
		full        bool
	}
	entries := make([]entry, 0, len(files))
	sum := 0
	for _, f := range files {
		e := entry{
			path:  f.Path,
			total: count(inference.FormatFileEntry(f)),
			diff:  count(diff.NumberDiff(f.Diff)),
			full:  f.ContentFetched,
		}
		if e.full {
			e.body = count(f.Content)
		}
		sum += e.total
		entries = append(entries, e)
	}
	sort.SliceStable(entries, func(i, j int) bool {
		if entries[i].total != entries[j].total {
			return entries[i].total > entries[j].total
		}
		return entries[i].path < entries[j].path
	})

	var b strings.Builder
	fmt.Fprintf(&b, "files sent: %d, %d tokens", len(entries), sum)
	if len(entries) == 0 {
		return b.String()
	}
	b.WriteString(" - ")
	shown := entries
	if len(shown) > maxLoggedPaths {
		shown = shown[:maxLoggedPaths]
	}
	for i, e := range shown {
		if i > 0 {
			b.WriteString(", ")
		}
		if e.full {
			fmt.Fprintf(&b, "%s %d (%d diff, %d body)", e.path, e.total, e.diff, e.body)
		} else {
			fmt.Fprintf(&b, "%s %d (diff only)", e.path, e.total)
		}
	}
	if rest := entries[len(shown):]; len(rest) > 0 {
		tail := 0
		for _, e := range rest {
			tail += e.total
		}
		fmt.Fprintf(&b, ", +%d more, %d tokens", len(rest), tail)
	}
	return b.String()
}
