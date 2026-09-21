package review

import (
	"context"
	"fmt"
	"sort"
	"strings"

	"github.com/trick77/noergler/internal/bitbucket"
	"github.com/trick77/noergler/internal/diff"
)

// tooLargeFile is one file of an over-cap PR as the log reports it.
type tooLargeFile struct {
	path string
	size int  // bytes in the truncated head; 0 when the file lies past the cap
	seen bool // size is known, i.e. the file was inside the head
	// short marks the head's final file, cut mid-diff by the cap, so its
	// size is a lower bound.
	short      bool
	reviewable bool
}

// tooLargeReport lists every file of a PR whose diff blew the byte cap, each
// marked REVIEWED or FILTERED, so the operator can tell from the log alone
// whether the PR held anything worth reviewing.
//
// The cap trips at the socket, before the diff is split into files, so nothing
// downstream can answer that. Two sources are combined:
//
//   - head: the bytes that did arrive, which carry per-file sizes. Only the
//     first Limit bytes, so on a 19.8 MB diff it covers roughly half the PR.
//   - paths: the full file list from `/changes`, which reaches past the cap
//     but carries no sizes. Empty when that call failed, and then the report
//     covers the head alone and says so.
//
// Status is what the review WOULD have done: REVIEWED means the file would
// have gone to the model, FILTERED means the skip lists would have dropped it.
// The PR is skipped either way.
//
// Returns "" when nothing can be reported, so the caller can omit the lines.
// The result is multi-line: one file per line, plus a totals line. The caller
// splits it so each file lands as its own log record, because the log is one
// JSON object per line and an embedded blob is not searchable in Splunk.
func tooLargeReport(head []byte, limit int, paths []string) string {
	files := headFiles(head)

	// Files past the cap are known by path only. Merge on path so a file the
	// head already sized is not listed twice.
	seen := make(map[string]bool, len(files))
	for _, f := range files {
		seen[f.path] = true
	}
	for _, p := range paths {
		if seen[p] {
			continue
		}
		files = append(files, tooLargeFile{path: p, reviewable: diff.PathIsReviewable(p)})
	}
	if len(files) == 0 {
		return ""
	}

	// Reviewable first: that is what the operator is looking for, and on a
	// fixture-heavy PR it would otherwise sort to the bottom. Then by size
	// descending, which names the file that blew the cap.
	sort.SliceStable(files, func(i, j int) bool {
		if files[i].reviewable != files[j].reviewable {
			return files[i].reviewable
		}
		return files[i].size > files[j].size
	})

	var b strings.Builder
	var reviewed, filtered, reviewedBytes, filteredBytes int
	for _, f := range files {
		status := "FILTERED"
		if f.reviewable {
			status = "REVIEWED"
		}
		size := "?"
		if f.seen {
			sep := ""
			if f.short {
				sep = ">="
			}
			size = fmt.Sprintf("%s%d", sep, f.size)
		}
		fmt.Fprintf(&b, "\n  %s  %s  %s", f.path, size, status)
		if f.reviewable {
			reviewed++
			reviewedBytes += f.size
		} else {
			filtered++
			filteredBytes += f.size
		}
	}

	// len(head) == limit whenever a head exists (it is buf[:max]), so the two
	// scopes quote the same number.
	scope := fmt.Sprintf("%d file(s) in the first %d bytes", len(files), limit)
	if len(paths) > 0 {
		scope = fmt.Sprintf("%d file(s) in the PR", len(files))
	}
	// Byte totals only when something was actually sized. With every file past
	// the cap they would both read "0 bytes", which looks like a bug rather
	// than the absence of a measurement.
	if reviewedBytes+filteredBytes > 0 {
		fmt.Fprintf(&b, "\n  %s: %d reviewable (%d bytes), %d filtered (%d bytes)",
			scope, reviewed, reviewedBytes, filtered, filteredBytes)
	} else {
		fmt.Fprintf(&b, "\n  %s: %d reviewable, %d filtered", scope, reviewed, filtered)
	}
	if len(paths) > 0 {
		fmt.Fprintf(&b, "\n  sizes are bytes within the first %d bytes; \"?\" is past the cap", limit)
	}
	if reviewed == 0 {
		b.WriteString("\n  no reviewable files: every file would have been filtered")
	}
	return b.String()
}

// headFiles parses the arrived prefix of an over-cap diff into per-file sizes.
func headFiles(head []byte) []tooLargeFile {
	if len(head) == 0 {
		return nil
	}
	parts := diff.SplitByFile(string(head))
	var files []tooLargeFile
	for i, part := range parts {
		path := diff.ExtractPath(part)
		if path == "" {
			// Preamble before the first `diff --git `, per SplitByFile.
			continue
		}
		files = append(files, tooLargeFile{
			path: path,
			size: len(part),
			seen: true,
			// The head's final file is cut mid-diff by the cap. It is kept,
			// marked ">=", because dropping it would report nothing at all in
			// the likeliest case: one bundle or fixture big enough to fill the
			// head by itself.
			short: i == len(parts)-1,
			// Full diff text available, so the stronger check applies: it also
			// catches binary markers and extensions the path regex misses.
			reviewable: diff.IsReviewable(part),
		})
	}
	return files
}

// logDiffTooLarge reports a PR diff that exceeded its byte cap: the error, then
// every file the PR touched with its reviewable status.
//
// The file list needs `/changes`, because the head stops at the cap. That call
// is best-effort: this is already a failure path and must not become a second
// one, so a failure degrades to the head-only report and says why.
func (r *Reviewer) logDiffTooLarge(ctx context.Context, prTag, project, repo string, prID int, err error, tooLarge *bitbucket.ContentTooLarge) {
	r.log.WarnContext(ctx, fmt.Sprintf("%s: %v - skipping review", prTag, err))

	paths, changesErr := r.bitbucket.FetchPRChanges(ctx, project, repo, prID)
	if changesErr != nil {
		r.log.WarnContext(ctx, fmt.Sprintf(
			"%s: full file list unavailable (%v) - reporting the first %d bytes only",
			prTag, changesErr, tooLarge.Limit))
		paths = nil
	}

	// One record per file: the log is one JSON object per line, so a single
	// blob with embedded newlines would arrive in Splunk as one unsearchable
	// field. Every line carries the pr tag for the same reason.
	report := tooLargeReport(tooLarge.Head, tooLarge.Limit, paths)
	for _, line := range strings.Split(report, "\n") {
		if line = strings.TrimSpace(line); line != "" {
			r.log.WarnContext(ctx, prTag+": "+line)
		}
	}
}
