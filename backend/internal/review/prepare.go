package review

import (
	"context"
	"errors"
	"fmt"
	"strings"
	"sync"

	"github.com/trick77/noergler/internal/bitbucket"
	"github.com/trick77/noergler/internal/diff"
)

// prepareFiles splits the diff, fetches each reviewable file's content and
// sorts the result.
//
// contentSkipped names files whose body was dropped: too large at the socket
// (ContentTooLarge) or over max_file_lines. A fetch that failed for any other
// reason is logged and the file is reviewed from its diff alone, which is not
// the same thing and is not reported to the reader.
//
// added and removed count every reviewable file's diff lines, size-gated
// ones included, exactly as the merge-time rollup does, in the same pass that
// classifies them: the raw diff is uncapped, so it is split once.
//
// oversized names the files the size gate kept out of the review, for the
// summary's "not reviewed" line; notReviewable the binary and generated ones
// the review never looks at; lines each reviewable or gated file's +/-, for
// the run page.
type prepared struct {
	files                                    []diff.FileReviewData
	contentSkipped, oversized, notReviewable []string
	added, removed                           int
	lines                                    map[string]lineCount
}

// lineCount is one file's added and removed lines.
type lineCount struct{ Added, Removed int }

func (r *Reviewer) prepareFiles(ctx context.Context, project, repo, rawDiff, sourceCommit, prTag string) prepared {
	all := diff.SplitByFile(rawDiff)
	var reviewable, skipped []string
	var p prepared
	p.lines = map[string]lineCount{}
	for _, fd := range all {
		path := diff.ExtractPath(fd)
		if diff.IsReviewable(fd) {
			// Counted for every reviewable file, size-gated ones included,
			// as the merge-time rollup (countDiffLines) counts them: not
			// reviewed, but changed.
			a, d := diff.CountChangedLines(fd)
			p.added, p.removed = p.added+a, p.removed+d
			p.lines[path] = lineCount{a, d}
			if changed, limit, over := diff.ExceedsSizeGate(path, fd); over {
				r.log.InfoContext(ctx, fmt.Sprintf("%s: skipped as oversized: %s (%d changed lines > %d)",
					prTag, path, changed, limit))
				p.oversized = append(p.oversized, path)
				continue
			}
			reviewable = append(reviewable, fd)
			continue
		}
		if path == "" {
			path = "<unparsed>"
		} else {
			// For the page's file table only: a binary or generated file is
			// not in the totals, which count reviewable files like the
			// rollup does.
			a, d := diff.CountChangedLines(fd)
			p.lines[path] = lineCount{a, d}
		}
		skipped = append(skipped, path)
	}
	r.log.InfoContext(ctx, fmt.Sprintf("%s: %d file(s) in diff, %d reviewable, %d skipped (binary/non-reviewable), %d oversized",
		prTag, len(all), len(reviewable), len(skipped), len(p.oversized)))
	p.notReviewable = skipped
	if len(skipped) > 0 {
		r.log.InfoContext(ctx, fmt.Sprintf("%s: skipped as binary/non-reviewable: %s", prTag, capPaths(skipped)))
	}
	if len(reviewable) == 0 {
		return p
	}

	type result struct {
		file    diff.FileReviewData
		ok      bool
		skipped string
	}
	results := make([]result, len(reviewable))

	// Bounds how many full file bodies are in flight at once; each one is
	// resident until the prompt is assembled, after which they are dropped:
	// the plan handed to the inference stage carries counts, not content.
	sem := make(chan struct{}, fileFetchConcurrency)
	var wg sync.WaitGroup
	var mu sync.Mutex

	for i, fd := range reviewable {
		wg.Add(1)
		go func(i int, fd string) {
			defer wg.Done()

			path := diff.ExtractPath(fd)
			if path == "" {
				first := fd
				if idx := strings.IndexByte(first, '\n'); idx >= 0 {
					first = first[:idx]
				}
				mu.Lock()
				r.log.WarnContext(ctx, "Could not extract path from diff chunk: "+truncateRunes(first, 200))
				mu.Unlock()
				return
			}

			deleted := diff.IsDeleted(fd)
			content, fetched := "", false
			if !deleted && sourceCommit != "" {
				sem <- struct{}{}
				body, err := r.bitbucket.FetchFileContent(ctx, project, repo, sourceCommit, path)
				<-sem

				var tooLarge *bitbucket.ContentTooLarge
				switch {
				case err == nil:
					content, fetched = body, true
				case errors.As(err, &tooLarge):
					mu.Lock()
					r.log.InfoContext(ctx, fmt.Sprintf("Skipping full content for %s (%s)", path, err))
					results[i].skipped = path
					mu.Unlock()
				default:
					mu.Lock()
					r.log.WarnContext(ctx, fmt.Sprintf("Failed to fetch content for %s, using diff only", path))
					mu.Unlock()
				}
			}

			// A body is measured in lines, not bytes: newlines + 1, so a
			// file at exactly the limit is kept
			// (TestFileAtMaxLinesKeepsItsContent).
			if fetched {
				lines := strings.Count(content, "\n") + 1
				if lines > r.cfg.MaxFileLines {
					mu.Lock()
					r.log.InfoContext(ctx, fmt.Sprintf("Skipping full content for %s (%d lines > limit %d)",
						path, lines, r.cfg.MaxFileLines))
					results[i].skipped = path
					mu.Unlock()
					content, fetched = "", false
				}
			}

			results[i] = result{
				file: diff.FileReviewData{
					Path: path, Diff: fd, Content: content, ContentFetched: fetched,
				},
				ok:      true,
				skipped: results[i].skipped,
			}
		}(i, fd)
	}
	wg.Wait()

	var files []diff.FileReviewData
	for _, res := range results {
		if res.skipped != "" {
			p.contentSkipped = append(p.contentSkipped, res.skipped)
		}
		if res.ok {
			files = append(files, res.file)
		}
	}

	// Stable ordering across re-reviews keeps unchanged files in the LLM's
	// prefix cache.
	files = diff.SortByLanguagePriority(files)
	p.files = files

	totalDiffLines, totalContentLines, withContent := 0, 0, 0
	for _, f := range files {
		totalDiffLines += strings.Count(f.Diff, "\n") + 1
		if f.ContentFetched {
			totalContentLines += strings.Count(f.Content, "\n") + 1
			withContent++
		}
	}
	// Per-file sizes are logged once the prompt's file set is final
	// (renderFilesSent, after compression and expansion), not here.
	r.log.InfoContext(ctx, fmt.Sprintf(
		"%s: %d file(s) for review - %d diff lines, %d content lines, %d with full content, %d diff-only",
		prTag, len(files), totalDiffLines, totalContentLines, withContent, len(files)-withContent))

	return p
}

// countDiffLines counts added and removed lines, skipping files the reviewer
// ignores so the count reflects real code changes rather than reformatted
// JSON or vendored bundles.
func countDiffLines(rawDiff string) (added, removed int) {
	for _, fd := range diff.SplitByFile(rawDiff) {
		if diff.IsReviewable(fd) {
			a, d := diff.CountChangedLines(fd)
			added, removed = added+a, removed+d
		}
	}
	return added, removed
}

// truncateRunes cuts s to at most n runes. Runes and not bytes: a byte cut
// would split a multi-byte character in the log prefix.
func truncateRunes(s string, n int) string {
	runes := []rune(s)
	if len(runes) <= n {
		return s
	}
	return string(runes[:n])
}
