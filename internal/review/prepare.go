package review

import (
	"context"
	"errors"
	"fmt"
	"strings"
	"sync"

	"github.com/trick77/noergler-go/internal/bitbucket"
	"github.com/trick77/noergler-go/internal/diff"
)

// prepareFiles splits the diff, fetches each reviewable file's content and
// sorts the result.
//
// contentSkipped names files whose body was dropped: too large at the socket
// (ContentTooLarge) or over max_file_lines. A fetch that failed for any other
// reason is logged and the file is reviewed from its diff alone, which is not
// the same thing and is not reported to the reader.
func (r *Reviewer) prepareFiles(ctx context.Context, project, repo, rawDiff, sourceCommit, prTag string) (files []diff.FileReviewData, contentSkipped []string) {
	all := diff.SplitByFile(rawDiff)
	var reviewable []string
	for _, fd := range all {
		if diff.IsReviewable(fd) {
			reviewable = append(reviewable, fd)
		}
	}
	r.log.InfoContext(ctx, fmt.Sprintf("%s: %d file(s) in diff, %d reviewable, %d skipped (binary/non-reviewable)",
		prTag, len(all), len(reviewable), len(all)-len(reviewable)))
	if len(reviewable) == 0 {
		return nil, nil
	}

	type result struct {
		file    diff.FileReviewData
		ok      bool
		skipped string
	}
	results := make([]result, len(reviewable))

	// Bounds how many full file bodies are in flight at once; each one is
	// resident until the whole review is rendered.
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

			// Python compares content.count("\n") + 1 against the limit, so
			// a body is measured in lines, not bytes.
			if fetched {
				lines := strings.Count(content, "\n") + 1
				if lines > r.cfg.MaxFileLines {
					mu.Lock()
					r.log.InfoContext(ctx, fmt.Sprintf("Skipping full content for %s (%d lines > limit %d)",
						path, lines, r.cfg.MaxFileLines))
					results[i].skipped = path
					mu.Unlock()
					content, fetched = "", false
				} else {
					mu.Lock()
					r.log.InfoContext(ctx, fmt.Sprintf("Including %s with full file content (%d lines)", path, lines))
					mu.Unlock()
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

	for _, res := range results {
		if res.skipped != "" {
			contentSkipped = append(contentSkipped, res.skipped)
		}
		if res.ok {
			files = append(files, res.file)
		}
	}

	// Stable ordering across re-reviews keeps unchanged files in the LLM's
	// prefix cache.
	files = diff.SortByLanguagePriority(files)

	totalDiffLines, totalContentLines, withContent := 0, 0, 0
	for _, f := range files {
		totalDiffLines += strings.Count(f.Diff, "\n") + 1
		if f.ContentFetched {
			totalContentLines += strings.Count(f.Content, "\n") + 1
			withContent++
		}
	}
	r.log.InfoContext(ctx, fmt.Sprintf(
		"%s: %d file(s) for review - %d diff lines, %d content lines, %d with full content, %d diff-only",
		prTag, len(files), totalDiffLines, totalContentLines, withContent, len(files)-withContent))

	return files, contentSkipped
}

// countDiffLines counts added and removed lines, skipping files the reviewer
// ignores so the count reflects real code changes rather than reformatted
// JSON or vendored bundles.
func countDiffLines(rawDiff string) (added, removed int) {
	for _, fd := range diff.SplitByFile(rawDiff) {
		if !diff.IsReviewable(fd) {
			continue
		}
		for _, line := range strings.Split(fd, "\n") {
			switch {
			case strings.HasPrefix(line, "+++"), strings.HasPrefix(line, "---"):
			case strings.HasPrefix(line, "+"):
				added++
			case strings.HasPrefix(line, "-"):
				removed++
			}
		}
	}
	return added, removed
}

// truncateRunes cuts s to at most n runes. Python's len() counts characters,
// so the log prefix is measured the same way.
func truncateRunes(s string, n int) string {
	runes := []rune(s)
	if len(runes) <= n {
		return s
	}
	return string(runes[:n])
}
