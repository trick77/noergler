package review

import (
	"context"
	"fmt"
	"strings"

	"github.com/trick77/noergler/internal/diff"
	"github.com/trick77/noergler/internal/store"
	"github.com/trick77/noergler/internal/webhook"
)

// searchHitFiles is how many files the one search may return.
const searchHitFiles = 25

// diffParts are a diff's per-file parts with their paths, unfiltered:
// definitions are read off removed lines too, deleted files included.
func diffParts(raw string) []diff.FileReviewData {
	var out []diff.FileReviewData
	for _, part := range diff.SplitByFile(raw) {
		if p := diff.ExtractPath(part); p != "" {
			out = append(out, diff.FileReviewData{Path: p, Diff: part})
		}
	}
	return out
}

// unchangedReferences finds, with one code search, lines of files the PR
// does not change that name a symbol the reviewed files define: the callers
// a changed signature breaks, which the review otherwise never sees.
//
// At most one search per review, on the worker like every Bitbucket call,
// never retried (user, 2026-11-20: never a burst of searches). Only for a PR
// into the repository's default branch: the search reads that branch alone,
// so an unchanged file there is the PR's own only when the PR goes there.
// prPaths are the whole PR's files: the default branch still holds their old
// version, so their lines are left out; prPathsKnown false means they could
// not be read, and there is no search. Any fault, or a search the instance
// does not run, is no block, logged; the review goes on.
func (r *Reviewer) unchangedReferences(ctx context.Context, pr webhook.PullRequest, key store.PRKey, prTag string,
	files []diff.FileReviewData, prPaths []string, prPathsKnown bool) []diff.UnchangedRef {
	symbols := diff.SearchSymbols(files)
	if len(symbols) == 0 {
		r.log.InfoContext(ctx, prTag+": unchanged references skipped: the diff changes or removes no searchable name")
		return nil
	}
	if !prPathsKnown {
		r.log.InfoContext(ctx, prTag+": unchanged references skipped: the PR's files are unknown (cumulative diff unread)")
		return nil
	}
	def, err := r.defaultBranch(ctx, key.Project, key.Repo)
	if err != nil {
		r.log.WarnContext(ctx, fmt.Sprintf("%s: unchanged references skipped: default branch unread: %v", prTag, err))
		return nil
	}
	if refID(pr.ToRef) != def {
		r.log.InfoContext(ctx, fmt.Sprintf("%s: unchanged references skipped: the PR goes to %s, the search reads %s only",
			prTag, pr.ToRef.DisplayID, def))
		return nil
	}
	query, used := diff.SearchQuery(symbols, key.Project, key.Repo)
	if query == "" {
		r.log.InfoContext(ctx, prTag+": unchanged references skipped: no name fits the search's 250-character query")
		return nil
	}
	hits, err := r.bitbucket.SearchCode(ctx, query, searchHitFiles)
	if err != nil {
		r.log.WarnContext(ctx, fmt.Sprintf("%s: unchanged references skipped: search failed: %v", prTag, err))
		return nil
	}
	skip := make(map[string]bool, len(prPaths))
	for _, p := range prPaths {
		skip[p] = true
	}
	var found []diff.SearchHit
	for _, h := range hits {
		// The query names the repository; a hit from another is the
		// search ignoring it, never this PR's code.
		if !strings.EqualFold(h.Project, key.Project) || !strings.EqualFold(h.Repo, key.Repo) {
			continue
		}
		sh := diff.SearchHit{File: h.File}
		for _, l := range h.Lines {
			sh.Lines = append(sh.Lines, diff.SearchLine{Line: l.Line, Text: l.Text})
		}
		found = append(found, sh)
	}
	refs := diff.MatchReferences(used, found, skip)
	r.log.InfoContext(ctx, fmt.Sprintf("%s: unchanged references: 1 search for %s, %d file(s) returned, %d line(s) kept",
		prTag, strings.Join(used, ", "), len(hits), len(refs)))
	return refs
}
