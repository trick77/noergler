package review

import (
	"context"
	"errors"
	"fmt"
	"slices"
	"strings"

	"github.com/trick77/noergler/internal/bitbucket"
	"github.com/trick77/noergler/internal/diff"
	"github.com/trick77/noergler/internal/inference"
	"github.com/trick77/noergler/internal/store"
)

// bytesPerTokenEstimate sizes a push without tokenizing it: the incremental
// diff is uncapped, and the estimate only scales a budget.
const bytesPerTokenEstimate = 4

// maxLoggedPaths caps a path list inside one log line.
const maxLoggedPaths = 10

// Cumulative tiers, in fill order.
const (
	tierTouched = iota // the push changed this file
	tierRelated        // references a symbol the push defines
	tierOther
)

// cumulativeContext is the PR-wide context of an incremental review.
type cumulativeContext struct {
	// diff holds whole per-file diffs, never a cut one.
	diff string
	// omitted names the reviewable PR files that did not fit.
	omitted []string
	// partial: something of the PR is not in diff, filtered or omitted. An
	// omitted part with no parseable path has no name, so omitted alone
	// cannot say it.
	partial bool
}

// fetchCumulativeDiff returns the part of the whole-PR diff worth sending as
// cross-file context for a push, or nothing when it is unavailable.
//
// Filter, rank, fill. Non-reviewable files go first: the raw diff carried
// lockfiles and generated code at up to 80k tokens. The budget follows the
// push, so a one-line commit no longer pays for the whole PR. What does not
// fit is left out file by file rather than dropping the block, which used to
// leave the largest PRs with no context at all.
func (r *Reviewer) fetchCumulativeDiff(ctx context.Context, key store.PRKey, prTag, pushDiff string) cumulativeContext {
	full, err := r.bitbucket.FetchPRDiff(ctx, key.Project, key.Repo, key.PRID, 0)
	var tooLarge *bitbucket.ContentTooLarge
	switch {
	case errors.As(err, &tooLarge):
		r.log.InfoContext(ctx, fmt.Sprintf("%s: cumulative PR diff dropped (%v)", prTag, err))
		return cumulativeContext{}
	case err != nil:
		r.log.WarnContext(ctx, fmt.Sprintf("%s: failed to fetch cumulative PR diff for context: %v", prTag, err))
		return cumulativeContext{}
	case strings.TrimSpace(full) == "":
		r.log.InfoContext(ctx, prTag+": cumulative PR diff is empty, no cross-file context")
		return cumulativeContext{}
	}

	push, _ := reviewableParts(pushDiff)
	pushBytes := 0
	touched := map[string]bool{}
	for _, f := range push {
		pushBytes += len(f.Diff)
		// An unparsed header has no path: keyed on "", one such part in the
		// push would rank every unparsed part of the PR as touched.
		if f.Path != "" {
			touched[f.Path] = true
		}
	}
	pushTokens := pushBytes / bytesPerTokenEstimate
	budget, bound := inference.ScaledCumulativeDiffBudget(pushTokens, r.llm.InputTokenBudget())

	candidates, filtered := reviewableParts(full)
	related := diff.ReferencingPaths(push, candidates)
	tier := func(f diff.FileReviewData) int {
		switch {
		case touched[f.Path]:
			return tierTouched
		case related[f.Path]:
			return tierRelated
		default:
			return tierOther
		}
	}
	// The path order first, then the tiers over it: stable, so each tier
	// keeps the order the focused files use.
	ranked := diff.SortByLanguagePriority(candidates)
	slices.SortStableFunc(ranked, func(a, b diff.FileReviewData) int { return tier(a) - tier(b) })

	var kept strings.Builder
	var keptPerTier [3]int
	var omitted []diff.FileReviewData
	used := 0
	for _, f := range ranked {
		remaining := budget - used
		// Tokenizing expands the text in RAM, so a part over what is left by
		// byte count alone is never tokenized.
		if len(f.Diff) > remaining*bytesPerTokenCeiling {
			omitted = append(omitted, f)
			continue
		}
		n := r.tokens.Count(f.Diff)
		if n > remaining {
			omitted = append(omitted, f)
			continue
		}
		kept.WriteString(f.Diff)
		if !strings.HasSuffix(f.Diff, "\n") {
			kept.WriteString("\n")
		}
		used += n
		keptPerTier[tier(f)]++
	}

	r.log.InfoContext(ctx, fmt.Sprintf(
		"%s: cumulative PR diff: push ~%d tokens, budget %d (%s), kept %d of %d file(s) in %d tokens "+
			"(%d touched by the push, %d related, %d other), %d filtered as non-reviewable, %d omitted for budget%s",
		prTag, pushTokens, budget, bound, len(ranked)-len(omitted), len(ranked), used,
		keptPerTier[tierTouched], keptPerTier[tierRelated], keptPerTier[tierOther],
		filtered, len(omitted), largestOmitted(omitted)))

	out := cumulativeContext{diff: kept.String(), partial: filtered > 0 || len(omitted) > 0}
	for _, f := range omitted {
		if f.Path != "" {
			out.omitted = append(out.omitted, f.Path)
		}
	}
	return out
}

// reviewableParts splits a combined diff into its reviewable per-file parts,
// carrying path and diff only, and counts the parts filtered out.
func reviewableParts(rawDiff string) (parts []diff.FileReviewData, filtered int) {
	for _, fd := range diff.SplitByFile(rawDiff) {
		if !diff.IsReviewable(fd) {
			filtered++
			continue
		}
		parts = append(parts, diff.FileReviewData{Path: diff.ExtractPath(fd), Diff: fd})
	}
	return parts, filtered
}

// largestOmitted names the three biggest omitted files, in bytes: a part
// dropped by the byte pre-check was never tokenized.
func largestOmitted(omitted []diff.FileReviewData) string {
	if len(omitted) == 0 {
		return ""
	}
	bySize := slices.Clone(omitted)
	slices.SortStableFunc(bySize, func(a, b diff.FileReviewData) int { return len(b.Diff) - len(a.Diff) })
	if len(bySize) > 3 {
		bySize = bySize[:3]
	}
	names := make([]string, len(bySize))
	for i, f := range bySize {
		path := f.Path
		if path == "" {
			path = "<unparsed>"
		}
		names[i] = fmt.Sprintf("%s %d bytes", path, len(f.Diff))
	}
	return " (largest: " + strings.Join(names, ", ") + ")"
}

// capPaths renders a path list for one log line, capped.
func capPaths(paths []string) string {
	if len(paths) <= maxLoggedPaths {
		return strings.Join(paths, ", ")
	}
	return fmt.Sprintf("%s, +%d more", strings.Join(paths[:maxLoggedPaths], ", "), len(paths)-maxLoggedPaths)
}
