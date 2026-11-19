package review

import (
	"context"
	"errors"
	"fmt"
	"net/http"
	"path"
	"slices"
	"strings"

	"github.com/trick77/noergler/internal/bitbucket"
	"github.com/trick77/noergler/internal/diff"
)

// An AGENTS.md below the root holds the rules of one part of a monorepo.
// The root file alone made such a rule apply to everything or nothing.
//
// Bounded, because every probe is a Bitbucket call on the single worker:
// candidates are the directories of the changed files down to
// nestedAgentsDepth, at most nestedAgentsProbes of them read, at most
// nestedAgentsKept kept, none over nestedAgentsMaxBytes (it is tokenized on
// the worker, and a rules file that size is not one).
const (
	nestedAgentsDepth    = 3
	nestedAgentsProbes   = 8
	nestedAgentsKept     = 3
	nestedAgentsMaxBytes = 32 << 10
)

// nestedAgentsMD is one AGENTS.md below the repo root.
type nestedAgentsMD struct {
	// Dir is its directory, with a trailing slash ("services/billing/").
	Dir     string
	Content string
}

// Path is the file's path in the repo.
func (n nestedAgentsMD) Path() string { return n.Dir + "AGENTS.md" }

// prPaths are the paths a diff names.
func prPaths(diffs ...string) []string {
	seen := map[string]bool{}
	var out []string
	for _, d := range diffs {
		for _, fd := range diff.SplitByFile(d) {
			if p := diff.ExtractPath(fd); p != "" && !seen[p] {
				seen[p] = true
				out = append(out, p)
			}
		}
	}
	return out
}

// nestedCandidates are the directories an AGENTS.md is looked for in: every
// ancestor of a changed file below the root, down to nestedAgentsDepth.
// Deepest first, since the nearest file is the one that applies and a
// parent always covers at least its children's files; then by how many
// changed files sit under it, then by path. Path only, never content: the
// same PR probes the same directories. cut is how many did not fit.
func nestedCandidates(paths []string) (dirs []string, cut int) {
	under := map[string]int{}
	for _, p := range paths {
		parts := strings.Split(path.Dir(p), "/")
		if parts[0] == "." {
			continue
		}
		for depth := 1; depth <= min(len(parts), nestedAgentsDepth); depth++ {
			under[strings.Join(parts[:depth], "/")+"/"]++
		}
	}
	for d := range under {
		dirs = append(dirs, d)
	}
	slices.SortFunc(dirs, func(a, b string) int {
		if da, db := strings.Count(a, "/"), strings.Count(b, "/"); da != db {
			return db - da
		}
		if under[a] != under[b] {
			return under[b] - under[a]
		}
		return strings.Compare(a, b)
	})
	if len(dirs) > nestedAgentsProbes {
		return dirs[:nestedAgentsProbes], len(dirs) - nestedAgentsProbes
	}
	return dirs, 0
}

// pickNested keeps the found files that are some changed file's nearest
// AGENTS.md, the ones nearest to most changed files first, at most
// nestedAgentsKept, in path order (the prompt's order depends on paths
// only). left are the found files not kept.
func pickNested(found []nestedAgentsMD, paths []string) (kept []nestedAgentsMD, left []string) {
	nearest := map[string]int{}
	for _, p := range paths {
		best := ""
		for _, f := range found {
			if strings.HasPrefix(p, f.Dir) && len(f.Dir) > len(best) {
				best = f.Dir
			}
		}
		if best != "" {
			nearest[best]++
		}
	}
	for _, f := range found {
		if nearest[f.Dir] > 0 {
			kept = append(kept, f)
		} else {
			left = append(left, f.Path()+" (not the nearest)")
		}
	}
	slices.SortStableFunc(kept, func(a, b nestedAgentsMD) int {
		if nearest[a.Dir] != nearest[b.Dir] {
			return nearest[b.Dir] - nearest[a.Dir]
		}
		return strings.Compare(a.Dir, b.Dir)
	})
	for _, f := range kept[min(len(kept), nestedAgentsKept):] {
		left = append(left, f.Path()+" (over the cap)")
	}
	kept = kept[:min(len(kept), nestedAgentsKept)]
	slices.SortFunc(kept, func(a, b nestedAgentsMD) int { return strings.Compare(a.Dir, b.Dir) })
	return kept, left
}

// fetchNestedAgentsMD reads the AGENTS.md files below the root that apply to
// the changed paths, at one commit (the PR head: a file the PR adds or
// changes is the PR's own rule set). A missing file is absent; a read that
// fails otherwise is absent too, but logged: the root rules still hold, and
// the next run may read it.
func (r *Reviewer) fetchNestedAgentsMD(ctx context.Context, project, repo, commit, prTag string, paths []string) []nestedAgentsMD {
	if commit == "" {
		return nil
	}
	dirs, cut := nestedCandidates(paths)
	if cut > 0 {
		r.log.InfoContext(ctx, fmt.Sprintf("%s: %d director(ies) not probed for an AGENTS.md (cap %d)", prTag, cut, nestedAgentsProbes))
	}
	var found []nestedAgentsMD
	for _, dir := range dirs {
		if ctx.Err() != nil {
			return nil
		}
		p := dir + "AGENTS.md"
		content, err := r.bitbucket.FetchFileContent(ctx, project, repo, commit, p)
		var status *bitbucket.StatusError
		switch {
		case errors.As(err, &status) && status.Status == http.StatusNotFound:
			continue
		case err != nil:
			r.log.WarnContext(ctx, fmt.Sprintf("%s: %s not read, reviewing without it: %v", prTag, p, err))
			continue
		case strings.TrimSpace(content) == "":
			continue
		case len(content) > nestedAgentsMaxBytes:
			r.log.WarnContext(ctx, fmt.Sprintf("%s: %s left out: %d bytes, over %d", prTag, p, len(content), nestedAgentsMaxBytes))
			continue
		}
		found = append(found, nestedAgentsMD{Dir: dir, Content: content})
	}
	kept, left := pickNested(found, paths)
	if len(left) > 0 {
		r.log.InfoContext(ctx, fmt.Sprintf("%s: AGENTS.md found but not used: %s", prTag, strings.Join(left, ", ")))
	}
	for _, n := range kept {
		r.log.InfoContext(ctx, fmt.Sprintf("%s: loaded %s", prTag, n.Path()))
		if strings.Contains(strings.ToLower(n.Content), "[blocking]") {
			// Blocking rules are the target branch's root file only: a
			// branch must not lift its own gate, and one place decides.
			r.log.InfoContext(ctx, fmt.Sprintf("%s: [blocking] lines in %s are ignored: only the root AGENTS.md of the target branch blocks", prTag, n.Path()))
		}
	}
	return kept
}

// nestedBlock is one nested file as the prompt carries it.
func nestedBlock(n nestedAgentsMD) string {
	return fmt.Sprintf("### %s (applies to files under `%s`)\n\n%s", n.Path(), n.Dir, strings.TrimRight(n.Content, "\n"))
}

// withNested is the root AGENTS.md followed by the nested ones, each under
// a line naming the directory it applies to. With a token limit (> 0) a
// nested file that would take the whole over it is left out, in path order,
// and named in dropped: the root decides whether a review runs, never a
// nested file. Each part is counted once.
func withNested(root string, nested []nestedAgentsMD, maxTokens int, count func(string) int) (combined string, dropped []string) {
	var parts []string
	total := 0
	if strings.TrimSpace(root) != "" {
		parts = append(parts, strings.TrimRight(root, "\n"))
		if maxTokens > 0 {
			total = count(root)
		}
	}
	for _, n := range nested {
		block := nestedBlock(n)
		if maxTokens > 0 {
			if c := count(block); total+c > maxTokens {
				dropped = append(dropped, n.Path())
				continue
			} else {
				total += c
			}
		}
		parts = append(parts, block)
	}
	return strings.Join(parts, "\n\n"), dropped
}

// rootTokens is the root AGENTS.md's token count for the footer: the
// prompt's own count when nothing was added to it, else counted alone.
func rootTokens(root, combined string, combinedTokens int, count func(string) int) int {
	if root == combined {
		return combinedTokens
	}
	if strings.TrimSpace(root) == "" {
		return 0
	}
	return count(root)
}
