package review

import (
	"context"
	"errors"
	"fmt"
	"net/http"
	"path"
	"slices"
	"strings"
	"sync"

	"github.com/trick77/noergler/internal/bitbucket"
	"github.com/trick77/noergler/internal/diff"
	"github.com/trick77/noergler/internal/render"
	"github.com/trick77/noergler/internal/store"
)

// An AGENTS.md below the root holds the rules of one part of a monorepo.
// The root file alone made such a rule apply to everything or nothing.
//
// Only an ancestor directory of a changed file can hold a file that applies
// to it, so those are probed, at any depth: a cap on depth missed a
// module's file below services/x/src/ (user, 2026-11-22). Bounded by count
// instead, because every probe is a Bitbucket call on the single worker:
// at most nestedAgentsProbes per lookup, of the order a review already
// spends on thread checks (maxThreadChecks). None over nestedAgentsMaxBytes
// is used (it is tokenized on the worker, and a rules file that size is
// not one). How many are kept is the token limit's alone (withNested).
const (
	nestedAgentsProbes   = 50
	nestedAgentsMaxBytes = 32 << 10
)

// nestedMemoMax and nestedMemoMaxBytes bound nestedMemo by commits held and
// by file content held (each commit keeps its own copy): past either the
// memo starts over (probes are repeated, nothing more).
const (
	nestedMemoMax      = 1000
	nestedMemoMaxBytes = 16 << 20
)

// nestedUnlimitedTokens caps the nested files together when the team turned
// the hard token limit off (0): that switch is about the root file, and
// without a cap a wide monorepo PR could carry dozens of 32 KiB files and
// overflow the model's context. The config default.
const nestedUnlimitedTokens = 7000

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
// ancestor of a changed file below the root, at any depth. Shallow first,
// then by how many changed files sit under it, then by path, so a cut drops
// deep package directories (services/x/src/main/java/...) before any module
// root, where such files live, however many files a package holds. Path
// only, never content: the same PR probes the same directories. cut is how
// many did not fit.
func nestedCandidates(paths []string) (dirs []string, cut int) {
	under := map[string]int{}
	for _, p := range paths {
		parts := strings.Split(path.Dir(p), "/")
		if parts[0] == "." {
			continue
		}
		for depth := 1; depth <= len(parts); depth++ {
			under[strings.Join(parts[:depth], "/")+"/"]++
		}
	}
	for d := range under {
		dirs = append(dirs, d)
	}
	slices.SortFunc(dirs, func(a, b string) int {
		if da, db := strings.Count(a, "/"), strings.Count(b, "/"); da != db {
			return da - db
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

// nestedProbe is what one directory holds at one commit: a usable file
// (content), or none, or one over nestedAgentsMaxBytes (oversize, its
// size). Never a failed read: that is not a fact about the commit.
type nestedProbe struct {
	content  string
	oversize int
}

// nestedMemo is each commit's probes, by "project/repo@commit" and
// directory. A commit never changes, so an answer holds for every later
// lookup at it (a finding reply, a Q&A, a retry) and never for another
// commit: a file added or removed is a new head, probed afresh, whether
// noergler saw the commit that did it or not.
type nestedMemo struct {
	mu    sync.Mutex
	m     map[string]map[string]nestedProbe
	bytes int
}

func (c *nestedMemo) get(key, dir string) (nestedProbe, bool) {
	c.mu.Lock()
	defer c.mu.Unlock()
	p, ok := c.m[key][dir]
	return p, ok
}

func (c *nestedMemo) put(key, dir string, p nestedProbe) {
	c.mu.Lock()
	defer c.mu.Unlock()
	// Starts over for a commit it does not hold past the commit cap, or for
	// content past the byte cap.
	_, held := c.m[key]
	if c.m == nil || (!held && len(c.m) >= nestedMemoMax) || c.bytes+len(p.content) > nestedMemoMaxBytes {
		c.m = map[string]map[string]nestedProbe{}
		c.bytes = 0
	}
	c.bytes += len(p.content)
	if c.m[key] == nil {
		c.m[key] = map[string]nestedProbe{}
	}
	c.m[key][dir] = p
}

// probeNested reads one directory's AGENTS.md at a commit, through the
// memo. ok is false for a read that failed for a reason other than a 404:
// logged, not remembered, so the next lookup at the same commit reads
// again.
func (r *Reviewer) probeNested(ctx context.Context, project, repo, commit, prTag, dir string) (p nestedProbe, ok bool) {
	key := project + "/" + repo + "@" + commit
	if p, held := r.nestedMemo.get(key, dir); held {
		return p, true
	}
	file := dir + "AGENTS.md"
	content, err := r.bitbucket.FetchFileContent(ctx, project, repo, commit, file)
	var status *bitbucket.StatusError
	switch {
	case errors.As(err, &status) && status.Status == http.StatusNotFound:
	case err != nil:
		if ctx.Err() == nil {
			r.log.WarnContext(ctx, fmt.Sprintf("%s: %s not read, reviewing without it: %v", prTag, file, err))
		}
		return nestedProbe{}, false
	case strings.TrimSpace(content) == "":
	case len(content) > nestedAgentsMaxBytes:
		p.oversize = len(content)
	default:
		p.content = content
	}
	r.nestedMemo.put(key, dir, p)
	return p, true
}

// fetchNestedAgentsMD reads the AGENTS.md files below the root that apply to
// the changed paths, at one commit (the PR head: a file the PR adds or
// changes is the PR's own rule set). Every file found applies: each is in
// an ancestor directory of a changed file, so a file gets the whole chain
// from the root down (user, 2026-11-22), the deeper one winning where they
// differ (nestedBlock says so). In path order, a parent before its
// children. A missing file is absent; one over the byte cap is in left,
// logged. A read that fails otherwise ends the lookup, logged once:
// Bitbucket failing is rarely one file's fault, and each further probe
// would hold the shared worker. incomplete says why not every candidate was
// probed (render.LookupUnreadable over render.LookupProbeCap), "" when all
// were: never a list of paths, since most candidates hold no file. The root
// rules still hold, and the next run may read the rest.
func (r *Reviewer) fetchNestedAgentsMD(ctx context.Context, project, repo, commit, prTag string, paths []string) (found []nestedAgentsMD, left []store.AgentsMDLeft, incomplete string) {
	return r.lookupNested(ctx, project, repo, commit, prTag, paths, false)
}

// lookupNested is fetchNestedAgentsMD; quiet leaves out the lines about
// what was found, cut or left out, for a lookup that only checks a rule
// (stillMarked): they would repeat the prompt's per task and read as if
// the files went to the model. A failed read is logged either way.
func (r *Reviewer) lookupNested(ctx context.Context, project, repo, commit, prTag string, paths []string, quiet bool) (found []nestedAgentsMD, left []store.AgentsMDLeft, incomplete string) {
	if commit == "" {
		return nil, nil, ""
	}
	dirs, cut := nestedCandidates(paths)
	if cut > 0 {
		if !quiet {
			r.log.InfoContext(ctx, fmt.Sprintf("%s: %d director(ies) not probed for an AGENTS.md (cap %d)", prTag, cut, nestedAgentsProbes))
		}
		incomplete = render.LookupProbeCap
	}
probe:
	for i, dir := range dirs {
		if ctx.Err() != nil {
			return nil, nil, ""
		}
		p, ok := r.probeNested(ctx, project, repo, commit, prTag, dir)
		switch {
		case !ok:
			if rest := len(dirs) - i - 1; rest > 0 {
				r.log.WarnContext(ctx, fmt.Sprintf("%s: nested AGENTS.md lookup stopped, %d director(ies) not probed", prTag, rest))
			}
			incomplete = render.LookupUnreadable
			break probe
		case p.oversize > 0:
			if !quiet {
				r.log.WarnContext(ctx, fmt.Sprintf("%s: %sAGENTS.md left out: %d bytes, over %d", prTag, dir, p.oversize, nestedAgentsMaxBytes))
			}
			left = append(left, store.AgentsMDLeft{Path: dir + "AGENTS.md", Reason: render.LeftOversize})
		case p.content != "":
			found = append(found, nestedAgentsMD{Dir: dir, Content: p.content})
		}
	}
	slices.SortFunc(found, func(a, b nestedAgentsMD) int { return strings.Compare(a.Dir, b.Dir) })
	slices.SortFunc(left, func(a, b store.AgentsMDLeft) int { return strings.Compare(a.Path, b.Path) })
	if !quiet {
		for _, n := range found {
			r.log.InfoContext(ctx, fmt.Sprintf("%s: loaded %s", prTag, n.Path()))
		}
	}
	return found, left, incomplete
}

// nestedBlock is one nested file as the prompt carries it. The heading
// tells the model the scope and that the deeper file wins, so no prompt
// needs a line of its own for it.
func nestedBlock(n nestedAgentsMD) string {
	return fmt.Sprintf("### %s (applies to files under `%s`; where it differs from the guidelines above, it wins for them)\n\n%s",
		n.Path(), n.Dir, strings.TrimRight(n.Content, "\n"))
}

// withNested is the root AGENTS.md followed by the nested ones, each under
// a line naming the directory it applies to. A nested file that would take
// the whole over the token limit is left out, in path order, and named in
// dropped: the root decides whether a review runs, never a nested file.
// With the limit off (<= 0) the root is not counted and the nested files
// alone stay under nestedUnlimitedTokens. kept are the carried ones with
// their tokens, for the footer and the run page. Each part is counted once.
func withNested(root string, nested []nestedAgentsMD, maxTokens int, count func(string) int) (combined string, kept []store.AgentsMDFile, dropped []string) {
	var parts []string
	total, limit := 0, maxTokens
	if limit <= 0 {
		limit = nestedUnlimitedTokens
	}
	if strings.TrimSpace(root) != "" {
		parts = append(parts, strings.TrimRight(root, "\n"))
		if maxTokens > 0 {
			total = count(root)
		}
	}
	for _, n := range nested {
		block := nestedBlock(n)
		c := count(block)
		if total+c > limit {
			dropped = append(dropped, n.Path())
			continue
		}
		total += c
		parts = append(parts, block)
		kept = append(kept, store.AgentsMDFile{Path: n.Path(), Tokens: c})
	}
	return strings.Join(parts, "\n\n"), kept, dropped
}

// leftOver is the nested files left out over the token limit, as
// store.AgentsMDLeft, after the ones fetchNestedAgentsMD left out; path
// order across both.
func leftOver(left []store.AgentsMDLeft, dropped []string) []store.AgentsMDLeft {
	out := slices.Clone(left)
	for _, p := range dropped {
		out = append(out, store.AgentsMDLeft{Path: p, Reason: render.LeftTokenLimit})
	}
	slices.SortStableFunc(out, func(a, b store.AgentsMDLeft) int { return strings.Compare(a.Path, b.Path) })
	return out
}

// renderNested and renderLeft hand the run's nested facts to the footer.
func renderNested(files []store.AgentsMDFile) []render.AgentsMDFile {
	out := make([]render.AgentsMDFile, len(files))
	for i, f := range files {
		out[i] = render.AgentsMDFile{Path: f.Path, Tokens: f.Tokens}
	}
	return out
}

func renderLeft(left []store.AgentsMDLeft) []render.AgentsMDLeft {
	out := make([]render.AgentsMDLeft, len(left))
	for i, l := range left {
		out[i] = render.AgentsMDLeft{Path: l.Path, Reason: l.Reason}
	}
	return out
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
