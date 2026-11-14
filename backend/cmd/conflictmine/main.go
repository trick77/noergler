// Command conflictmine turns real merge commits into merge-conflict eval
// cases (internal/evals/conflicts). It is a dev tool beside cmd/evals: it
// reads a local clone with the git binary and makes no model call.
//
//	git clone --bare --filter=blob:none https://github.com/git/git.git /tmp/git.git
//	go run ./cmd/conflictmine -repo /tmp/git.git -url https://github.com/git/git \
//	  -scan 2000 -out /tmp/cases
//
// For every two-parent merge it asks git whether the parents conflict, then
// merges each conflicted file again with noergler's own merge.Merge3 (the
// first parent is the branch merged into, noergler's source; the second the
// branch merged in, its target). The merge commit's own file is the truth:
// the lines its author kept where each conflict was. A conflict the model
// would be asked about becomes an expected answer when ids can write the
// author's lines, and makes the case a control when they cannot (the
// author wrote a line neither side has).
//
// A merge is skipped, never bent to fit: a file one side added, deleted or
// renamed, a lock or generated file, a merge Merge3 calls ambiguous or
// conflict-free, more conflicts than production resolves, or an author who
// also changed settled lines (an evil merge: the truth is not where the
// conflict was).
//
// Every case is a draft: read it, add the other correct answers by hand,
// and write its why before committing it.
//
// It also counts the conflicts code resolves on its own (merge.Trivial)
// and how often the author resolved one differently: a measure of the
// trivial rule on real merges.
package main

import (
	"bytes"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"slices"
	"sort"
	"strings"

	"github.com/trick77/noergler/internal/evals"
	"github.com/trick77/noergler/internal/merge"
)

func main() {
	os.Exit(run(os.Args[1:], os.Stdout, os.Stderr))
}

// run is the command, exit code returned so a test can drive it.
func run(args []string, stdout, stderr io.Writer) int {
	fs := flag.NewFlagSet("conflictmine", flag.ContinueOnError)
	fs.SetOutput(stderr)
	repo := fs.String("repo", "", "local clone (bare is fine)")
	url := fs.String("url", "", "the repo's web URL, for each case's source")
	rev := fs.String("rev", "HEAD", "where to start walking merges")
	scan := fs.Int("scan", 500, "merges to look at")
	merges := fs.String("merges", "", "comma-separated merge commits instead of -scan")
	out := fs.String("out", "", "directory for the case files")
	if err := fs.Parse(args); err != nil {
		return 2
	}
	if *repo == "" || *url == "" || *out == "" {
		_, _ = fmt.Fprintln(stderr, "conflictmine: -repo, -url and -out are required")
		return 2
	}
	m := miner{repo: *repo, url: strings.TrimSuffix(*url, "/"), out: *out, skipped: map[string]int{}, trivialKinds: map[string]int{},
		stdout: stdout, stderr: stderr}
	var list []string
	if *merges != "" {
		list = strings.Split(*merges, ",")
	} else {
		raw, err := m.git("rev-list", "--min-parents=2", "--max-parents=2", "-n", fmt.Sprint(*scan), *rev)
		if err != nil {
			_, _ = fmt.Fprintln(stderr, "conflictmine:", err)
			return 2
		}
		list = strings.Fields(string(raw))
	}
	if err := os.MkdirAll(*out, 0o750); err != nil {
		_, _ = fmt.Fprintln(stderr, "conflictmine:", err)
		return 2
	}
	for _, sha := range list {
		sha = strings.TrimSpace(sha)
		why, err := m.mine(sha)
		switch {
		case err != nil:
			_, _ = fmt.Fprintf(stderr, "%s: %v\n", sha, err)
			m.skipped["error"]++
		case why != "":
			m.skipped[why]++
		}
	}
	_, _ = fmt.Fprintf(stdout, "%d merges, %d conflicted, %d cases written\n", len(list), m.conflicted, m.written)
	// Conflicts counted are those of merges within production's bounds:
	// what noergler would resolve, by code or by the model.
	_, _ = fmt.Fprintf(stdout, "conflicts in bounds: %d, code resolves %d, the model is asked %d\n",
		m.trivial+m.asked, m.trivial, m.asked)
	_, _ = fmt.Fprintf(stdout, "code against the author: %d same, %d order only, %d blank lines (order aside), %d content\n",
		m.trivialKinds[kindSame], m.trivialKinds[kindOrder], m.trivialKinds[kindBlank], m.trivialKinds[kindContent])
	whys := make([]string, 0, len(m.skipped))
	for why := range m.skipped {
		whys = append(whys, why)
	}
	sort.Strings(whys)
	for _, why := range whys {
		_, _ = fmt.Fprintf(stdout, "  skipped %-28s %d\n", why, m.skipped[why])
	}
	return 0
}

type miner struct {
	repo, url, out      string
	conflicted, written int
	// trivial counts conflicts code resolves, asked those for the model;
	// trivialKinds how code's resolution compares with the author's.
	trivial, asked int
	trivialKinds   map[string]int
	skipped        map[string]int
	stdout, stderr io.Writer
}

// How a resolution compares with the author's lines.
const (
	kindSame    = "same"
	kindOrder   = "order only"
	kindBlank   = "blank lines, order aside"
	kindContent = "content"
)

// compare says how got differs from want: not at all, only in the order
// of its lines, only in blank lines (order aside), or in content.
func compare(got, want []string) string {
	if slices.Equal(got, want) {
		return kindSame
	}
	sorted := func(ls []string, dropBlank bool) []string {
		var out []string
		for _, l := range ls {
			if dropBlank && strings.TrimSpace(l) == "" {
				continue
			}
			out = append(out, l)
		}
		slices.Sort(out)
		return out
	}
	switch {
	case slices.Equal(sorted(got, false), sorted(want, false)):
		return kindOrder
	case slices.Equal(sorted(got, true), sorted(want, true)):
		return kindBlank
	}
	return kindContent
}

func (m *miner) git(args ...string) ([]byte, error) {
	cmd := exec.Command("git", append([]string{"-C", m.repo}, args...)...) //nolint:gosec // G204: a dev tool's own git calls
	var stderr bytes.Buffer
	cmd.Stderr = &stderr
	b, err := cmd.Output()
	if err != nil {
		return b, fmt.Errorf("git %s: %w: %s", strings.Join(args, " "), err, strings.TrimSpace(stderr.String()))
	}
	return b, nil
}

// conflicts reports whether git's own merge of the parents conflicts: a
// cheap filter, the files themselves are merged again by merge.Merge3.
func (m *miner) conflicts(p1, p2 string) (bool, error) {
	cmd := exec.Command("git", "-C", m.repo, "merge-tree", "--write-tree", "--name-only", "--no-messages", p1, p2) //nolint:gosec // G204
	err := cmd.Run()
	var exit *exec.ExitError
	switch {
	case err == nil:
		return false, nil
	case errors.As(err, &exit) && exit.ExitCode() == 1:
		return true, nil
	default:
		return false, err
	}
}

// changed is the paths that differ between two commits, renames as a
// delete and an add, NUL-separated so no path comes back quoted.
func (m *miner) changed(from, to string) (map[string]bool, error) {
	raw, err := m.git("diff", "--name-only", "-z", "--no-renames", from, to)
	if err != nil {
		return nil, err
	}
	out := map[string]bool{}
	for _, p := range strings.Split(string(raw), "\x00") {
		if p != "" {
			out[p] = true
		}
	}
	return out, nil
}

// entry is a path's mode at a commit, "" when the commit has no such path.
func (m *miner) entry(rev, path string) (string, error) {
	raw, err := m.git("ls-tree", "--format=%(objectmode)", rev, "--", path)
	return strings.TrimSpace(string(raw)), err
}

func (m *miner) blob(rev, path string) ([]byte, error) {
	return m.git("cat-file", "blob", rev+":"+path)
}

var mergeSubject = regexp.MustCompile(`^Merge (?:remote-tracking )?branch '([^']+)'(?: of \S+)?(?: into (\S+))?`)

// mine looks at one merge; skip names why it gave no case.
func (m *miner) mine(rev string) (skip string, err error) {
	// The full id: a short hash or a ref name given to -merges would make
	// the case name and its commit URL.
	full, err := m.git("rev-parse", "--verify", rev+"^{commit}")
	if err != nil {
		return "", err
	}
	sha := strings.TrimSpace(string(full))
	raw, err := m.git("rev-parse", sha+"^1", sha+"^2")
	if err != nil {
		return "", err
	}
	parents := strings.Fields(string(raw))
	source, target := parents[0], parents[1]
	conflicted, err := m.conflicts(source, target)
	if err != nil {
		return "", err
	}
	if !conflicted {
		return "clean", nil
	}
	m.conflicted++
	bases, err := m.git("merge-base", "--all", source, target)
	if err != nil {
		return "", err
	}
	if len(strings.Fields(string(bases))) != 1 {
		return "criss_cross", nil
	}
	base := strings.TrimSpace(string(bases))
	files, skip, err := m.mergeFiles(base, source, target, sha)
	if skip != "" || err != nil {
		return skip, err
	}

	subject, _ := m.git("log", "-1", "--format=%s", sha)
	tb, sb := "topic", "master"
	if g := mergeSubject.FindStringSubmatch(strings.TrimSpace(string(subject))); g != nil {
		tb = g[1]
		if g[2] != "" {
			sb = g[2]
		}
	}
	c := evals.ConflictCase{
		Name:         "mined-" + filepath.Base(strings.TrimSuffix(m.url, ".git")) + "-" + sha[:10],
		Origin:       evals.OriginMined,
		Source:       m.url + "/commit/" + sha,
		TargetBranch: tb,
		SourceBranch: sb,
	}
	asked, inexpressible := 0, 0
	for _, f := range files {
		cf := evals.ConflictFile{Path: f.path, Merged: diff3(f.chunks)}
		for k, x := range merge.Conflicts(f.chunks) {
			if x.Trimmed {
				cf.Trimmed = append(cf.Trimmed, k+1)
			}
		}
		c.Files = append(c.Files, cf)
		for k, cf := range merge.Conflicts(f.chunks) {
			want := trimAll(f.truth[k])
			if cf.Trivial() {
				m.trivial++
				kind := compare(trimAll(merge.Pick(cf, merge.TrivialResolution(cf))), want)
				m.trivialKinds[kind]++
				if kind != kindSame {
					_, _ = fmt.Fprintf(m.stderr, "trivial resolved differently (%s): %s %s#%d\n--- target\n%s\n--- source\n%s\n--- author\n%s\n",
						kind, c.Source, f.path, k+1, strings.Join(trimAll(cf.Target), "\n"), strings.Join(trimAll(cf.Source), "\n"),
						strings.Join(want, "\n"))
				}
				continue
			}
			m.asked++
			asked++
			if !evals.Expressible(cf, want) {
				inexpressible++
				continue
			}
			text := strings.Join(want, "\n")
			if len(want) > 0 {
				text += "\n"
			}
			c.Accept = append(c.Accept, evals.ConflictAccept{File: f.path, Conflict: k + 1, Texts: []string{text}})
		}
	}
	if asked == 0 {
		return "trivial_only", nil
	}
	if inexpressible > 0 {
		c.Control, c.Accept = true, nil
		c.Why = "Control (mined): the merge's author wrote lines neither side has. TODO: check by hand."
	} else {
		c.Why = "Mined: the merge's author kept these lines. TODO: check by hand, add other correct answers."
	}
	if _, err := evals.LoadConflictCase(c); err != nil {
		return "", fmt.Errorf("case does not load: %w", err)
	}
	// Code is full of <, > and &: escaped, a case is unreadable by hand.
	var blob bytes.Buffer
	enc := json.NewEncoder(&blob)
	enc.SetEscapeHTML(false)
	enc.SetIndent("", "  ")
	if err := enc.Encode(c); err != nil {
		return "", err
	}
	if err := os.WriteFile(filepath.Join(m.out, c.Name+".json"), blob.Bytes(), 0o600); err != nil {
		return "", err
	}
	m.written++
	_, _ = fmt.Fprintf(m.stdout, "%s: %d asked, control=%v\n", c.Name, asked, c.Control)
	return "", nil
}

// mined is one file of a case.
type mined struct {
	path   string
	chunks []merge.Chunk
	truth  [][]string // per conflict: the author's lines
}

// mergeFiles checks every path both parents changed as production does
// (the same refusals, in the same bounds) and merges it with merge.Merge3;
// the files with a conflict are the case. skip names a refusal.
func (m *miner) mergeFiles(base, source, target, sha string) (files []mined, skip string, err error) {
	bs, err := m.changed(base, source)
	if err != nil {
		return nil, "", err
	}
	bt, err := m.changed(base, target)
	if err != nil {
		return nil, "", err
	}
	var both []string
	for p := range bs {
		if bt[p] {
			both = append(both, p)
		}
	}
	sort.Strings(both)
	if len(both) > merge.MaxBothChanged {
		return nil, "too_many_both_changed", nil
	}
	total := 0
	for _, p := range both {
		var mode [3]string
		for i, rev := range []string{base, source, target} {
			if mode[i], err = m.entry(rev, p); err != nil {
				return nil, "", err
			}
		}
		switch {
		case mode[0] == "" || mode[1] == "" || mode[2] == "":
			return nil, "added_deleted_renamed", nil
		case mode[0] != mode[1] || mode[1] != mode[2] || (mode[1] != "100644" && mode[1] != "100755"):
			return nil, "mode", nil
		}
		var v [3][]byte
		for i, rev := range []string{base, source, target} {
			if v[i], err = m.blob(rev, p); err != nil {
				return nil, "", err
			}
		}
		if bytes.Equal(v[1], v[2]) {
			continue // the same change on both: no merge
		}
		if why := merge.Excluded(p, v[0], v[1], v[2]); why != "" {
			return nil, why, nil
		}
		chunks, err := merge.Merge3(merge.Lines(v[0]), merge.Lines(v[2]), merge.Lines(v[1]))
		switch {
		case errors.Is(err, merge.ErrAmbiguous):
			return nil, "ambiguous", nil
		case errors.Is(err, merge.ErrTooDifferent):
			return nil, "too_different", nil
		case err != nil:
			return nil, "", err
		}
		cs := merge.Conflicts(chunks)
		if len(cs) == 0 {
			continue
		}
		for _, c := range cs {
			if c.Oversized() {
				return nil, "oversized", nil
			}
		}
		// A case file ends every line; one without (a file's last) could
		// make two sides equal that production keeps apart.
		for _, ch := range chunks {
			lines := ch.Lines
			if ch.Conflict != nil {
				lines = slices.Concat(ch.Conflict.Base, ch.Conflict.Target, ch.Conflict.Source)
			}
			for _, l := range lines {
				if !strings.HasSuffix(l, "\n") {
					return nil, "no_final_newline", nil
				}
			}
		}
		merged, err := m.blob(sha, p)
		if err != nil {
			return nil, "evil_merge_or_unclear", nil //nolint:nilerr // the merge deleted the file: no truth
		}
		truth, ok := authorRegions(chunks, merge.Lines(merged))
		if !ok {
			return nil, "evil_merge_or_unclear", nil
		}
		total += len(cs)
		files = append(files, mined{path: p, chunks: chunks, truth: truth})
	}
	switch {
	case len(files) == 0:
		return nil, "no_conflict_here", nil
	case len(files) > merge.MaxConflictFiles:
		return nil, "too_many_files", nil
	case total > merge.MaxConflicts:
		return nil, "too_many_conflicts", nil
	}
	return files, "", nil
}

// authorRegions finds, in the merge commit's file, the lines where each
// conflict was: every settled chunk must appear there in order, unchanged,
// and the file must end with the last chunk. Every split is tried, not the
// first place the next settled chunk appears: an author's region may
// contain that chunk's text. ok is false when no split fits (an evil
// merge) or more than one does (the truth is ambiguous).
func authorRegions(chunks []merge.Chunk, merged []string) (regions [][]string, ok bool) {
	var found [][][]string
	var walk func(i, p int, acc [][]string)
	walk = func(i, p int, acc [][]string) {
		if len(found) > 1 {
			return
		}
		if i == len(chunks) {
			if p == len(merged) {
				found = append(found, slices.Clone(acc))
			}
			return
		}
		if c := chunks[i]; c.Conflict == nil {
			if hasAt(merged, p, c.Lines) {
				walk(i+1, p+len(c.Lines), acc)
			}
			return
		}
		for q := p; q <= len(merged) && len(found) < 2; q++ {
			walk(i+1, q, append(acc[:len(acc):len(acc)], merged[p:q]))
		}
	}
	walk(0, 0, nil)
	if len(found) != 1 {
		return nil, false
	}
	return found[0], true
}

func hasAt(ls []string, at int, want []string) bool {
	if at+len(want) > len(ls) {
		return false
	}
	for i, w := range want {
		if ls[at+i] != w {
			return false
		}
	}
	return true
}

// trimAll is merge.TrimEOLs.
var trimAll = merge.TrimEOLs

// diff3 writes chunks in the corpus's diff3 form, settled runs cut to
// the context production shows around a conflict.
func diff3(chunks []merge.Chunk) string {
	var b strings.Builder
	n := merge.ContextLines
	for i, c := range chunks {
		if c.Conflict == nil {
			ls := c.Lines
			switch {
			case i == 0:
				ls = ls[max(0, len(ls)-n):]
			case i == len(chunks)-1:
				ls = ls[:min(len(ls), n)]
			case len(ls) > 2*n:
				ls = append(append([]string(nil), ls[:n]...), ls[len(ls)-n:]...)
			}
			for _, l := range ls {
				b.WriteString(eol(l))
			}
			continue
		}
		b.WriteString("<<<<<<< source\n")
		for _, l := range c.Conflict.Source {
			b.WriteString(eol(l))
		}
		b.WriteString("||||||| base\n")
		for _, l := range c.Conflict.Base {
			b.WriteString(eol(l))
		}
		b.WriteString("=======\n")
		for _, l := range c.Conflict.Target {
			b.WriteString(eol(l))
		}
		b.WriteString(">>>>>>> target\n")
	}
	return b.String()
}

// eol ends a line the corpus way; a file's unterminated last line gets
// one, which no conflict's choice depends on.
func eol(l string) string {
	if strings.HasSuffix(l, "\n") {
		return l
	}
	return l + "\n"
}
