package review

import (
	"context"
	"fmt"
	"slices"
	"strings"
	"sync"
	"time"

	"github.com/trick77/noergler/internal/bitbucket"
	"github.com/trick77/noergler/internal/lang"
	"github.com/trick77/noergler/internal/render"
	snap "github.com/trick77/noergler/internal/snapshot"
	"github.com/trick77/noergler/internal/store"
	"github.com/trick77/noergler/internal/webhook"
)

// The SNAPSHOT task (user, 2026-11-15): a PR into the repo's default branch
// whose delta adds an unreleased Maven/Gradle build (snapshot.AddedIn) gets
// a Bitbucket task that the author must resolve; where the repo's
// "incomplete tasks" merge check is on, it blocks the merge. Code decides,
// never the model: a task that blocks a merge hangs on a blunt rule.
//
// Run in prepare on every review run past the author gates, the stop and
// the opt-out branch (a stopped PR gets no check, user: decided), before
// any guard that can end a run early: no model, so the AGENTS.md gates and
// the cost cap never skip it. A retarget (pr:modified) runs it alone
// (HandleRetargeted). Its own diff, one build file at a time: the review's
// diff may be cut, a push's is not the delta.
//
// Every task names the lines it was posted for (pull_requests.snapshot_tasks),
// so a task the author resolved is never posted again for them; a line no
// task named gets a new task. A line gone from the delta leaves its task's
// list, and an open task's text follows its list; an open task with no line
// left is resolved by noergler. A SNAPSHOT removed and added again is new.
// Any read that fails leaves everything as it is: an unknown is never "no
// hit".
//
// Off is off (user, 2026-11-15: "off means off"): with the instance's
// REVIEW_CREATE_TASKS or the team's snapshot_task off, no task call at all.
//
// The package is imported as snap: this package's tests have a snapshot()
// helper.

// defaultBranchTTL is how long a repo's default branch is taken as read: a
// repo changes it rarely, and every push would ask otherwise.
const defaultBranchTTL = time.Hour

// maxSnapshotBuildFiles caps the build files whose diff one check reads, a
// Bitbucket call each on the single worker; past it (a reactor's version
// bump) the whole PR diff is read once instead.
const maxSnapshotBuildFiles = 50

// hitsMemo is each PR's last hits by the build files' blob ids: a push
// that changes no build file reads no diff again.
type hitsMemo struct {
	mu sync.Mutex
	m  map[store.PRKey]hitsEntry
}

type hitsEntry struct {
	sig  string
	hits []snap.Hit
}

func (c *hitsMemo) get(k store.PRKey, sig string) ([]snap.Hit, bool) {
	c.mu.Lock()
	defer c.mu.Unlock()
	e, ok := c.m[k]
	if !ok || e.sig != sig {
		return nil, false
	}
	return e.hits, true
}

func (c *hitsMemo) put(k store.PRKey, sig string, hits []snap.Hit) {
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.m == nil {
		c.m = map[store.PRKey]hitsEntry{}
	}
	c.m[k] = hitsEntry{sig: sig, hits: hits}
}

// buildSig is the build files' identity for hitsMemo, "" when any lacks a
// blob id (nothing remembered then).
func buildSig(build []bitbucket.Change) string {
	var b strings.Builder
	for _, c := range build {
		if c.ContentID == "" {
			return ""
		}
		fmt.Fprintf(&b, "%s\x00%s\x00%s\x00%s\n", c.Path, c.SrcPath, c.FromContentID, c.ContentID)
	}
	return b.String()
}

// branchCache is each repo's default branch ref id, by project/repo.
type branchCache struct {
	mu sync.Mutex
	m  map[string]branchEntry
}

type branchEntry struct {
	ref string
	at  time.Time
}

func (c *branchCache) get(key string, now time.Time) (string, bool) {
	c.mu.Lock()
	defer c.mu.Unlock()
	e, ok := c.m[key]
	if !ok || now.Sub(e.at) > defaultBranchTTL {
		return "", false
	}
	return e.ref, true
}

func (c *branchCache) put(key, ref string, now time.Time) {
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.m == nil {
		c.m = map[string]branchEntry{}
	}
	c.m[key] = branchEntry{ref: ref, at: now}
}

// defaultBranch is the repo's default branch ref id (refs/heads/main).
func (r *Reviewer) defaultBranch(ctx context.Context, project, repo string) (string, error) {
	key, now := project+"/"+repo, time.Now()
	if ref, ok := r.defaultBranches.get(key, now); ok {
		return ref, nil
	}
	ref, err := r.bitbucket.DefaultBranch(ctx, project, repo)
	if err != nil {
		return "", err
	}
	r.defaultBranches.put(key, ref, now)
	return ref, nil
}

// refID is a ref's full id; the webhook may carry only the display id.
func refID(ref webhook.Ref) string {
	if ref.ID != "" {
		return ref.ID
	}
	return "refs/heads/" + ref.DisplayID
}

// snapshotHits is the PR delta's hits, nil when the PR goes elsewhere than
// the default branch: a task there is never wanted, and noergler resolves
// its own. Only the changed build files' diffs are read.
func (r *Reviewer) snapshotHits(ctx context.Context, pr webhook.PullRequest, key store.PRKey) ([]snap.Hit, error) {
	def, err := r.defaultBranch(ctx, key.Project, key.Repo)
	if err != nil {
		return nil, fmt.Errorf("default branch: %w", err)
	}
	if refID(pr.ToRef) != def {
		return nil, nil
	}
	cs, err := r.bitbucket.FetchPRChangeSet(ctx, key.Project, key.Repo, key.PRID)
	if err != nil {
		return nil, fmt.Errorf("changes: %w", err)
	}
	var build []bitbucket.Change
	for _, c := range cs.Changes {
		if c.Type != "DELETE" && snap.BuildFile(c.Path) {
			build = append(build, c)
		}
	}
	if len(build) == 0 {
		return nil, nil
	}
	sig := buildSig(build)
	if sig != "" {
		if hits, ok := r.snapshotMemo.get(key, sig); ok {
			return hits, nil
		}
	}
	hits, err := r.readHits(ctx, key, build)
	if err != nil {
		return nil, err
	}
	if sig != "" {
		r.snapshotMemo.put(key, sig, hits)
	}
	return hits, nil
}

// readHits reads the build files' diffs, one call each, or the whole PR
// diff once past maxSnapshotBuildFiles.
func (r *Reviewer) readHits(ctx context.Context, key store.PRKey, build []bitbucket.Change) ([]snap.Hit, error) {
	if len(build) > maxSnapshotBuildFiles {
		raw, err := r.bitbucket.FetchPRDiff(ctx, key.Project, key.Repo, key.PRID, 0)
		if err != nil {
			return nil, fmt.Errorf("diff (%d build files): %w", len(build), err)
		}
		return snap.Added(raw), nil
	}
	var hits []snap.Hit
	for _, c := range build {
		src := ""
		if c.Type == "MOVE" || c.Type == "COPY" {
			src = c.SrcPath
		}
		raw, err := r.bitbucket.FetchPRFileDiff(ctx, key.Project, key.Repo, key.PRID, c.Path, src)
		if err != nil {
			return nil, fmt.Errorf("diff of %s: %w", c.Path, err)
		}
		hits = append(hits, snap.FileHits(c.Path, raw)...)
	}
	return hits, nil
}

// checkSnapshotTask keeps the PR's SNAPSHOT tasks in step with its delta.
func (r *Reviewer) checkSnapshotTask(ctx context.Context, pr webhook.PullRequest, key store.PRKey,
	upsert store.PRUpsert, prTag string, l lang.Lang) {
	if !r.createTasks || !r.snapshotTask.Load() || r.store == nil {
		return
	}
	unknown := func(format string, args ...any) {
		if ctx.Err() == nil {
			r.log.WarnContext(ctx, fmt.Sprintf("%s: snapshot task: %s, left as it is", prTag, fmt.Sprintf(format, args...)))
		}
	}
	hits, err := r.snapshotHits(ctx, pr, key)
	if err != nil {
		unknown("%v", err)
		return
	}
	rowID, tasks, err := r.store.SnapshotTasks(ctx, key)
	if err != nil {
		unknown("read failed: %v", err)
		return
	}
	if len(hits) == 0 && len(tasks) == 0 {
		return
	}

	// Every open task on the PR, one call: a task a person resolved, deleted
	// or reopened reads as it is now, and noergler's own open task the row
	// lost (a write that failed past a post) is found by its text below.
	openList, err := r.bitbucket.OpenTasks(ctx, key.Project, key.Repo, key.PRID)
	if err != nil {
		unknown("open tasks: %v", err)
		return
	}
	open := make(map[int]bitbucket.Comment, len(openList))
	for _, c := range openList {
		open[c.ID] = c
	}

	cur := make(map[string]snap.Hit, len(hits))
	for _, h := range hits {
		cur[h.Key()] = h
	}
	branch := pr.ToRef.Branch()
	named, known := map[string]bool{}, map[int]bool{}
	next := make([]store.SnapshotTask, 0, len(tasks)+1)
	for _, t := range tasks {
		known[t.ID] = true
		var keys []string
		for _, k := range t.Keys {
			if _, ok := cur[k]; ok {
				keys = append(keys, k)
				named[k] = true
			}
		}
		c, isOpen := open[t.ID]
		if isOpen != t.Open {
			r.log.InfoContext(ctx, fmt.Sprintf("%s: snapshot task %d is %s, by a person", prTag, t.ID,
				map[bool]string{true: "open again", false: "resolved or deleted"}[isOpen]))
		}
		t.Open = isOpen
		switch {
		case t.Open && len(keys) == 0:
			if err := r.bitbucket.ResolveTask(ctx, key.Project, key.Repo, key.PRID, t.ID); err != nil {
				r.log.WarnContext(ctx, fmt.Sprintf("%s: snapshot task %d: resolve failed: %v, tried again next run", prTag, t.ID, err))
				next = append(next, store.SnapshotTask{ID: t.ID, Open: true})
				continue
			}
			r.log.InfoContext(ctx, fmt.Sprintf("%s: snapshot task %d resolved: none of its lines left in the delta", prTag, t.ID))
			continue
		case len(keys) == 0:
			continue // done by a person, its lines gone: nothing to remember
		case t.Open:
			// The text follows the task's lines: one gone or moved shows.
			text := render.SnapshotTask(l, branch, snapshotLines(keys, cur))
			if strings.TrimSpace(render.NoEmDash(text)) != strings.TrimSpace(c.Text) {
				if _, err := r.bitbucket.UpdatePRComment(ctx, key.Project, key.Repo, key.PRID, t.ID, c.Version, text); err != nil {
					r.log.WarnContext(ctx, fmt.Sprintf("%s: snapshot task %d: update failed: %v, tried again next run", prTag, t.ID, err))
				}
			}
		}
		next = append(next, store.SnapshotTask{ID: t.ID, Keys: keys, Open: t.Open})
	}

	var fresh []string
	for _, h := range hits {
		if k := h.Key(); !named[k] {
			fresh = append(fresh, k)
		}
	}
	if len(fresh) > 0 {
		lines := snapshotLines(fresh, cur)
		text := render.SnapshotTask(l, branch, lines)
		if id := r.lostTask(openList, known, text); id != 0 {
			r.log.InfoContext(ctx, fmt.Sprintf("%s: snapshot task %d found open with this text, taken over", prTag, id))
			next = append(next, store.SnapshotTask{ID: id, Keys: fresh, Open: true})
		} else if id, err := r.bitbucket.PostPRTask(ctx, key.Project, key.Repo, key.PRID, text); err != nil {
			r.log.WarnContext(ctx, fmt.Sprintf("%s: snapshot task: post failed: %v, tried again next run", prTag, err))
		} else {
			r.log.InfoContext(ctx, fmt.Sprintf("%s: snapshot task %d posted: %d SNAPSHOT line(s) %s", prTag, id, len(fresh), lineList(lines)))
			next = append(next, store.SnapshotTask{ID: id, Keys: fresh, Open: true})
		}
	}
	if slices.EqualFunc(next, tasks, sameTask) {
		return
	}
	// Past a post or a resolve, the row must say so: a lost id would post
	// the task again and never resolve the first. A shutdown must not cut
	// the write short.
	wctx := context.WithoutCancel(ctx)
	if rowID == 0 {
		if rowID, _ = r.upsertKeepingPrior(wctx, upsert, key); rowID == 0 {
			r.log.ErrorContext(ctx, fmt.Sprintf("%s: snapshot task: no PR row to record %d task(s) in", prTag, len(next)))
			return
		}
	}
	if err := r.store.SetSnapshotTasks(wctx, rowID, next); err != nil {
		r.log.ErrorContext(ctx, fmt.Sprintf("%s: snapshot task: write failed: %v", prTag, err))
	}
}

// lostTask is noergler's own open task with exactly text that no stored
// task names, 0 for none: one posted whose id a failed write lost.
func (r *Reviewer) lostTask(openList []bitbucket.Comment, known map[int]bool, text string) int {
	want := strings.TrimSpace(render.NoEmDash(text))
	for _, c := range openList {
		if !known[c.ID] && c.Author.Name == r.bitbucket.BotUsername() && strings.TrimSpace(c.Text) == want {
			return c.ID
		}
	}
	return 0
}

func sameTask(a, b store.SnapshotTask) bool {
	return a.ID == b.ID && a.Open == b.Open && slices.Equal(a.Keys, b.Keys)
}

// snapshotLines is the task lines of keys, in their order.
func snapshotLines(keys []string, cur map[string]snap.Hit) []render.SnapshotLine {
	out := make([]render.SnapshotLine, 0, len(keys))
	for _, k := range keys {
		h := cur[k]
		out = append(out, render.SnapshotLine{Path: h.Path, Line: h.Line, Text: h.Text})
	}
	return out
}

// lineList is the lines' places for a log line, capped like the task.
func lineList(lines []render.SnapshotLine) string {
	var out []string
	for i, s := range lines {
		if i == render.SnapshotTaskMaxListed {
			out = append(out, fmt.Sprintf("+%d", len(lines)-i))
			break
		}
		out = append(out, fmt.Sprintf("%s:%d", s.Path, s.Line))
	}
	return strings.Join(out, ",")
}

// HandleRetargeted runs the SNAPSHOT task alone for a pr:modified that
// changed the PR's target: a retarget onto the default branch with no push
// would get none, one away from it would keep its task. The review's gates
// that come before the task apply (claim, author, actor, stop, opt-out).
func (r *Reviewer) HandleRetargeted(ctx context.Context, p *webhook.Payload) {
	if !r.createTasks || !r.snapshotTask.Load() || r.store == nil {
		return
	}
	pr := p.PullRequest
	ctx, key, prTag, ok := prContext(ctx, p)
	if !ok {
		return
	}
	author := pr.Author.User.Name
	switch {
	case pr.State != "" && pr.State != "OPEN":
		return
	case r.owns != nil && !r.owns(key.Project, key.Repo):
		return
	case !r.IsAutoReviewAuthor(author):
		return
	case p.Actor != nil && r.isIgnoredAuthor(p.Actor.Name):
		return
	case r.checkSkipState(ctx, key, prTag) != SkipNone:
		return
	}
	if kw := r.cfg.OptOutBranchKeyword; kw != "" && strings.Contains(strings.ToLower(pr.FromRef.Branch()), strings.ToLower(kw)) {
		return
	}
	r.log.InfoContext(ctx, fmt.Sprintf("%s: retargeted to %s, checking the snapshot task", prTag, pr.ToRef.Branch()))
	upsert := store.PRUpsert{Key: key, TeamSlug: r.TeamSlug, Author: &author, Title: &pr.Title, OpenedAt: epochMSToTime(pr.CreatedDate)}
	r.checkSnapshotTask(ctx, pr, key, upsert, prTag, r.languageFor(ctx, author))
}
