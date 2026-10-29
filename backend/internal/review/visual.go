package review

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"maps"
	"math"
	"net/http"
	"slices"
	"strings"

	"github.com/trick77/noergler/internal/bitbucket"
	"github.com/trick77/noergler/internal/diff"
	"github.com/trick77/noergler/internal/inference"
	"github.com/trick77/noergler/internal/render"
	"github.com/trick77/noergler/internal/store"
	"github.com/trick77/noergler/internal/visual"
)

// maxVisualFetched bounds the screenshot reads one review makes, on the
// single worker. A pair known by its blob ids is looked up without a read,
// so only pairs on their way to the model (or with no ids) count.
const maxVisualFetched = 20

// lfsPointer starts the text the raw endpoint serves for a file kept in Git
// LFS: the pointer, not the image.
var lfsPointer = []byte("version https://git-lfs.github.com/spec/")

// visualPlan is one review's screenshots: the lines the summary shows and the
// images the call carries. The images are the one exception to the plan
// carrying no file content: they are part of the call, as the prompt is.
type visualPlan struct {
	lines []render.VisualLine
	// unread: the PR's changes could not be fetched, so its screenshots
	// are unknown this run.
	unread bool
	images []inference.VisualImage
	// tokens is the images' estimate plus their labels', added to the fit
	// check.
	tokens int
	// pending is each sent id's line and cache key, for post.
	pending map[string]visualPending
	// same are pairs found unchanged, stored by post: prepare runs before a
	// first review's PR row exists, and an insert then matches no row.
	same []store.VisualPair
}

type visualPending struct {
	line int
	pair store.VisualPair
}

// screenshotsOf is the PR's changed test screenshots, from its changes: the
// type says added, deleted or moved, and the blob ids key the cache.
func screenshotsOf(cs bitbucket.ChangeSet) []diff.Screenshot {
	var out []diff.Screenshot
	for _, ch := range cs.Changes {
		if !diff.IsScreenshot(ch.Path) {
			continue
		}
		s := diff.Screenshot{Path: ch.Path, OldPath: ch.Path, BeforeID: ch.FromContentID, AfterID: ch.ContentID}
		switch strings.ToUpper(ch.Type) {
		case "ADD":
			s.Change, s.BeforeID = diff.ScreenshotAdded, ""
		case "DELETE":
			s.Change = diff.ScreenshotDeleted
		case "MOVE", "COPY":
			if ch.SrcPath != "" {
				s.OldPath = ch.SrcPath
			}
		}
		out = append(out, s)
	}
	slices.SortFunc(out, func(a, b diff.Screenshot) int { return strings.Compare(a.Path, b.Path) })
	return out
}

// prepareVisual reads the PR's changed screenshots and decides, for each,
// what the summary says: a stored line, an image to send, or why not.
//
// The list and the base commit come from the PR's changes, every run, so
// the section stays the PR's and "before" is the PR's base, never the
// target's tip, which may carry another PR's baseline: without a base, a
// pair that needs one is not compared. "after" is head, the commit whose
// code this run reviews; the PR moved on since means its blob ids are not
// head's, and the bytes key the store instead. room is what the fit
// ceiling leaves after the prompt.
func (r *Reviewer) prepareVisual(ctx context.Context, key store.PRKey, prTag, head string, room int) visualPlan {
	var vp visualPlan
	cs, err := r.bitbucket.FetchPRChangeSet(ctx, key.Project, key.Repo, key.PRID)
	if err != nil {
		r.log.WarnContext(ctx, fmt.Sprintf("%s: screenshots not read: the PR's changes did not fetch: %v", prTag, err))
		vp.unread = true
		return vp
	}
	shots := screenshotsOf(cs)
	if len(shots) == 0 {
		return vp
	}
	if cs.FromHash == "" {
		r.log.WarnContext(ctx, prTag+": the PR's changes name no base commit: a screenshot needing a before side is not compared")
	}
	if cs.ToHash != "" && cs.ToHash != head {
		r.log.InfoContext(ctx, fmt.Sprintf("%s: the PR moved on to %s since %s: screenshots keyed by their bytes",
			prTag, shortSHA(cs.ToHash, 10), shortSHA(head, 10)))
		for i := range shots {
			shots[i].BeforeID, shots[i].AfterID = "", ""
		}
	}
	before, after := cs.FromHash, head
	// An unreadable cache reads as empty: its pairs are sent again, which
	// costs tokens (safeDB logs the fault) but never loses a line.
	cache := safeDB(ctx, r.log, "VisualChanges", func() (map[store.VisualPair]string, error) {
		return r.store.VisualChanges(ctx, key)
	})
	vs := &visualRun{r: r, ctx: ctx, key: key, prTag: prTag, before: before, after: after,
		room: room, cache: cache, vp: &vp}
	vp.pending = map[string]visualPending{}
	notCompared := map[render.VisualStatus][]string{}
	unchanged := 0
	var removed []string

	// One line per group of browser and platform variants: the first that
	// changed, chromium first. A variant whose pixels did not change, or
	// that is gone, hands over to the next, so a change only firefox shows
	// still reaches the model. A removed baseline gets no line (user,
	// 2026-10-28: a list of "removed" says nothing a reviewer needs); the
	// log names it.
	for _, group := range diff.GroupVariants(shots) {
		for i, s := range group {
			line, out := vs.one(s)
			if out == visualRemoved {
				removed = append(removed, s.Path)
			}
			if out != visualShown && i < len(group)-1 {
				continue
			}
			switch out {
			case visualUnchanged:
				unchanged++
			case visualRemoved:
			default:
				line.Variants = len(group) - 1
				if line.Status != render.VisualDescribed && line.Status != render.VisualNoAnswer {
					notCompared[line.Status] = append(notCompared[line.Status], s.Path)
				}
				if line.Status == render.VisualNoAnswer {
					vp.pending[vs.lastID] = visualPending{line: len(vp.lines), pair: vs.lastPair}
				}
				vp.lines = append(vp.lines, line)
			}
			break
		}
	}

	r.log.InfoContext(ctx, fmt.Sprintf("%s: screenshots: %d changed, %d shown, %d sent (~%d tokens), %d from earlier runs, %d unchanged pixels, %d removed (not shown), %d read(s)",
		prTag, len(shots), len(vp.lines), len(vp.images), vp.tokens, vs.cached, unchanged, len(removed), vs.fetched))
	if len(removed) > 0 {
		r.log.InfoContext(ctx, fmt.Sprintf("%s: %d screenshot(s) removed, not listed: %s", prTag, len(removed), capPaths(removed)))
	}
	vp.same = vs.same
	for _, st := range slices.Sorted(maps.Keys(notCompared)) {
		paths := notCompared[st]
		r.log.InfoContext(ctx, fmt.Sprintf("%s: %d screenshot(s) not compared (%s): %s",
			prTag, len(paths), visualReason(st), capPaths(paths)))
	}
	return vp
}

// visualRun is one prepareVisual's state across its screenshots.
type visualRun struct {
	r             *Reviewer
	ctx           context.Context
	key           store.PRKey
	prTag         string
	before, after string
	room          int
	cache         map[store.VisualPair]string
	vp            *visualPlan
	cached        int
	fetched       int
	// lastID and lastPair are the image one just queued, for the caller to
	// file under its line.
	lastID   string
	lastPair store.VisualPair
	// same are the pairs found unchanged, for post to store.
	same []store.VisualPair
}

// visualOutcome is what one screenshot comes to: a line, or nothing to show
// because its pixels did not change or it was removed.
type visualOutcome int

const (
	visualShown visualOutcome = iota
	visualUnchanged
	visualRemoved
)

// one decides a single screenshot: its line, or why it shows none.
func (vs *visualRun) one(s diff.Screenshot) (line render.VisualLine, out visualOutcome) {
	r, ctx, vp := vs.r, vs.ctx, vs.vp
	line = render.VisualLine{Path: s.Path}
	done := func(st render.VisualStatus) (render.VisualLine, visualOutcome) {
		line.Status = st
		return line, visualShown
	}
	if s.Change == diff.ScreenshotDeleted {
		return line, visualRemoved
	}
	// Before any read: a model without vision never wrote a stored line,
	// so the reads would buy nothing. A team that moves off a vision model
	// loses its stored lines with it.
	if !r.llm.Vision() {
		return done(render.VisualNoVision)
	}
	// Known by its blob ids: the cache answers without a read.
	pair := store.VisualPair{Path: s.Path, BeforeID: s.BeforeID, AfterID: s.AfterID}
	known := s.AfterID != "" && (s.Change == diff.ScreenshotAdded || s.BeforeID != "")
	lookup := func() (render.VisualLine, visualOutcome, bool) {
		text, ok := vs.cache[pair]
		switch {
		case !ok:
			return line, visualShown, false
		case text == "":
			return line, visualUnchanged, true
		}
		vs.cached++
		line.Text = text
		l, out := done(render.VisualDescribed)
		return l, out, true
	}
	if known {
		if l, out, hit := lookup(); hit {
			return l, out
		}
	}
	// No base, no before side: never the target's tip in its place.
	if s.Change != diff.ScreenshotAdded && vs.before == "" {
		return done(render.VisualUnreadable)
	}
	// Past either cap the pair waits for a later run, which finds the ones
	// this run sent in the cache.
	if len(vp.images) == inference.MaxVisualImages || vs.fetched >= maxVisualFetched {
		return done(render.VisualOverCap)
	}
	afterBytes, st, missing := r.fetchScreenshot(ctx, vs.key, vs.prTag, vs.after, s.Path)
	vs.fetched++
	switch {
	case missing:
		return line, visualRemoved
	case st != render.VisualDescribed:
		return done(st)
	}
	var beforeBytes []byte
	if s.Change != diff.ScreenshotAdded {
		beforeBytes, st, missing = r.fetchScreenshot(ctx, vs.key, vs.prTag, vs.before, s.OldPath)
		vs.fetched++
		if !missing && st != render.VisualDescribed {
			return done(st)
		}
	}
	if bytes.HasPrefix(afterBytes, lfsPointer) || bytes.HasPrefix(beforeBytes, lfsPointer) {
		return done(render.VisualLFS)
	}
	if !known {
		pair.BeforeID, pair.AfterID = contentID(beforeBytes), contentID(afterBytes)
		if l, out, hit := lookup(); hit {
			return l, out
		}
	}
	res, err := visual.Compare(beforeBytes, afterBytes)
	switch {
	case errors.Is(err, visual.ErrTooLarge):
		return done(render.VisualTooLarge)
	case err != nil:
		r.log.WarnContext(ctx, fmt.Sprintf("%s: screenshot %s not compared: %v", vs.prTag, s.Path, err))
		return done(render.VisualUnreadable)
	case res.Identical:
		// Re-encoded, same pixels: not a visual change. Stored as such by
		// post, once a first review's PR row exists, so the pair is never
		// read again.
		vs.same = append(vs.same, pair)
		return line, visualUnchanged
	}
	img, cost := inference.NewVisualImage(len(vp.images), s.Path, res, r.tokens.Count)
	if cost > vs.room-vp.tokens {
		return done(render.VisualNoRoom)
	}
	vp.images = append(vp.images, img)
	vp.tokens += cost
	vs.lastID, vs.lastPair = img.ID, pair
	// Until the model answers for it.
	return done(render.VisualNoAnswer)
}

// contentID keys a pair whose blob ids Bitbucket did not give; "" for nil.
func contentID(b []byte) string {
	if b == nil {
		return ""
	}
	return "sha256:" + visual.Hash(b)
}

// fetchScreenshot reads one version. The status is VisualDescribed when the
// bytes came back, else why not; missing is a 404, the file not at that
// commit, which is an answer rather than a fault and is not logged.
func (r *Reviewer) fetchScreenshot(ctx context.Context, key store.PRKey, prTag, commit, path string) (b []byte, st render.VisualStatus, missing bool) {
	b, err := r.bitbucket.FetchFileBytes(ctx, key.Project, key.Repo, commit, path)
	var tooLarge *bitbucket.ContentTooLarge
	var status *bitbucket.StatusError
	switch {
	case errors.As(err, &tooLarge):
		return nil, render.VisualTooLarge, false
	case errors.As(err, &status) && status.Status == http.StatusNotFound:
		return nil, render.VisualUnreadable, true
	case err != nil:
		r.log.WarnContext(ctx, fmt.Sprintf("%s: screenshot %s at %s not fetched: %v", prTag, path, shortSHA(commit, 10), err))
		return nil, render.VisualUnreadable, false
	}
	return b, render.VisualDescribed, false
}

// visualReason names a status for the log.
func visualReason(st render.VisualStatus) string {
	switch st {
	case render.VisualNoVision:
		return "the model's profile has no vision"
	case render.VisualOverCap:
		return fmt.Sprintf("over %d sent or %d reads per review", inference.MaxVisualImages, maxVisualFetched)
	case render.VisualNoRoom:
		return "no room under the fit ceiling"
	case render.VisualTooLarge:
		return "image too large"
	case render.VisualUnreadable:
		return "unreadable"
	case render.VisualLFS:
		return "stored in Git LFS, the raw endpoint serves the pointer"
	}
	return "no answer"
}

// visualRoom is what the fit ceiling leaves for images after the prompt;
// unbounded before the window is known, as the call's own check is.
func (r *Reviewer) visualRoom(promptTokens int) int {
	if !r.llm.Ready() {
		return math.MaxInt
	}
	return r.llm.FitCeiling() - promptTokens
}

// visualLines fills in what the model said for the pairs this run sent and
// stores each, so a later run reuses it; a pair the model saw no difference
// in loses its line. An id the run did not send, and a sent one the model
// skipped, are logged.
func (r *Reviewer) visualLines(ctx context.Context, plan *reviewPlan, answers []inference.VisualChange) []render.VisualLine {
	vp := plan.visual
	for _, p := range vp.same {
		safeDBErr(ctx, r.log, "InsertVisualChange", func() error {
			return r.store.InsertVisualChange(ctx, plan.key, p, "", r.llm.Model())
		})
	}
	lines := append([]render.VisualLine(nil), vp.lines...)
	answered := map[string]bool{}
	var unknown []string
	// hidden are the lines of pairs the model saw no difference in (user,
	// 2026-10-30: noise). Stored as "", like unchanged pixels, so a later
	// run neither shows nor re-sends them.
	hidden := map[int]bool{}
	var invisible []string
	for _, a := range answers {
		p, ok := vp.pending[a.ID]
		if !ok || answered[a.ID] {
			unknown = append(unknown, a.ID)
			continue
		}
		answered[a.ID] = true
		text := a.Change
		if a.Invisible {
			text = ""
			hidden[p.line] = true
			invisible = append(invisible, p.pair.Path)
		} else {
			lines[p.line].Status, lines[p.line].Text = render.VisualDescribed, a.Change
		}
		safeDBErr(ctx, r.log, "InsertVisualChange", func() error {
			return r.store.InsertVisualChange(ctx, plan.key, p.pair, text, r.llm.Model())
		})
	}
	if len(invisible) > 0 {
		slices.Sort(invisible)
		r.log.InfoContext(ctx, fmt.Sprintf("%s: %d screenshot(s) with no visible change, not listed: %s",
			plan.prTag, len(invisible), capPaths(invisible)))
		kept := lines[:0]
		for i, l := range lines {
			if !hidden[i] {
				kept = append(kept, l)
			}
		}
		lines = kept
	}
	if len(unknown) > 0 {
		r.log.InfoContext(ctx, fmt.Sprintf("%s: visual_changes for unknown or repeated id(s) dropped: %s",
			plan.prTag, strings.Join(unknown, ", ")))
	}
	var skipped []string
	for id, p := range vp.pending {
		if !answered[id] {
			skipped = append(skipped, p.pair.Path)
		}
	}
	slices.Sort(skipped)
	if len(skipped) > 0 {
		r.log.InfoContext(ctx, fmt.Sprintf("%s: %d screenshot(s) sent and not described by the model: %s",
			plan.prTag, len(skipped), capPaths(skipped)))
	}
	return lines
}
