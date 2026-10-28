package review

import (
	"bytes"
	"context"
	"errors"
	"fmt"
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

// hasScreenshots reports whether this run should look at screenshots at
// all, without a call: the diff it reviews or the whole PR's (an
// incremental run's cumulative diff) names one, or the whole PR's could not
// be read.
func hasScreenshots(rawDiff string, cumulative cumulativeContext, incremental bool) bool {
	if len(diff.Screenshots(rawDiff)) > 0 {
		return true
	}
	return incremental && (cumulative.scope == nil || len(cumulative.screenshots) > 0)
}

// prepareVisual reads the PR's changed screenshots and decides, for each,
// what the summary says: a stored line, an image to send, or why not.
//
// The list and both commits come from the PR's changes, every run, so the
// section stays the PR's and "before" is the PR's base, not the target's
// tip, which may carry another PR's baseline. target and head stand in
// when Bitbucket leaves the commits out. room is what the fit ceiling
// leaves after the prompt.
func (r *Reviewer) prepareVisual(ctx context.Context, key store.PRKey, prTag, target, head string, room int) visualPlan {
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
	before, after := cs.FromHash, cs.ToHash
	if before == "" {
		r.log.InfoContext(ctx, prTag+": the PR's changes name no base commit, screenshots compared against the target's tip")
		before = target
	}
	if after == "" {
		after = head
	}
	// An unreadable cache reads as empty: its pairs are sent again, which
	// costs tokens (safeDB logs the fault) but never loses a line.
	cache := safeDB(ctx, r.log, "VisualChanges", func() (map[store.VisualPair]string, error) {
		return r.store.VisualChanges(ctx, key)
	})
	vs := &visualRun{r: r, ctx: ctx, key: key, prTag: prTag, before: before, after: after,
		room: room, cache: cache, vp: &vp}
	vp.pending = map[string]visualPending{}
	notCompared := map[render.VisualStatus][]string{}
	var unchanged []string

	// One line per group of browser and platform variants: the first that
	// changed, chromium first. A variant whose pixels did not change hands
	// over to the next, so a change only firefox shows still reaches the
	// model.
	for _, group := range diff.GroupVariants(shots) {
		for i, s := range group {
			line, same := vs.one(s)
			if same && i < len(group)-1 {
				continue
			}
			if same {
				unchanged = append(unchanged, group[0].Path)
				break
			}
			line.Variants = len(group) - 1
			if line.Status != render.VisualDescribed && line.Status != render.VisualDeleted && line.Status != render.VisualNoAnswer {
				notCompared[line.Status] = append(notCompared[line.Status], s.Path)
			}
			if line.Status == render.VisualNoAnswer {
				vp.pending[vs.lastID] = visualPending{line: len(vp.lines), pair: vs.lastPair}
			}
			vp.lines = append(vp.lines, line)
			break
		}
	}

	r.log.InfoContext(ctx, fmt.Sprintf("%s: screenshots: %d changed, %d shown, %d sent (~%d tokens), %d from earlier runs, %d unchanged pixels, %d read(s)",
		prTag, len(shots), len(vp.lines), len(vp.images), vp.tokens, vs.cached, len(unchanged), vs.fetched))
	for _, st := range slices.Sorted(mapKeys(notCompared)) {
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
}

// one decides a single screenshot: its line, or same when its pixels did
// not change (known from the cache or found now), which shows nothing.
func (vs *visualRun) one(s diff.Screenshot) (line render.VisualLine, same bool) {
	r, ctx, vp := vs.r, vs.ctx, vs.vp
	line = render.VisualLine{Path: s.Path}
	done := func(st render.VisualStatus) (render.VisualLine, bool) {
		line.Status = st
		return line, false
	}
	if s.Change == diff.ScreenshotDeleted {
		return done(render.VisualDeleted)
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
	lookup := func() (render.VisualLine, bool, bool) {
		text, ok := vs.cache[pair]
		switch {
		case !ok:
			return line, false, false
		case text == "":
			return line, true, true
		}
		vs.cached++
		line.Text = text
		l, _ := done(render.VisualDescribed)
		return l, false, true
	}
	if known {
		if l, same, hit := lookup(); hit {
			return l, same
		}
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
		return done(render.VisualDeleted)
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
		if l, same, hit := lookup(); hit {
			return l, same
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
		// Re-encoded, same pixels: not a visual change, and stored as such
		// so the pair is never read again.
		safeDBErr(ctx, r.log, "InsertVisualChange", func() error {
			return r.store.InsertVisualChange(ctx, vs.key, pair, "", r.llm.Model())
		})
		return line, true
	}
	id := fmt.Sprintf("V%d", len(vp.images)+1)
	img := inference.VisualImage{ID: id, Path: s.Path, After: visual.DataURI(res.After), Cropped: res.Cropped}
	if res.Before != nil {
		img.Before = visual.DataURI(res.Before)
	}
	// The text around the images is billed and fitted as well.
	cost := res.Tokens + r.tokens.Count(inference.VisualLabel(img))
	if len(vp.images) == 0 {
		cost += r.tokens.Count(inference.VisualIntro)
	}
	if cost > vs.room-vp.tokens {
		return done(render.VisualNoRoom)
	}
	vp.images = append(vp.images, img)
	vp.tokens += cost
	vs.lastID, vs.lastPair = id, pair
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

func mapKeys[K comparable, V any](m map[K]V) func(func(K) bool) {
	return func(yield func(K) bool) {
		for k := range m {
			if !yield(k) {
				return
			}
		}
	}
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
// stores each, so a later run reuses it. An id the run did not send, and a
// sent one the model skipped, are logged.
func (r *Reviewer) visualLines(ctx context.Context, plan *reviewPlan, answers []inference.VisualChange) []render.VisualLine {
	vp := plan.visual
	lines := append([]render.VisualLine(nil), vp.lines...)
	answered := map[string]bool{}
	var unknown []string
	for _, a := range answers {
		p, ok := vp.pending[a.ID]
		if !ok || answered[a.ID] {
			unknown = append(unknown, a.ID)
			continue
		}
		answered[a.ID] = true
		lines[p.line].Status, lines[p.line].Text = render.VisualDescribed, a.Change
		safeDBErr(ctx, r.log, "InsertVisualChange", func() error {
			return r.store.InsertVisualChange(ctx, plan.key, p.pair, a.Change, r.llm.Model())
		})
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
