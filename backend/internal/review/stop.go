package review

import (
	"cmp"
	"context"
	"fmt"
	"net/http"
	"regexp"
	"strings"

	"github.com/trick77/noergler/internal/bitbucket"
	"github.com/trick77/noergler/internal/lang"
	"github.com/trick77/noergler/internal/render"
	"github.com/trick77/noergler/internal/store"
)

// A stop request tells noergler to stop reviewing one PR (user, 2026-11-04):
// no more reviews on push, so no more review cost. The summary stays, with
// a banner; questions are still answered; the merge's riptide rollup and
// Almanach send run as for any PR. Only a review or resume request brings
// the reviews back, never a question.
//
// Decided by code, never by the model, when the comment is a short command
// ("stop monitoring", "Höre auf, diesen PR zu beobachten"): free, so it
// works at the cost cap too. Wording the pattern misses reaches the Q&A,
// whose stop_reviewing tool code checks against stopIntent.

// Unicode boundaries throughout: RE2's \b is ASCII (INVARIANTS: adapters).
const (
	wb  = `(?:^|[^\pL\pN_])`
	wbe = `(?:[^\pL\pN_]|$)`
)

// The word lists, as regexp alternations.
const (
	// stopWords: an imperative to stop. Alone ("stop", "stopp bitte") it
	// is a command; code talk uses it too ("stop the goroutine leak"), so
	// anything beyond filler needs a target next to it.
	stopWords = `stop|stopp|stoppe|stoppen|pause|pausiere|pausieren|hör\pL* auf|aufhören|aufzuhören|beende|beenden`
	// noMoreWords: the other way to say it ("don't review this anymore",
	// "nicht mehr prüfen"). Code talk says it as often ("this no longer
	// checks nil"), so it always needs a target.
	noMoreWords = `no longer|no more|anymore|any more|nicht mehr|nicht länger`
	// targetWords: what is stopped, the bot's reviewing or watching, as a
	// gerund or an infinitive, never a code verb's third person ("checks",
	// "prüft"), which describes code.
	targetWords = `review|reviews|reviewing|monitor|monitoring|watch|watching|tracking|checking|` +
		`reviewen|prüfen|beobachten|überwachen|monitoren|kommentieren`
	resumeWords = `resume|continue|restart|start again|weiter|wieder|weitermachen|fortsetzen|fortfahren|wieder aufnehmen`
)

func words(alt string) *regexp.Regexp {
	return regexp.MustCompile(`(?i)` + wb + `(?:` + alt + `)` + wbe)
}

// near matches a and b, either order, at most gap words apart: "stop
// reviewing", "Reviews stoppen", "Höre auf, diesen PR zu beobachten" (3).
// Words far apart are a sentence that happens to hold both ("review again,
// fixed the stop condition").
func near(a, b string, gap int) *regexp.Regexp {
	g := fmt.Sprintf(`(?:[^\pL\pN_]+[\pL\pN_]+){0,%d}[^\pL\pN_]+`, gap)
	return regexp.MustCompile(`(?i)` + wb + `(?:(?:` + a + `)` + g + `(?:` + b + `)|(?:` + b + `)` + g + `(?:` + a + `))` + wbe)
}

var (
	stopVerb   = words(stopWords)
	noMore     = words(noMoreWords)
	resumeVerb = words(resumeWords)
	stopNear   = near(stopWords+`|`+noMoreWords, targetWords, 3)
	// resumeNear is tighter: "continue explaining the review finding" is
	// a question, and a resume costs a review.
	resumeNear = near(resumeWords, targetWords, 1)
	// explicitStop needs nothing else.
	explicitStop = words(`keine reviews mehr`)
	// keepGoing negates a stop: "don't stop reviewing", "please don't
	// pause reviews", "Reviews bitte nicht stoppen", "nicht aufhören". Not
	// a bare "keep": "stop reviewing, keep the summary" is still a stop.
	keepGoing = regexp.MustCompile(`(?i)` + wb + `(?:` +
		`(?:don't|dont|do not|never|nicht|nie)(?:[^\pL\pN_]+[\pL\pN_]+)?[^\pL\pN_]+(?:` + stopWords + `)|` +
		`keep (?:reviewing|monitoring|watching|going|checking)|hör\pL* nicht auf` +
		`)` + wbe)
	// codeSpan is code the comment quotes: `stop()`, stop(). A word in it
	// is an identifier, never an order.
	codeSpan = regexp.MustCompile("`[^`]*`|[\\pL_][\\pL\\pN_.]*\\(\\)?")
)

// commandFiller are the words a verb-only command may carry ("stop please",
// "bitte weiter"). Anything else ("weiter so", "continue explaining") makes
// it a sentence.
var commandFiller = map[string]bool{
	"please": true, "pls": true, "now": true, "it": true, "that": true, "this": true, "here": true,
	"thanks": true, "ok": true, "okay": true, "bitte": true, "jetzt": true, "das": true, "hier": true,
	"danke": true, "noergler": true,
}

// stopCommandWords is how long a comment may be and still be a stop or
// resume command decided by code. A longer one is a sentence about the PR
// more often than an order; the Q&A model reads it.
const stopCommandWords = 10

// normalize is the comment as the patterns read it: typographic
// apostrophes as ASCII (iOS and macOS type "don’t"), quoted code as a
// placeholder word, so "stop()" is no order and the word gaps still count.
func normalize(text string) string {
	t := strings.NewReplacer("’", "'", "‘", "'").Replace(strings.TrimSpace(text))
	return codeSpan.ReplaceAllString(t, " code ")
}

// short reports whether t can be a command decided by code: no question,
// at most stopCommandWords words.
func short(t string) bool {
	return !strings.Contains(t, "?") && len(strings.Fields(t)) <= stopCommandWords
}

// verbOnly reports whether t holds verb and nothing else but filler.
func verbOnly(t string, verb *regexp.Regexp) bool {
	if !verb.MatchString(t) {
		return false
	}
	for _, w := range strings.Fields(verb.ReplaceAllString(t, " ")) {
		if w = strings.ToLower(strings.Trim(w, ".,:;!-")); w != "" && !commandFiller[w] {
			return false
		}
	}
	return true
}

// stopRequest reports whether text, the comment with the mention stripped,
// is an order to stop reviewing the PR. A question is not one: "should I
// stop reviewing this?" is for the Q&A.
func stopRequest(text string) bool {
	t := normalize(text)
	return short(t) && !keepGoing.MatchString(t) &&
		(explicitStop.MatchString(t) || stopNear.MatchString(t) || verbOnly(t, stopVerb))
}

// resumeRequest reports whether text asks to take reviews up again. Read
// only on a stopped PR: elsewhere "@bot continue" is a question.
func resumeRequest(text string) bool {
	t := normalize(text)
	return short(t) && !stopVerb.MatchString(t) && !noMore.MatchString(t) &&
		(resumeNear.MatchString(t) || verbOnly(t, resumeVerb))
}

// stopIntent is the stop_reviewing tool's guard: the commenter's own text
// must ask to stop the reviewing, a stop word next to a target, at any
// length. The model reads the diff and the description, so a call either
// steered it into finds no such words and changes nothing; "why does this
// loop stop early?" names no target.
func stopIntent(text string) bool {
	t := normalize(text)
	return !keepGoing.MatchString(t) && (explicitStop.MatchString(t) || stopNear.MatchString(t))
}

// stopOutcome is what a stop did.
type stopOutcome int

const (
	// stopFailed: the state could not be written.
	stopFailed stopOutcome = iota
	// stopDone: the PR is stopped now.
	stopDone
	// stopAlready: it was stopped, or ignored after its summary was
	// deleted, before this.
	stopAlready
)

// stopReviews stops reviews on the PR: the state, then the banner on the
// summary in the PR author's language. by is the commenter. The reply is
// the caller's. Fail open like every store call: a fault stops nothing and
// says so.
//
// newRow writes the PR's row when it has none, for a PR never reviewed;
// nil when the caller knows the row exists.
func (r *Reviewer) stopReviews(ctx context.Context, key store.PRKey, prTag, by, prAuthor string, newRow func()) stopOutcome {
	if r.store == nil {
		return stopFailed
	}
	id, err := r.store.MarkStopped(ctx, key, by)
	if err == nil && id == 0 && newRow != nil {
		// No row yet, or ignored already: only the first gets a row, and
		// the second stop then matches nothing again.
		if safeDB(ctx, r.log, "GetSkipState", func() (*store.SkipState, error) {
			return r.store.GetSkipState(ctx, key)
		}) == nil {
			newRow()
			id, err = r.store.MarkStopped(ctx, key, by)
		}
	}
	if err != nil {
		r.log.WarnContext(ctx, fmt.Sprintf("%s: stop by %s not stored: %v", prTag, by, err))
		return stopFailed
	}
	if id == 0 {
		r.log.InfoContext(ctx, fmt.Sprintf("%s: stop by %s: PR already ignored", prTag, by))
		return stopAlready
	}
	r.log.InfoContext(ctx, fmt.Sprintf("%s: reviews stopped by %s - pushes are no longer reviewed", prTag, by))
	// No summary: bannerSummary posts nothing for an edit-only banner, and
	// the confirmation is the notice.
	banner := render.StoppedBanner(r.languageFor(ctx, prAuthor), by, r.bitbucket.BotUsername())
	r.bannerSummary(ctx, key.Project, key.Repo, key.PRID, id, "", func(live string) (string, bool) {
		// Below any cost banner: its stripper reads the top of the body only.
		return render.PlaceStoppedBanner(live, banner), true
	}, "stopped banner")
	return stopDone
}

// resume brings a stopped PR back and takes the banner off its summary
// before the review runs: a review that then fails puts its stale banner on
// top and would leave the stopped one standing. The fetch doubles as the
// deletion check the review guard skips on a stopped PR: a summary deleted
// meanwhile (no pr:comment:deleted hook) makes it the deleted case, so the
// review posts a fresh one instead of 404ing on the old id.
func (r *Reviewer) resume(ctx context.Context, key store.PRKey, prTag string, state *store.SkipState) {
	var live *string
	if state.Summary != nil {
		c, err := r.bitbucket.FetchPRComment(ctx, key.Project, key.Repo, key.PRID, state.Summary.ID)
		switch {
		case bitbucket.Status(err) == http.StatusNotFound:
			r.log.InfoContext(ctx, fmt.Sprintf("%s: summary comment %d of the stopped PR was deleted - resuming with a fresh one",
				prTag, state.Summary.ID))
			safeDBErr(ctx, r.log, "MarkIgnored", func() error { return r.store.MarkIgnored(ctx, key) })
		case err != nil:
			r.log.WarnContext(ctx, fmt.Sprintf("%s: summary of the stopped PR not read, banner left for the review to replace: %v", prTag, err))
		default:
			live = &c.Text
			state.Summary.Version = cmp.Or(c.Version, state.Summary.Version)
		}
	}
	if err := r.store.Reactivate(ctx, key); err != nil {
		r.log.WarnContext(ctx, fmt.Sprintf("%s: resume not stored, PR stays stopped: %v", prTag, err))
		return
	}
	r.log.InfoContext(ctx, fmt.Sprintf("%s: reviews resumed via @mention (stopped by %s)", prTag, state.StoppedBy))
	if live == nil {
		return
	}
	body := render.PlaceStoppedBanner(*live, "")
	if body == *live {
		return
	}
	v, err := r.bitbucket.UpdatePRComment(ctx, key.Project, key.Repo, key.PRID, state.Summary.ID, state.Summary.Version, body)
	if err != nil {
		r.log.WarnContext(ctx, fmt.Sprintf("%s: stopped banner not removed, the review replaces it: %v", prTag, err))
		return
	}
	r.trackSummary(ctx, state.ID, state.Summary.ID, v)
}

// stopReply is the answer to a stop request, in the asker's language.
func stopReply(l lang.Lang, o stopOutcome, bot string) string {
	switch o {
	case stopDone:
		return fmt.Sprintf(l.Pick(
			"Alles klar: Diesen PR reviewe ich nicht mehr, neue Pushes kosten nichts mehr. Fragen beantworte ich weiterhin. `@%s review` nimmt die Reviews wieder auf.",
			"Got it: I no longer review this PR, so new pushes cost nothing. Questions are still answered. `@%s review` resumes the reviews."), bot)
	case stopAlready:
		return fmt.Sprintf(l.Pick(
			"Diesen PR reviewe ich schon nicht mehr. `@%s review` nimmt die Reviews wieder auf.",
			"I already stopped reviewing this PR. `@%s review` resumes the reviews."), bot)
	}
	return l.Pick("Der Stopp konnte gerade nicht gespeichert werden. Bitte später nochmals versuchen.",
		"The stop could not be saved right now. Please try again later.")
}

// handleStop is a stop request decided by code: the state, the banner, the
// reply. No model call, no attempt row, nothing counted toward the cost.
// A PR never seen gets its row first, so the stop has somewhere to live.
func (r *Reviewer) handleStop(ctx context.Context, key store.PRKey, prTag string, commentID int, by string, author, title string, openedAt int64) {
	o := r.stopReviews(ctx, key, prTag, by, author, func() { r.ensurePRRow(ctx, key, author, title, openedAt) })
	r.reply(ctx, key.Project, key.Repo, key.PRID, commentID, stopReply(r.languageFor(ctx, by), o, r.bitbucket.BotUsername()))
}
