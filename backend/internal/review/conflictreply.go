package review

import (
	"context"
	"encoding/json"
	"fmt"
	"strconv"
	"strings"
	"unicode"

	"github.com/trick77/noergler/internal/httpstats"
	"github.com/trick77/noergler/internal/inference"
	"github.com/trick77/noergler/internal/lang"
	"github.com/trick77/noergler/internal/store"
	"github.com/trick77/noergler/internal/webhook"
)

// Replies to the merge-conflict flow, in one place. A reply is read in this
// order, the first row that matches decides:
//
//	reply                                  under         read by  then
//	the command ("behebe den Konflikt")    anywhere      code     request (conflictRequest, before routeConflictReply)
//	a short yes ("ja", "ok bitte")         the offer     code     request
//	any other wording                      the offer     Q&A      resolve_merge_conflict: request
//	the sides ("T", "1T 2S")               the question  code     settle the merge kept with it
//	reads like sides, settles none         the question  code     the author is asked again for the letters
//	any other wording, at the cost cap     the question  code     the author is asked for the letters (free)
//	any other wording                      the question  Q&A      pick_conflict_side: settle the merge
//
// Code reads first and never guesses: letters, never a version's text
// ("nicht 1.0.31" would pick it). The model reads only what code could not,
// only directly under the comment it answers, only which way, and only a
// question it sees whole (not cut as a parent is); code vetoes a no or a
// put-off in the commenter's own words (conflictVeto) and sides that do
// not fit the question. Whoever read it, handleConflict checks the author
// and answers by itself; the model's sentence is never posted. Under the
// question, an answer of the author's that settled nothing ends with
// the letters' hint, whatever the model did.

// conflictRoute is where routeConflictReply left a reply: done when code
// handled it (staged as handleConflict reports), else the Q&A goes on,
// with the tool its parent allows.
type conflictRoute struct {
	done, staged bool
	underOffer   bool
	pickOpen     *store.ConflictChoice
	pickTool     bool
}

// routeConflictReply applies the code rows of the table to a mention or a
// reply whose parent may be the open offer or the open question. state is
// the PR's skip state, read once by the caller.
func (r *Reviewer) routeConflictReply(ctx context.Context, payload *webhook.Payload, key store.PRKey, prTag, team string,
	sched Scheduler, counter *httpstats.Counter, state *store.SkipState, question string, engaged func(string)) conflictRoute {
	comment, pr := payload.Comment, payload.PullRequest
	if r.underOpenOffer(ctx, state, payload.CommentParentID) {
		if conflictConsent(question) {
			r.log.InfoContext(ctx, prTag+": yes under the merge conflict offer, taken as the request")
			return conflictRoute{done: true, staged: r.handleConflict(ctx, payload, key, prTag, team, sched, counter, nil)}
		}
		return conflictRoute{underOffer: true}
	}
	choice := r.openPick(ctx, key, payload.CommentParentID)
	if choice == nil {
		return conflictRoute{}
	}
	kind, target := readPickReply(question, choice.Asked())
	if kind == pickSides {
		r.log.InfoContext(ctx, prTag+": sides under the merge conflict question, read by code")
		return conflictRoute{done: true,
			staged: r.handleConflict(ctx, payload, key, prTag, team, sched, counter, &conflictPick{choice: choice, target: target})}
	}
	if strings.EqualFold(comment.Author.Name, pr.Author.User.Name) {
		why := ""
		switch kind {
		case pickNudge:
			why = "settles no side"
		default:
			// The model costs money, the letters do not: at the cap the
			// author is pointed to them, never left with the banner alone.
			if _, _, over := r.overCap(ctx, key); over {
				why = "needs the model, the PR is at its cost cap"
			}
		}
		if why != "" {
			r.log.InfoContext(ctx, prTag+": reply under the merge conflict question "+why+": asked for the letters")
			r.reply(ctx, key.Project, key.Repo, pr.ID, comment.ID, pickHint(r.languageFor(ctx, comment.Author.Name), choice.Asked()))
			engaged(store.EngagementAnswered)
			return conflictRoute{done: true}
		}
	}
	// The model picks only from a question it sees whole: a parent is cut
	// at maxParentRunes, and sides for spots it never saw are a guess.
	return conflictRoute{pickOpen: choice, pickTool: choice.QuestionRunes > 0 && choice.QuestionRunes <= maxParentRunes}
}

// openPick is the PR's open question when parentID is its comment: a reply
// directly under it is an answer to it. nil otherwise, and on a failed
// read (the reply goes on as any other).
func (r *Reviewer) openPick(ctx context.Context, key store.PRKey, parentID int) *store.ConflictChoice {
	if parentID == 0 {
		return nil
	}
	// The question is a notice: under any other bot reply, read already
	// by the reply route, there is none to answer.
	if b, read := parentBotReplyFrom(ctx, parentID); read && (b == nil || b.Kind != store.BotReplyQuestion) {
		return nil
	}
	c := safeDB(ctx, r.log, "ConflictChoice", func() (*store.ConflictChoice, error) { return r.store.ConflictChoice(ctx, key) })
	if c == nil || c.CommentID != parentID || c.Asked() == 0 {
		return nil
	}
	return c
}

// pickRead is what code reads in a reply under the question.
type pickRead int

const (
	// pickOther: words beyond sides and filler, a question, a no: the
	// Q&A's to read.
	pickOther pickRead = iota
	// pickSides: one side per asked spot.
	pickSides
	// pickNudge: sides, numbers and filler only, yet no answer: a
	// go-ahead naming none ("Lets go!"), the wrong count ("T" for two),
	// a number twice. Numbers with no side at all are a version, other.
	pickNudge
)

// pickFiller is what a reply may carry besides its sides.
var pickFiller = func() map[string]bool {
	m := map[string]bool{
		"nimm": true, "nehme": true, "nehmen": true, "take": true, "use": true, "pick": true,
		"version": true, "seite": true, "side": true, "die": true, "der": true, "the": true,
		"von": true, "from": true, "ich": true, "i": true, "and": true, "und": true,
		"lets": true, "let's": true,
	}
	for w := range consentWords {
		m[w] = true
	}
	for w := range consentFiller {
		m[w] = true
	}
	return m
}()

// readPickReply reads a reply under a question about n spots, in one pass:
// `T` or `S` for one, the letters in order (`T S`) or numbered (`1T 2S`,
// `1: T, 2: S`) for several, nothing beside them but filler. target is,
// per spot, whether target's side stands (pickSides only).
func readPickReply(text string, n int) (kind pickRead, target []bool) {
	t := pickText(text)
	words := pickWords(t)
	if n < 1 || len(words) == 0 || strings.Contains(t, "?") || len(strings.Fields(t)) > stopCommandWords+2*n || conflictVeto(t) {
		return pickOther, nil
	}
	type side struct {
		num    int
		target bool
	}
	var sides []side
	num, settled, numbers := 0, true, false
	for _, w := range words {
		if pickFiller[w] {
			settled = settled && num == 0
			continue
		}
		digits := strings.TrimRightFunc(w, unicode.IsLetter)
		rest := w[len(digits):]
		// A version ("1.0.31"), an apostrophe's word ("it's"), anything
		// mixed: the model's to read.
		if strings.IndexFunc(digits, func(r rune) bool { return !unicode.IsDigit(r) }) >= 0 {
			return pickOther, nil
		}
		switch {
		case rest == "" && digits != "":
			// A number, for the side after it.
			settled = settled && num == 0
			num, _ = strconv.Atoi(digits)
			numbers = true
		case rest == "t" || rest == "s":
			s := side{num: num, target: rest == "t"}
			if digits != "" {
				settled = settled && num == 0
				s.num, _ = strconv.Atoi(digits)
			}
			sides = append(sides, s)
			num = 0
		default:
			return pickOther, nil
		}
	}
	switch {
	case len(sides) == 0 && numbers:
		// Numbers and no side: a version ("nimm 1.0.31"), the model's to
		// read, never code's.
		return pickOther, nil
	case !settled || num != 0 || len(sides) != n:
		return pickNudge, nil
	}
	target = make([]bool, n)
	numbered := 0
	for _, s := range sides {
		if s.num != 0 {
			numbered++
		}
	}
	switch numbered {
	case 0:
		for i, s := range sides {
			target[i] = s.target
		}
	case n:
		seen := make([]bool, n)
		for _, s := range sides {
			if s.num < 1 || s.num > n || seen[s.num-1] {
				return pickNudge, nil
			}
			seen[s.num-1] = true
			target[s.num-1] = s.target
		}
	default:
		return pickNudge, nil
	}
	return pickSides, target
}

// pickText is a reply as readPickReply reads it: code spans kept ("`T`" is
// an answer, unlike normalize's placeholder), markup gone.
func pickText(text string) string {
	return strings.NewReplacer("`", " ", "*", " ").Replace(asciiQuotes(text))
}

// pickWords are t's words, lower case. An apostrophe or a dot stays inside
// its word: "it's T" is no "s" and a "t", "1.0.31" no three numbers; one
// that ends a word is punctuation ("T.", "1.").
func pickWords(t string) []string {
	var out []string
	for _, w := range strings.FieldsFunc(strings.ToLower(t), func(r rune) bool {
		return !unicode.IsLetter(r) && !unicode.IsDigit(r) && r != '\'' && r != '.'
	}) {
		if w = strings.Trim(w, ".'"); w != "" {
			out = append(out, w)
		}
	}
	return out
}

// resolveConflictTool is the model reading a reply as a yes to the offer
// and code deciding it: no veto in the commenter's own text. The posting
// stage starts the request.
func (r *Reviewer) resolveConflictTool(ctx context.Context, ask *mentionAsk) map[string]string {
	if conflictVeto(ask.text) {
		r.log.WarnContext(ctx, fmt.Sprintf("%s: merge conflict resolution for %s rejected: the reply turns the offer down or puts it off",
			ask.prTag, ask.user))
		ask.consentRefused = true
		return map[string]string{"result": "rejected: the user's reply does not agree to the offer"}
	}
	ask.resolveConflict = true
	return map[string]string{"result": "ok", "note": "noergler starts the resolution once this turn ends and answers " +
		"in its own reply; your answer is not posted, keep it short"}
}

// pickConflictSide is the model reading which version the reply keeps and
// code deciding it may stand: no veto in the commenter's own text, one
// side per asked spot. The posting stage settles the merge.
func (r *Reviewer) pickConflictSide(ctx context.Context, ask *mentionAsk, args string) map[string]string {
	rejected := func(why string) map[string]string {
		r.log.WarnContext(ctx, fmt.Sprintf("%s: merge conflict pick for %s rejected: %s", ask.prTag, ask.user, why))
		return map[string]string{"result": "rejected: " + why}
	}
	var in struct {
		Sides []string `json:"sides"`
	}
	if err := json.Unmarshal([]byte(args), &in); err != nil {
		return rejected(`arguments are not {"sides": ["T" or "S", ...]}`)
	}
	if conflictVeto(ask.text) {
		return rejected("the user's reply turns it down or puts it off")
	}
	n := ask.pickOpen.Asked()
	if len(in.Sides) != n {
		return rejected(fmt.Sprintf("the question asks about %d spots, %d sides given", n, len(in.Sides)))
	}
	target := make([]bool, n)
	for i, s := range in.Sides {
		switch s {
		case "T":
			target[i] = true
		case "S":
		default:
			return rejected(fmt.Sprintf("side %q is neither T nor S", s))
		}
	}
	ask.pickTarget = target
	return map[string]string{"result": "ok", "note": "noergler merges with these sides once this turn ends and " +
		"answers in its own reply; your answer is not posted, keep it short"}
}

// postConflictTools is the posting stage's part of the table, on a loop
// that ended well: a tool code let stand starts the request or settles the
// merge, after what this turn confirmed first (prefix), never with the
// model's sentence (started). Otherwise the model may promise what nobody
// does: under the offer, a model without the tool or one code vetoed adds
// what starts it; under the question the author's answer always ends with
// the letters' hint, whatever the model did or did not call.
func (r *Reviewer) postConflictTools(ctx context.Context, sched Scheduler, j *mentionJob, prefix string,
	result *inference.MentionResult, l lang.Lang) (started, staged bool) {
	ask := j.ask
	var pick *conflictPick
	switch {
	case ask.resolveConflict:
		r.log.InfoContext(ctx, ask.prTag+": the model read the reply as a yes to the merge conflict offer")
	case ask.pickTarget != nil:
		r.log.InfoContext(ctx, ask.prTag+": the model read the sides under the merge conflict question")
		pick = &conflictPick{choice: ask.pickOpen, target: ask.pickTarget}
	default:
		switch {
		case ask.underOffer && (result.ToolsOff != "" || ask.consentRefused):
			result.Answer = strings.TrimSpace(result.Answer + "\n\n" + offerHint(l, r.bitbucket.BotUsername()))
		case ask.pickOpen != nil && strings.EqualFold(ask.user, ask.prAuthor):
			result.Answer = strings.TrimSpace(result.Answer + "\n\n" + pickHint(l, ask.pickOpen.Asked()))
		}
		return false, false
	}
	if prefix != "" {
		r.reply(ctx, j.project, j.repo, j.prID, j.replyTo, strings.TrimSpace(prefix))
	}
	return true, r.handleConflict(ctx, j.payload, ask.key, ask.prTag, j.team, sched, j.counter, pick)
}
