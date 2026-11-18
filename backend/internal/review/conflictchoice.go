package review

import (
	"context"
	"fmt"
	"slices"
	"strconv"
	"strings"
	"unicode"

	"github.com/trick77/noergler/internal/lang"
	"github.com/trick77/noergler/internal/merge"
	"github.com/trick77/noergler/internal/store"
)

// A conflict where both branches hold two whole versions of the same lines,
// neither carrying the other's change (two versions of one dependency), is
// the PR author's to pick, never the model's (user, 2026-11-18: the
// request used to fail whole on it, and IntelliJ leaves it to the developer
// too). The model names it in `ask`; code checks either side alone stands
// (merge.Choosable), resolves every other conflict as before, and asks
// "T or S?" instead of giving up. The answer, read by code (readPicks),
// settles the merge kept from that request: same heads, same conflicts,
// no second call, so one answer ends it. Heads moved: the question is
// cleared and the author asks for the merge again.

// conflictPick is the author's answer to the open question: per asked
// conflict, in order, whether target's side stands.
type conflictPick struct {
	choice *store.ConflictChoice
	target []bool
}

// pickFiller is what an answer may carry besides its letters.
var pickFiller = func() map[string]bool {
	m := map[string]bool{
		"nimm": true, "nehme": true, "nehmen": true, "take": true, "use": true, "pick": true,
		"version": true, "seite": true, "side": true, "die": true, "der": true, "the": true,
		"von": true, "from": true, "ich": true, "i": true, "and": true, "und": true,
	}
	for w := range consentWords {
		m[w] = true
	}
	for w := range consentFiller {
		m[w] = true
	}
	return m
}()

// readPicks reads the author's answer to a question about n conflicts:
// `T` or `S` for one, the letters in order (`T S`) or numbered (`1T 2S`,
// `1: T, 2: S`) for several. Letters only, never a version's text ("nicht
// 1.0.31" would pick it), and nothing but filler beside them; a no or a
// put-off is no answer.
func readPicks(text string, n int) ([]bool, bool) {
	t := pickText(text)
	if n < 1 || !pickShort(t, n) || conflictVeto(t) {
		return nil, false
	}
	type pick struct {
		num    int
		target bool
	}
	var picks []pick
	num := 0
	isDigits := func(s string) bool {
		return s != "" && strings.IndexFunc(s, func(r rune) bool { return !unicode.IsDigit(r) }) < 0
	}
	for _, w := range pickWords(t) {
		last := w[len(w)-1]
		prefix := w[:len(w)-1]
		switch {
		case isDigits(w):
			if num != 0 {
				return nil, false
			}
			num, _ = strconv.Atoi(w)
		case (last == 't' || last == 's') && (prefix == "" || isDigits(prefix)):
			p := pick{num: num, target: last == 't'}
			if prefix != "" {
				if num != 0 {
					return nil, false
				}
				p.num, _ = strconv.Atoi(prefix)
			}
			picks = append(picks, p)
			num = 0
		case pickFiller[w] && num == 0:
		default:
			return nil, false
		}
	}
	if num != 0 || len(picks) != n {
		return nil, false
	}
	out := make([]bool, n)
	numbered := 0
	for _, p := range picks {
		if p.num != 0 {
			numbered++
		}
	}
	switch numbered {
	case 0:
		for i, p := range picks {
			out[i] = p.target
		}
	case n:
		seen := make([]bool, n)
		for _, p := range picks {
			if p.num < 1 || p.num > n || seen[p.num-1] {
				return nil, false
			}
			seen[p.num-1] = true
			out[p.num-1] = p.target
		}
	default:
		return nil, false
	}
	return out, true
}

// pickText is a reply as readPicks reads it: code spans kept ("`T`" is an
// answer, unlike normalize's placeholder), markup gone.
func pickText(text string) string {
	return strings.NewReplacer("`", " ", "*", " ").Replace(asciiQuotes(text))
}

// pickWords are t's words, lower case. An apostrophe stays inside its word:
// "it's T" is no "s" and a "t".
func pickWords(t string) []string {
	return strings.FieldsFunc(strings.ToLower(t), func(r rune) bool {
		return !unicode.IsLetter(r) && !unicode.IsDigit(r) && r != '\''
	})
}

// pickShort is short for an answer about n conflicts: no question, and a
// word budget that grows with n ("1: T, 2: S, ..." takes two a spot).
func pickShort(t string, n int) bool {
	return !strings.Contains(t, "?") && len(strings.Fields(t)) <= stopCommandWords+2*n
}

// pickNudge reports a reply under the question about n conflicts that is
// no answer but reads like one: a go-ahead naming no side ("Lets go!",
// "ja"), or letters of the wrong count ("T" for two). Asked again for the
// letters. Anything else (a command, a question, a no) goes on as any
// reply.
func pickNudge(text string, n int) bool {
	t := pickText(text)
	ws := pickWords(t)
	if !pickShort(t, n) || conflictVeto(t) || len(ws) == 0 {
		return false
	}
	for _, w := range ws {
		side := strings.TrimLeftFunc(w, unicode.IsDigit)
		if !pickFiller[w] && w != "lets" && w != "let's" && side != "" && side != "t" && side != "s" {
			return false
		}
	}
	return true
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

// pickMoved answers a pick whose merge is gone: a branch moved, or the
// conflicts are no longer the ones asked about.
func pickMoved(l lang.Lang) string {
	return l.Pick("Seit meiner Frage hat sich der Stand geändert. Bitte frag nochmals nach dem Merge-Konflikt.",
		"Things changed since my question. Please ask for the merge conflict again.")
}

// pickHint asks for the answer, for n conflicts.
func pickHint(l lang.Lang, n int) string {
	if n == 1 {
		return l.Pick("Antworte hier mit **T** oder **S**.", "Reply **T** or **S** here.")
	}
	return l.Pick("Antworte hier der Reihe nach, z. B. `1T 2S`.", "Reply here in order, e.g. `1T 2S`.")
}

// askPick posts the question for the conflicts the model left to the
// author (choose, in j.pending's order) and keeps the merge with it. The
// request's fix row is written when the answer settles it, under the
// request's comment; one never answered keeps its call row only.
func (r *Reviewer) askPick(ctx context.Context, j *conflictJob, choose []bool) {
	pick := j.lang.Pick
	asked := map[*merge.Conflict]bool{}
	// side sets every asked conflict to one whole side, by pick: target's
	// or not.
	side := func(target func(c *merge.Conflict) bool) {
		for i, pc := range j.pending {
			if choose[i] {
				c := pc.file.conflicts[pc.k]
				pc.file.resolutions[pc.k] = merge.Whole(c, target(c))
			}
		}
	}
	for i, pc := range j.pending {
		if choose[i] {
			asked[pc.file.conflicts[pc.k]] = true
			pc.file.byModel[pc.k], pc.file.byAuthor[pc.k] = false, true
		}
	}
	// A question the answer could not settle is never asked: either pick
	// must assemble, and its confirmation quote every dropped line (the
	// worst case: each spot keeping its shorter side).
	for _, target := range []bool{true, false} {
		side(func(*merge.Conflict) bool { return target })
		for _, f := range j.merged {
			if _, err := merge.Assemble(f.chunks, f.resolutions); err != nil {
				r.log.InfoContext(ctx, fmt.Sprintf("%s: merge conflict resolution dropped before asking: %s: %v", j.prTag, f.path, err))
				r.finishConflict(ctx, j, store.FixRejected, "", conflictUnsafe(j.lang))
				return
			}
		}
	}
	side(func(c *merge.Conflict) bool {
		return len(strings.Join(c.Source, "")) >= len(strings.Join(c.Target, ""))
	})
	if !r.answerFits(ctx, j) {
		return
	}
	// The PR's version, for the line numbers the question shows.
	side(func(*merge.Conflict) bool { return false })
	choice := &store.ConflictChoice{RequestID: j.commentID, SourceHead: j.sourceHead.String(), TargetHead: j.targetHead.String()}
	var b strings.Builder
	n := len(asked)
	if n == 1 {
		b.WriteString(pick("Den Merge-Konflikt löse ich auf, bis auf eine Stelle: dort haben beide Branches eine eigene Version, und welche gilt, entscheidest du.",
			"I'll resolve the merge conflict except for one spot: both branches have their own version there, and you decide which one stands."))
	} else {
		fmt.Fprintf(&b, pick("Den Merge-Konflikt löse ich auf, bis auf %d Stellen: dort haben beide Branches eine eigene Version, und welche gilt, entscheidest du.",
			"I'll resolve the merge conflict except for %d spots: both branches have their own version there, and you decide which one stands."), n)
	}
	b.WriteString("\n")
	q := 0
	for _, f := range j.merged {
		for k, c := range f.conflicts {
			cc := store.ChoiceConflict{Path: f.path, Target: merge.TrimEOLs(c.Target), Source: merge.TrimEOLs(c.Source)}
			switch {
			case asked[c]:
				cc.How = store.ChoiceAsk
				q++
				where := fmt.Sprintf("`%s:%d`", f.path, merge.LineOf(f.chunks, f.resolutions, k))
				if n > 1 {
					where = fmt.Sprintf("**%d.** %s", q, where)
				}
				fmt.Fprintf(&b, "\n%s\n", where)
				for _, side := range []struct {
					id, branch string
					lines      []string
				}{{"T", j.target, cc.Target}, {"S", j.source, cc.Source}} {
					body := strings.Join(side.lines, "\n")
					fence := lang.Fence(body)
					fmt.Fprintf(&b, "\n**%s** %s `%s`:\n\n%s\n%s\n%s\n", side.id, pick("aus", "from"), side.branch, fence, body, fence)
				}
			case f.byModel[k]:
				cc.How = store.ChoiceModel
			default:
				cc.How = store.ChoiceRule
			}
			if cc.How != store.ChoiceAsk {
				for _, ref := range f.resolutions[k] {
					cc.Refs = append(cc.Refs, ref.String())
				}
			}
			choice.Conflicts = append(choice.Conflicts, cc)
		}
	}
	b.WriteString("\n" + pickHint(j.lang, n))
	text := b.String()
	if size := len([]rune(text)); size > conflictAnswerRunes {
		r.log.InfoContext(ctx, fmt.Sprintf("%s: merge conflict not resolved: question of %d runes, over %d", j.prTag, size, conflictAnswerRunes))
		r.finishConflict(ctx, j, store.FixNotFeasible, "", conflictCannot(j.lang, pick(
			"die Versionen, zwischen denen du wählen müsstest, sind zu lang, um sie hier zu zeigen.",
			"the versions you would choose between are too long to show here.")))
		return
	}
	id, ok := r.postReply(ctx, j.key.Project, j.key.Repo, j.key.PRID, j.replyTo, text)
	if !ok {
		// Nothing to answer: an older question is not this merge either.
		r.log.WarnContext(ctx, j.prTag+": merge conflict: the question to the author failed to post, not resolved")
		safeDBErr(ctx, r.log, "SetConflictChoice", func() error { return r.store.SetConflictChoice(ctx, j.key, nil) })
		return
	}
	// A plain reply under it is addressed to the bot, as under any notice.
	r.recordBotReply(ctx, j.key, id, store.BotReplyQuestion)
	choice.CommentID = id
	safeDBErr(ctx, r.log, "SetConflictChoice", func() error { return r.store.SetConflictChoice(ctx, j.key, choice) })
	r.log.InfoContext(ctx, fmt.Sprintf("%s: merge conflict: asked the author to pick %d of %d conflicts (comment %d)", j.prTag, n, len(choice.Conflicts), id))
}

// bindPick settles j's merge from the open question and the author's
// answer: the same heads and, in order, the same conflicts as when asked,
// each resolved as then, the asked ones by the answer. why is log
// vocabulary, empty when it binds.
func (j *conflictJob) bindPick() (why string) {
	c, picks := j.pick.choice, j.pick.target
	if j.sourceHead.String() != c.SourceHead || j.targetHead.String() != c.TargetHead {
		return "a branch moved"
	}
	i, a := 0, 0
	for _, f := range j.merged {
		for k, cf := range f.conflicts {
			if i == len(c.Conflicts) {
				return "more conflicts than asked about"
			}
			cc := c.Conflicts[i]
			i++
			if cc.Path != f.path || !slices.Equal(merge.TrimEOLs(cf.Target), cc.Target) || !slices.Equal(merge.TrimEOLs(cf.Source), cc.Source) {
				return fmt.Sprintf("conflict %d changed", i)
			}
			f.byModel[k], f.byAuthor[k] = false, false
			if cc.How == store.ChoiceAsk {
				if a == len(picks) {
					return "more asked than answered"
				}
				f.resolutions[k] = merge.Whole(cf, picks[a])
				f.byAuthor[k] = true
				a++
				continue
			}
			refs := make([]merge.Ref, 0, len(cc.Refs))
			for _, id := range cc.Refs {
				ref, ok := merge.ParseRef(id)
				if !ok {
					return fmt.Sprintf("conflict %d: bad ref %q", i, id)
				}
				refs = append(refs, ref)
			}
			if why := merge.ValidateResolution(cf, refs); why != "" {
				return fmt.Sprintf("conflict %d: %s", i, why)
			}
			f.resolutions[k] = refs
			f.byModel[k] = cc.How == store.ChoiceModel
		}
	}
	switch {
	case i != len(c.Conflicts):
		return "fewer conflicts than asked about"
	case a != len(picks):
		return "more answered than asked"
	}
	j.pending = nil
	return ""
}
