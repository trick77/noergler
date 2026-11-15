package review

import (
	"context"

	"github.com/trick77/noergler/internal/lang"
	"github.com/trick77/noergler/internal/store"
)

// Author-only actions (user, 2026-11-15, a general rule): only the PR's
// author can have noergler change the PR, a fix or a merge-conflict
// resolution. Anyone else who asks it, with a mention or in a reply to its
// comment, gets a kind answer that only the author may, never silence.
// Decided by code: a short order and nothing else ("fix it", "bitte
// beheben"), never a question ("how would you fix this?" is the Q&A's),
// never a remark that names a change ("this change is fine"), never
// someone speaking of their own doing ("I'll fix it"). Wording it misses
// goes on as before.

var (
	// changeVerbs are the order's verb; the rest of the comment may only
	// be what it names (changeObjects) and filler.
	changeVerbs = map[string]bool{
		"fix": true, "fixe": true, "fixen": true, "behebe": true, "beheben": true, "korrigiere": true,
		"korrigieren": true, "repariere": true, "reparieren": true, "apply": true, "anwenden": true,
		"umsetzen": true, "übernimm": true, "übernehmen": true, "implement": true, "implementiere": true,
		"ändere": true, "ändern": true,
	}
	changeObjects = map[string]bool{
		"it": true, "this": true, "that": true, "das": true, "es": true, "dies": true, "dieses": true,
		"diesen": true, "the": true, "den": true, "finding": true, "suggestion": true, "vorschlag": true,
		"please": true, "pls": true, "bitte": true, "now": true, "jetzt": true, "noergler": true,
	}
)

// changeRequest reports whether text, the comment with the mention
// stripped, orders noergler to change the PR: "fix it", "bitte beheben",
// "apply".
func changeRequest(text string) bool {
	t := normalize(text)
	return short(t) && !conflictNot.MatchString(t) && !conflictSelf.MatchString(t) &&
		onlyOf(t, changeVerbs, nil, changeObjects)
}

// underQAAnswer reports whether parentID is a Q&A answer of noergler's,
// whose "apply it" flow answers anyone but the author itself.
func (r *Reviewer) underQAAnswer(ctx context.Context, key store.PRKey, parentID int) bool {
	if parentID == 0 {
		return false
	}
	b, read := parentBotReplyFrom(ctx, parentID)
	if !read && r.store != nil {
		b = safeDB(ctx, r.log, "BotReplyByComment", func() (*store.BotReply, error) {
			return r.store.BotReplyByComment(ctx, key, parentID)
		})
	}
	return b != nil && b.Kind == store.BotReplyAnswer
}

// authorOnlyReply is the kind answer to anyone but the PR's author who asks
// for a change.
func authorOnlyReply(l lang.Lang) string {
	return l.Pick("Gerne, aber Änderungen am PR setze ich nur um, wenn der PR-Autor darum bittet.", //nolint:misspell // German "Autor"
		"Happy to help, but I only change the PR when its author asks.")
}
