package review

import (
	"context"
	"fmt"
	"strings"

	"github.com/trick77/noergler/internal/lang"
	"github.com/trick77/noergler/internal/render"
)

// languageKeywords are the exact comment texts (after the @mention is
// stripped) that set the commenter's language. Exact, like reviewKeywords:
// a question that merely mentions a language is a question.
var languageKeywords = map[string]lang.Lang{
	"deutsch":           lang.DE,
	"german":            lang.DE,
	"de":                lang.DE,
	"auf deutsch":       lang.DE,
	"in german":         lang.DE,
	"answer in german":  lang.DE,
	"english":           lang.EN,
	"englisch":          lang.EN,
	"en":                lang.EN,
	"auf englisch":      lang.EN,
	"in english":        lang.EN,
	"answer in english": lang.EN,
}

// languageKeyword reports whether text is a language switch. Trailing
// punctuation is forgiven ("English!", "deutsch.").
func languageKeyword(text string) (lang.Lang, bool) {
	t := strings.ToLower(strings.TrimRight(strings.TrimSpace(text), ".!? "))
	l, ok := languageKeywords[t]
	return l, ok
}

// sharpS is render.SwissSS for German text the model wrote; English is
// left as it came.
func sharpS(l lang.Lang, s string) string {
	if l == lang.EN {
		return s
	}
	return render.SwissSS(s)
}

// languageFor is the language noergler writes to user in: their own choice,
// else the team's default. Fails open to the team default: a DB fault must
// never fail a review, and a wrong language is not worth one.
func (r *Reviewer) languageFor(ctx context.Context, user string) lang.Lang {
	def := r.cfg.Language.Or(lang.Default)
	if r.store == nil || user == "" {
		return def
	}
	return safeDB(ctx, r.log, "UserLanguage", func() (lang.Lang, error) {
		return r.store.UserLanguage(ctx, user)
	}).Or(def)
}

// setLanguage stores the commenter's language and confirms it, in that
// language, under their comment. Only ever the commenter's own row: the
// webhook names who wrote the comment, nothing in the text can name anyone
// else.
func (r *Reviewer) setLanguage(ctx context.Context, project, repo string, prID, commentID int, prTag, user string, l lang.Lang) {
	if err := r.store.SetUserLanguage(ctx, user, l); err != nil {
		r.log.WarnContext(ctx, fmt.Sprintf("%s: language for %s not stored: %v", prTag, user, err))
		r.reply(ctx, project, repo, prID, commentID, l.Pick(
			"Die Sprache konnte gerade nicht gespeichert werden. Bitte später nochmals versuchen.",
			"The language could not be saved right now. Please try again later."))
		return
	}
	r.log.InfoContext(ctx, fmt.Sprintf("%s: language set to %s for %s", prTag, l, user))
	r.reply(ctx, project, repo, prID, commentID, languageConfirmation(l))
}

// languageConfirmation is the switch's confirmation, in the new language.
func languageConfirmation(l lang.Lang) string {
	return l.Pick("Alles klar: Reviews deiner PRs und Antworten an dich schreibe ich ab jetzt auf Deutsch.",
		"Got it: from now on I write reviews of your PRs and replies to you in English.")
}
