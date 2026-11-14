package render

import (
	"regexp"

	"github.com/trick77/noergler/internal/lang"
)

// The merge-conflict offer: a comment of its own that a review posts when
// the PR has a conflict noergler could resolve (user, 2026-11-14). Its own
// comment, not a summary line: the summary is often left unread, and the
// mention notifies the author. It only offers; the author's command
// resolves.
//
// The commands are ones the conflict request recognises
// (review.conflictRequest; a test pins it).
const (
	ConflictCommandDE = "behebe den Merge-Konflikt"
	ConflictCommandEN = "resolve the merge conflict"
)

// plainName is a username Bitbucket mentions bare; any other is quoted.
var plainName = regexp.MustCompile(`^[A-Za-z0-9._-]+$`)

// Mention is how a comment mentions user, so Bitbucket notifies them.
func Mention(user string) string {
	if plainName.MatchString(user) {
		return "@" + user
	}
	return `@"` + user + `"`
}

// ConflictOffer is the offer's text. byCode: code resolves every conflict,
// no model; else the conflicts are within bounds but need the model, which
// may still decline.
func ConflictOffer(l lang.Lang, byCode bool, author, bot, target string) string {
	cmd := inlineCode("@" + bot + " " + l.Pick(ConflictCommandDE, ConflictCommandEN))
	head := Mention(author) + " " + l.Pick("Dieser PR hat einen Merge-Konflikt mit ", "This PR has a merge conflict with ") +
		inlineCode(target) + "."
	if byCode {
		return head + l.Pick(
			" Ich kann ihn ohne Modell auflösen. Schreib "+cmd+", dann pushe ich den Merge-Commit auf deinen Branch.",
			" I can resolve it without the model. Write "+cmd+" and I push the merge commit to your branch.")
	}
	return head + l.Pick(
		" Er sieht einfach genug aus, dass ich versuchen kann, ihn aufzulösen. Schreib "+cmd+
			". Geht es nicht sicher, sage ich dir, warum.",
		" It looks simple enough for me to try to resolve it. Write "+cmd+
			". If I can't do it safely, I'll tell you why.")
}
