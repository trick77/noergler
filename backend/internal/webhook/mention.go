package webhook

import (
	"regexp"
	"strings"

	"github.com/trick77/noergler/internal/lang"
)

// userMention is a Bitbucket mention as a comment's raw text holds it: a
// username bare (`@sjh`) or quoted when it holds anything else
// (`@"jan.saner@example.com"`), the inverse of render.Mention. The character
// before the @ is no letter, digit or address part, so "foo@bar.ch" is no
// mention. Unicode-aware on purpose: RE2's classes are ASCII.
var userMention = regexp.MustCompile(`(?:^|[^\pL\pN_.@"/])@(?:"([^"\n]+)"|([\pL\pN_][\pL\pN._-]*))`)

// MentionsOther reports whether text mentions a user other than bot. A
// reply that does is addressed to that user, not to the bot (user,
// 2026-11-16), unless it mentions the bot too; the caller checks that.
//
// Code does not count: "`@Transactional` fehlt" mentions nobody. A bare
// annotation in prose ("@Override fehlt") does, accepted: whoever means the
// bot writes its name.
func MentionsOther(text, bot string) bool {
	for _, m := range userMention.FindAllStringSubmatch(lang.WithoutCode(text), -1) {
		name := m[1]
		if name == "" {
			// A sentence may end right after the name: "frag @sjh."
			name = strings.TrimRight(m[2], ".-")
		}
		if name != "" && !strings.EqualFold(name, bot) {
			return true
		}
	}
	return false
}
