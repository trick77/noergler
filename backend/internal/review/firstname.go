package review

import (
	"regexp"
	"strings"
	"unicode"
	"unicode/utf8"
)

// firstNameMaxRunes bounds what a pattern may capture as a first name.
const firstNameMaxRunes = 40

// firstName is the "first" group of re on a Bitbucket displayName, or ""
// when names are off, the pattern does not match, or the capture is not a
// name. The display name is author-controlled and the operator's pattern may
// capture anything, so only letters, marks, inner spaces, ' and - pass.
func firstName(re *regexp.Regexp, displayName string) string {
	if re == nil {
		return ""
	}
	m := re.FindStringSubmatch(displayName)
	if m == nil {
		return ""
	}
	name := strings.TrimSpace(m[re.SubexpIndex("first")])
	if name == "" || utf8.RuneCountInString(name) > firstNameMaxRunes {
		return ""
	}
	for _, c := range name {
		if !unicode.IsLetter(c) && !unicode.IsMark(c) && c != ' ' && c != '\'' && c != '-' {
			return ""
		}
	}
	return name
}

// nameIntents are the reply intents that may address the author by name; a
// question or a rebuttal ("other") reads as talking down with one.
var nameIntents = map[string]bool{"acknowledges": true, "disputes": true}

// guardName enforces nameIntents on the model's reply: the prompt asks for
// it, this is the blunt rule behind it. A leading address ("Jan, ",
// "Ok Jan - ") on another intent is cut; the name anywhere else is left as
// written, a rewrite mid-sentence would garble it. state goes into the reply's
// log line, never the name.
func guardName(reply, name, intent string) (out, state string) {
	if name == "" {
		return reply, "none"
	}
	q := regexp.QuoteMeta(name)
	if !regexp.MustCompile(`(?i)(?:^|[^\pL\pN_])` + q + `(?:[^\pL\pN_]|$)`).MatchString(reply) {
		return reply, "unused"
	}
	if nameIntents[intent] {
		return reply, "used"
	}
	lead := regexp.MustCompile(`(?i)^[\s\p{Zs}]*(?:(?:ok(?:ay)?|thanks|hi|hey)[,!]?[\s\p{Zs}]+)?` + q +
		`[\s\p{Zs}]*[,:!.\x{2013}\x{2014}-][\s\p{Zs}]*`)
	loc := lead.FindStringIndex(reply)
	if loc == nil {
		return reply, "kept (not leading)"
	}
	rest := reply[loc[1]:]
	if rest == "" {
		return reply, "kept (not leading)"
	}
	c, size := utf8.DecodeRuneInString(rest)
	return string(unicode.ToUpper(c)) + rest[size:], "cut"
}
