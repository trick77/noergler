package review

import (
	"regexp"
	"strings"
	"unicode"
	"unicode/utf8"

	"golang.org/x/text/unicode/norm"
)

// firstNameMaxRunes bounds what a pattern may capture as a first name.
const firstNameMaxRunes = 40

// firstName is the "first" group of re on a Bitbucket displayName, or ""
// when names are off, the pattern does not match, or the capture is not a
// name. The display name is author-controlled and the operator's pattern may
// capture anything, so only letters, marks, inner spaces, ' and - pass, with
// at least two letters: an initial ("J.") or a lone "-" is no name. NFC, so
// it compares with what the model writes.
func firstName(re *regexp.Regexp, displayName string) string {
	if re == nil {
		return ""
	}
	m := re.FindStringSubmatch(norm.NFC.String(displayName))
	if m == nil {
		return ""
	}
	name := strings.TrimSpace(m[re.SubexpIndex("first")])
	if utf8.RuneCountInString(name) > firstNameMaxRunes {
		return ""
	}
	letters := 0
	for _, c := range name {
		switch {
		case unicode.IsLetter(c):
			letters++
		case unicode.IsMark(c) || c == ' ' || c == '\'' || c == '-':
		default:
			return ""
		}
	}
	if letters < 2 {
		return ""
	}
	return name
}

// nameIntents are the reply intents that may address the author by name; a
// question or a rebuttal ("other") reads as talking down with one.
var nameIntents = map[string]bool{"acknowledges": true, "disputes": true}

// leadGreeting is what may stand before a leading address ("Ok Jan - ").
const leadGreeting = `(?:ok(?:ay)?|sure|thanks|thank you|hi|hey|hello|right|got it|yes|no|alright)`

// guardName enforces nameIntents on the model's reply: the prompt asks for
// it, this is the blunt rule behind it. A leading address ("Jan, ",
// "Ok Jan - ", "Hi Jan ") on another intent is cut; the name anywhere else
// is left as written, a rewrite mid-sentence would garble it. state goes
// into the reply's log line, never the name; patternSet tells a name off
// from one the pattern did not find.
func guardName(reply, name, intent string, patternSet bool) (out, state string) {
	if name == "" {
		if patternSet {
			return reply, "unmatched"
		}
		return reply, "none"
	}
	reply = norm.NFC.String(reply)
	q := regexp.QuoteMeta(name)
	if !regexp.MustCompile(`(?i)(?:^|[^\pL\pN_])` + q + `(?:[^\pL\pN_]|$)`).MatchString(reply) {
		return reply, "unused"
	}
	if nameIntents[intent] {
		return reply, "used"
	}
	// After a greeting the punctuation is optional ("Hi Jan the check ..."),
	// alone the name needs it: "Jan said" is no address.
	sep := `[\s\p{Zs}]*[,:!.\x{2026}\x{2013}\x{2014}-][\s\p{Zs}]*`
	lead := regexp.MustCompile(`(?i)^[\s\p{Zs}]*(?:` + leadGreeting + `[,!]?[\s\p{Zs}]+` + q +
		`(?:` + sep + `|[\s\p{Zs}]+)|` + q + sep + `)`)
	loc := lead.FindStringIndex(reply)
	if loc == nil || loc[1] == len(reply) {
		return reply, "kept"
	}
	return capitalizeWord(reply[loc[1]:]), "cut"
}

// plainWordRE is a lower-case word of two letters or more and a space: the
// only start capitalizeWord touches, so `x`, `iOS` or `nil()` stay as written.
var plainWordRE = regexp.MustCompile(`^\p{Ll}{2,}[\s\p{Zs}]`)

// capitalizeWord upper-cases the first letter of s when it starts a plain
// word: the cut left "the check is ..." mid-sentence.
func capitalizeWord(s string) string {
	if !plainWordRE.MatchString(s) {
		return s
	}
	c, size := utf8.DecodeRuneInString(s)
	return string(unicode.ToUpper(c)) + s[size:]
}
