package render

import "regexp"

// securityKeywords flags a finding as security-relevant so the summary can
// surface a count above the Issues list.
//
// Python's \b is Unicode and RE2's is ASCII, so a naive port diverges: to
// Python "üinsecure" does NOT match (ü is a word character, so there is no
// boundary) while a naive RE2 port matches it. Verified against the venv.
// (?:^|[^\pL\pN_]) and its mirror restore Python's behaviour; the boundary
// characters are consumed, which is fine here because only MatchString is
// used, never a substitution.
//
// The alternation is quirky in ways that are deliberate parity, not bugs to
// fix: "secret leak" matches and "secrets leak" does not, because the class
// is secret[s ]?leak, a single optional character.
var securityKeywords = regexp.MustCompile(
	`(?i)(?:^|[^\pL\pN_])(?:injection|xss|sql[_ -]?injection|authentication|` +
		`authorization|credentials?|secret[s ]?leak|csrf|ssrf|` +
		`path[_ -]?traversal|insecure|vulnerability|sanitiz(?:e|ation)|` +
		`privilege[_ -]?escalation|deserialization|token[_ -]?leak|` +
		`exposed?[_ -]?(?:secret|credential|key|token))(?:[^\pL\pN_]|$)`)

// IsSecurityFinding reports whether a finding's comment mentions a security
// keyword.
func IsSecurityFinding(comment string) bool {
	return securityKeywords.MatchString(comment)
}
