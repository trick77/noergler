package render

import "regexp"

// securityKeywords flags a finding as security-relevant so the summary can
// surface a count above the Issues list.
//
// The word boundary must be Unicode-aware: "üinsecure" must NOT match,
// because ü is a word character and there is no boundary before "insecure".
// RE2's \b is ASCII and would match it, so the boundary is spelled out as
// (?:^|[^\pL\pN_]) and its mirror (TestSecurityKeywordsUnicodeBoundary). The
// boundary characters are consumed, which is fine here because only
// MatchString is used, never a substitution.
//
// The alternation is quirky in ways that are deliberate, not bugs to fix:
// "secret leak" matches and "secrets leak" does not, because the class is
// secret[s ]?leak, a single optional character (TestSecurityKeywordsQuirks).
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
