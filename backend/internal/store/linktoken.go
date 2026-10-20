package store

import (
	"crypto/rand"
	"encoding/base64"
	"regexp"
)

// NewLinkToken is a PR page's link token: 16 random bytes, base64url, 22
// characters. The only key to the page, so never derived from the PR.
func NewLinkToken() string {
	var b [16]byte
	_, _ = rand.Read(b[:]) // never fails (crypto/rand panics instead)
	return base64.RawURLEncoding.EncodeToString(b[:])
}

// LinkPattern is what NewLinkToken produces, unanchored: the one
// spelling the web route builds on (ui/src/routing.ts mirrors it).
const LinkPattern = `[A-Za-z0-9_-]{22}`

// LinkTokenRE matches a whole link token and nothing else.
var LinkTokenRE = regexp.MustCompile(`^` + LinkPattern + `$`)
