package store

import (
	"crypto/rand"
	"encoding/base64"
	"regexp"
)

// NewLinkToken is a run page's link token: 16 random bytes, base64url, 22
// characters. The only key to the page, so never derived from the attempt.
func NewLinkToken() string {
	var b [16]byte
	_, _ = rand.Read(b[:]) // never fails (crypto/rand panics instead)
	return base64.RawURLEncoding.EncodeToString(b[:])
}

// LinkTokenRE is what NewLinkToken produces, nothing else.
var LinkTokenRE = regexp.MustCompile(`^[A-Za-z0-9_-]{22}$`)
