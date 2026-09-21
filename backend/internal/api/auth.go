package api

import (
	"context"
	"crypto/hmac"
	"crypto/sha256"
	"encoding/hex"
	"net/http"
	"strings"

	"github.com/trick77/noergler-go/internal/httpapi"
	"github.com/trick77/noergler-go/internal/teams"
)

// runtimeFor resolves the slug in the path and answers 404 or 503 itself.
//
// The path is the only source of team identity. A payload's project.key is
// unauthenticated and is never consulted; the HMAC below proves the sender
// holds this team's secret, and the ownership check proves the PR is this
// team's.
//
// 404 and 503 come before any auth check on every route: they leak nothing
// the webhook route does not already leak, and "disabled" rather than
// "unknown" is the difference between reading the startup log and hunting a
// typo.
func (d Deps) runtimeFor(ctx context.Context, w http.ResponseWriter, slug string) (*teams.Runtime, bool) {
	rt, reason, ok := d.Teams.Lookup(slug)
	if ok {
		return rt, true
	}
	if reason != "" {
		d.Log.WarnContext(ctx, "request rejected: team is disabled", "reason", reason)
		httpapi.WriteDetail(w, http.StatusServiceUnavailable,
			"team "+slug+" is disabled, see the noergler startup log")
		return nil, false
	}
	httpapi.WriteDetail(w, http.StatusNotFound, "unknown team")
	return nil, false
}

// verifySignature checks Bitbucket's X-Hub-Signature against the team secret.
//
// Bitbucket sends "sha256=<lowercase hex>". The comparison is on the hex
// STRINGS, constant-time and case-sensitive, matching Python's
// hmac.compare_digest over two hexdigests: uppercase hex must fail rather
// than decode to the same bytes, and a malformed signature must compare
// false rather than error.
func verifySignature(body []byte, signature, secret string) bool {
	mac := hmac.New(sha256.New, []byte(secret))
	mac.Write(body)
	expected := hex.EncodeToString(mac.Sum(nil))
	signature = strings.TrimPrefix(signature, "sha256=")
	return hmac.Equal([]byte(expected), []byte(signature))
}
