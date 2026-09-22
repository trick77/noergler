package api

import (
	"crypto/sha256"
	"encoding/hex"
	"net/http"
	"strings"

	spec "github.com/trick77/noergler/api"
)

// specETag is the spec's hash, computed once at startup. The bytes are
// embedded in the binary, so they cannot change while it runs: a conditional
// request can be answered without touching them.
var specETag = func() string {
	sum := sha256.Sum256(spec.OpenAPISpec)
	return `"` + hex.EncodeToString(sum[:8]) + `"`
}()

// openapiSpec serves the contract itself.
func (d Deps) openapiSpec(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("ETag", specETag)
	w.Header().Set("Cache-Control", "public, max-age=300")
	if matchesETag(r.Header.Get("If-None-Match"), specETag) {
		w.WriteHeader(http.StatusNotModified)
		return
	}
	w.Header().Set("Content-Type", "application/yaml; charset=utf-8")
	_, _ = w.Write(spec.OpenAPISpec)
}

// matchesETag reports whether an If-None-Match header covers the tag. It
// handles the three forms a client may send: "*", a comma-separated list, and
// a weak "W/" prefix.
func matchesETag(header, tag string) bool {
	header = strings.TrimSpace(header)
	if header == "" {
		return false
	}
	if header == "*" {
		return true
	}
	for _, candidate := range strings.Split(header, ",") {
		candidate = strings.TrimSpace(candidate)
		candidate = strings.TrimPrefix(candidate, "W/")
		if candidate == tag {
			return true
		}
	}
	return false
}

// docsHTML is Swagger UI, loaded from a CDN and pointed at the spec route.
//
// The script is third-party and runs on this origin, which matters because
// "Try it out" is enabled: a credential typed into that panel is readable by
// it. That credential is the team's webhook secret - which signs PR events -
// and a Bitbucket token with project admin. The alternative is vendoring
// swagger-ui-dist through //go:embed, which costs a few MB in the image and a
// bump to keep current; it is deliberately not taken here, on the same
// reasoning as ../lens-gateway. Revisit it if this ever serves a surface
// where the credential is not already the caller's own.
//
// Same-origin, so the panel needs no CORS and no proxy.
const docsHTML = `<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>noergler — team self-service API</title>
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/swagger-ui-dist@5/swagger-ui.css" />
  </head>
  <body>
    <div id="app"></div>
    <script src="https://cdn.jsdelivr.net/npm/swagger-ui-dist@5/swagger-ui-bundle.js"></script>
    <script>
      window.ui = SwaggerUIBundle({ url: '/api/openapi.yaml', dom_id: '#app' })
    </script>
  </body>
</html>
`

// docs serves the API reference page.
func (d Deps) docs(w http.ResponseWriter, _ *http.Request) {
	w.Header().Set("Content-Type", "text/html; charset=utf-8")
	w.Header().Set("Cache-Control", "public, max-age=300")
	_, _ = w.Write([]byte(docsHTML))
}
