// Package web serves the built dashboard SPA out of the binary.
package web

import (
	"embed"
	"io/fs"
	"net/http"
	"regexp"

	"github.com/trick77/noergler/internal/store"
	"github.com/trick77/ship/kit/spa"
)

// The SPA is built into dist/ by `npm run build` (ui/vite.config.ts points
// its outDir here). A fresh clone has no build, so dist/ holds a tracked
// .gitkeep: `//go:embed all:dist` fails at COMPILE time on a missing or
// empty directory, which would break `go build` for anyone who has not run
// the frontend build.
//
//go:embed all:dist
var distFS embed.FS

// placeholderHTML ships separately from dist/. vite runs with
// emptyOutDir: false so the tracked .gitkeep survives a build; embedding the
// placeholder inside dist/ would mean the first `make fe-build` permanently
// overwrote the tracked placeholder with the built shell.
//
//go:embed placeholder.html
var placeholderHTML []byte

// routes mirrors ui/src/routing.ts. An explicit allowlist, so an unknown
// path is a real 404 rather than 200 plus the shell: a soft 404 makes a typo
// look like an empty page. One pattern beside it: a PR page and its runs.
var routes = map[string]bool{
	"/":        true,
	"/live":    true,
	"/runs":    true,
	"/metrics": true,
	"/teams":   true,
	"/faq":     true,
}

// prPath is the PR page, /prs/ and a link token (store.NewLinkToken: 22
// base64url characters), and each of its runs under it by number. The
// token, handed out only in the PR's summary comment, is the pages' one
// key; an old /runs/{token} link is the 404 an unknown path is (decided).
var prPath = regexp.MustCompile(`^/prs/` + store.LinkPattern + `(?:/runs/[1-9][0-9]{0,8})?$`)

// teamPath is one team on the Teams page, /teams/{slug}: the page scrolled
// to that team with its configuration open, the link the # on a team card
// copies. The slug is config.TeamSlugRE's shape; an unknown slug is the
// page's to report, so any well-formed one gets the shell.
var teamPath = regexp.MustCompile(`^/teams/[a-z0-9][a-z0-9-]*$`)

func dist() fs.FS {
	sub, err := fs.Sub(distFS, "dist")
	if err != nil {
		panic(err) // unreachable: the embed pattern guarantees the directory
	}
	return sub
}

// HasBuiltIndex reports whether the embedded dist/ holds a real build rather
// than just the tracked .gitkeep.
func HasBuiltIndex() bool { return spa.Built(dist()) }

// Handler serves the SPA. It is registered on "/" as the catch-all; Go's
// ServeMux prefers the most specific pattern, so every explicit route
// (/webhook/{team}, /healthz, /readyz, /api/...) still wins over it.
func Handler() http.Handler { return handler(dist()) }

// handler is kit's SPA server with the route allowlist: /api and a missing
// /assets/ file stay 404s, the shell is revalidated on every load, and an
// unbuilt binary answers its routes with the placeholder and a 503.
func handler(sub fs.FS) http.Handler {
	h := spa.Handler(sub, spa.Options{
		Routes:      func(p string) bool { return routes[p] || prPath.MatchString(p) || teamPath.MatchString(p) },
		Placeholder: placeholderHTML,
	})
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		// A PR or run page's URL carries its key: keep it out of search indexes,
		// out of the Referer of every link the page opens, and out of every
		// cache. Set as the headers go out: spa.Handler sets its own
		// Cache-Control for the shell, which would win if set before it.
		if prPath.MatchString(r.URL.Path) {
			w = &runPageWriter{ResponseWriter: w}
		}
		h.ServeHTTP(w, r)
	})
}

// runPageWriter stamps a PR or run page's headers over whatever the SPA handler
// set, right before they are written.
type runPageWriter struct {
	http.ResponseWriter
	wrote bool
}

func (w *runPageWriter) WriteHeader(code int) {
	if !w.wrote {
		w.wrote = true
		h := w.Header()
		h.Set("X-Robots-Tag", "noindex, nofollow")
		h.Set("Referrer-Policy", "no-referrer")
		h.Set("Cache-Control", "no-store")
		h.Del("ETag")
	}
	w.ResponseWriter.WriteHeader(code)
}

func (w *runPageWriter) Write(b []byte) (int, error) {
	if !w.wrote {
		w.WriteHeader(http.StatusOK)
	}
	return w.ResponseWriter.Write(b)
}
