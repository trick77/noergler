// Package web serves the built dashboard SPA out of the binary.
package web

import (
	"embed"
	"io/fs"
	"net/http"
	"strings"
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
// look like an empty page.
var routes = map[string]bool{
	"/":        true,
	"/live":    true,
	"/runs":    true,
	"/metrics": true,
	"/teams":   true,
	"/faq":     true,
}

// HasBuiltIndex reports whether the embedded dist/ holds a real build rather
// than just the tracked .gitkeep. Tests use it to tell the two legitimate
// states apart without duplicating the embed lookup.
func HasBuiltIndex() bool {
	sub, err := fs.Sub(distFS, "dist")
	if err != nil {
		return false
	}
	_, err = fs.Stat(sub, "index.html")
	return err == nil
}

// Handler serves the SPA. It is registered on "/" as the catch-all; Go's
// ServeMux prefers the most specific pattern, so every explicit route
// (/webhook/{team}, /health, /ready, /api/...) still wins over it.
func Handler() http.Handler {
	sub, err := fs.Sub(distFS, "dist")
	if err != nil {
		// Unreachable with a valid embed, but a nil handler would panic at
		// request time rather than here.
		return http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
			http.Error(w, "web assets unavailable", http.StatusInternalServerError)
		})
	}
	return handler(sub)
}

func handler(sub fs.FS) http.Handler {
	files := http.FileServer(http.FS(sub))
	shell, shellErr := fs.ReadFile(sub, "index.html")

	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		// A mistyped API path must stay a 404 and never answer with HTML: a
		// caller expecting JSON should see the error, not a page. The bare
		// form needs its own check, because a prefix test on "/api/" lets
		// "/api" fall through to the shell.
		//
		// /api/docs is HTML and is served by its own mux pattern, which is
		// more specific than this catch-all and so never reaches here. What
		// this rejects is only the paths nothing claimed.
		if r.URL.Path == "/api" || strings.HasPrefix(r.URL.Path, "/api/") {
			http.NotFound(w, r)
			return
		}

		if shellErr != nil {
			// No build in this binary. Say so on a route; everything else is
			// still a 404, so a missing asset does not answer with prose.
			if routes[r.URL.Path] {
				w.Header().Set("Content-Type", "text/html; charset=utf-8")
				_, _ = w.Write(placeholderHTML)
				return
			}
			http.NotFound(w, r)
			return
		}

		if _, err := fs.Stat(sub, strings.TrimPrefix(r.URL.Path, "/")); err != nil {
			// A hashed asset that is gone must 404 rather than receive the
			// shell: a browser holding a stale index asks for a chunk that no
			// longer exists, and text/html for a JS module request turns a
			// reload-on-stale heuristic into a syntax error.
			if strings.HasPrefix(r.URL.Path, "/assets/") || !routes[r.URL.Path] {
				http.NotFound(w, r)
				return
			}
			serveShell(w, shell)
			return
		}

		files.ServeHTTP(w, r)
	})
}

func serveShell(w http.ResponseWriter, shell []byte) {
	w.Header().Set("Content-Type", "text/html; charset=utf-8")
	// The shell names hashed assets, so it must not be cached itself: a
	// cached shell outlives the assets it points at.
	w.Header().Set("Cache-Control", "no-store")
	_, _ = w.Write(shell)
}
