// Package web serves the built dashboard SPA out of the binary.
package web

import (
	"embed"
	"io/fs"
	"net/http"

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
// look like an empty page.
var routes = map[string]bool{
	"/":        true,
	"/live":    true,
	"/runs":    true,
	"/metrics": true,
	"/teams":   true,
	"/faq":     true,
}

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
	return spa.Handler(sub, spa.Options{
		Routes:      func(p string) bool { return routes[p] },
		Placeholder: placeholderHTML,
	})
}
