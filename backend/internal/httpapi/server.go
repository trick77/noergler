// Package httpapi is the HTTP surface's mux: probes and middleware from kit's
// httpx, routes added by the packages that own them. net/http only, Go 1.22
// method patterns on a ServeMux.
package httpapi

import (
	"context"
	"errors"
	"log/slog"
	"net/http"
	"regexp"

	"github.com/trick77/ship/kit/httpx"
)

// TeamStatus reports the enabled and disabled slugs. Slugs only: the disable
// reasons carry internal detail (gateway URLs, error bodies, env var names)
// and belong in the log, not in a probe's body.
type TeamStatus func() (enabled, disabled []string)

// errNoTeams fails readiness while no team can take traffic.
var errNoTeams = errors.New("no team is enabled")

// Server holds what the handlers share.
type Server struct {
	log *slog.Logger
	mux *http.ServeMux
}

// New builds the mux with the probes registered: /healthz is liveness (a
// config fault that leaves every team disabled is not fixed by a restart, so
// it never fails), /readyz is readiness and 503 while no team is enabled.
// Further routes are added by the packages that own them through Handle.
func New(status TeamStatus, log *slog.Logger) *Server {
	s := &Server{log: log, mux: http.NewServeMux()}
	httpx.Health(s.mux, log, httpx.Check{Name: "teams", Run: func(context.Context) error {
		if enabled, _ := status(); len(enabled) == 0 {
			return errNoTeams
		}
		return nil
	}})
	return s
}

// Handle registers a pattern on the mux.
func (s *Server) Handle(pattern string, h http.Handler) { s.mux.Handle(pattern, h) }

// HandleFunc registers a handler function on the mux.
func (s *Server) HandleFunc(pattern string, h http.HandlerFunc) { s.mux.HandleFunc(pattern, h) }

// quietPaths are polled on a timer and would otherwise be the bulk of the
// access log; a failing one is still logged. The dashboard's pages poll
// themselves while anyone has one open. The probes are kit's ProbePaths: a
// probe is never access-logged, Health logs readiness when it changes; one
// kit refused (not a local caller) logs as any 404.
var quietPaths = []string{
	"/api/dashboard/live",
	"/api/dashboard/runs",
	"/api/dashboard/metrics",
	"/api/dashboard/teams",
}

// runTokenRE is whatever follows /runs/ in a path (the SPA's /runs/{token}
// and the API's /api/dashboard/runs/{token}, a trailing slash or a further
// segment included): a run page's only key, so never written to a log.
var runTokenRE = regexp.MustCompile(`/runs/.+`)

// logPath is the path as the access log and a panic's line write it, a run
// page's token masked.
func logPath(r *http.Request) string {
	return runTokenRE.ReplaceAllString(r.URL.Path, "/runs/{token}")
}

// Handler is the mux wrapped in the middleware chain: access log outermost,
// so a panic's log line and its 500 carry the request id.
func (s *Server) Handler() http.Handler {
	return httpx.Chain(s.mux,
		httpx.AccessLog(s.log, httpx.AccessLogOptions{Quiet: quietPaths, Path: logPath}),
		httpx.RecoverWith(s.log, httpx.RecoverOptions{Path: logPath}),
	)
}
