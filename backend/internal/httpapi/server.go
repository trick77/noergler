// Package httpapi is the HTTP surface: probes, the webhook, the team
// self-service API. net/http only, Go 1.22 method patterns on a ServeMux.
package httpapi

import (
	"context"
	"crypto/rand"
	"encoding/hex"
	"encoding/json"
	"errors"
	"log/slog"
	"net/http"
	"regexp"
	"runtime/debug"
	"time"

	"github.com/trick77/noergler-go/internal/logging"
)

// TeamStatus reports the slugs the probes show. Slugs only: the disable
// reasons carry internal detail (gateway URLs, error bodies, env var names)
// and belong in the log, not on an unauthenticated probe.
type TeamStatus func() (enabled, disabled []string)

// Server holds what the handlers share.
type Server struct {
	status TeamStatus
	log    *slog.Logger
	mux    *http.ServeMux
}

// New builds the mux with the probes registered. Further routes are added
// by the packages that own them through Handle.
func New(status TeamStatus, log *slog.Logger) *Server {
	s := &Server{status: status, log: log, mux: http.NewServeMux()}
	s.mux.HandleFunc("GET /health", s.health)
	s.mux.HandleFunc("GET /ready", s.ready)
	return s
}

// Handle registers a pattern on the mux.
func (s *Server) Handle(pattern string, h http.Handler) { s.mux.Handle(pattern, h) }

// HandleFunc registers a handler function on the mux.
func (s *Server) HandleFunc(pattern string, h http.HandlerFunc) { s.mux.HandleFunc(pattern, h) }

// Handler is the mux wrapped in the middleware chain: request id and access
// log outermost, so a panic's log line and its 500 carry the request id.
func (s *Server) Handler() http.Handler {
	return s.accessLog(s.recoverer(s.mux))
}

// --- probes ------------------------------------------------------------------

func (s *Server) teamStatus() map[string]any {
	enabled, disabled := s.status()
	if enabled == nil {
		enabled = []string{}
	}
	if disabled == nil {
		disabled = []string{}
	}
	return map[string]any{"enabled": enabled, "disabled": disabled}
}

// health is liveness: 200 while the process is up. A config fault that
// leaves every team disabled is not fixed by a restart, so it never fails.
func (s *Server) health(w http.ResponseWriter, _ *http.Request) {
	WriteJSON(w, http.StatusOK, map[string]any{"status": "ok", "teams": s.teamStatus()})
}

// ready is readiness: 503 while no team can take traffic.
func (s *Server) ready(w http.ResponseWriter, _ *http.Request) {
	enabled, _ := s.status()
	status, code := "ok", http.StatusOK
	if len(enabled) == 0 {
		status, code = "no-teams", http.StatusServiceUnavailable
	}
	WriteJSON(w, code, map[string]any{"status": status, "teams": s.teamStatus()})
}

// WriteJSON renders a JSON body with the status.
func WriteJSON(w http.ResponseWriter, status int, body any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(body)
}

// WriteDetail is FastAPI's error shape, {"detail": "..."}, which the
// onboarding scripts and the .http files match on.
func WriteDetail(w http.ResponseWriter, status int, detail string) {
	WriteJSON(w, status, map[string]string{"detail": detail})
}

// --- middleware --------------------------------------------------------------

var silentPaths = map[string]bool{"/health": true, "/ready": true}

// requestIDRE bounds a caller-supplied X-Request-Id: untrusted input must
// not become an indexed field (newlines, huge strings).
var requestIDRE = regexp.MustCompile(`^[A-Za-z0-9._-]{1,64}$`)

func newRequestID() string {
	var b [16]byte
	_, _ = rand.Read(b[:])
	return hex.EncodeToString(b[:])
}

type statusWriter struct {
	http.ResponseWriter
	status int
}

func (w *statusWriter) WriteHeader(code int) {
	w.status = code
	w.ResponseWriter.WriteHeader(code)
}

func (w *statusWriter) Write(b []byte) (int, error) {
	if w.status == 0 {
		w.status = http.StatusOK
	}
	return w.ResponseWriter.Write(b)
}

// accessLog binds request_id/method/path into the context and emits one
// `http_request` line per request with status_code and duration_ms. Probes
// fire every few seconds and pass through unobserved.
func (s *Server) accessLog(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if silentPaths[r.URL.Path] {
			next.ServeHTTP(w, r)
			return
		}
		id := r.Header.Get("X-Request-Id")
		if !requestIDRE.MatchString(id) {
			id = newRequestID()
		}
		ctx := logging.With(r.Context(), "request_id", id, "method", r.Method, "path", r.URL.Path)
		sw := &statusWriter{ResponseWriter: w}
		started := time.Now()
		defer func() {
			// A handler that wrote nothing still answered 200 (net/http
			// sends it at the end of the handler).
			status := sw.status
			if status == 0 {
				status = http.StatusOK
			}
			ms := float64(time.Since(started).Microseconds()) / 1000
			s.log.InfoContext(ctx, "http_request", "status_code", status, "duration_ms", float64(int(ms*10+0.5))/10)
		}()
		next.ServeHTTP(sw, r.WithContext(ctx))
	})
}

// recoverer turns a panic into a 500 with the stack logged, so one bad
// request never takes the single replica down.
func (s *Server) recoverer(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		defer func() {
			if rec := recover(); rec != nil {
				err, ok := rec.(error)
				if ok && errors.Is(err, http.ErrAbortHandler) {
					panic(rec)
				}
				s.log.ErrorContext(r.Context(), "panic in handler", "panic", rec, "stack", string(debug.Stack()))
				WriteDetail(w, http.StatusInternalServerError, "Internal Server Error")
			}
		}()
		next.ServeHTTP(w, r)
	})
}

// Run serves until ctx is cancelled, then drains for up to the grace period.
func Run(ctx context.Context, addr string, h http.Handler, log *slog.Logger) error {
	srv := &http.Server{
		Addr:              addr,
		Handler:           h,
		ReadHeaderTimeout: 10 * time.Second,
	}
	errc := make(chan error, 1)
	go func() {
		log.Info("listening", "addr", addr)
		errc <- srv.ListenAndServe()
	}()
	select {
	case err := <-errc:
		return err
	case <-ctx.Done():
		shutdownCtx, cancel := context.WithTimeout(context.Background(), 15*time.Second)
		defer cancel()
		return srv.Shutdown(shutdownCtx)
	}
}
