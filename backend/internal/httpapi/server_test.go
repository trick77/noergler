package httpapi

import (
	"bytes"
	"encoding/json"
	"errors"
	"log/slog"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/trick77/noergler-go/internal/logging"
)

func newTestServer(t *testing.T, enabled, disabled []string) (*Server, *bytes.Buffer) {
	t.Helper()
	var buf bytes.Buffer
	log := slog.New(logging.NewHandler(&buf, slog.LevelDebug, "test"))
	s := New(func() ([]string, []string) { return enabled, disabled }, log)
	return s, &buf
}

func get(h http.Handler, path string, headers ...string) *httptest.ResponseRecorder {
	req := httptest.NewRequest(http.MethodGet, path, nil)
	for i := 0; i+1 < len(headers); i += 2 {
		req.Header.Set(headers[i], headers[i+1])
	}
	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, req)
	return rec
}

func body(t *testing.T, rec *httptest.ResponseRecorder) map[string]any {
	t.Helper()
	var m map[string]any
	if err := json.Unmarshal(rec.Body.Bytes(), &m); err != nil {
		t.Fatalf("not JSON: %s", rec.Body.String())
	}
	return m
}

func TestHealth_Always200WithSlugs(t *testing.T) {
	s, buf := newTestServer(t, nil, []string{"payments"})
	rec := get(s.Handler(), "/health")
	if rec.Code != 200 {
		t.Fatalf("status = %d", rec.Code)
	}
	m := body(t, rec)
	teams := m["teams"].(map[string]any)
	if m["status"] != "ok" || len(teams["enabled"].([]any)) != 0 || teams["disabled"].([]any)[0] != "payments" {
		t.Errorf("body = %v", m)
	}
	if buf.Len() != 0 {
		t.Errorf("probes must not be access-logged: %s", buf.String())
	}
}

func TestReady_503WhileNoTeamEnabled(t *testing.T) {
	s, _ := newTestServer(t, nil, nil)
	rec := get(s.Handler(), "/ready")
	if rec.Code != 503 || body(t, rec)["status"] != "no-teams" {
		t.Errorf("status = %d body = %s", rec.Code, rec.Body.String())
	}
	s, _ = newTestServer(t, []string{"platform"}, nil)
	rec = get(s.Handler(), "/ready")
	if rec.Code != 200 || body(t, rec)["status"] != "ok" {
		t.Errorf("status = %d body = %s", rec.Code, rec.Body.String())
	}
}

func TestAccessLog_RequestIDHonouredOnlyWhenSane(t *testing.T) {
	s, buf := newTestServer(t, nil, nil)
	s.HandleFunc("GET /x", func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusTeapot)
	})
	h := s.Handler()
	get(h, "/x", "X-Request-Id", "abc.123-XYZ")
	var m map[string]any
	if err := json.Unmarshal(buf.Bytes(), &m); err != nil {
		t.Fatalf("log: %s", buf.String())
	}
	if m["msg"] != "http_request" || m["request_id"] != "abc.123-XYZ" || m["method"] != "GET" || m["path"] != "/x" || m["status_code"] != float64(418) {
		t.Errorf("access line = %v", m)
	}
	if _, ok := m["duration_ms"].(float64); !ok {
		t.Errorf("duration_ms missing: %v", m)
	}
	buf.Reset()
	get(h, "/x", "X-Request-Id", "bad id\nwith newline")
	_ = json.Unmarshal(buf.Bytes(), &m)
	if id, _ := m["request_id"].(string); id == "bad id\nwith newline" || len(id) != 32 {
		t.Errorf("unsafe id must be replaced: %q", id)
	}
}

func TestRecoverer_PanicIs500AndLogged(t *testing.T) {
	s, buf := newTestServer(t, nil, nil)
	s.HandleFunc("GET /boom", func(_ http.ResponseWriter, _ *http.Request) { panic("kaboom") })
	rec := get(s.Handler(), "/boom")
	if rec.Code != 500 || body(t, rec)["detail"] != "Internal Server Error" {
		t.Errorf("status = %d body = %s", rec.Code, rec.Body.String())
	}
	lines := strings.Split(strings.TrimSpace(buf.String()), "\n")
	if len(lines) != 2 || !strings.Contains(lines[0], "kaboom") {
		t.Fatalf("want the panic line then the access line, got %s", buf.String())
	}
	var panicLine, accessLine map[string]any
	_ = json.Unmarshal([]byte(lines[0]), &panicLine)
	_ = json.Unmarshal([]byte(lines[1]), &accessLine)
	if panicLine["request_id"] == nil || panicLine["request_id"] != accessLine["request_id"] {
		t.Errorf("panic line must carry the request id: %v / %v", panicLine, accessLine)
	}
	if accessLine["status_code"] != float64(500) {
		t.Errorf("access line status = %v", accessLine["status_code"])
	}
}

func TestAccessLog_HandlerThatWritesNothingIs200(t *testing.T) {
	s, buf := newTestServer(t, nil, nil)
	s.HandleFunc("GET /empty", func(_ http.ResponseWriter, _ *http.Request) {})
	rec := get(s.Handler(), "/empty")
	var m map[string]any
	_ = json.Unmarshal(buf.Bytes(), &m)
	if rec.Code != 200 || m["status_code"] != float64(200) {
		t.Errorf("code = %d, logged = %v", rec.Code, m["status_code"])
	}
}

func TestUnknownRouteIs404(t *testing.T) {
	s, _ := newTestServer(t, nil, nil)
	if rec := get(s.Handler(), "/docs"); rec.Code != 404 {
		t.Errorf("status = %d", rec.Code)
	}
}

// The recoverer must re-panic on http.ErrAbortHandler rather than turning it
// into a 500: net/http treats that sentinel as "the handler deliberately gave
// up on this connection" and suppresses its own logging for it. Swallowing it
// would convert a silent abort into a spurious error line plus a response body
// written to a connection the handler meant to drop.
//
// Covered because the guard was rewritten from `rec == http.ErrAbortHandler`
// to an errors.As/Is form, which also has to keep matching the bare sentinel.
func TestRecoverer_rePanicsOnErrAbortHandler(t *testing.T) {
	s, _ := newTestServer(t, nil, nil)
	h := s.recoverer(http.HandlerFunc(func(http.ResponseWriter, *http.Request) {
		panic(http.ErrAbortHandler)
	}))

	defer func() {
		rec := recover()
		if rec == nil {
			t.Fatal("ErrAbortHandler was swallowed; net/http must see it to abort quietly")
		}
		if !errors.Is(rec.(error), http.ErrAbortHandler) {
			t.Fatalf("re-panicked with %v, want http.ErrAbortHandler", rec)
		}
	}()
	h.ServeHTTP(httptest.NewRecorder(), httptest.NewRequest(http.MethodGet, "/", nil))
}

// Every other panic, including a non-error value, must become a 500 and be
// logged rather than escaping. A panic(string) does not satisfy the error
// interface, so this is the case a type-assertion-first guard could drop.
func TestRecoverer_turnsOtherPanicsInto500(t *testing.T) {
	for _, tc := range []struct {
		name  string
		value any
	}{
		{"error value", errors.New("boom")},
		{"plain string", "boom"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			s, buf := newTestServer(t, nil, nil)
			h := s.recoverer(http.HandlerFunc(func(http.ResponseWriter, *http.Request) {
				panic(tc.value)
			}))

			rr := httptest.NewRecorder()
			h.ServeHTTP(rr, httptest.NewRequest(http.MethodGet, "/", nil))

			if rr.Code != http.StatusInternalServerError {
				t.Errorf("status = %d, want 500", rr.Code)
			}
			if !strings.Contains(buf.String(), "panic in handler") {
				t.Errorf("the panic was not logged: %s", buf.String())
			}
		})
	}
}
