package httpapi

import (
	"bytes"
	"encoding/json"
	"log/slog"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/trick77/ship/kit/logging"
)

func newTestServer(t *testing.T, enabled, disabled []string) (*Server, *bytes.Buffer) {
	t.Helper()
	var buf bytes.Buffer
	log := slog.New(logging.NewSplunkHandler(&buf, slog.LevelDebug, "noergler", "test"))
	s := New(func() ([]string, []string) { return enabled, disabled }, log)
	return s, &buf
}

func get(h http.Handler, path string) *httptest.ResponseRecorder {
	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, httptest.NewRequest(http.MethodGet, path, nil))
	return rec
}

// probe GETs a probe path as a direct local caller: kit's probes answer no
// one else.
func probe(h http.Handler, path string) *httptest.ResponseRecorder {
	rec := httptest.NewRecorder()
	req := httptest.NewRequest(http.MethodGet, path, nil)
	req.RemoteAddr = "127.0.0.1:40000"
	h.ServeHTTP(rec, req)
	return rec
}

func TestHealthz_Always200(t *testing.T) {
	s, buf := newTestServer(t, nil, []string{"payments"})
	rec := probe(s.Handler(), "/healthz")
	if rec.Code != 200 {
		t.Fatalf("status = %d", rec.Code)
	}
	// Slugs never reach the unauthenticated probe.
	if strings.Contains(rec.Body.String(), "payments") {
		t.Errorf("body names a team: %s", rec.Body.String())
	}
	if buf.Len() != 0 {
		t.Errorf("a passing probe must not be access-logged: %s", buf.String())
	}
}

func TestReadyz_503WhileNoTeamEnabled(t *testing.T) {
	s, _ := newTestServer(t, nil, []string{"payments"})
	rec := probe(s.Handler(), "/readyz")
	var m map[string]string
	_ = json.Unmarshal(rec.Body.Bytes(), &m)
	if rec.Code != 503 || m["check"] != "teams" {
		t.Errorf("status = %d body = %s", rec.Code, rec.Body.String())
	}
	s, _ = newTestServer(t, []string{"platform"}, nil)
	if rec := probe(s.Handler(), "/readyz"); rec.Code != 200 {
		t.Errorf("status = %d body = %s", rec.Code, rec.Body.String())
	}
}

func TestOldProbePathsAreGone(t *testing.T) {
	s, _ := newTestServer(t, []string{"platform"}, nil)
	for _, p := range []string{"/health", "/ready"} {
		if rec := get(s.Handler(), p); rec.Code != 404 {
			t.Errorf("%s = %d, want 404", p, rec.Code)
		}
	}
}

func TestAccessLogAndRecoverAreWired(t *testing.T) {
	s, buf := newTestServer(t, nil, nil)
	s.HandleFunc("GET /boom", func(http.ResponseWriter, *http.Request) { panic("kaboom") })
	s.Handle("GET /x", http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusTeapot)
	}))
	h := s.Handler()
	if rec := get(h, "/boom"); rec.Code != 500 || !strings.Contains(rec.Body.String(), `"error"`) {
		t.Errorf("panic: status = %d body = %s", rec.Code, rec.Body.String())
	}
	if !strings.Contains(buf.String(), "kaboom") || !strings.Contains(buf.String(), `"service":"noergler"`) {
		t.Errorf("panic not logged: %s", buf.String())
	}
	buf.Reset()
	get(h, "/x")
	var m map[string]any
	if err := json.Unmarshal(buf.Bytes(), &m); err != nil {
		t.Fatalf("log: %s", buf.String())
	}
	if m["msg"] != "http_request" || m["path"] != "/x" || m["request_id"] == nil {
		t.Errorf("access line = %v", m)
	}
}

func TestUnknownRouteIs404(t *testing.T) {
	s, _ := newTestServer(t, nil, nil)
	if rec := get(s.Handler(), "/docs"); rec.Code != 404 {
		t.Errorf("status = %d", rec.Code)
	}
}
