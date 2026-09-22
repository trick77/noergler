package web

import (
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"testing/fstest"
)

func built() fstest.MapFS {
	return fstest.MapFS{
		"index.html":           {Data: []byte("<!doctype html><div id=root>")},
		"assets/index-abc.js":  {Data: []byte("console.log(1)")},
		"assets/index-abc.css": {Data: []byte("body{}")},
	}
}

func req(t *testing.T, h http.Handler, path string) *httptest.ResponseRecorder {
	t.Helper()
	w := httptest.NewRecorder()
	h.ServeHTTP(w, httptest.NewRequest(http.MethodGet, path, nil))
	return w
}

// A client route is served the shell so the SPA can render it; anything else
// is a real 404. A catch-all that answered 200 for every path would turn a
// typo into a blank page and hide it from any uptime check.
func TestKnownRoutesGetTheShellAndUnknownOnesA404(t *testing.T) {
	h := handler(built())

	// Driven off routes itself, not a copy of it: a hardcoded list stops
	// covering a page the moment one is added to the allowlist.
	if len(routes) < 5 {
		t.Fatalf("routes has %d entries, expected the dashboard's pages", len(routes))
	}
	for path := range routes {
		w := req(t, h, path)
		if w.Code != http.StatusOK {
			t.Errorf("%s: status = %d, want 200", path, w.Code)
		}
		if !strings.Contains(w.Body.String(), "id=root") {
			t.Errorf("%s: want the shell, got %q", path, w.Body.String())
		}
	}

	for _, path := range []string{"/nope", "/live/deeper", "/Runs"} {
		if code := req(t, h, path).Code; code != http.StatusNotFound {
			t.Errorf("%s: status = %d, want 404", path, code)
		}
	}
}

// A browser holding a stale index asks for a chunk that no longer exists.
// Answering text/html for a JS module request turns a reload-on-stale
// heuristic into a syntax error, so a missing asset must 404.
func TestMissingAssetIsA404NotTheShell(t *testing.T) {
	h := handler(built())

	w := req(t, h, "/assets/index-OLD.js")
	if w.Code != http.StatusNotFound {
		t.Fatalf("status = %d, want 404", w.Code)
	}
	if strings.Contains(w.Body.String(), "id=root") {
		t.Error("a missing asset must not receive the shell")
	}
}

func TestExistingAssetIsServed(t *testing.T) {
	w := req(t, handler(built()), "/assets/index-abc.js")
	if w.Code != http.StatusOK {
		t.Fatalf("status = %d, want 200", w.Code)
	}
	if !strings.Contains(w.Body.String(), "console.log") {
		t.Errorf("body = %q", w.Body.String())
	}
}

// A mistyped API path must stay a 404. The bare "/api" needs its own check:
// a prefix test on "/api/" alone lets it fall through to the shell, and a
// caller expecting JSON would parse a page.
func TestAPIPathsNeverReceiveTheShell(t *testing.T) {
	h := handler(built())

	for _, path := range []string{"/api", "/api/", "/api/dashboard/typo"} {
		w := req(t, h, path)
		if w.Code != http.StatusNotFound {
			t.Errorf("%s: status = %d, want 404", path, w.Code)
		}
		if strings.Contains(w.Body.String(), "id=root") {
			t.Errorf("%s received the shell", path)
		}
	}
}

// A binary built without running the frontend build still starts and still
// answers: it says what to run rather than 404ing its own pages.
func TestUnbuiltBinaryServesThePlaceholder(t *testing.T) {
	h := handler(fstest.MapFS{".gitkeep": {Data: []byte("")}})

	w := req(t, h, "/")
	if w.Code != http.StatusOK {
		t.Fatalf("status = %d, want 200", w.Code)
	}
	if !strings.Contains(w.Body.String(), "make fe-build") {
		t.Errorf("placeholder should name the build command, got %q", w.Body.String())
	}

	// Still a 404 for everything that is not a route: a missing build must
	// not turn every path into prose.
	if code := req(t, h, "/assets/index-abc.js").Code; code != http.StatusNotFound {
		t.Errorf("asset status = %d, want 404", code)
	}
	if code := req(t, h, "/nope").Code; code != http.StatusNotFound {
		t.Errorf("unknown path status = %d, want 404", code)
	}
}

// The shell names hashed assets, so a cached shell outlives what it points
// at.
func TestShellIsNotCached(t *testing.T) {
	if got := req(t, handler(built()), "/").Header().Get("Cache-Control"); got != "no-store" {
		t.Errorf("Cache-Control = %q, want no-store", got)
	}
}

// The embed must compile and resolve in every checkout, built or not.
func TestEmbedResolves(t *testing.T) {
	if Handler() == nil {
		t.Fatal("Handler() must never be nil")
	}
	// HasBuiltIndex is whichever the working tree is; it must not panic and
	// must agree with what the handler does.
	w := req(t, Handler(), "/")
	if w.Code != http.StatusOK {
		t.Errorf("/ status = %d, want 200 in either state", w.Code)
	}
	if HasBuiltIndex() && strings.Contains(w.Body.String(), "make fe-build") {
		t.Error("a built tree served the placeholder")
	}
}
