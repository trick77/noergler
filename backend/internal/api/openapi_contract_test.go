package api

import (
	"bytes"
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/getkin/kin-openapi/openapi3"
	"github.com/getkin/kin-openapi/openapi3filter"
	"github.com/getkin/kin-openapi/routers"
	"github.com/getkin/kin-openapi/routers/legacy"

	spec "github.com/trick77/noergler/api"
)

// The spec has to be valid before anything can be checked against it.
func TestOpenAPISpecIsValid(t *testing.T) {
	doc := loadSpec(t)
	if err := doc.Validate(context.Background()); err != nil {
		t.Fatalf("openapi.yaml is not a valid OpenAPI document: %v", err)
	}
}

// The spec is served as written, with a stable ETag, and answers 304 for it.
func TestSpecIsServedWithAnETag(t *testing.T) {
	h := newHarness(t, nil)

	w := h.do(t, http.MethodGet, "/api/openapi.yaml", "", "")
	if w.Code != http.StatusOK {
		t.Fatalf("status = %d, want 200", w.Code)
	}
	if !bytes.Equal(w.Body.Bytes(), spec.OpenAPISpec) {
		t.Error("served bytes differ from the embedded spec")
	}
	tag := w.Header().Get("ETag")
	if tag == "" {
		t.Fatal("no ETag")
	}

	r := newRequest(http.MethodGet, "/api/openapi.yaml", "")
	r.Header.Set("If-None-Match", tag)
	again := httptest.NewRecorder()
	h.srv.Handler().ServeHTTP(again, r)
	if again.Code != http.StatusNotModified {
		t.Errorf("conditional request = %d, want 304", again.Code)
	}
	if again.Body.Len() != 0 {
		t.Error("304 carried a body")
	}
}

func TestMatchesETag(t *testing.T) {
	for _, c := range []struct {
		header, tag string
		want        bool
	}{
		{`"abc"`, `"abc"`, true},
		{`*`, `"abc"`, true},
		{`W/"abc"`, `"abc"`, true},
		{`"other", "abc"`, `"abc"`, true},
		{`"other"`, `"abc"`, false},
		{``, `"abc"`, false},
	} {
		if got := matchesETag(c.header, c.tag); got != c.want {
			t.Errorf("matchesETag(%q, %q) = %v, want %v", c.header, c.tag, got, c.want)
		}
	}
}

// The docs page exists, is HTML, and points at the route that actually
// serves the spec. Nothing else checks that those two agree.
func TestDocsPageIsServedAndPointsAtTheSpec(t *testing.T) {
	h := newHarness(t, nil)
	w := h.do(t, http.MethodGet, "/api/docs", "", "")
	if w.Code != http.StatusOK {
		t.Fatalf("status = %d, want 200", w.Code)
	}
	if ct := w.Header().Get("Content-Type"); !strings.HasPrefix(ct, "text/html") {
		t.Errorf("content-type = %q", ct)
	}
	if !strings.Contains(w.Body.String(), "'/api/openapi.yaml'") {
		t.Error("the page does not point at /api/openapi.yaml")
	}
}

// The contract test proper: drive the REAL handler and validate every
// response against the spec. A route whose behaviour drifts from what
// openapi.yaml promises fails here rather than in a caller.
func TestResponsesMatchOpenAPISpec(t *testing.T) {
	doc := loadSpec(t)
	// httptest's host is not one of the documented servers, and a "/" server
	// makes every path relative to it. Dropping them matches on path alone.
	doc.Servers = nil
	router, err := legacy.NewRouter(doc)
	if err != nil {
		t.Fatalf("build router: %v", err)
	}

	t.Run("getTeam", func(t *testing.T) {
		h := newHarness(t, nil)
		w := h.do(t, http.MethodGet, "/teams/"+testSlug, bearer(), "")
		validate(t, router, http.MethodGet, "/teams/"+testSlug, w)
	})

	t.Run("getTeam unauthorized", func(t *testing.T) {
		h := newHarness(t, nil)
		w := h.do(t, http.MethodGet, "/teams/"+testSlug, "Bearer wrong", "")
		validate(t, router, http.MethodGet, "/teams/"+testSlug, w)
	})

	t.Run("getTeam unknown team", func(t *testing.T) {
		h := newHarness(t, nil)
		w := h.do(t, http.MethodGet, "/teams/nope", bearer(), "")
		if w.Code != http.StatusNotFound {
			t.Fatalf("status = %d, want 404", w.Code)
		}
		validate(t, router, http.MethodGet, "/teams/nope", w)
	})

	t.Run("putSettings", func(t *testing.T) {
		h := newHarness(t, nil)
		h.setStore(t, &fakeSettingsStore{})
		w := h.do(t, http.MethodPut, "/teams/"+testSlug+"/settings", bearer(),
			`{"ignore_authors":["renovate"],"exclude_repos":[]}`)
		if w.Code != http.StatusOK {
			t.Fatalf("status = %d: %s", w.Code, w.Body)
		}
		validate(t, router, http.MethodPut, "/teams/"+testSlug+"/settings", w)
	})

	t.Run("putSettings unknown field", func(t *testing.T) {
		h := newHarness(t, nil)
		h.setStore(t, &fakeSettingsStore{})
		w := h.do(t, http.MethodPut, "/teams/"+testSlug+"/settings", bearer(), `{"nope":1}`)
		if w.Code != http.StatusUnprocessableEntity {
			t.Fatalf("status = %d, want 422", w.Code)
		}
		validate(t, router, http.MethodPut, "/teams/"+testSlug+"/settings", w)
	})

	t.Run("putSettings over cap", func(t *testing.T) {
		h := newHarness(t, nil)
		h.setStore(t, &fakeSettingsStore{})
		big := `{"ignore_authors":["` + strings.Repeat("a", 1<<20) + `"]}`
		w := h.do(t, http.MethodPut, "/teams/"+testSlug+"/settings", bearer(), big)
		if w.Code != http.StatusRequestEntityTooLarge {
			t.Fatalf("status = %d, want 413", w.Code)
		}
		validate(t, router, http.MethodPut, "/teams/"+testSlug+"/settings", w)
	})

	// Both shapes of `rows` are exercised: the oneOf is only meaningful if
	// each branch is actually produced by the real handler.
	t.Run("onboard status rows", func(t *testing.T) {
		body := onboardBody(t, `{"action":"status"}`)
		if len(body.Rows) == 0 {
			t.Fatal("no rows, so the StatusRow schema was never exercised")
		}
		if _, ok := body.Rows[0]["bot_can_write"]; !ok {
			t.Errorf("row is not a StatusRow: %v", body.Rows[0])
		}
	})

	t.Run("onboard action rows", func(t *testing.T) {
		body := onboardBody(t, `{"action":"grant-bot","dry_run":true,"projects":[{"key":"PLAT"}]}`)
		if len(body.Rows) == 0 {
			t.Fatal("no rows, so the TargetResult schema was never exercised")
		}
		if _, ok := body.Rows[0]["status"]; !ok {
			t.Errorf("row is not a TargetResult: %v", body.Rows[0])
		}
	})
}

// onboardBody runs one /onboard request through the real handler, checks it
// against the spec, and returns the decoded body.
func onboardBody(t *testing.T, req string) struct {
	Rows []map[string]any `json:"rows"`
} {
	t.Helper()
	doc := loadSpec(t)
	doc.Servers = nil
	router, err := legacy.NewRouter(doc)
	if err != nil {
		t.Fatalf("build router: %v", err)
	}

	h := newHarness(t, nil)
	h.setOnboard(t, &fakeBB{hooks: []map[string]any{}}, &fakeClaims{}, "https://n.example.com")
	w := h.onboardPost(t, bearer(), "tok", req)
	if w.StatusCode != http.StatusOK {
		t.Fatalf("status = %d", w.StatusCode)
	}

	rec := httptest.NewRecorder()
	rec.Code = w.StatusCode
	for k, v := range w.Header {
		rec.Header()[k] = v
	}
	var buf bytes.Buffer
	if w.Body != nil {
		_, _ = buf.ReadFrom(w.Body)
	}
	rec.Body = &buf
	validate(t, router, http.MethodPost, "/onboard/"+testSlug, rec)

	var body struct {
		Rows []map[string]any `json:"rows"`
	}
	if err := json.Unmarshal(buf.Bytes(), &body); err != nil {
		t.Fatalf("decode: %v", err)
	}
	return body
}

// validate runs one recorded response through the spec, status included.
func validate(t *testing.T, router routers.Router, method, path string, w *httptest.ResponseRecorder) {
	t.Helper()
	req := httptest.NewRequest(method, path, nil)
	route, pathParams, err := router.FindRoute(req)
	if err != nil {
		t.Fatalf("%s %s: no route in the spec: %v", method, path, err)
	}

	input := &openapi3filter.ResponseValidationInput{
		RequestValidationInput: &openapi3filter.RequestValidationInput{
			Request:    req,
			PathParams: pathParams,
			Route:      route,
			Options: &openapi3filter.Options{
				// The responses are what is under test; the fake requests
				// here carry no credentials.
				AuthenticationFunc: openapi3filter.NoopAuthenticationFunc,
			},
		},
		Status: w.Code,
		Header: w.Header(),
		Options: &openapi3filter.Options{
			IncludeResponseStatus: true,
			AuthenticationFunc:    openapi3filter.NoopAuthenticationFunc,
		},
	}
	input.SetBodyBytes(w.Body.Bytes())

	if err := openapi3filter.ValidateResponse(context.Background(), input); err != nil {
		t.Errorf("%s %s -> %d does not match the spec: %v\nbody: %s",
			method, path, w.Code, err, w.Body)
	}
}

func loadSpec(t *testing.T) *openapi3.T {
	t.Helper()
	loader := openapi3.NewLoader()
	loader.IsExternalRefsAllowed = false
	doc, err := loader.LoadFromData(spec.OpenAPISpec)
	if err != nil {
		t.Fatalf("load openapi.yaml: %v", err)
	}
	return doc
}
