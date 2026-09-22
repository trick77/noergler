package api

import (
	"bytes"
	"context"
	"errors"
	"net/http"
	"net/http/httptest"
	"reflect"
	"strings"
	"testing"

	"github.com/trick77/noergler/internal/config"
	"github.com/trick77/noergler/internal/store"
)

// fakeSettingsStore records the write and can fail on demand.
type fakeSettingsStore struct {
	last store.TeamSettings
	by   string
	err  error
}

func (f *fakeSettingsStore) PutSettings(_ context.Context, _ string, t store.TeamSettings, by string) error {
	if f.err != nil {
		return f.err
	}
	f.last, f.by = t, by
	return nil
}

func (h *harness) do(t *testing.T, method, path, auth, body string) *httptest.ResponseRecorder {
	t.Helper()
	r := httptest.NewRequest(method, path, bytes.NewReader([]byte(body)))
	if auth != "" {
		r.Header.Set("Authorization", auth)
	}
	w := httptest.NewRecorder()
	h.srv.Handler().ServeHTTP(w, r)
	return w
}

func bearer() string { return "Bearer " + testSecret }

// 404 and 503 for the slug come before the 401: they leak nothing the
// webhook route does not, and they tell an operator which problem it is.
func TestTeams_SlugErrorsPrecedeAuth(t *testing.T) {
	h := newHarness(t, nil)

	if w := h.do(t, http.MethodGet, "/teams/nope", "", ""); w.Code != http.StatusNotFound {
		t.Errorf("unknown slug without auth = %d, want 404", w.Code)
	}
	if w := h.do(t, http.MethodGet, "/teams/payments", "", ""); w.Code != http.StatusServiceUnavailable {
		t.Errorf("disabled slug without auth = %d, want 503", w.Code)
	}
}

func TestTeams_BearerAuth(t *testing.T) {
	cases := []struct {
		name string
		auth string
		want int
	}{
		{"valid", bearer(), http.StatusOK},
		{"lowercase scheme", "bearer " + testSecret, http.StatusOK},
		{"surrounding space", "Bearer   " + testSecret + "  ", http.StatusOK},
		{"missing", "", http.StatusUnauthorized},
		{"wrong secret", "Bearer nope", http.StatusUnauthorized},
		{"empty secret", "Bearer ", http.StatusUnauthorized},
		{"wrong scheme", "Basic " + testSecret, http.StatusUnauthorized},
		{"bare secret", testSecret, http.StatusUnauthorized},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			h := newHarness(t, nil)
			w := h.do(t, http.MethodGet, "/teams/"+testSlug, c.auth, "")
			if w.Code != c.want {
				t.Errorf("status = %d, want %d (body %s)", w.Code, c.want, w.Body.String())
			}
		})
	}
}

// A whole-project scope serialises with no repos key at all. Empty lists stay
// [], never null.
func TestTeams_GetRendersTheSnapshot(t *testing.T) {
	h := newHarness(t, func(team *config.Team) {
		team.Projects = []config.ProjectScope{{Key: "PLAT"}, {Key: "OPS", Repos: []string{"a", "b"}}}
		team.Review.ExcludeRepos = []string{"*-infra"}
	})
	w := h.do(t, http.MethodGet, "/teams/"+testSlug, bearer(), "")
	wantBody(t, w, http.StatusOK, map[string]any{
		"team": testSlug,
		"projects": []any{
			map[string]any{"key": "PLAT"},
			map[string]any{"key": "OPS", "repos": []any{"a", "b"}},
		},
		"auto_review_authors": []any{},
		"ignore_authors":      []any{},
		"exclude_repos":       []any{"*-infra"},
	})
}

// The partial update: absent keeps, a list replaces, [] clears, and an
// explicit null behaves exactly like absent.
func TestTeams_PutIsAPartialUpdate(t *testing.T) {
	cases := []struct {
		name string
		body string
		want store.TeamSettings
	}{
		{"absent keeps everything", `{}`, store.TeamSettings{
			AutoReviewAuthors: []string{"alice"},
			IgnoreAuthors:     []string{"ci-bot"},
			ExcludeRepos:      []string{"*-infra"},
		}},
		{"explicit null keeps", `{"auto_review_authors":null}`, store.TeamSettings{
			AutoReviewAuthors: []string{"alice"},
			IgnoreAuthors:     []string{"ci-bot"},
			ExcludeRepos:      []string{"*-infra"},
		}},
		{"a list replaces", `{"auto_review_authors":["bob"]}`, store.TeamSettings{
			AutoReviewAuthors: []string{"bob"},
			IgnoreAuthors:     []string{"ci-bot"},
			ExcludeRepos:      []string{"*-infra"},
		}},
		{"empty list clears", `{"exclude_repos":[]}`, store.TeamSettings{
			AutoReviewAuthors: []string{"alice"},
			IgnoreAuthors:     []string{"ci-bot"},
			ExcludeRepos:      []string{},
		}},
		{"entries are trimmed and blanks dropped", `{"ignore_authors":["  bob  ","","   "]}`, store.TeamSettings{
			AutoReviewAuthors: []string{"alice"},
			IgnoreAuthors:     []string{"bob"},
			ExcludeRepos:      []string{"*-infra"},
		}},
		{"all three at once", `{"auto_review_authors":["x"],"ignore_authors":[],"exclude_repos":["*-test"]}`,
			store.TeamSettings{
				AutoReviewAuthors: []string{"x"},
				IgnoreAuthors:     []string{},
				ExcludeRepos:      []string{"*-test"},
			}},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			h := newHarness(t, func(team *config.Team) {
				team.Review.AutoReviewAuthors = []string{"alice"}
				team.Review.IgnoreAuthors = []string{"ci-bot"}
				team.Review.ExcludeRepos = []string{"*-infra"}
			})
			db := &fakeSettingsStore{}
			h.setStore(t, db)

			w := h.do(t, http.MethodPut, "/teams/"+testSlug+"/settings", bearer(), c.body)
			if w.Code != http.StatusOK {
				t.Fatalf("status = %d, want 200 (body %s)", w.Code, w.Body.String())
			}
			if !reflect.DeepEqual(db.last, c.want) {
				t.Errorf("stored = %+v, want %+v", db.last, c.want)
			}
			if db.by != "team:"+testSlug {
				t.Errorf("updated_by = %q, want the team", db.by)
			}
		})
	}
}

// The write takes effect immediately, on the snapshot and on the reviewer.
func TestTeams_PutAppliesToTheRuntime(t *testing.T) {
	h := newHarness(t, nil)
	h.setStore(t, &fakeSettingsStore{})

	w := h.do(t, http.MethodPut, "/teams/"+testSlug+"/settings", bearer(),
		`{"exclude_repos":["*-infra"]}`)
	if w.Code != http.StatusOK {
		t.Fatalf("status = %d, want 200", w.Code)
	}
	if h.rt.Team().ReviewsRepo("PLAT", "core-infra") {
		t.Error("the new exclude pattern did not reach the snapshot")
	}
	// And the response is the updated view, not the old one.
	if got := decodeBody(t, w)["exclude_repos"]; !reflect.DeepEqual(got, []any{"*-infra"}) {
		t.Errorf("exclude_repos = %v, want the new list", got)
	}
}

// extra="forbid": an unknown key is an operator typo, not something to
// shrug at.
func TestTeams_UnknownFieldIs422(t *testing.T) {
	h := newHarness(t, nil)
	db := &fakeSettingsStore{}
	h.setStore(t, db)

	w := h.do(t, http.MethodPut, "/teams/"+testSlug+"/settings", bearer(), `{"nope":1}`)
	if w.Code != http.StatusUnprocessableEntity {
		t.Errorf("status = %d, want 422 (body %s)", w.Code, w.Body.String())
	}
	if db.by != "" {
		t.Error("a rejected body must not be written")
	}
}

func TestTeams_MalformedBodyIs422(t *testing.T) {
	for _, body := range []string{``, `not json`, `[]`, `{} {}`} {
		h := newHarness(t, nil)
		h.setStore(t, &fakeSettingsStore{})
		w := h.do(t, http.MethodPut, "/teams/"+testSlug+"/settings", bearer(), body)
		if w.Code != http.StatusUnprocessableEntity {
			t.Errorf("body %q = %d, want 422", body, w.Code)
		}
	}
}

// A failed write must not leave the in-memory copy ahead of the DB.
func TestTeams_StoreFailureDoesNotApply(t *testing.T) {
	h := newHarness(t, nil)
	h.setStore(t, &fakeSettingsStore{err: errors.New("connection refused")})

	w := h.do(t, http.MethodPut, "/teams/"+testSlug+"/settings", bearer(),
		`{"exclude_repos":["*-infra"]}`)
	if w.Code != http.StatusServiceUnavailable {
		t.Errorf("status = %d, want 503", w.Code)
	}
	if !h.rt.Team().ReviewsRepo("PLAT", "core-infra") {
		t.Error("a failed write must not change the snapshot")
	}
}

// Auth runs before the body is decoded, so an unauthenticated caller cannot
// probe the schema: a bad body behind a bad secret answers 401, never 422.
// An over-cap body is 413, not a 422 carrying Go's own "http: request body
// too large". The webhook route has always answered 413; these did not, so
// the status said "your JSON is malformed" and the detail leaked an
// internal string.
func TestTeams_OverCapBodyIs413(t *testing.T) {
	h := newHarness(t, nil)
	h.setStore(t, &fakeSettingsStore{})
	big := `{"auto_review_authors":["` + strings.Repeat("a", 1<<20) + `"]}`

	// The settings route: /onboard shares decodeStrict but answers 503 here
	// first, because this harness sets no PublicURL and that guard runs
	// before the body is read.
	w := h.do(t, http.MethodPut, "/teams/"+testSlug+"/settings", "Bearer "+testSecret, big)
	if w.Code != http.StatusRequestEntityTooLarge {
		t.Errorf("status = %d, want 413", w.Code)
	}
	if strings.Contains(w.Body.String(), "request body too large") {
		t.Errorf("detail leaks the Go error: %s", w.Body)
	}
}

func TestTeams_AuthPrecedesBodyValidation(t *testing.T) {
	h := newHarness(t, nil)
	h.setStore(t, &fakeSettingsStore{})
	w := h.do(t, http.MethodPut, "/teams/"+testSlug+"/settings", "Bearer wrong", `{"nope":1}`)
	if w.Code != http.StatusUnauthorized {
		t.Errorf("status = %d, want 401", w.Code)
	}
}
