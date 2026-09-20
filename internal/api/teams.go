package api

import (
	"context"
	"crypto/subtle"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"strings"

	"github.com/trick77/noergler-go/internal/config"
	"github.com/trick77/noergler-go/internal/httpapi"
	"github.com/trick77/noergler-go/internal/logging"
	"github.com/trick77/noergler-go/internal/store"
	"github.com/trick77/noergler-go/internal/teams"
)

// maxAPIBodyBytes bounds the authenticated request bodies.
const maxAPIBodyBytes = 1 << 20

// SettingsStore is the persistence the team API needs.
type SettingsStore interface {
	PutSettings(ctx context.Context, teamSlug string, t store.TeamSettings, updatedBy string) error
}

// teamAuth resolves the slug and checks the bearer token.
//
// The team's webhook secret is its credential everywhere: Bitbucket signs
// events with it, the team admin authenticates API calls with it. 404 and
// 503 for the slug come first; they leak nothing the webhook route does not.
func (d Deps) teamAuth(w http.ResponseWriter, ctx context.Context, slug, authorization string) (*teams.Runtime, bool) {
	rt, ok := d.runtimeFor(w, ctx, slug)
	if !ok {
		return nil, false
	}
	if !strings.HasPrefix(strings.ToLower(authorization), "bearer ") {
		httpapi.WriteDetail(w, http.StatusUnauthorized,
			"Authorization: Bearer <the team's webhook secret> required")
		return nil, false
	}
	secret := strings.TrimSpace(authorization[len("bearer "):])
	want := rt.Team().WebhookSecret
	if secret == "" || subtle.ConstantTimeCompare([]byte(secret), []byte(want)) != 1 {
		httpapi.WriteDetail(w, http.StatusUnauthorized, "wrong team secret")
		return nil, false
	}
	return rt, true
}

// projectScopeView mirrors Pydantic's exclude_none: a whole-project scope
// serialises as {"key": "PROJ"} with no repos field at all, not null.
type projectScopeView struct {
	Key   string   `json:"key"`
	Repos []string `json:"repos,omitempty"`
}

type teamView struct {
	Team              string             `json:"team"`
	Projects          []projectScopeView `json:"projects"`
	AutoReviewAuthors []string           `json:"auto_review_authors"`
	IgnoreAuthors     []string           `json:"ignore_authors"`
	ExcludeRepos      []string           `json:"exclude_repos"`
}

func viewOf(t *config.Team) teamView {
	projects := make([]projectScopeView, 0, len(t.Projects))
	for _, p := range t.Projects {
		projects = append(projects, projectScopeView{Key: p.Key, Repos: p.Repos})
	}
	return teamView{
		Team:              t.Slug,
		Projects:          projects,
		AutoReviewAuthors: nonNil(t.Review.AutoReviewAuthors),
		IgnoreAuthors:     nonNil(t.Review.IgnoreAuthors),
		ExcludeRepos:      nonNil(t.Review.ExcludeRepos),
	}
}

// nonNil keeps an empty list rendering as [] rather than null, matching
// Python, where these are always lists.
func nonNil(s []string) []string {
	if s == nil {
		return []string{}
	}
	return s
}

// getTeam answers the team's own settings. Read entirely from the snapshot;
// it never touches the DB.
func (d Deps) getTeam(w http.ResponseWriter, r *http.Request) {
	slug := r.PathValue("team")
	ctx := logging.WithTeam(r.Context(), slug)
	rt, ok := d.teamAuth(w, ctx, slug, r.Header.Get("Authorization"))
	if !ok {
		return
	}
	httpapi.WriteJSON(w, http.StatusOK, viewOf(rt.Team()))
}

// settingsRequest is a PARTIAL update.
//
// Pointers, not slices, because a field left out must stay as it is while an
// empty list clears it. Probed against the venv: Python's _clean treats an
// explicit null exactly like an absent field, so a nil pointer covers both.
type settingsRequest struct {
	AutoReviewAuthors *[]string `json:"auto_review_authors"`
	IgnoreAuthors     *[]string `json:"ignore_authors"`
	ExcludeRepos      *[]string `json:"exclude_repos"`
}

// clean is Python's _clean: nil keeps the current list, otherwise each entry
// is trimmed and the blanks are dropped.
func clean(items *[]string, current []string) []string {
	if items == nil {
		return current
	}
	out := []string{}
	for _, s := range *items {
		if s = strings.TrimSpace(s); s != "" {
			out = append(out, s)
		}
	}
	return out
}

// putTeamSettings updates the team's three lists. Takes effect immediately.
func (d Deps) putTeamSettings(w http.ResponseWriter, r *http.Request) {
	slug := r.PathValue("team")
	ctx := logging.WithTeam(r.Context(), slug)
	// Auth before the body is decoded, so an unauthenticated caller cannot
	// probe the schema. FastAPI validates the body first and would answer 422
	// ahead of the 401; no test pins that ordering.
	rt, ok := d.teamAuth(w, ctx, slug, r.Header.Get("Authorization"))
	if !ok {
		return
	}

	var body settingsRequest
	if !decodeStrict(w, r, &body) {
		return
	}

	current := rt.Settings()
	next := store.TeamSettings{
		AutoReviewAuthors: clean(body.AutoReviewAuthors, current.AutoReviewAuthors),
		IgnoreAuthors:     clean(body.IgnoreAuthors, current.IgnoreAuthors),
		ExcludeRepos:      clean(body.ExcludeRepos, current.ExcludeRepos),
	}
	if err := d.Store.PutSettings(ctx, slug, next, "team:"+slug); err != nil {
		d.Log.ErrorContext(ctx, "settings update failed", "error", err)
		httpapi.WriteDetail(w, http.StatusServiceUnavailable, "database not ready")
		return
	}
	rt.ApplySettings(next)
	httpapi.WriteJSON(w, http.StatusOK, viewOf(rt.Team()))
}

// decodeStrict rejects an unknown field with a 422.
//
// Both request models are Pydantic extra="forbid", where an unknown key is
// an operator typo rather than something to shrug at, and FastAPI answers
// 422. The detail is a plain string; Python's is a list of error objects,
// which nothing parses.
func decodeStrict(w http.ResponseWriter, r *http.Request, into any) bool {
	// Behind auth, so the cap is only a backstop, but an authenticated caller
	// is still not a reason to read without a bound.
	dec := json.NewDecoder(http.MaxBytesReader(w, r.Body, maxAPIBodyBytes))
	dec.DisallowUnknownFields()
	if err := dec.Decode(into); err != nil {
		httpapi.WriteDetail(w, http.StatusUnprocessableEntity, decodeDetail(err))
		return false
	}
	// A second value in the stream is as malformed as a bad first one.
	if err := dec.Decode(new(json.RawMessage)); !errors.Is(err, io.EOF) {
		httpapi.WriteDetail(w, http.StatusUnprocessableEntity, "body must be a single JSON object")
		return false
	}
	return true
}

func decodeDetail(err error) string {
	if errors.Is(err, io.EOF) {
		return "body must be a JSON object"
	}
	return err.Error()
}
