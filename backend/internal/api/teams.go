package api

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"strings"
	"time"

	"github.com/trick77/noergler/internal/config"
	"github.com/trick77/noergler/internal/store"
	"github.com/trick77/noergler/internal/teams"
	"github.com/trick77/ship/kit/httpx"
	"github.com/trick77/ship/kit/logging"
	"github.com/trick77/ship/kit/secret"
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
func (d Deps) teamAuth(w http.ResponseWriter, r *http.Request) (rt *teams.Runtime, slug string, ctx context.Context, ok bool) {
	slug = r.PathValue("team")
	ctx = logging.With(r.Context(), "team", slug)
	rt, ok = d.runtimeFor(ctx, w, slug)
	if !ok {
		return nil, slug, ctx, false
	}
	presented, ok := httpx.Bearer(r)
	if !ok {
		httpx.WriteError(w, http.StatusUnauthorized,
			"Authorization: Bearer <the team's webhook secret> required")
		return nil, slug, ctx, false
	}
	if !secret.Match(presented, rt.Team().WebhookSecret) {
		httpx.WriteError(w, http.StatusUnauthorized, "wrong team secret")
		return nil, slug, ctx, false
	}
	return rt, slug, ctx, true
}

// projectScopeView omits a null field rather than emitting it: a
// whole-project scope serialises as {"key": "PROJ"} with no repos key at all.
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
	// Branch globs; a match skips auto-review only, a mention still works.
	ExcludeSourceBranches []string `json:"exclude_source_branches"`
	ExcludeTargetBranches []string `json:"exclude_target_branches"`
	UsePRDescription      bool     `json:"use_pr_description"`
	// Which accounts the team works with: its own (pushed) or the shared
	// ones. Never a token.
	BitbucketAccount bitbucketAccountView `json:"bitbucket_account"`
	JiraAccount      jiraAccountView      `json:"jira_account"`
}

// bitbucketAccountView is the team's Bitbucket account. Username is the
// handle the team mentions: its own bot's, or the shared bot's. Status and
// since are an own account's health, null on the shared one.
type bitbucketAccountView struct {
	Source   string  `json:"source"`
	Username string  `json:"username"`
	Status   *string `json:"status"`
	Since    *string `json:"since"`
}

type jiraAccountView struct {
	Source string  `json:"source"`
	Status *string `json:"status"`
	Since  *string `json:"since"`
}

// accountState renders an account's source and health. The status is the
// fixed set of teams.Status, never a reason: this reaches the
// unauthenticated dashboard too.
func accountState(a teams.Account) (source string, status, since *string) {
	if !a.Own {
		return "shared", nil, nil
	}
	s, t := string(a.Status), a.Since.UTC().Format(time.RFC3339)
	return "team", &s, &t
}

func accountViews(rt *teams.Runtime, sharedBot string) (bitbucketAccountView, jiraAccountView) {
	bb, jr := rt.Accounts()
	var bv bitbucketAccountView
	bv.Source, bv.Status, bv.Since = accountState(bb)
	bv.Username = sharedBot
	if bb.Own {
		bv.Username = bb.Username
	}
	var jv jiraAccountView
	jv.Source, jv.Status, jv.Since = accountState(jr)
	return bv, jv
}

// viewOf renders one snapshot of the team. rt is read for the accounts only;
// the caller took t from it once.
func viewOf(t *config.Team, rt *teams.Runtime, sharedBot string) teamView {
	projects := make([]projectScopeView, 0, len(t.Projects))
	for _, p := range t.Projects {
		projects = append(projects, projectScopeView{Key: p.Key, Repos: p.Repos})
	}
	bv, jv := accountViews(rt, sharedBot)
	return teamView{
		BitbucketAccount:      bv,
		JiraAccount:           jv,
		Team:                  t.Slug,
		Projects:              projects,
		AutoReviewAuthors:     nonNil(t.Review.AutoReviewAuthors),
		IgnoreAuthors:         nonNil(t.Review.IgnoreAuthors),
		ExcludeRepos:          nonNil(t.Review.ExcludeRepos),
		ExcludeSourceBranches: nonNil(t.Review.ExcludeSourceBranches),
		ExcludeTargetBranches: nonNil(t.Review.ExcludeTargetBranches),
		UsePRDescription:      t.Review.UsePRDescription,
	}
}

// nonNil keeps an empty list rendering as [] rather than null: these three
// fields are always lists on the wire.
func nonNil(s []string) []string {
	if s == nil {
		return []string{}
	}
	return s
}

// getTeam answers the team's own settings. Read entirely from the snapshot;
// it never touches the DB.
func (d Deps) getTeam(w http.ResponseWriter, r *http.Request) {
	rt, _, _, ok := d.teamAuth(w, r)
	if !ok {
		return
	}
	httpx.WriteJSON(w, http.StatusOK, viewOf(rt.Team(), rt, d.BotUsername))
}

// settingsRequest is a PARTIAL update.
//
// Pointers, not slices, because a field left out must stay as it is while an
// empty list clears it (and false switches the PR description off). An explicit null behaves exactly like an absent
// field, so a nil pointer covers both; pinned by
// TestTeams_PutIsAPartialUpdate.
type settingsRequest struct {
	AutoReviewAuthors     *[]string `json:"auto_review_authors"`
	IgnoreAuthors         *[]string `json:"ignore_authors"`
	ExcludeRepos          *[]string `json:"exclude_repos"`
	ExcludeSourceBranches *[]string `json:"exclude_source_branches"`
	ExcludeTargetBranches *[]string `json:"exclude_target_branches"`
	UsePRDescription      *bool     `json:"use_pr_description"`
}

// clean resolves one partial-update field: nil keeps the current list,
// otherwise each entry is trimmed and the blanks are dropped.
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

// keep resolves a partial-update switch: nil keeps the current value.
func keep(v *bool, current bool) bool {
	if v == nil {
		return current
	}
	return *v
}

// putTeamSettings updates the team's lists and the PR description switch. Takes effect immediately.
func (d Deps) putTeamSettings(w http.ResponseWriter, r *http.Request) {
	// Auth before the body is decoded, so an unauthenticated caller cannot
	// probe the schema: a bad body behind a bad secret answers 401, never 422.
	// Pinned by TestTeams_AuthPrecedesBodyValidation.
	rt, slug, ctx, ok := d.teamAuth(w, r)
	if !ok {
		return
	}

	var body settingsRequest
	if !decodeStrict(w, r, &body) {
		return
	}

	err := rt.UpdateSettings(func(current store.TeamSettings) store.TeamSettings {
		return store.TeamSettings{
			AutoReviewAuthors:     clean(body.AutoReviewAuthors, current.AutoReviewAuthors),
			IgnoreAuthors:         clean(body.IgnoreAuthors, current.IgnoreAuthors),
			ExcludeRepos:          clean(body.ExcludeRepos, current.ExcludeRepos),
			ExcludeSourceBranches: clean(body.ExcludeSourceBranches, current.ExcludeSourceBranches),
			ExcludeTargetBranches: clean(body.ExcludeTargetBranches, current.ExcludeTargetBranches),
			UsePRDescription:      keep(body.UsePRDescription, current.UsePRDescription),
		}
	}, func(next store.TeamSettings) error {
		return d.Store.PutSettings(ctx, slug, next, "team:"+slug)
	})
	if err != nil {
		d.Log.ErrorContext(ctx, "settings update failed", logging.Err(err))
		httpx.WriteError(w, http.StatusServiceUnavailable, errDBNotReady)
		return
	}
	httpx.WriteJSON(w, http.StatusOK, viewOf(rt.Team(), rt, d.BotUsername))
}

// decodeStrict rejects an unknown field with a 422.
//
// An unknown key is an operator typo rather than something to shrug at, so
// both request models reject it instead of ignoring it. The detail is a plain
// string, not a list of error objects: nothing parses it.
func decodeStrict(w http.ResponseWriter, r *http.Request, into any) bool {
	// Behind auth, so the cap is only a backstop, but an authenticated caller
	// is still not a reason to read without a bound.
	dec := json.NewDecoder(http.MaxBytesReader(w, r.Body, maxAPIBodyBytes))
	dec.DisallowUnknownFields()
	if err := dec.Decode(into); err != nil {
		// An over-cap body is 413, the same as the webhook route answers.
		// Folded into the 422 it reported Go's own "http: request body too
		// large" as if the JSON had been malformed, which is both the wrong
		// status and an internal string on the wire.
		var tooLarge *http.MaxBytesError
		if errors.As(err, &tooLarge) {
			httpx.WriteError(w, http.StatusRequestEntityTooLarge, "payload too large")
			return false
		}
		httpx.WriteError(w, http.StatusUnprocessableEntity, decodeDetail(err))
		return false
	}
	// A second value in the stream is as malformed as a bad first one.
	if err := dec.Decode(new(json.RawMessage)); !errors.Is(err, io.EOF) {
		httpx.WriteError(w, http.StatusUnprocessableEntity, "body must be a single JSON object")
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
