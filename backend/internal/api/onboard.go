package api

import (
	"context"
	"errors"
	"net/http"
	"strings"

	"github.com/trick77/noergler-go/internal/config"
	"github.com/trick77/noergler-go/internal/httpapi"
	"github.com/trick77/noergler-go/internal/logging"
	"github.com/trick77/noergler-go/internal/onboarding"
	"github.com/trick77/noergler-go/internal/store"
)

// onboardRequest is Pydantic's OnboardRequest, extra="forbid".
type onboardRequest struct {
	Action   *string             `json:"action"`
	Projects *[]projectScopeView `json:"projects"`
	Targets  *[]string           `json:"targets"`
	DryRun   *bool               `json:"dry_run"`
	Name     *string             `json:"name"`
	Prune    *bool               `json:"prune"`
}

var validActions = map[string]bool{"status": true, "onboard": true, "grant-bot": true, "remove": true}

// onboard puts noergler's webhook on the team's projects and repos, and with
// `projects` in the body claims them first (or, with remove, gives them up
// along with every PR record on them).
//
// Authenticated by the team's webhook secret. The Bitbucket calls run on the
// caller's own token, used for this request and dropped: a claim needs
// project admin on the target, proven with that token.
func (d Deps) onboard(w http.ResponseWriter, r *http.Request) {
	slug := r.PathValue("team")
	ctx := logging.WithTeam(r.Context(), slug)

	rt, ok := d.teamAuth(ctx, w, slug, r.Header.Get("Authorization"))
	if !ok {
		return
	}
	if d.PublicURL == "" {
		httpapi.WriteDetail(w, http.StatusServiceUnavailable,
			"NOERGLER_PUBLIC_URL is not set on this instance; onboarding via API is disabled")
		return
	}

	var body onboardRequest
	if !decodeStrict(w, r, &body) {
		return
	}
	action := "status"
	if body.Action != nil {
		action = *body.Action
	}
	if !validActions[action] {
		httpapi.WriteDetail(w, http.StatusUnprocessableEntity,
			"action must be one of status, onboard, grant-bot, remove")
		return
	}
	if body.Projects != nil && action == "status" {
		httpapi.WriteDetail(w, http.StatusBadRequest, "projects only with onboard, grant-bot or remove")
		return
	}
	if body.Projects != nil && body.Targets != nil {
		httpapi.WriteDetail(w, http.StatusBadRequest, "projects and targets are exclusive")
		return
	}

	// The caller's own Bitbucket token, for this request only. Never logged,
	// never stored: it proves itself per target.
	token := strings.TrimSpace(r.Header.Get("X-Bitbucket-Token"))
	if token == "" {
		httpapi.WriteDetail(w, http.StatusUnauthorized,
			"X-Bitbucket-Token: <your Bitbucket HTTP access token with project admin> required")
		return
	}
	admin := d.Bitbucket.WithToken(token)

	opts := onboarding.Options{
		GrantBot: action == "grant-bot",
		DryRun:   body.DryRun != nil && *body.DryRun,
		NoPrune:  body.Prune != nil && !*body.Prune,
	}
	if body.Name != nil {
		opts.WebhookName = *body.Name
	}

	webhookURL := d.PublicURL + "/webhook/" + slug
	caller := "team:" + slug
	// A copy: the orchestrators mutate the team they are given, and the
	// snapshot the webhook route reads must never be written through.
	team := *rt.Team()

	resp := map[string]any{
		"team":        slug,
		"action":      action,
		"webhook_url": webhookURL,
	}

	switch {
	case body.Projects != nil && action != "remove":
		scopes, err := scopesOf(*body.Projects)
		if err != nil {
			httpapi.WriteDetail(w, http.StatusUnprocessableEntity, err.Error())
			return
		}
		res, err := onboarding.ClaimAndOnboard(ctx, admin, d.Bitbucket, d.Claims,
			&team, scopes, caller, webhookURL, opts, d.Log)
		if err != nil {
			d.writeError(ctx, w, err)
			return
		}
		if res.Claims != nil {
			rt.ApplyClaims(res.Claims)
		}
		resp["claimed"] = res.Claimed
		fill(resp, res.Results, res.Text, res.Healthy)

	case body.Projects != nil:
		scopes, err := scopesOf(*body.Projects)
		if err != nil {
			httpapi.WriteDetail(w, http.StatusUnprocessableEntity, err.Error())
			return
		}
		res, err := onboarding.RemoveAndUnclaim(ctx, admin, d.Bitbucket, d.Claims,
			&team, scopes, caller, webhookURL, opts, d.Log)
		if err != nil {
			d.writeError(ctx, w, err)
			return
		}
		if res.Claims != nil {
			rt.ApplyClaims(res.Claims)
		}
		resp["unclaimed"] = res.Unclaimed
		resp["purged_prs"] = res.PurgedPRs
		fill(resp, res.Results, res.Text, res.Healthy)

	default:
		var subset []string
		if body.Targets != nil {
			subset = *body.Targets
		}
		targets, err := onboarding.TargetsFor(&team, subset)
		if err != nil {
			d.writeError(ctx, w, err)
			return
		}
		if len(targets) == 0 {
			httpapi.WriteDetail(w, http.StatusBadRequest,
				"no targets: claim a project first (projects in the body)")
			return
		}
		o := onboarding.New(admin, d.Bitbucket, &team, webhookURL, opts, d.Log)
		rows, results, text, healthy := onboarding.Run(ctx, o, onboarding.Action(action), targets)
		if rows != nil {
			resp["rows"] = rows
		} else {
			resp["rows"] = results
		}
		resp["text"] = text
		resp["healthy"] = healthy
		httpapi.WriteJSON(w, http.StatusOK, resp)
		return
	}

	httpapi.WriteJSON(w, http.StatusOK, resp)
}

func fill(resp map[string]any, results []onboarding.TargetResult, text string, healthy bool) {
	resp["rows"] = results
	resp["text"] = text
	resp["healthy"] = healthy
}

// scopesOf applies Pydantic's ProjectScope rules to the JSON path: the key is
// trimmed and required, each repo is trimmed, blanks are dropped, and a
// present-but-empty repos list is refused.
func scopesOf(in []projectScopeView) ([]config.ProjectScope, error) {
	out := make([]config.ProjectScope, 0, len(in))
	for _, p := range in {
		key := strings.TrimSpace(p.Key)
		if key == "" {
			return nil, errors.New("projects[].key must not be empty")
		}
		scope := config.ProjectScope{Key: key}
		if p.Repos != nil {
			repos := []string{}
			for _, repo := range p.Repos {
				if repo = strings.TrimSpace(repo); repo != "" {
					repos = append(repos, repo)
				}
			}
			if len(repos) == 0 {
				return nil, errors.New("projects[].repos must not be empty when given")
			}
			scope.Repos = repos
		}
		out = append(out, scope)
	}
	return out, nil
}

// writeError maps a domain error to a status. The onboarding package knows
// nothing about HTTP; this is the only place that translation happens.
func (d Deps) writeError(ctx context.Context, w http.ResponseWriter, err error) {
	var unknown *onboarding.UnknownTarget
	var noClaim *onboarding.NoClaim
	var upstream *onboarding.UpstreamError
	var conflict *store.ClaimConflict
	switch {
	case errors.As(err, &unknown):
		httpapi.WriteDetail(w, http.StatusBadRequest, unknown.Error())
	case errors.As(err, &noClaim):
		httpapi.WriteDetail(w, http.StatusBadRequest, noClaim.Error())
	case errors.As(err, &upstream):
		httpapi.WriteDetail(w, http.StatusBadGateway, upstream.Error())
	case errors.As(err, &conflict):
		target := conflict.Project
		if conflict.Repo != nil {
			target += "/" + *conflict.Repo
		}
		// The one place the detail is an object rather than a string.
		httpapi.WriteJSON(w, http.StatusConflict, map[string]any{
			"detail": map[string]any{
				"message":  conflict.Error(),
				"conflict": map[string]any{"target": target, "team": conflict.OtherTeam},
			},
		})
	default:
		d.Log.ErrorContext(ctx, "onboard failed", "error", err)
		httpapi.WriteDetail(w, http.StatusInternalServerError, "Internal Server Error")
	}
}
