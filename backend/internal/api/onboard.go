package api

import (
	"errors"
	"fmt"
	"net/http"
	"strings"

	"github.com/trick77/noergler/internal/config"
	"github.com/trick77/noergler/internal/onboarding"
	"github.com/trick77/noergler/internal/store"
	"github.com/trick77/ship/kit/httpx"
	"github.com/trick77/ship/kit/logging"
)

// onboardRequest is the /onboard body. Decoded strictly: an unknown field is
// rejected rather than ignored, so an operator typo is not silently dropped.
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
	ctx := logging.With(r.Context(), "team", slug)

	rt, ok := d.teamAuth(ctx, w, r, slug)
	if !ok {
		return
	}
	if d.PublicURL == "" {
		httpx.WriteError(w, http.StatusServiceUnavailable,
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
		httpx.WriteError(w, http.StatusUnprocessableEntity,
			"action must be one of status, onboard, grant-bot, remove")
		return
	}
	if body.Projects != nil && action == "status" {
		httpx.WriteError(w, http.StatusBadRequest, "projects only with onboard, grant-bot or remove")
		return
	}
	if body.Projects != nil && body.Targets != nil {
		httpx.WriteError(w, http.StatusBadRequest, "projects and targets are exclusive")
		return
	}

	// The caller's own Bitbucket token, for this request only. Never logged,
	// never stored: it proves itself per target.
	token := strings.TrimSpace(r.Header.Get("X-Bitbucket-Token"))
	if token == "" {
		httpx.WriteError(w, http.StatusUnauthorized,
			"X-Bitbucket-Token: <your Bitbucket HTTP access token with project admin> required")
		return
	}
	admin := d.Bitbucket.WithToken(token)
	// An own bot that cannot be worked with would prove nothing and could
	// grant rights to a name its token does not belong to: say so instead
	// of reporting every target unreachable.
	if bb, blocked := rt.BitbucketBlocked(ctx); blocked {
		httpx.WriteError(w, http.StatusUnprocessableEntity, "the team's own Bitbucket account is "+string(bb.Status)+
			"; push a working token to PUT /teams/"+slug+"/credentials first")
		return
	}
	// The bot whose access is checked and granted: the team's own once it
	// pushed one, so grant-bot after a switch gives the new bot its rights.
	var bot onboarding.BotClient = d.Bitbucket
	if own, _ := rt.OwnBot(); own != nil {
		bot = own
	}

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
			httpx.WriteError(w, http.StatusUnprocessableEntity, err.Error())
			return
		}
		res, err := onboarding.ClaimAndOnboard(ctx, admin, bot, d.Claims,
			&team, scopes, caller, webhookURL, opts, d.Log)
		if err != nil {
			d.writeError(w, r.WithContext(ctx), err)
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
			httpx.WriteError(w, http.StatusUnprocessableEntity, err.Error())
			return
		}
		res, err := onboarding.RemoveAndUnclaim(ctx, admin, bot, d.Claims,
			&team, scopes, caller, webhookURL, opts, d.Log)
		if err != nil {
			d.writeError(w, r.WithContext(ctx), err)
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
			d.writeError(w, r.WithContext(ctx), err)
			return
		}
		if len(targets) == 0 {
			httpx.WriteError(w, http.StatusBadRequest,
				"no targets: claim a project first (projects in the body)")
			return
		}
		o := onboarding.New(admin, bot, &team, webhookURL, opts, d.Log)
		rows, results, text, healthy := onboarding.Run(ctx, o, onboarding.Action(action), targets)
		if rows != nil {
			resp["rows"] = rows
		} else {
			resp["rows"] = results
		}
		resp["text"] = text
		resp["healthy"] = healthy
		httpx.WriteJSON(w, http.StatusOK, resp)
		return
	}

	httpx.WriteJSON(w, http.StatusOK, resp)
}

func fill(resp map[string]any, results []onboarding.TargetResult, text string, healthy bool) {
	resp["rows"] = results
	resp["text"] = text
	resp["healthy"] = healthy
}

// scopesOf applies the project-scope rules to the JSON path: the key is
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
func (d Deps) writeError(w http.ResponseWriter, r *http.Request, err error) {
	var unknown *onboarding.UnknownTarget
	var noClaim *onboarding.NoClaim
	var upstream *onboarding.UpstreamError
	var conflict *store.ClaimConflict
	switch {
	case errors.As(err, &unknown):
		httpx.WriteError(w, http.StatusBadRequest, unknown.Error())
	case errors.As(err, &noClaim):
		httpx.WriteError(w, http.StatusBadRequest, noClaim.Error())
	case errors.As(err, &upstream):
		httpx.WriteError(w, http.StatusBadGateway, upstream.Error())
	case errors.As(err, &conflict):
		target := conflict.Project
		if conflict.Repo != nil {
			target += "/" + *conflict.Repo
		}
		// The one error body with more than the message: who holds the target.
		httpx.WriteJSON(w, http.StatusConflict, conflictBody{
			Error:    conflict.Error(),
			Conflict: conflictHolder{Target: target, Team: conflict.OtherTeam},
		})
	default:
		httpx.ServerError(w, r, fmt.Errorf("onboard: %w", err), "Internal Server Error")
	}
}

// conflictBody is the 409: the canonical {"error"} plus the holder.
type conflictBody struct {
	Error    string         `json:"error"`
	Conflict conflictHolder `json:"conflict"`
}

type conflictHolder struct {
	Target string `json:"target"`
	Team   string `json:"team"`
}
