package api

import (
	"errors"
	"net/http"
	"strings"

	"github.com/trick77/noergler/internal/teams"
	"github.com/trick77/ship/kit/httpx"
	"github.com/trick77/ship/kit/logging"
)

// credentialsRequest is a PARTIAL update: a block left out (or null) keeps
// that account as it is. A token is write-only: no response ever carries it.
type credentialsRequest struct {
	Bitbucket *bitbucketCredentials `json:"bitbucket"`
	Jira      *jiraCredentials      `json:"jira"`
}

type bitbucketCredentials struct {
	Username string `json:"username"`
	Token    string `json:"token"`
}

type jiraCredentials struct {
	Token string `json:"token"`
}

// credentialsResponse is the team view, plus after a Bitbucket push the
// claimed targets the new bot cannot read yet: what grant-bot still has to
// fix.
type credentialsResponse struct {
	teamView
	UnreachableClaims []string `json:"unreachable_claims,omitempty"`
}

// putTeamCredentials pushes the team's own Bitbucket and/or Jira account.
//
// Every part is proven against Bitbucket or Jira before anything is stored;
// a refused part changes nothing. Takes effect immediately. The tokens are
// never logged, never echoed, and stored sealed (internal/secretbox).
func (d Deps) putTeamCredentials(w http.ResponseWriter, r *http.Request) {
	// Auth before the body is decoded, as for settings.
	rt, slug, ctx, ok := d.teamAuth(w, r)
	if !ok {
		return
	}

	var body credentialsRequest
	if !decodeStrict(w, r, &body) {
		return
	}
	if body.Bitbucket == nil && body.Jira == nil {
		httpx.WriteError(w, http.StatusUnprocessableEntity, "nothing to push: give bitbucket, jira or both")
		return
	}
	var bb *teams.BitbucketPush
	if body.Bitbucket != nil {
		user, token := strings.TrimSpace(body.Bitbucket.Username), strings.TrimSpace(body.Bitbucket.Token)
		if user == "" || token == "" {
			httpx.WriteError(w, http.StatusUnprocessableEntity,
				"bitbucket needs both username (the bot's, which the team then @mentions) and token")
			return
		}
		bb = &teams.BitbucketPush{Username: user, Token: token}
	}
	var jiraToken *string
	if body.Jira != nil {
		token := strings.TrimSpace(body.Jira.Token)
		if token == "" {
			httpx.WriteError(w, http.StatusUnprocessableEntity, "jira needs a token")
			return
		}
		jiraToken = &token
	}

	if err := rt.PushCredentials(ctx, bb, jiraToken, "team:"+slug); err != nil {
		var pe *teams.PushError
		if errors.As(err, &pe) {
			// The detail is for the caller, who holds the token; the log
			// gets the fixed status only.
			d.Log.WarnContext(ctx, "credentials push refused", "kind", string(pe.Kind), "status", string(pe.Status))
			code := http.StatusUnprocessableEntity
			if pe.Status == teams.StatusUnreachable {
				code = http.StatusBadGateway
			}
			httpx.WriteError(w, code, pe.Error())
			return
		}
		d.Log.ErrorContext(ctx, "credentials push failed", logging.Err(err))
		httpx.WriteError(w, http.StatusServiceUnavailable, errDBNotReady)
		return
	}

	resp := credentialsResponse{teamView: viewOf(rt.Team(), rt, d.BotUsername)}
	if bb != nil {
		resp.UnreachableClaims = rt.UnreachableClaims(ctx)
	}
	httpx.WriteJSON(w, http.StatusOK, resp)
}

// deleteTeamCredentials goes back to the shared account for one kind.
func (d Deps) deleteTeamCredentials(w http.ResponseWriter, r *http.Request) {
	rt, slug, ctx, ok := d.teamAuth(w, r)
	if !ok {
		return
	}
	kind, ok := teams.ParseKind(r.PathValue("kind"))
	if !ok {
		httpx.WriteError(w, http.StatusNotFound, "unknown credential kind: bitbucket or jira")
		return
	}
	removed, err := rt.ResetCredentials(ctx, kind, "team:"+slug)
	if err != nil {
		d.Log.ErrorContext(ctx, "credentials reset failed", logging.Err(err))
		httpx.WriteError(w, http.StatusServiceUnavailable, errDBNotReady)
		return
	}
	if !removed {
		httpx.WriteError(w, http.StatusNotFound, "the team has no own "+string(kind)+" account")
		return
	}
	httpx.WriteJSON(w, http.StatusOK, viewOf(rt.Team(), rt, d.BotUsername))
}
