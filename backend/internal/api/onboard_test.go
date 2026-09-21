package api

import (
	"context"
	"errors"
	"net/http"
	"testing"

	"github.com/trick77/noergler-go/internal/bitbucket"
	"github.com/trick77/noergler-go/internal/config"
	"github.com/trick77/noergler-go/internal/onboarding"
	"github.com/trick77/noergler-go/internal/store"
)

// fakeBB stands in for both the bot client and the per-request admin clone.
type fakeBB struct {
	listErr    error
	lastToken  string
	hooks      []map[string]any
	grantCalls int
}

func (f *fakeBB) GetProject(context.Context, string) (map[string]any, error) {
	return map[string]any{"key": "PLAT"}, nil
}
func (f *fakeBB) GetRepo(context.Context, string, string) (map[string]any, error) {
	return map[string]any{"slug": "svc"}, nil
}
func (f *fakeBB) BotUsername() string { return testBot }

func (f *fakeBB) WithToken(token string) onboarding.AdminClient {
	f.lastToken = token
	return f
}

func (f *fakeBB) ListWebhooks(context.Context, string, string) ([]map[string]any, error) {
	if f.listErr != nil {
		return nil, f.listErr
	}
	return f.hooks, nil
}
func (f *fakeBB) CreateWebhook(_ context.Context, _, _ string, b bitbucket.Webhook) (*bitbucket.Webhook, error) {
	b.ID = 1
	return &b, nil
}
func (f *fakeBB) UpdateWebhook(_ context.Context, _, _ string, id int, b bitbucket.Webhook) (*bitbucket.Webhook, error) {
	b.ID = id
	return &b, nil
}
func (f *fakeBB) DeleteWebhook(context.Context, string, string, int) error { return nil }
func (f *fakeBB) GrantUserPermission(context.Context, string, string, string, string) error {
	f.grantCalls++
	return nil
}
func (f *fakeBB) ListRepos(context.Context, string) ([]map[string]any, error) { return nil, nil }

// fakeClaims records claim writes.
type fakeClaims struct {
	addErr  error
	added   []string
	removed []string
	claims  []config.ProjectScope
}

func (f *fakeClaims) AddClaims(_ context.Context, _ string, scopes []config.ProjectScope, _ string) ([]string, error) {
	if f.addErr != nil {
		return nil, f.addErr
	}
	for _, s := range scopes {
		f.added = append(f.added, s.Key)
	}
	return f.added, nil
}
func (f *fakeClaims) RemoveClaims(_ context.Context, _ string, scopes []config.ProjectScope) ([]string, error) {
	for _, s := range scopes {
		f.removed = append(f.removed, s.Key)
	}
	return f.removed, nil
}
func (f *fakeClaims) ListClaims(context.Context, string) ([]config.ProjectScope, error) {
	return f.claims, nil
}
func (f *fakeClaims) PurgeProject(context.Context, string, string, *string) (int, error) {
	return 3, nil
}
func (f *fakeClaims) CountProjectPRs(context.Context, string, string, *string) (int, error) {
	return 3, nil
}

// setOnboard rebuilds the server with the onboarding dependencies.
func (h *harness) setOnboard(t *testing.T, bb *fakeBB, cl *fakeClaims, publicURL string) {
	t.Helper()
	h.srv = newServer(h)
	Register(h.srv, Deps{
		Teams: h.reg, Queue: h.q, Log: h.log, BotUsername: testBot,
		Store: &fakeSettingsStore{}, Claims: cl, Bitbucket: bb, PublicURL: publicURL,
	})
}

func (h *harness) onboardPost(t *testing.T, auth, token, body string) *http.Response {
	t.Helper()
	r := newRequest(http.MethodPost, "/onboard/"+testSlug, body)
	if auth != "" {
		r.Header.Set("Authorization", auth)
	}
	if token != "" {
		r.Header.Set("X-Bitbucket-Token", token)
	}
	return h.serve(r)
}

func TestOnboard_RequiresPublicURL(t *testing.T) {
	h := newHarness(t, nil)
	h.setOnboard(t, &fakeBB{}, &fakeClaims{}, "")
	w := h.onboardPost(t, bearer(), "tok", `{}`)
	if w.StatusCode != http.StatusServiceUnavailable {
		t.Errorf("status = %d, want 503", w.StatusCode)
	}
}

func TestOnboard_RequiresTheCallersBitbucketToken(t *testing.T) {
	h := newHarness(t, nil)
	h.setOnboard(t, &fakeBB{}, &fakeClaims{}, "https://n.example.com")
	w := h.onboardPost(t, bearer(), "", `{}`)
	if w.StatusCode != http.StatusUnauthorized {
		t.Errorf("status = %d, want 401", w.StatusCode)
	}
}

// The caller's token is used for this request and never the bot's.
func TestOnboard_UsesTheCallersTokenForAdminCalls(t *testing.T) {
	h := newHarness(t, nil)
	bb := &fakeBB{}
	h.setOnboard(t, bb, &fakeClaims{}, "https://n.example.com")
	h.onboardPost(t, bearer(), "caller-token", `{"action":"status"}`)
	if bb.lastToken != "caller-token" {
		t.Errorf("admin token = %q, want the caller's", bb.lastToken)
	}
}

func TestOnboard_BodyValidation(t *testing.T) {
	cases := []struct {
		name string
		body string
		want int
	}{
		{"unknown field", `{"nope":1}`, http.StatusUnprocessableEntity},
		{"bad action", `{"action":"explode"}`, http.StatusUnprocessableEntity},
		{"projects with status", `{"projects":[{"key":"A"}]}`, http.StatusBadRequest},
		{"projects and targets", `{"action":"onboard","projects":[{"key":"A"}],"targets":["A"]}`, http.StatusBadRequest},
		{"blank project key", `{"action":"onboard","projects":[{"key":"  "}]}`, http.StatusUnprocessableEntity},
		{"empty repos list", `{"action":"onboard","projects":[{"key":"A","repos":[]}]}`, http.StatusUnprocessableEntity},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			h := newHarness(t, nil)
			h.setOnboard(t, &fakeBB{}, &fakeClaims{}, "https://n.example.com")
			w := h.onboardPost(t, bearer(), "tok", c.body)
			if w.StatusCode != c.want {
				t.Errorf("status = %d, want %d", w.StatusCode, c.want)
			}
		})
	}
}

// A team with no claims has nothing to act on.
func TestOnboard_NoTargetsIs400(t *testing.T) {
	h := newHarness(t, func(team *config.Team) { team.Projects = nil })
	h.setOnboard(t, &fakeBB{}, &fakeClaims{}, "https://n.example.com")
	w := h.onboardPost(t, bearer(), "tok", `{"action":"status"}`)
	if w.StatusCode != http.StatusBadRequest {
		t.Errorf("status = %d, want 400", w.StatusCode)
	}
}

// A target the team does not hold is a 400 naming what it does hold.
func TestOnboard_UnknownTargetIs400(t *testing.T) {
	h := newHarness(t, nil)
	h.setOnboard(t, &fakeBB{}, &fakeClaims{}, "https://n.example.com")
	w := h.onboardPost(t, bearer(), "tok", `{"action":"status","targets":["NOPE"]}`)
	if w.StatusCode != http.StatusBadRequest {
		t.Errorf("status = %d, want 400 (body %s)", w.StatusCode, h.body(w))
	}
}

// Bitbucket answering with anything but 401/403 is not the caller's fault:
// the whole request aborts with a 502 rather than recording one failed row.
func TestOnboard_UpstreamFailureIs502(t *testing.T) {
	h := newHarness(t, nil)
	bb := &fakeBB{listErr: &bitbucket.StatusError{Status: http.StatusInternalServerError}}
	h.setOnboard(t, bb, &fakeClaims{}, "https://n.example.com")

	w := h.onboardPost(t, bearer(), "tok", `{"action":"onboard","projects":[{"key":"PLAT"}]}`)
	if w.StatusCode != http.StatusBadGateway {
		t.Errorf("status = %d, want 502 (body %s)", w.StatusCode, h.body(w))
	}
}

// A 403 from Bitbucket means the caller lacks admin: one failed row, not a
// 502, and nothing is claimed.
func TestOnboard_NoAdminIsAFailedRowNotAnError(t *testing.T) {
	h := newHarness(t, nil)
	bb := &fakeBB{listErr: &bitbucket.StatusError{Status: http.StatusForbidden}}
	cl := &fakeClaims{}
	h.setOnboard(t, bb, cl, "https://n.example.com")

	w := h.onboardPost(t, bearer(), "tok", `{"action":"onboard","projects":[{"key":"PLAT"}]}`)
	if w.StatusCode != http.StatusOK {
		t.Fatalf("status = %d, want 200 (body %s)", w.StatusCode, h.body(w))
	}
	if len(cl.added) != 0 {
		t.Errorf("claimed %v without admin", cl.added)
	}
}

// The 409 is the one response whose detail is an object rather than a string.
func TestOnboard_ClaimConflictIs409WithAnObjectDetail(t *testing.T) {
	h := newHarness(t, nil)
	repo := "svc"
	cl := &fakeClaims{addErr: &store.ClaimConflict{Project: "PLAT", Repo: &repo, OtherTeam: "payments"}}
	h.setOnboard(t, &fakeBB{}, cl, "https://n.example.com")

	w := h.onboardPost(t, bearer(), "tok", `{"action":"onboard","projects":[{"key":"PLAT","repos":["svc"]}]}`)
	if w.StatusCode != http.StatusConflict {
		t.Fatalf("status = %d, want 409 (body %s)", w.StatusCode, h.body(w))
	}
	got := h.decode(t, w)
	detail, ok := got["detail"].(map[string]any)
	if !ok {
		t.Fatalf("detail = %#v, want an object", got["detail"])
	}
	conflict, ok := detail["conflict"].(map[string]any)
	if !ok {
		t.Fatalf("conflict = %#v, want an object", detail["conflict"])
	}
	if conflict["target"] != "PLAT/svc" || conflict["team"] != "payments" {
		t.Errorf("conflict = %v, want PLAT/svc held by payments", conflict)
	}
}

// A successful claim refreshes the runtime, so the webhook route sees the
// new repo immediately.
func TestOnboard_SuccessfulClaimRefreshesTheRuntime(t *testing.T) {
	h := newHarness(t, func(team *config.Team) { team.Projects = nil })
	cl := &fakeClaims{claims: []config.ProjectScope{{Key: "PLAT"}}}
	h.setOnboard(t, &fakeBB{}, cl, "https://n.example.com")

	w := h.onboardPost(t, bearer(), "tok", `{"action":"onboard","projects":[{"key":"PLAT"}]}`)
	if w.StatusCode != http.StatusOK {
		t.Fatalf("status = %d, want 200 (body %s)", w.StatusCode, h.body(w))
	}
	if !h.rt.Team().Owns("PLAT", "svc") {
		t.Error("the new claim did not reach the runtime snapshot")
	}
}

// The orchestrators mutate the team they are handed, so the route must pass
// a copy: the snapshot the webhook route reads is shared.
func TestOnboard_DoesNotWriteThroughTheSnapshot(t *testing.T) {
	h := newHarness(t, nil)
	before := h.rt.Team()
	cl := &fakeClaims{addErr: errors.New("nope")}
	h.setOnboard(t, &fakeBB{}, cl, "https://n.example.com")

	h.onboardPost(t, bearer(), "tok", `{"action":"onboard","projects":[{"key":"OTHER"}],"dry_run":true}`)

	if len(before.Projects) != 1 || before.Projects[0].Key != "PLAT" {
		t.Errorf("the published snapshot was mutated: %+v", before.Projects)
	}
}

func TestOnboard_RemoveReportsUnclaimedAndPurged(t *testing.T) {
	h := newHarness(t, nil)
	cl := &fakeClaims{}
	h.setOnboard(t, &fakeBB{}, cl, "https://n.example.com")

	w := h.onboardPost(t, bearer(), "tok", `{"action":"remove","projects":[{"key":"PLAT"}]}`)
	if w.StatusCode != http.StatusOK {
		t.Fatalf("status = %d, want 200 (body %s)", w.StatusCode, h.body(w))
	}
	got := h.decode(t, w)
	if _, ok := got["unclaimed"]; !ok {
		t.Error("a remove must report unclaimed")
	}
	if _, ok := got["purged_prs"]; !ok {
		t.Error("a remove must report purged_prs")
	}
}

// The response carries the rendered table and the health verdict.
func TestOnboard_StatusReportsRowsAndText(t *testing.T) {
	h := newHarness(t, nil)
	h.setOnboard(t, &fakeBB{}, &fakeClaims{}, "https://n.example.com")

	w := h.onboardPost(t, bearer(), "tok", `{"action":"status"}`)
	if w.StatusCode != http.StatusOK {
		t.Fatalf("status = %d, want 200 (body %s)", w.StatusCode, h.body(w))
	}
	got := h.decode(t, w)
	if got["webhook_url"] != "https://n.example.com/webhook/"+testSlug {
		t.Errorf("webhook_url = %v", got["webhook_url"])
	}
	for _, key := range []string{"rows", "text", "healthy", "team", "action"} {
		if _, ok := got[key]; !ok {
			t.Errorf("response missing %q", key)
		}
	}
}

// Auth precedes everything, including the public-URL check.
func TestOnboard_AuthFirst(t *testing.T) {
	h := newHarness(t, nil)
	h.setOnboard(t, &fakeBB{}, &fakeClaims{}, "")
	if w := h.onboardPost(t, "Bearer wrong", "tok", `{}`); w.StatusCode != http.StatusUnauthorized {
		t.Errorf("status = %d, want 401", w.StatusCode)
	}
}

// The real client must satisfy the interface the fake stands in for.
var _ onboarding.AdminClient = (*bitbucket.Client)(nil)
var _ BotClient = bitbucketClient{}
