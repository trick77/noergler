package api

import (
	"bytes"
	"context"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"

	"github.com/getkin/kin-openapi/routers/legacy"

	"github.com/trick77/noergler/internal/bitbucket"
	"github.com/trick77/noergler/internal/config"
	"github.com/trick77/noergler/internal/secretbox"
	"github.com/trick77/noergler/internal/store"
	"github.com/trick77/noergler/internal/teams"
	"github.com/trick77/noergler/internal/webhook"
)

// memCreds is the credential store in memory.
type memCreds struct {
	mu   sync.Mutex
	rows map[string]store.SealedCredential
	chk  [2][]byte
	has  bool
}

func (m *memCreds) ListCredentials(context.Context) ([]store.SealedCredential, error) {
	m.mu.Lock()
	defer m.mu.Unlock()
	var out []store.SealedCredential
	for _, r := range m.rows {
		out = append(out, r)
	}
	return out, nil
}
func (m *memCreds) PutCredentials(_ context.Context, creds []store.SealedCredential, _ string) error {
	m.mu.Lock()
	defer m.mu.Unlock()
	for _, c := range creds {
		m.rows[c.Team+"|"+c.Kind] = c
	}
	return nil
}
func (m *memCreds) DeleteCredential(_ context.Context, team, kind, _ string) (bool, error) {
	m.mu.Lock()
	defer m.mu.Unlock()
	_, ok := m.rows[team+"|"+kind]
	delete(m.rows, team+"|"+kind)
	return ok, nil
}
func (m *memCreds) KeyCheck(context.Context) ([]byte, []byte, bool, error) {
	return m.chk[0], m.chk[1], m.has, nil
}
func (m *memCreds) PutKeyCheck(_ context.Context, n, c []byte) error {
	m.chk, m.has = [2][]byte{n, c}, true
	return nil
}

// upstream is Bitbucket and Jira at once: a known token answers as its
// account (X-AUSERNAME), anything else is 401.
func upstream(t *testing.T, tokens map[string]string) *httptest.Server {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		who, ok := tokens[strings.TrimPrefix(r.Header.Get("Authorization"), "Bearer ")]
		if !ok {
			http.Error(w, `{"errors":[{"message":"Authentication failed"}]}`, http.StatusUnauthorized)
			return
		}
		w.Header().Set("X-AUSERNAME", who)
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"name":"` + who + `"}`))
	}))
	t.Cleanup(srv.Close)
	return srv
}

// withAccounts gives the harness team real accounts against a fake upstream.
func (h *harness) withAccounts(t *testing.T, opened map[teams.Kind]teams.Opened) *memCreds {
	t.Helper()
	up := upstream(t, map[string]string{"shared": testBot, "plat-tok": "plat-bot", "jira-tok": "plat-jira", "shared-jira": "rop"})
	shared, err := bitbucket.New(config.Bitbucket{BaseURL: up.URL, Token: "shared", Username: testBot}, h.log)
	if err != nil {
		t.Fatal(err)
	}
	box, err := secretbox.New(bytes.Repeat([]byte{7}, secretbox.KeySize))
	if err != nil {
		t.Fatal(err)
	}
	creds := &memCreds{rows: map[string]store.SealedCredential{}}
	if err := h.rt.StartAccounts(context.Background(), teams.AccountDeps{
		Store: creds, Box: box, Shared: shared,
		Jira: config.Jira{URL: up.URL, Token: "shared-jira"}, Log: h.log,
	}, opened); err != nil {
		t.Fatal(err)
	}
	return creds
}

const pushBB = `{"bitbucket":{"username":"plat-bot","token":"plat-tok"}}`

func TestCredentials_AuthPrecedesBodyValidation(t *testing.T) {
	h := newHarness(t, nil)
	h.withAccounts(t, nil)
	w := h.do(t, http.MethodPut, "/teams/"+testSlug+"/credentials", "Bearer wrong", `{"nope":1}`)
	if w.Code != http.StatusUnauthorized {
		t.Errorf("status = %d, want 401 before the body is looked at", w.Code)
	}
	w = h.do(t, http.MethodDelete, "/teams/"+testSlug+"/credentials/bitbucket", "Bearer wrong", "")
	if w.Code != http.StatusUnauthorized {
		t.Errorf("delete status = %d, want 401", w.Code)
	}
}

func TestCredentials_PushSwitchesTheBotAndNeverEchoesTheToken(t *testing.T) {
	h := newHarness(t, nil)
	creds := h.withAccounts(t, nil)

	w := h.do(t, http.MethodPut, "/teams/"+testSlug+"/credentials", bearer(), pushBB)
	if w.Code != http.StatusOK {
		t.Fatalf("status = %d: %s", w.Code, w.Body)
	}
	got := decodeBody(t, w)
	bb := got["bitbucket_account"].(map[string]any)
	if bb["source"] != "team" || bb["username"] != "plat-bot" || bb["status"] != "ok" || bb["since"] == nil {
		t.Errorf("bitbucket_account = %v", bb)
	}
	if _, ok := got["unreachable_claims"]; ok {
		t.Errorf("every claim is readable, yet unreachable_claims = %v", got["unreachable_claims"])
	}
	if strings.Contains(w.Body.String(), "plat-tok") || strings.Contains(h.logs.String(), "plat-tok") {
		t.Error("the token reached the response or the log")
	}
	if row, ok := creds.rows[testSlug+"|bitbucket"]; !ok || bytes.Contains(row.Ciphertext, []byte("plat-tok")) {
		t.Errorf("stored row = %+v", row)
	}

	// GET agrees, and so does the mention trigger: the team's own bot now,
	// the shared one no longer.
	got = decodeBody(t, h.do(t, http.MethodGet, "/teams/"+testSlug, bearer(), ""))
	if got["bitbucket_account"].(map[string]any)["username"] != "plat-bot" {
		t.Errorf("GET = %v", got["bitbucket_account"])
	}
	mention := func(text string) string {
		extra := fmt.Sprintf(`,"comment":{"id":7,"text":%q,"author":{"name":"bob"}}`, text)
		return fmt.Sprint(decodeBody(t, h.signedPost(t, payloadJSON(webhook.EventCommentAdded, extra)))["reason"])
	}
	if r := mention("@plat-bot please review"); r != "mention" {
		t.Errorf("own bot mention: reason = %s", r)
	}
	if r := mention("@" + testBot + " please review"); r != "comment without mention" {
		t.Errorf("shared bot mention on a team with its own bot: reason = %s", r)
	}

	// DELETE goes back to the shared bot, once.
	w = h.do(t, http.MethodDelete, "/teams/"+testSlug+"/credentials/bitbucket", bearer(), "")
	if w.Code != http.StatusOK || decodeBody(t, w)["bitbucket_account"].(map[string]any)["source"] != "shared" {
		t.Errorf("delete = %d %s", w.Code, w.Body)
	}
	if r := mention("@" + testBot + " please review"); r != "mention" {
		t.Errorf("after delete, shared mention: reason = %s", r)
	}
	if w := h.do(t, http.MethodDelete, "/teams/"+testSlug+"/credentials/bitbucket", bearer(), ""); w.Code != http.StatusNotFound {
		t.Errorf("second delete = %d, want 404", w.Code)
	}
	if w := h.do(t, http.MethodDelete, "/teams/"+testSlug+"/credentials/riptide", bearer(), ""); w.Code != http.StatusNotFound {
		t.Errorf("unknown kind = %d, want 404", w.Code)
	}
}

func TestCredentials_RefusedPushChangesNothing(t *testing.T) {
	cases := []struct{ name, body, want string }{
		{"rejected token", `{"bitbucket":{"username":"plat-bot","token":"revoked"}}`, "rejected the token"},
		{"another account", `{"bitbucket":{"username":"someone","token":"plat-tok"}}`, "belongs to plat-bot"},
		{"jira refused, bitbucket fine", `{"bitbucket":{"username":"plat-bot","token":"plat-tok"},"jira":{"token":"bad"}}`, "jira"},
		{"nothing", `{}`, "nothing to push"},
		{"missing username", `{"bitbucket":{"token":"plat-tok"}}`, "needs both"},
		{"blank jira", `{"jira":{"token":" "}}`, "jira needs a token"},
		{"unknown field", `{"riptide":{"token":"x"}}`, "unknown field"},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			h := newHarness(t, nil)
			creds := h.withAccounts(t, nil)
			w := h.do(t, http.MethodPut, "/teams/"+testSlug+"/credentials", bearer(), c.body)
			if w.Code != http.StatusUnprocessableEntity || !strings.Contains(w.Body.String(), c.want) {
				t.Errorf("= %d %s, want 422 naming %q", w.Code, w.Body, c.want)
			}
			if len(creds.rows) != 0 {
				t.Error("a refused push stored something")
			}
			if bb, _ := h.rt.Accounts(); bb.Own {
				t.Error("a refused push swapped the account")
			}
		})
	}
}

// A team whose own Bitbucket account cannot be worked with keeps answering
// webhooks, processes none, and the team API stays open so it can push a
// working token. Another team is untouched.
func TestCredentials_BlockedTeamIsAcknowledgedNotProcessed(t *testing.T) {
	h := newHarness(t, nil)
	h.withAccounts(t, map[teams.Kind]teams.Opened{teams.KindBitbucket: {Username: "plat-bot", Undecryptable: true}})

	w := h.signedPost(t, payloadJSON(webhook.EventOpened, ""))
	got := decodeBody(t, w)
	if w.Code != http.StatusOK || got["status"] != "ignored" || !strings.Contains(fmt.Sprint(got["reason"]), "undecryptable") {
		t.Errorf("= %d %v", w.Code, got)
	}
	if len(h.q.keys)+len(h.q.tags) != 0 {
		t.Error("a blocked team's webhook reached the queue")
	}
	bb := decodeBody(t, h.do(t, http.MethodGet, "/teams/"+testSlug, bearer(), ""))["bitbucket_account"].(map[string]any)
	if bb["status"] != "undecryptable" {
		t.Errorf("GET = %v", bb)
	}

	if w := h.do(t, http.MethodPut, "/teams/"+testSlug+"/credentials", bearer(), pushBB); w.Code != http.StatusOK {
		t.Fatalf("push to heal = %d %s", w.Code, w.Body)
	}
	if got := decodeBody(t, h.signedPost(t, payloadJSON(webhook.EventOpened, ""))); got["status"] != "accepted" {
		t.Errorf("after the push = %v", got)
	}
}

// Lifecycle events pass a blocked team's gate: Bitbucket never redelivers
// them, so a merge rollup or a purge dropped here would be lost for good.
func TestCredentials_BlockedTeamStillRecordsLifecycle(t *testing.T) {
	for _, event := range []string{webhook.EventMerged, webhook.EventDeclined, webhook.EventDeleted} {
		h := newHarness(t, nil)
		h.withAccounts(t, map[teams.Kind]teams.Opened{teams.KindBitbucket: {Username: "plat-bot", Undecryptable: true}})
		if got := decodeBody(t, h.signedPost(t, payloadJSON(event, ""))); got["status"] != "accepted" {
			t.Errorf("%s on a blocked team = %v, want accepted", event, got)
		}
	}
}

// Onboarding with an own bot that cannot be worked with says so, instead of
// reporting every target unreachable or granting a name the token is not.
func TestCredentials_OnboardRefusesABlockedBot(t *testing.T) {
	h := newHarness(t, nil)
	h.withAccounts(t, map[teams.Kind]teams.Opened{teams.KindBitbucket: {Username: "plat-bot", Undecryptable: true}})
	h.setOnboard(t, &fakeBB{hooks: []map[string]any{}}, &fakeClaims{}, "https://n.example.com")
	w := h.onboardPost(t, bearer(), "tok", `{"action":"status"}`)
	if w.StatusCode != http.StatusUnprocessableEntity || !strings.Contains(h.body(w), "undecryptable") {
		t.Errorf("= %d %s, want 422 naming the account's state", w.StatusCode, h.body(w))
	}
}

func TestCredentials_MatchOpenAPISpec(t *testing.T) {
	doc := loadSpec(t)
	doc.Servers = nil
	router, err := legacy.NewRouter(doc)
	if err != nil {
		t.Fatal(err)
	}
	h := newHarness(t, nil)
	h.withAccounts(t, nil)
	path := "/teams/" + testSlug + "/credentials"

	w := h.do(t, http.MethodPut, path, bearer(), pushBB+"\n")
	if w.Code != http.StatusOK {
		t.Fatalf("push = %d", w.Code)
	}
	validate(t, router, http.MethodPut, path, w)
	w = h.do(t, http.MethodPut, path, bearer(), `{"jira":{"token":"jira-tok"}}`)
	validate(t, router, http.MethodPut, path, w)
	w = h.do(t, http.MethodPut, path, bearer(), `{"bitbucket":{"username":"x","token":"bad"}}`)
	if w.Code != http.StatusUnprocessableEntity {
		t.Fatalf("refused push = %d", w.Code)
	}
	validate(t, router, http.MethodPut, path, w)
	w = h.do(t, http.MethodDelete, path+"/jira", bearer(), "")
	validate(t, router, http.MethodDelete, path+"/jira", w)
	w = h.do(t, http.MethodDelete, path+"/jira", bearer(), "")
	validate(t, router, http.MethodDelete, path+"/jira", w)
}
