package onboarding

import (
	"context"
	"fmt"
	"os"
	"strings"
	"sync"
	"sync/atomic"
	"testing"

	"github.com/trick77/noergler/internal/bitbucket"
	"github.com/trick77/noergler/internal/config"
)

const (
	publicURL  = "https://noergler.test"
	team       = "platform"
	secret     = "whsec"
	botName    = "noergler"
	webhookURL = publicURL + "/webhook/" + team
)

// The real clients must keep satisfying the consumer-side interfaces.
var (
	_ AdminClient = (*bitbucket.Client)(nil)
	_ BotClient   = (*bitbucket.Client)(nil)
)

func statusErr(code int, body string) error {
	return &bitbucket.StatusError{Method: "GET", Path: "/x", Status: code, Body: body}
}

// transportErr stands in for a dial/TLS failure: no status at all.
func transportErr(msg string) error { return fmt.Errorf("%s", msg) }

func newTeam(scopes ...config.ProjectScope) *config.Team {
	return &config.Team{Slug: team, Name: team, WebhookSecret: secret, Projects: scopes}
}

func whole(key string) config.ProjectScope { return config.ProjectScope{Key: key} }

func repos(key string, rs ...string) config.ProjectScope {
	return config.ProjectScope{Key: key, Repos: rs}
}

// goodHook is a hook that needs no changes, with the given overrides applied.
func goodHook(over map[string]any) map[string]any {
	events := make([]any, 0, len(config.RequiredWebhookEvents))
	for _, e := range config.RequiredWebhookEvents {
		events = append(events, e)
	}
	h := map[string]any{
		"id": float64(7), "name": botName, "url": webhookURL, "active": true,
		"events": events, "configuration": map[string]any{"secret": "x"},
		"sslVerificationRequired": true,
	}
	for k, v := range over {
		h[k] = v
	}
	return h
}

// hookKey addresses one webhook listing: "PROJ" or "PROJ/repo".
func hookKey(project, repo string) string {
	if repo == "" {
		return project
	}
	return project + "/" + repo
}

// fakeAdmin answers from canned tables. A missing key is a 404, which is what
// Bitbucket does and what the code must never mistake for "no admin".
type fakeAdmin struct {
	mu sync.Mutex

	hooks    map[string][]map[string]any
	hookErr  map[string]error
	repos    map[string][]map[string]any
	reposErr map[string]error

	created  []bitbucket.Webhook
	updated  []int
	deleted  []string
	grants   []string
	createFn func() (*bitbucket.Webhook, error)
	deleteFn func(project, repo string, id int) error

	// inFlight tracks the concurrency of ListWebhooks.
	inFlight, maxInFlight atomic.Int64
	// onList runs inside ListWebhooks, for ordering tests.
	onList func(project, repo string)
}

func (f *fakeAdmin) ListWebhooks(_ context.Context, project, repo string) ([]map[string]any, error) {
	n := f.inFlight.Add(1)
	for {
		m := f.maxInFlight.Load()
		if n <= m || f.maxInFlight.CompareAndSwap(m, n) {
			break
		}
	}
	defer f.inFlight.Add(-1)
	if f.onList != nil {
		f.onList(project, repo)
	}
	k := hookKey(project, repo)
	f.mu.Lock()
	defer f.mu.Unlock()
	if err, ok := f.hookErr[k]; ok {
		return nil, err
	}
	if h, ok := f.hooks[k]; ok {
		return h, nil
	}
	return nil, statusErr(404, "no such target")
}

func (f *fakeAdmin) CreateWebhook(_ context.Context, _, _ string, body bitbucket.Webhook) (*bitbucket.Webhook, error) {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.created = append(f.created, body)
	if f.createFn != nil {
		return f.createFn()
	}
	out := body
	out.ID = 42
	return &out, nil
}

func (f *fakeAdmin) UpdateWebhook(_ context.Context, _, _ string, id int, body bitbucket.Webhook) (*bitbucket.Webhook, error) {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.updated = append(f.updated, id)
	out := body
	out.ID = id
	return &out, nil
}

func (f *fakeAdmin) DeleteWebhook(_ context.Context, project, repo string, id int) error {
	if f.deleteFn != nil {
		if err := f.deleteFn(project, repo, id); err != nil {
			return err
		}
	}
	f.mu.Lock()
	defer f.mu.Unlock()
	f.deleted = append(f.deleted, fmt.Sprintf("%s#%d", hookKey(project, repo), id))
	return nil
}

func (f *fakeAdmin) GrantUserPermission(_ context.Context, project, repo, username, permission string) error {
	f.mu.Lock()
	defer f.mu.Unlock()
	if err, ok := f.hookErr["grant:"+hookKey(project, repo)]; ok {
		return err
	}
	f.grants = append(f.grants, fmt.Sprintf("%s:%s:%s", hookKey(project, repo), username, permission))
	return nil
}

func (f *fakeAdmin) ListRepos(_ context.Context, project string) ([]map[string]any, error) {
	f.mu.Lock()
	defer f.mu.Unlock()
	if err, ok := f.reposErr[project]; ok {
		return nil, err
	}
	return f.repos[project], nil
}

// fakeBot answers the bot read proof. readable lists the keys it can read;
// everything else fails with err (default 404).
type fakeBot struct {
	readable map[string]bool
	err      error
}

func (f *fakeBot) answer(key string) (map[string]any, error) {
	if f.readable[key] {
		return map[string]any{"key": key}, nil
	}
	if f.err != nil {
		return nil, f.err
	}
	return nil, statusErr(404, "not found")
}

func (f *fakeBot) GetProject(_ context.Context, project string) (map[string]any, error) {
	return f.answer(project)
}

func (f *fakeBot) GetRepo(_ context.Context, project, repo string) (map[string]any, error) {
	return f.answer(project + "/" + repo)
}

func (f *fakeBot) BotUsername() string { return botName }

func botReading(keys ...string) *fakeBot {
	m := map[string]bool{}
	for _, k := range keys {
		m[k] = true
	}
	return &fakeBot{readable: m}
}

// golden reads a testdata file, dropping the one trailing newline a text
// editor keeps and the renderers never emit.
func golden(t *testing.T, name string) string {
	t.Helper()
	b, err := os.ReadFile("testdata/" + name)
	if err != nil {
		t.Fatalf("read golden %s: %v", name, err)
	}
	return strings.TrimSuffix(string(b), "\n")
}
