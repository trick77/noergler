package bitbucket

import (
	"context"
	"errors"
	"io"
	"log/slog"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/trick77/noergler/internal/config"
)

func jsonServer(t *testing.T, routes map[string]string) *Client {
	t.Helper()
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, ok := routes[r.URL.Path]
		if !ok {
			http.NotFound(w, r)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, body)
	}))
	t.Cleanup(srv.Close)
	c, err := New(config.Bitbucket{BaseURL: srv.URL, Token: "test-token", Username: "bot-user"}, slog.New(slog.DiscardHandler))
	if err != nil {
		t.Fatal(err)
	}
	return c
}

func TestMergeStatus(t *testing.T) {
	c := jsonServer(t, map[string]string{
		"/rest/api/1.0/projects/P/repos/r/pull-requests/7/merge": `{"canMerge":false,"conflicted":true,"outcome":"CONFLICTED","vetoes":[]}`,
	})
	got, err := c.MergeStatus(context.Background(), "P", "r", 7)
	if err != nil || got.CanMerge || !got.Conflicted {
		t.Fatalf("got %+v, %v", got, err)
	}
}

func TestGitRemoteTakesTheHTTPCloneLink(t *testing.T) {
	c := jsonServer(t, map[string]string{
		"/rest/api/1.0/projects/P/repos/r": `{"links":{"clone":[` +
			`{"href":"ssh://git@bb.example:7999/p/r.git","name":"ssh"},` +
			`{"href":"https://bot-user@bb.example/scm/p/r.git","name":"http"}]}}`,
		"/rest/api/1.0/projects/P/repos/noclone": `{"links":{"clone":[{"href":"ssh://x","name":"ssh"}]}}`,
	})
	r, err := c.GitRemote(context.Background(), "P", "r", 99)
	if err != nil {
		t.Fatal(err)
	}
	// The account name in the link never rides along: auth is a header.
	if r.URL != "https://bb.example/scm/p/r.git" || r.Username != "bot-user" || r.Token != "test-token" || r.MaxBytes != 99 {
		t.Fatalf("remote = %+v", r)
	}
	if r.HTTP == nil || r.HTTP.CheckRedirect == nil {
		t.Fatal("the adapter's client, with its redirect policy, is not passed on")
	}
	if _, err := c.GitRemote(context.Background(), "P", "noclone", 0); !errors.Is(err, ErrNoHTTPClone) {
		t.Fatalf("err = %v, want ErrNoHTTPClone", err)
	}
}

func TestBotIdentity(t *testing.T) {
	c := jsonServer(t, map[string]string{
		"/rest/api/1.0/users/bot-user": `{"name":"bot-user","displayName":"Nörgler Bot","emailAddress":"bot@example.invalid"}`,
	})
	name, email, err := c.BotIdentity(context.Background())
	if err != nil || name != "Nörgler Bot" || email != "bot@example.invalid" {
		t.Fatalf("got %q %q %v", name, email, err)
	}
}
