package bitbucket

import (
	"context"
	"io"
	"log/slog"
	"net/http"
	"testing"
)

// The account comes from X-AUSERNAME; without it, from the whoami servlet;
// without either, "". The token goes in every request, the observer sees
// every outcome.
func TestIdentity(t *testing.T) {
	cases := []struct {
		name   string
		header string
		whoami int
		want   string
	}{
		{"header", "plat-bot", 0, "plat-bot"},
		{"whoami fallback", "", http.StatusOK, "plat-bot"},
		{"neither", "", http.StatusNotFound, ""},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			f, cl := newFake(t, func(w http.ResponseWriter, r *http.Request) {
				if r.URL.Path == "/plugins/servlet/applinks/whoami" {
					w.WriteHeader(c.whoami)
					_, _ = io.WriteString(w, "plat-bot\n")
					return
				}
				if c.header != "" {
					w.Header().Set("X-AUSERNAME", c.header)
				}
				_, _ = io.WriteString(w, `{"version":"8"}`)
			})
			var seen []int
			own := cl.WithAccount("own-token", "plat-bot", slog.New(slog.DiscardHandler), func(st int, _ error) { seen = append(seen, st) })
			got, err := own.Identity(context.Background())
			if err != nil || got != c.want {
				t.Fatalf("Identity = %q, %v; want %q", got, err, c.want)
			}
			for _, r := range f.seen {
				if r.Header.Get("Authorization") != "Bearer own-token" {
					t.Errorf("%s sent %q", r.Path, r.Header.Get("Authorization"))
				}
			}
			if len(seen) != f.calls() {
				t.Errorf("observer saw %d outcomes for %d calls", len(seen), f.calls())
			}
		})
	}
}

func TestIdentity_RejectedTokenIsAStatusError(t *testing.T) {
	_, cl := newFake(t, jsonReply(http.StatusUnauthorized, `{"errors":[]}`))
	if _, err := cl.Identity(context.Background()); Status(err) != http.StatusUnauthorized {
		t.Fatalf("err = %v, want a 401 StatusError", err)
	}
}
