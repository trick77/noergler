package forward

import (
	"context"
	"encoding/json"
	"io"
	"log/slog"
	"net/http"
	"net/http/httptest"
	"testing"
)

type fake struct {
	srv    *httptest.Server
	paths  []string
	header http.Header
	body   map[string]any
}

func newFake(t *testing.T, handler func(w http.ResponseWriter, r *http.Request)) (*fake, *Client) {
	t.Helper()
	f := &fake{}
	f.srv = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		f.paths = append(f.paths, r.URL.Path)
		f.header = r.Header.Clone()
		if raw, _ := io.ReadAll(r.Body); len(raw) > 0 {
			_ = json.Unmarshal(raw, &f.body)
		}
		handler(w, r)
	}))
	t.Cleanup(f.srv.Close)
	return f, New("Receiver", f.srv.URL, "t", slog.New(slog.DiscardHandler))
}

func reply(status int, body string) func(http.ResponseWriter, *http.Request) {
	return func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(status)
		_, _ = io.WriteString(w, body)
	}
}

func TestEnabledAndOff(t *testing.T) {
	log := slog.New(slog.DiscardHandler)
	for _, tc := range []struct {
		name, url, token string
		want             bool
	}{
		{"both set", "http://a", "t", true},
		{"no url", "", "t", false},
		{"no token", "http://a", "", false},
		{"url is only a slash", "/", "t", false},
		{"whitespace only", "   ", "t", false},
	} {
		if got := New("R", tc.url, tc.token, log).Enabled(); got != tc.want {
			t.Errorf("%s: Enabled() = %v, want %v", tc.name, got, tc.want)
		}
	}
	var none *Client
	if none.Enabled() || none.Status() != StatusOff {
		t.Error("a nil client is off")
	}
	off := New("R", "", "", log)
	if team, unauthorized := off.Ping(context.Background(), "/auth/ping"); team != "" || unauthorized {
		t.Error("an off client pinged")
	}
	if off.Post(context.Background(), "/x", map[string]any{}, "x") {
		t.Error("an off client posted")
	}
}

// The ping sends the bearer, reads the team, and sets the word: ok only
// when the receiver named a team, rejected on 401 and 403, failing for
// every other miss, with 401 alone reported to the caller.
func TestPing(t *testing.T) {
	f, c := newFake(t, reply(200, `{"status":"ok","team":"checkout"}`))
	team, unauthorized := c.Ping(context.Background(), "/auth/ping")
	if team != "checkout" || unauthorized || c.Status() != StatusOK {
		t.Errorf("ping = %q %v %s", team, unauthorized, c.Status())
	}
	if f.paths[0] != "/auth/ping" || f.header.Get("Authorization") != "Bearer t" {
		t.Errorf("request = %s %q", f.paths[0], f.header.Get("Authorization"))
	}
	cases := []struct {
		status       int
		body         string
		want         string
		unauthorized bool
	}{
		{200, `not json`, StatusFailing, false},
		{200, `{}`, StatusFailing, false},
		{200, `null`, StatusFailing, false},
		{503, `nope`, StatusFailing, false},
		{302, ``, StatusFailing, false},
		{403, `forbidden`, StatusRejected, false},
		{401, `bad token`, StatusRejected, true},
	}
	for _, tc := range cases {
		_, c := newFake(t, reply(tc.status, tc.body))
		c.Set(StatusOK)
		team, unauthorized := c.Ping(context.Background(), "/auth/ping")
		if team != "" || unauthorized != tc.unauthorized || c.Status() != tc.want {
			t.Errorf("%d %q: team %q unauthorized %v status %s, want %v %s", tc.status, tc.body, team, unauthorized, c.Status(), tc.unauthorized, tc.want)
		}
	}
	dead := New("R", "http://127.0.0.1:1", "t", slog.New(slog.DiscardHandler))
	if team, unauthorized := dead.Ping(context.Background(), "/auth/ping"); team != "" || unauthorized || dead.Status() != StatusFailing {
		t.Errorf("unreachable: %q %v %s", team, unauthorized, dead.Status())
	}
}

// A post reports taken on a 2xx and nothing else; each outcome sets the
// word, 401/403 alone saying why; our own cancellation says nothing about
// the receiver.
func TestPost(t *testing.T) {
	f, c := newFake(t, reply(202, ``))
	if !c.Post(context.Background(), "/webhooks/x", map[string]any{"k": "v"}, "x") || c.Status() != StatusOK {
		t.Fatalf("202: not taken or status %s", c.Status())
	}
	if f.paths[0] != "/webhooks/x" || f.header.Get("Content-Type") != "application/json" || f.body["k"] != "v" {
		t.Errorf("request = %s %v %v", f.paths[0], f.header, f.body)
	}
	for _, tc := range []struct {
		status int
		want   string
	}{{422, StatusFailing}, {401, StatusRejected}, {403, StatusRejected}, {404, StatusFailing}, {302, StatusFailing}, {500, StatusFailing}} {
		_, c := newFake(t, reply(tc.status, `nope`))
		c.Set(StatusOK)
		if c.Post(context.Background(), "/x", map[string]any{}, "x") || c.Status() != tc.want {
			t.Errorf("%d: taken or status %s, want %s", tc.status, c.Status(), tc.want)
		}
	}
	dead := New("R", "http://127.0.0.1:1", "t", slog.New(slog.DiscardHandler))
	dead.Set(StatusOK)
	if dead.Post(context.Background(), "/x", map[string]any{}, "x") || dead.Status() != StatusFailing {
		t.Errorf("transport error: taken or status %s", dead.Status())
	}
	_, c = newFake(t, reply(202, ``))
	c.Set(StatusOK)
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if c.Post(ctx, "/x", map[string]any{}, "x") || c.Status() != StatusOK {
		t.Errorf("cancelled: taken or status %s, want ok unchanged", c.Status())
	}
	if c.Post(context.Background(), "/x", map[string]any{"bad": make(chan int)}, "x") || c.Status() != StatusFailing {
		t.Errorf("unencodable body: taken or status %s", c.Status())
	}
}
