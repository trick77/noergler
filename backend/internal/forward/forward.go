// Package forward is the HTTP core the riptide and Almanach emitters share:
// one bearer per team, a short timeout, no redirects, a startup ping that
// returns the team the receiver knows us as, a POST that reports whether
// the receiver took the body, and a fixed status word for the Teams page.
//
// What to send, when, and whether a miss is retried are the emitters'
// policies; this package knows only the wire.
package forward

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"strings"
	"sync/atomic"
	"time"
)

const (
	// requestTimeout is deliberately short: a slow receiver must not hold a
	// review worker.
	requestTimeout = 2 * time.Second
	// errBodyLimit caps how much of a rejection is logged.
	errBodyLimit = 200
)

// Status words, as the dashboard shows them: fixed, never a reason. Each is
// the outcome of the last call; no word guesses why a call failed beyond
// the one the receiver states itself (401/403: the token).
const (
	StatusOff      = "off"
	StatusOK       = "ok"
	StatusFailing  = "failing"
	StatusRejected = "rejected"
)

// Client is one team's connection to one receiver. A client without both a
// URL and a token is off, and every method is then a no-op.
type Client struct {
	// Name is the receiver as the log names it ("Riptide", "Almanach").
	name    string
	url     string
	token   string
	enabled bool
	// last is the last call's outcome, a Status word: ok from a startup ping
	// the receiver answered with a team name or a 2xx post, rejected from a
	// 401 or 403 (ping or post), failing from any other lost call. A
	// cancelled post keeps it. Nil reads as failing. Atomic because the
	// dashboard reads it while the worker writes.
	last atomic.Pointer[string]
	http *http.Client
	log  *slog.Logger
}

// New builds a client. Off unless both url and token are set.
func New(name, rawURL, token string, log *slog.Logger) *Client {
	url := strings.TrimSuffix(strings.TrimSpace(rawURL), "/")
	token = strings.TrimSpace(token)
	c := &Client{name: name, url: url, token: token, enabled: url != "" && token != "", log: log}
	if c.enabled {
		c.http = &http.Client{
			Timeout: requestTimeout,
			// Redirects are not followed: a 3xx would replay the bearer token
			// at the new host. Non-2xx, so a redirect cannot read as success.
			CheckRedirect: func(*http.Request, []*http.Request) error { return http.ErrUseLastResponse },
		}
	}
	return c
}

// Enabled reports whether this client forwards at all.
func (c *Client) Enabled() bool { return c != nil && c.enabled }

// URL is the receiver's base URL, for an error message.
func (c *Client) URL() string { return c.url }

// Status is the forwarding state as of the last call: off, ok, failing or
// rejected. A fixed word, never a reason.
func (c *Client) Status() string {
	if c == nil || !c.enabled {
		return StatusOff
	}
	if s := c.last.Load(); s != nil {
		return *s
	}
	return StatusFailing
}

// Set records a status word. Exported for the emitters' tests.
func (c *Client) Set(status string) { c.last.Store(&status) }

// Ping checks the token against pingPath and returns the team the receiver
// knows us as. unauthorized is true on a 401: the emitter decides whether
// that is fatal. Anything else, an unreachable receiver included, returns
// an empty team and is logged as a warning; a 403 reads as rejected on the
// dashboard.
func (c *Client) Ping(ctx context.Context, pingPath string) (team string, unauthorized bool) {
	if !c.Enabled() {
		return "", false
	}
	c.Set(StatusFailing)
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, c.url+pingPath, nil)
	if err != nil {
		c.log.WarnContext(ctx, c.name+" ping failed: "+err.Error())
		return "", false
	}
	req.Header.Set("Authorization", "Bearer "+c.token)

	resp, err := c.http.Do(req)
	if err != nil {
		c.log.WarnContext(ctx, fmt.Sprintf("%s unreachable at startup (url=%s): %v", c.name, c.url, err))
		return "", false
	}
	defer func() { _ = resp.Body.Close() }()

	if resp.StatusCode == http.StatusUnauthorized {
		c.Set(StatusRejected)
		return "", true
	}
	// 3xx included: a redirected URL is a misconfiguration, and reporting it
	// as an unreadable body would send the reader looking at the receiver.
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		if resp.StatusCode == http.StatusForbidden {
			c.Set(StatusRejected)
		}
		c.log.WarnContext(ctx, fmt.Sprintf("%s ping returned unexpected status %d: %s",
			c.name, resp.StatusCode, ReadErrorBody(resp.Body, errBodyLimit)))
		return "", false
	}

	var out struct {
		Team string `json:"team"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&out); err != nil {
		c.log.WarnContext(ctx, c.name+" ping returned an unreadable body: "+err.Error())
		return "", false
	}
	// A 200 without a team is not the receiver answering for us (a proxy, a
	// stub): it must not read as forwarding on the dashboard.
	if out.Team == "" {
		c.log.WarnContext(ctx, c.name+" ping named no team")
		return "", false
	}
	c.Set(StatusOK)
	c.log.InfoContext(ctx, c.name+": OK (team="+out.Team+")")
	return out.Team, false
}

// NullableString renders an empty string as JSON null: a field that is always
// present in a payload and null when there is nothing to say.
func NullableString(s string) any {
	if s == "" {
		return nil
	}
	return s
}

// OrEmpty keeps a list field an array rather than null.
func OrEmpty(s []string) []string {
	if s == nil {
		return []string{}
	}
	return s
}

// ReadErrorBody keeps at most limit bytes of a failed response's body for the
// error message and drains the rest, so the connection can be reused. The
// Bitbucket and Jira clients read their error bodies through it too.
func ReadErrorBody(r io.Reader, limit int64) string {
	body, _ := io.ReadAll(io.LimitReader(r, limit))
	_, _ = io.Copy(io.Discard, r)
	return strings.TrimSpace(strings.ToValidUTF8(string(body), "�"))
}

// Post sends body as JSON to path and reports whether the receiver took it
// (2xx). what names the body in the log ("event_type=pr_completed", "note
// PROJ/repo#1"). Never an error: a transport failure or a rejection is
// logged and the status word set; a cancelled context says nothing about
// the receiver, so the word stays.
func (c *Client) Post(ctx context.Context, path string, body any, what string) bool {
	if !c.Enabled() {
		return false
	}
	raw, err := json.Marshal(body)
	if err != nil {
		c.Set(StatusFailing)
		c.log.WarnContext(ctx, c.name+" emit failed to encode: "+err.Error())
		return false
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, c.url+path, bytes.NewReader(raw))
	if err != nil {
		c.Set(StatusFailing)
		c.log.WarnContext(ctx, c.name+" emit failed: "+err.Error())
		return false
	}
	req.Header.Set("Authorization", "Bearer "+c.token)
	req.Header.Set("Content-Type", "application/json")

	resp, err := c.http.Do(req)
	if err != nil {
		if ctx.Err() != nil {
			c.log.WarnContext(ctx, fmt.Sprintf("%s emit cancelled (%s): %v", c.name, what, err))
			return false
		}
		c.Set(StatusFailing)
		c.log.WarnContext(ctx, fmt.Sprintf("%s emit failed (%s): %v", c.name, what, err))
		return false
	}
	defer func() { _ = resp.Body.Close() }()
	// Anything but a 2xx is a miss, 3xx included: redirects are not
	// followed, so a redirected endpoint would otherwise drop everything
	// without a word.
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		word := "failed"
		if resp.StatusCode == http.StatusUnauthorized || resp.StatusCode == http.StatusForbidden {
			c.Set(StatusRejected)
			word = "rejected"
		} else {
			c.Set(StatusFailing)
		}
		c.log.WarnContext(ctx, fmt.Sprintf("%s emit %s (%s, status=%d): %s",
			c.name, word, what, resp.StatusCode, ReadErrorBody(resp.Body, errBodyLimit)))
		return false
	}
	c.Set(StatusOK)
	_, _ = io.Copy(io.Discard, resp.Body)
	return true
}
