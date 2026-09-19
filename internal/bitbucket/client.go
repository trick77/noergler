// Package bitbucket is the Bitbucket Server (Data Center) REST client: diffs,
// comments, webhooks and permissions. It holds no review logic — callers hand
// it finished comment bodies and decide what a finding means.
package bitbucket

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"net"
	"net/http"
	"net/url"
	"strings"
	"time"

	"github.com/trick77/noergler-go/internal/config"
	"github.com/trick77/noergler-go/internal/httpstats"
)

// apiBase prefixes every path below.
const apiBase = "/rest/api/1.0"

// errBodyLimit caps how much of an error response is kept for the message.
const errBodyLimit = 2000

// dialTimeout and headerTimeout bound connecting and waiting for the response
// head. Deliberately not an http.Client.Timeout: that one covers reading the
// body too, and a 10 MiB diff over a slow link would trip it where the Python
// service (httpx per-operation timeouts) succeeded. The caller's context bounds
// the total instead.
const (
	dialTimeout   = 30 * time.Second
	headerTimeout = 30 * time.Second
)

// StatusError is a non-2xx response. Callers branch on Status: 404 means a PR
// is gone, 401/403 means a token lacks rights, 409 is an optimistic-lock miss.
type StatusError struct {
	Method string
	Path   string
	Status int
	Body   string
}

func (e *StatusError) Error() string {
	if e.Body == "" {
		return fmt.Sprintf("%s %s: HTTP %d", e.Method, e.Path, e.Status)
	}
	return fmt.Sprintf("%s %s: HTTP %d: %s", e.Method, e.Path, e.Status, e.Body)
}

// Status returns the HTTP status of err, or 0 if err is not a StatusError.
func Status(err error) int {
	var se *StatusError
	if errors.As(err, &se) {
		return se.Status
	}
	return 0
}

// ContentTooLarge is a diff or file body over its byte cap. What names the
// resource. Size is the declared length when the server sent one, nil when the
// cap tripped mid-stream.
type ContentTooLarge struct {
	What  string
	Limit int
	Size  *int
}

func (e *ContentTooLarge) Error() string {
	seen := fmt.Sprintf("> %d bytes", e.Limit)
	if e.Size != nil {
		seen = fmt.Sprintf("%d bytes", *e.Size)
	}
	return fmt.Sprintf("%s: %s exceeds cap of %d bytes", e.What, seen, e.Limit)
}

// ErrIncrementalDiffUnavailable means Bitbucket cannot diff two commits: the
// source branch was rebased so the old commit is unreachable, or the two live
// on unrelated histories. Signalled by 406 on compare/diff; the caller falls
// back to a full review.
var ErrIncrementalDiffUnavailable = errors.New("incremental diff unavailable")

// ErrVersionConflict means a comment was edited concurrently (HTTP 409). The
// caller posts a fresh comment instead of updating.
var ErrVersionConflict = errors.New("comment version conflict")

// Client talks to one Bitbucket instance as one account.
type Client struct {
	base         *url.URL
	token        string
	botUsername  string
	http         *http.Client
	log          *slog.Logger
	maxDiffBytes int
	maxFileBytes int
}

// New builds a client for the shared service account.
func New(cfg config.Bitbucket, log *slog.Logger) (*Client, error) {
	base, err := url.Parse(strings.TrimSuffix(cfg.BaseURL, "/"))
	if err != nil {
		return nil, fmt.Errorf("parse BITBUCKET_URL: %w", err)
	}
	if base.Scheme == "" || base.Host == "" {
		return nil, fmt.Errorf("parse BITBUCKET_URL: %q is not an absolute URL", cfg.BaseURL)
	}
	return &Client{
		base:        base,
		token:       cfg.Token,
		botUsername: cfg.Username,
		http: &http.Client{
			Transport: httpstats.Transport("bitbucket", &http.Transport{
				Proxy:                 http.ProxyFromEnvironment,
				DialContext:           (&net.Dialer{Timeout: dialTimeout}).DialContext,
				TLSHandshakeTimeout:   dialTimeout,
				ResponseHeaderTimeout: headerTimeout,
				ForceAttemptHTTP2:     true,
			}),
			// httpx does not follow redirects, so a 3xx is an error there. Go
			// would follow it by default and replay the Authorization header to
			// whatever host it points at.
			CheckRedirect: func(*http.Request, []*http.Request) error { return http.ErrUseLastResponse },
		},
		log:          log,
		maxDiffBytes: cfg.MaxDiffBytes,
		maxFileBytes: cfg.MaxFileBytes,
	}, nil
}

// BotUsername is the service account's name, which doubles as the @mention
// trigger.
func (c *Client) BotUsername() string { return c.botUsername }

// WithToken returns a copy acting as a different user, for onboarding writes
// made with a team admin's own token. The transport (and its pool) is shared.
func (c *Client) WithToken(token string) *Client {
	clone := *c
	clone.token = token
	return &clone
}

// url builds an absolute URL. Path segments are escaped, so a file path with a
// space or a '#' survives; query is set verbatim by the caller where a contract
// demands it.
func (c *Client) url(path string, query url.Values) string {
	u := *c.base
	u.Path = c.base.Path + path
	if len(query) > 0 {
		u.RawQuery = query.Encode()
	}
	return u.String()
}

// do issues a request and decodes a JSON response into out (nil to discard).
func (c *Client) do(ctx context.Context, method, path string, query url.Values, body, out any) error {
	var reader io.Reader
	if body != nil {
		buf, err := json.Marshal(body)
		if err != nil {
			return fmt.Errorf("encode %s %s: %w", method, path, err)
		}
		reader = strings.NewReader(string(buf))
	}
	req, err := http.NewRequestWithContext(ctx, method, c.url(path, query), reader)
	if err != nil {
		return err
	}
	req.Header.Set("Authorization", "Bearer "+c.token)
	req.Header.Set("Accept", "application/json")
	if body != nil {
		req.Header.Set("Content-Type", "application/json")
	}
	resp, err := c.http.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()

	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		return statusError(method, path, resp)
	}
	if out == nil {
		_, _ = io.Copy(io.Discard, resp.Body)
		return nil
	}
	if err := json.NewDecoder(resp.Body).Decode(out); err != nil {
		return fmt.Errorf("decode %s %s: %w", method, path, err)
	}
	return nil
}

// statusError drains the body (so the connection can be reused) and wraps it.
func statusError(method, path string, resp *http.Response) error {
	body, _ := io.ReadAll(io.LimitReader(resp.Body, errBodyLimit))
	_, _ = io.Copy(io.Discard, resp.Body)
	return &StatusError{
		Method: method,
		Path:   path,
		Status: resp.StatusCode,
		Body:   strings.TrimSpace(strings.ToValidUTF8(string(body), "�")),
	}
}

// getTextCapped GETs path and returns the body as text, refusing to hold more
// than max bytes. Order matters and is load-bearing:
//
//  1. a non-2xx wins over the cap: the body is drained and a StatusError
//     returned, so a huge error page is never reported as ContentTooLarge;
//  2. a declared Content-Length over the cap fails before the body is read,
//     with Size set;
//  3. otherwise the read itself stops one byte past the cap, with Size nil.
//
// The comparison is strictly greater than: a body of exactly max passes.
func (c *Client) getTextCapped(ctx context.Context, path string, query url.Values, accept, what string, max int) (string, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, c.url(path, query), nil)
	if err != nil {
		return "", err
	}
	req.Header.Set("Authorization", "Bearer "+c.token)
	if accept == "" {
		accept = "application/json"
	}
	req.Header.Set("Accept", accept)

	resp, err := c.http.Do(req)
	if err != nil {
		return "", err
	}
	defer resp.Body.Close()

	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		return "", statusError(http.MethodGet, path, resp)
	}
	if resp.ContentLength >= 0 && resp.ContentLength > int64(max) {
		size := int(resp.ContentLength)
		return "", &ContentTooLarge{What: what, Limit: max, Size: &size}
	}
	buf, err := io.ReadAll(io.LimitReader(resp.Body, int64(max)+1))
	if err != nil {
		return "", fmt.Errorf("read %s: %w", what, err)
	}
	if len(buf) > max {
		return "", &ContentTooLarge{What: what, Limit: max}
	}
	// Python decoded with errors="replace"; Go leaves invalid bytes alone.
	return strings.ToValidUTF8(string(buf), "�"), nil
}
