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

	"github.com/trick77/noergler/internal/config"
	"github.com/trick77/noergler/internal/httpstats"
)

// apiBase prefixes every path below.
const apiBase = "/rest/api/1.0"

// errBodyLimit caps how much of an error response is kept for the message.
const errBodyLimit = 2000

// dialTimeout and headerTimeout bound connecting and waiting for the response
// head. Deliberately not an http.Client.Timeout: that one covers reading the
// body too, and a 10 MiB diff over a slow link would trip it mid-download. The
// caller's context bounds the total instead.
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
// resource.
//
// Size is the total body size: the declared Content-Length when the server
// sent one, else the count measured by draining the body past the cap.
// getTextCapped always sets it. Truncated means the drain stopped at its
// ceiling or errored, so Size is a lower bound.
//
// Head holds the first Limit bytes, so a caller can say which files were in
// the part it did see. Both matter: discarding the running byte count leaves
// the size unreportable, and ignoring the partial body loses the file names.
type ContentTooLarge struct {
	What      string
	Limit     int
	Size      *int
	Truncated bool
	Head      []byte
}

func (e *ContentTooLarge) Error() string {
	seen := fmt.Sprintf("> %d bytes", e.Limit)
	if e.Size != nil {
		seen = fmt.Sprintf("%d bytes", *e.Size)
		if e.Truncated {
			seen = fmt.Sprintf("> %d bytes", *e.Size)
		}
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
			// Redirects are not followed: a 3xx would replay the bearer token
			// at the new host. Non-2xx, so a redirect cannot read as success.
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
	defer func() { _ = resp.Body.Close() }()

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
//  3. otherwise the read stops one byte past the cap, and the rest of the
//     body is drained and counted so Size reports what was actually sent,
//     up to a ceiling of drainCeilingFactor times the cap.
//
// The comparison is strictly greater than: a body of exactly max passes.
//
// max <= 0 is unlimited: the body is read whole and ContentTooLarge is never
// returned. That is the default for the PR diff, whose size is dominated by
// files IsReviewable discards before anything expensive touches them.
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
	defer func() { _ = resp.Body.Close() }()

	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		return "", statusError(http.MethodGet, path, resp)
	}
	// Unlimited: read the body whole, no Content-Length check, no limit reader,
	// no drain. Returning early keeps the capped path below exactly as it was.
	if max <= 0 {
		buf, err := io.ReadAll(resp.Body)
		if err != nil {
			return "", fmt.Errorf("read %s: %w", what, err)
		}
		return strings.ToValidUTF8(string(buf), "�"), nil
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
		// Bound the drain in time as well as bytes: a stalled connection
		// sends no more bytes, so the ceiling would never be reached and the
		// single review worker would block for every team. Closing the body
		// is what interrupts a Read already blocked in the kernel.
		stop := time.AfterFunc(drainTimeout, func() { _ = resp.Body.Close() })
		size, truncated := drainSize(resp.Body, len(buf), max*drainCeilingFactor)
		stop.Stop()
		return "", &ContentTooLarge{
			What: what, Limit: max, Size: &size, Truncated: truncated,
			Head: buf[:max],
		}
	}
	// Invalid UTF-8 is replaced with U+FFFD so the body is valid text.
	return strings.ToValidUTF8(string(buf), "�"), nil
}

// drainCeilingFactor bounds the drain at this multiple of the cap. There is a
// single review worker and the job's ctx is shared, so reading a multi-GB diff
// to its end would stall every team's queue for bytes that are thrown away.
// Ten times the cap still separates "slightly over" from "wildly over".
const drainCeilingFactor = 10

// drainTimeout bounds the drain in wall-clock time. The byte ceiling alone is
// not a bound: a connection that stalls after the cap sends no further bytes,
// so the ceiling is never reached and the read blocks forever, holding the
// single review worker. The caller's ctx does not help either, because the
// review path runs without a deadline.
var drainTimeout = 10 * time.Second

// drainSize reads the rest of body to measure what the cap refused, counting
// from seen and discarding as it goes. It returns the total and whether that
// total is a lower bound, which it is once the ceiling is reached or the read
// fails part way.
//
// It never reports an error. The caller has already decided this body is
// ContentTooLarge, and a drain that fails must not turn that into a generic
// error: the review path posts no notice and writes no row for one, so a torn
// connection here would lose the skip comment the user gets today.
// A body that ends exactly at the ceiling is exact, not a lower bound, so the
// error says "N bytes" rather than "> N bytes": the count is only truncated
// once a byte BEYOND the ceiling is seen.
func drainSize(body io.Reader, seen, ceiling int) (int, bool) {
	total := seen
	if total > ceiling {
		return total, true
	}
	scratch := make([]byte, 32*1024)
	for {
		n, err := body.Read(scratch)
		total += n
		if total > ceiling {
			return total, true
		}
		if err != nil {
			return total, err != io.EOF
		}
	}
}
