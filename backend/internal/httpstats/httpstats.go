// Package httpstats counts outbound HTTP round-trips per scope, so a log line
// can say how many calls one review run made against Bitbucket and Jira.
//
// The counter rides in the context, like the Python service's ContextVar, so a
// fan-out inherits it without threading state through every signature. Unlike
// the asyncio original this counter is mutex-guarded: goroutines really do run
// concurrently, where the event loop only interleaved.
package httpstats

import (
	"context"
	"net/http"
	"sort"
	"strings"
	"sync"
)

type ctxKey struct{}

// Counter tallies requests by "<label>:<METHOD>".
type Counter struct {
	mu     sync.Mutex
	counts map[string]int
}

// WithScope returns a context carrying a fresh counter. A nested scope gets its
// own: the outer never sees the inner's counts.
func WithScope(ctx context.Context) (context.Context, *Counter) {
	c := &Counter{counts: make(map[string]int)}
	return context.WithValue(ctx, ctxKey{}, c), c
}

// FromContext returns the scope's counter, or nil outside a scope.
func FromContext(ctx context.Context) *Counter {
	c, _ := ctx.Value(ctxKey{}).(*Counter)
	return c
}

// Record bumps one request. A nil receiver is a no-op, so callers outside a
// scope need no guard.
func (c *Counter) Record(label, method string) {
	if c == nil {
		return
	}
	key := label + ":" + strings.ToUpper(method)
	c.mu.Lock()
	defer c.mu.Unlock()
	c.counts[key]++
}

// Methods returns a copy of the per-"<label>:<METHOD>" counts.
func (c *Counter) Methods() map[string]int {
	if c == nil {
		return nil
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	out := make(map[string]int, len(c.counts))
	for k, v := range c.counts {
		out[k] = v
	}
	return out
}

// Summarize rolls the per-method counts up into per-label totals. The method
// breakdown stays in the counter for the caller to log alongside.
func (c *Counter) Summarize() map[string]int {
	if c == nil {
		return nil
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	totals := make(map[string]int)
	for key, n := range c.counts {
		label, _, _ := strings.Cut(key, ":")
		if label != "" {
			totals[label] += n
		}
	}
	return totals
}

// Labels returns the keys of a Summarize result sorted, for a stable log line.
func Labels(totals map[string]int) []string {
	out := make([]string, 0, len(totals))
	for k := range totals {
		out = append(out, k)
	}
	sort.Strings(out)
	return out
}

// Transport counts every request it forwards under label. It fires per request
// actually sent, so a retried POST counts twice, each page of a paged call
// counts, and a request answered with a non-2xx still counts.
func Transport(label string, next http.RoundTripper) http.RoundTripper {
	if next == nil {
		next = http.DefaultTransport
	}
	return &transport{label: label, next: next}
}

type transport struct {
	label string
	next  http.RoundTripper
}

func (t *transport) RoundTrip(req *http.Request) (*http.Response, error) {
	FromContext(req.Context()).Record(t.label, req.Method)
	return t.next.RoundTrip(req)
}

// Unwrap returns the wrapped RoundTripper, so a caller that supplied a tuned
// transport can assert on the one actually installed rather than on a second
// copy built the same way.
func (t *transport) Unwrap() http.RoundTripper { return t.next }
