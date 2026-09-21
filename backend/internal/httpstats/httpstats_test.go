package httpstats

import (
	"context"
	"net/http"
	"net/http/httptest"
	"reflect"
	"sync"
	"testing"
)

func TestRecordKeysByLabelAndUpperMethod(t *testing.T) {
	_, c := WithScope(context.Background())
	c.Record("bitbucket", "get")
	c.Record("bitbucket", "GET")
	c.Record("bitbucket", "post")
	c.Record("jira", "GET")

	want := map[string]int{"bitbucket:GET": 2, "bitbucket:POST": 1, "jira:GET": 1}
	if got := c.Methods(); !reflect.DeepEqual(got, want) {
		t.Fatalf("Methods() = %v, want %v", got, want)
	}
}

func TestSummarizeRollsUpPerLabel(t *testing.T) {
	_, c := WithScope(context.Background())
	c.Record("bitbucket", "GET")
	c.Record("bitbucket", "POST")
	c.Record("jira", "GET")

	want := map[string]int{"bitbucket": 2, "jira": 1}
	if got := c.Summarize(); !reflect.DeepEqual(got, want) {
		t.Fatalf("Summarize() = %v, want %v", got, want)
	}
	if got, want := Labels(c.Summarize()), []string{"bitbucket", "jira"}; !reflect.DeepEqual(got, want) {
		t.Fatalf("Labels() = %v, want %v", got, want)
	}
}

// Summarize must leave the method breakdown intact: the caller logs both.
func TestSummarizeDoesNotConsumeMethods(t *testing.T) {
	_, c := WithScope(context.Background())
	c.Record("bitbucket", "GET")
	_ = c.Summarize()
	if got := c.Methods()["bitbucket:GET"]; got != 1 {
		t.Fatalf("method breakdown lost after Summarize: %v", c.Methods())
	}
}

// Outside a scope, recording is a silent no-op rather than a panic: adapters
// are called from startup paths that never open one.
func TestNoScopeIsNoOp(t *testing.T) {
	if c := FromContext(context.Background()); c != nil {
		t.Fatalf("FromContext outside a scope = %v, want nil", c)
	}
	FromContext(context.Background()).Record("bitbucket", "GET") // must not panic
	var nilC *Counter
	if got := nilC.Summarize(); got != nil {
		t.Fatalf("nil.Summarize() = %v, want nil", got)
	}
	if got := nilC.Methods(); got != nil {
		t.Fatalf("nil.Methods() = %v, want nil", got)
	}
}

// A nested scope gets its own counter; the outer never sees the inner's counts.
func TestNestedScopeIsIsolated(t *testing.T) {
	outerCtx, outer := WithScope(context.Background())
	outer.Record("bitbucket", "GET")

	_, inner := WithScope(outerCtx)
	inner.Record("bitbucket", "POST")

	if got := outer.Methods(); !reflect.DeepEqual(got, map[string]int{"bitbucket:GET": 1}) {
		t.Fatalf("outer saw inner counts: %v", got)
	}
	if got := inner.Methods(); !reflect.DeepEqual(got, map[string]int{"bitbucket:POST": 1}) {
		t.Fatalf("inner = %v", got)
	}
}

func TestTransportCountsEveryRequestSent(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		// A non-2xx still counts: the request went out.
		if r.URL.Path == "/boom" {
			w.WriteHeader(http.StatusInternalServerError)
			return
		}
		w.WriteHeader(http.StatusOK)
	}))
	defer srv.Close()

	client := &http.Client{Transport: Transport("bitbucket", nil)}
	ctx, c := WithScope(context.Background())

	for _, path := range []string{"/ok", "/ok", "/boom"} {
		req, err := http.NewRequestWithContext(ctx, http.MethodGet, srv.URL+path, nil)
		if err != nil {
			t.Fatal(err)
		}
		resp, err := client.Do(req)
		if err != nil {
			t.Fatal(err)
		}
		resp.Body.Close()
	}

	if got := c.Summarize()["bitbucket"]; got != 3 {
		t.Fatalf("counted %d requests, want 3 (a non-2xx still counts)", got)
	}
}

// Without a scope in the request context the transport must still forward.
func TestTransportWithoutScopeStillForwards(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusOK)
	}))
	defer srv.Close()

	client := &http.Client{Transport: Transport("jira", nil)}
	resp, err := client.Get(srv.URL)
	if err != nil {
		t.Fatalf("request without a scope failed: %v", err)
	}
	resp.Body.Close()
}

// Concurrent requests share the counter, so it must be atomic on its own and
// survive -race.
func TestCounterIsConcurrencySafe(t *testing.T) {
	_, c := WithScope(context.Background())
	var wg sync.WaitGroup
	for i := 0; i < 50; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			c.Record("bitbucket", "GET")
		}()
	}
	wg.Wait()
	if got := c.Methods()["bitbucket:GET"]; got != 50 {
		t.Fatalf("counted %d, want 50", got)
	}
}
