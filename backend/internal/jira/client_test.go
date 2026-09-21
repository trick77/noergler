package jira

import (
	"context"
	"io"
	"log/slog"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/trick77/noergler/internal/config"
	"github.com/trick77/noergler/internal/httpstats"
)

type fake struct {
	srv   *httptest.Server
	paths []string
	query []string
}

func newFake(t *testing.T, handler func(w http.ResponseWriter, r *http.Request)) (*fake, *Client) {
	t.Helper()
	f := &fake{}
	f.srv = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		f.paths = append(f.paths, r.URL.Path)
		f.query = append(f.query, r.URL.RawQuery)
		handler(w, r)
	}))
	t.Cleanup(f.srv.Close)

	c, err := New(config.Jira{
		URL:                        f.srv.URL,
		Token:                      "test-token",
		AcceptanceCriteriaPrefixes: defaultPrefixes,
	}, slog.New(slog.DiscardHandler))
	if err != nil {
		t.Fatal(err)
	}
	return f, c
}

func jsonReply(status int, body string) func(http.ResponseWriter, *http.Request) {
	return func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(status)
		_, _ = io.WriteString(w, body)
	}
}

// --- connectivity ------------------------------------------------------------

func TestCheckConnectivity(t *testing.T) {
	f, c := newFake(t, jsonReply(200, `{"name":"jdoe","displayName":"John Doe"}`))
	if err := c.CheckConnectivity(context.Background()); err != nil {
		t.Fatalf("CheckConnectivity: %v", err)
	}
	if f.paths[0] != "/rest/api/2/myself" {
		t.Errorf("path = %q", f.paths[0])
	}
}

// The startup check is strict, unlike ticket reads.
func TestCheckConnectivity401Propagates(t *testing.T) {
	_, c := newFake(t, jsonReply(401, `{"errorMessages":["denied"]}`))
	if err := c.CheckConnectivity(context.Background()); Status(err) != 401 {
		t.Fatalf("err = %v, want a 401 StatusError", err)
	}
}

func TestCheckConnectivityConnectionErrorPropagates(t *testing.T) {
	c, err := New(config.Jira{URL: "http://127.0.0.1:1", Token: "t"}, slog.New(slog.DiscardHandler))
	if err != nil {
		t.Fatal(err)
	}
	if err := c.CheckConnectivity(context.Background()); err == nil {
		t.Fatal("an unreachable Jira must fail the startup check")
	}
}

// --- fetching ----------------------------------------------------------------

const fullIssue = `{
  "key": "SEP-22888",
  "fields": {
    "summary": "Add auth filter",
    "description": "h1. Overview\nAK-1: Must handle auth",
    "labels": ["security", "backend"],
    "subtasks": [
      {"key": "SEP-22889", "fields": {"summary": "Implement auth filter"}},
      {"key": "SEP-22890", "fields": {"summary": "Add tests"}}
    ],
    "issuetype": {"name": "Story"},
    "status": {"name": "In Progress"},
    "parent": {"key": "SEP-100"}
  }
}`

func TestFetchTicket(t *testing.T) {
	f, c := newFake(t, jsonReply(200, fullIssue))
	got, err := c.FetchTicket(context.Background(), "SEP-22888")
	if err != nil {
		t.Fatal(err)
	}
	if got == nil {
		t.Fatal("ticket = nil")
	}
	if got.Key != "SEP-22888" || got.Title != "Add auth filter" {
		t.Errorf("key/title = %q/%q", got.Key, got.Title)
	}
	if len(got.Labels) != 2 || got.Labels[0] != "security" {
		t.Errorf("labels = %v", got.Labels)
	}
	if len(got.Subtasks) != 2 || got.Subtasks[0] != "SEP-22889: Implement auth filter" {
		t.Errorf("subtasks = %v", got.Subtasks)
	}
	if got.IssueType != "Story" || got.Status != "In Progress" || got.ParentKey != "SEP-100" {
		t.Errorf("type/status/parent = %q/%q/%q", got.IssueType, got.Status, got.ParentKey)
	}
	if want := f.srv.URL + "/browse/SEP-22888"; got.URL != want {
		t.Errorf("url = %q, want %q", got.URL, want)
	}
	// Markup is stripped and the criteria extracted from the stripped text.
	if strings.Contains(got.Description, "h1.") {
		t.Errorf("description still has markup: %q", got.Description)
	}
	if got.AcceptanceCriteria != "AK-1: Must handle auth" {
		t.Errorf("acceptance criteria = %q", got.AcceptanceCriteria)
	}
	// The field list is a contract; the commas must not be escaped.
	if f.query[0] != fieldsQuery {
		t.Errorf("query = %q, want %q", f.query[0], fieldsQuery)
	}
}

// A branch name may reference a ticket that does not exist. Not an error.
func TestFetchTicketNotFound(t *testing.T) {
	_, c := newFake(t, jsonReply(404, `{"errorMessages":["Issue does not exist"]}`))
	got, err := c.FetchTicket(context.Background(), "SEP-1")
	if err != nil || got != nil {
		t.Fatalf("got (%v, %v), want (nil, nil)", got, err)
	}
}

// Nor is a Jira that is down, or one that rejects the read.
func TestFetchTicketAPIErrorIsSwallowed(t *testing.T) {
	_, c := newFake(t, jsonReply(500, `{"errorMessages":["boom"]}`))
	got, err := c.FetchTicket(context.Background(), "SEP-1")
	if err != nil || got != nil {
		t.Fatalf("got (%v, %v), want (nil, nil)", got, err)
	}
}

func TestFetchTicketConnectionRefusedIsSwallowed(t *testing.T) {
	c, err := New(config.Jira{URL: "http://127.0.0.1:1", Token: "t"}, slog.New(slog.DiscardHandler))
	if err != nil {
		t.Fatal(err)
	}
	got, err := c.FetchTicket(context.Background(), "SEP-1")
	if err != nil || got != nil {
		t.Fatalf("got (%v, %v), want (nil, nil): a refused connection is not a review failure", got, err)
	}
}

// A cancelled or timed-out call is a real failure: it says the call went wrong,
// not that the ticket is absent.
func TestFetchTicketContextCancellationPropagates(t *testing.T) {
	_, c := newFake(t, jsonReply(200, fullIssue))
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if _, err := c.FetchTicket(ctx, "SEP-1"); err == nil {
		t.Fatal("a cancelled context must surface as an error")
	}
}

// A malformed body is a real failure too.
func TestFetchTicketMalformedJSONPropagates(t *testing.T) {
	_, c := newFake(t, jsonReply(200, `{"key": "SEP-1", "fields":`))
	if _, err := c.FetchTicket(context.Background(), "SEP-1"); err == nil {
		t.Fatal("a truncated body must surface as an error")
	}
}

// Truncation happens on the raw markup, before stripping, and appends "...".
func TestFetchTicketTruncatesDescription(t *testing.T) {
	long := strings.Repeat("x", 6000)
	_, c := newFake(t, jsonReply(200, `{"key":"S-1","fields":{"description":"`+long+`"}}`))
	got, err := c.FetchTicket(context.Background(), "S-1")
	if err != nil {
		t.Fatal(err)
	}
	if len(got.Description) != maxDescriptionLength+3 {
		t.Errorf("length = %d, want %d", len(got.Description), maxDescriptionLength+3)
	}
	if !strings.HasSuffix(got.Description, "...") {
		t.Errorf("description does not end in an ellipsis: %q", got.Description[len(got.Description)-10:])
	}
}

func TestFetchTicketNullDescription(t *testing.T) {
	_, c := newFake(t, jsonReply(200, `{"key":"S-1","fields":{"description":null,"summary":"t"}}`))
	got, err := c.FetchTicket(context.Background(), "S-1")
	if err != nil {
		t.Fatal(err)
	}
	if got.Description != "" || got.AcceptanceCriteria != "" {
		t.Errorf("description = %q, criteria = %q", got.Description, got.AcceptanceCriteria)
	}
}

// Absent issuetype/status/parent are empty, not a crash.
func TestFetchTicketMissingOptionalFields(t *testing.T) {
	_, c := newFake(t, jsonReply(200, `{"key":"S-1","fields":{"summary":"t"}}`))
	got, err := c.FetchTicket(context.Background(), "S-1")
	if err != nil {
		t.Fatal(err)
	}
	if got.IssueType != "" || got.Status != "" || got.ParentKey != "" {
		t.Errorf("optional fields = %q/%q/%q", got.IssueType, got.Status, got.ParentKey)
	}
	if got.Subtasks != nil {
		t.Errorf("subtasks = %v, want nil", got.Subtasks)
	}
}

// A subtask entry without a key is skipped rather than rendered as ": summary".
func TestFetchTicketSkipsKeylessSubtasks(t *testing.T) {
	body := `{"key":"S-1","fields":{"subtasks":[{"fields":{"summary":"no key"}},{"key":"S-2","fields":{"summary":"ok"}}]}}`
	_, c := newFake(t, jsonReply(200, body))
	got, err := c.FetchTicket(context.Background(), "S-1")
	if err != nil {
		t.Fatal(err)
	}
	if len(got.Subtasks) != 1 || got.Subtasks[0] != "S-2: ok" {
		t.Errorf("subtasks = %v", got.Subtasks)
	}
}

// The response key wins over the requested id, but a response without one falls
// back rather than producing an empty key.
func TestFetchTicketKeyFallsBackToTheRequestedID(t *testing.T) {
	_, c := newFake(t, jsonReply(200, `{"fields":{"summary":"t"}}`))
	got, err := c.FetchTicket(context.Background(), "SEP-7")
	if err != nil {
		t.Fatal(err)
	}
	if got.Key != "SEP-7" {
		t.Errorf("key = %q, want the requested id", got.Key)
	}
}

// --- parent ------------------------------------------------------------------

// Acceptance criteria often live on the parent story, not the subtask.
func TestFetchTicketWithParent(t *testing.T) {
	var n int
	_, c := newFake(t, func(w http.ResponseWriter, _ *http.Request) {
		n++
		if n == 1 {
			jsonReply(200, fullIssue)(w, nil)
			return
		}
		jsonReply(200, `{"key":"SEP-100","fields":{"summary":"Parent story"}}`)(w, nil)
	})

	ticket, parent, err := c.FetchTicketWithParent(context.Background(), "SEP-22888")
	if err != nil {
		t.Fatal(err)
	}
	if ticket == nil || ticket.ParentKey != "SEP-100" {
		t.Fatalf("ticket = %+v", ticket)
	}
	if parent == nil || parent.Title != "Parent story" {
		t.Fatalf("parent = %+v", parent)
	}
}

func TestFetchTicketWithParentNoParent(t *testing.T) {
	f, c := newFake(t, jsonReply(200, `{"key":"S-1","fields":{"summary":"t"}}`))
	ticket, parent, err := c.FetchTicketWithParent(context.Background(), "S-1")
	if err != nil {
		t.Fatal(err)
	}
	if ticket == nil || parent != nil {
		t.Fatalf("got (%v, %v), want the ticket and no parent", ticket, parent)
	}
	if len(f.paths) != 1 {
		t.Errorf("made %d requests, want 1", len(f.paths))
	}
}

// A parent that cannot be read must not cost us the child.
func TestFetchTicketWithParentParentMissing(t *testing.T) {
	var n int
	_, c := newFake(t, func(w http.ResponseWriter, _ *http.Request) {
		n++
		if n == 1 {
			jsonReply(200, `{"key":"S-1","fields":{"summary":"t","parent":{"key":"SEP-300"}}}`)(w, nil)
			return
		}
		jsonReply(404, `{"errorMessages":["gone"]}`)(w, nil)
	})

	ticket, parent, err := c.FetchTicketWithParent(context.Background(), "S-1")
	if err != nil {
		t.Fatal(err)
	}
	if ticket == nil || ticket.ParentKey != "SEP-300" {
		t.Fatalf("ticket = %+v, want the child intact", ticket)
	}
	if parent != nil {
		t.Errorf("parent = %+v, want nil", parent)
	}
}

// A parent whose fetch fails outright (timeout, bad JSON) must not cost the
// caller the child: the usual `if err != nil { return }` would throw it away.
func TestFetchTicketWithParentParentErrorKeepsTheChild(t *testing.T) {
	var n int
	_, c := newFake(t, func(w http.ResponseWriter, _ *http.Request) {
		n++
		if n == 1 {
			jsonReply(200, `{"key":"S-1","fields":{"summary":"child","parent":{"key":"SEP-300"}}}`)(w, nil)
			return
		}
		jsonReply(200, `{"key": "SEP-300", "fields":`)(w, nil) // truncated
	})

	ticket, parent, err := c.FetchTicketWithParent(context.Background(), "S-1")
	if err != nil {
		t.Fatalf("err = %v, want nil: a bad parent is not the child's fault", err)
	}
	if ticket == nil || ticket.Title != "child" {
		t.Fatalf("ticket = %+v, want the child intact", ticket)
	}
	if parent != nil {
		t.Errorf("parent = %+v, want nil", parent)
	}
}

func TestFetchTicketWithParentTicketMissing(t *testing.T) {
	_, c := newFake(t, jsonReply(404, `{"errorMessages":["gone"]}`))
	ticket, parent, err := c.FetchTicketWithParent(context.Background(), "S-1")
	if err != nil || ticket != nil || parent != nil {
		t.Fatalf("got (%v, %v, %v), want all nil", ticket, parent, err)
	}
}

// --- plumbing ----------------------------------------------------------------

func TestRequestsCarryTheBearerTokenAndAreCounted(t *testing.T) {
	var auth string
	_, c := newFake(t, func(w http.ResponseWriter, r *http.Request) {
		auth = r.Header.Get("Authorization")
		jsonReply(200, `{"key":"S-1","fields":{}}`)(w, nil)
	})

	ctx, counter := httpstats.WithScope(context.Background())
	if _, err := c.FetchTicket(ctx, "S-1"); err != nil {
		t.Fatal(err)
	}
	if auth != "Bearer test-token" {
		t.Errorf("Authorization = %q", auth)
	}
	if got := counter.Methods()["jira:GET"]; got != 1 {
		t.Errorf("counted %d GETs, want 1", got)
	}
}

func TestNewRejectsARelativeURL(t *testing.T) {
	if _, err := New(config.Jira{URL: "jira.example.com"}, slog.New(slog.DiscardHandler)); err == nil {
		t.Fatal("a URL without a scheme must be refused")
	}
}

func TestNewTrimsTrailingSlash(t *testing.T) {
	c, err := New(config.Jira{URL: "https://jira.example.com/"}, slog.New(slog.DiscardHandler))
	if err != nil {
		t.Fatal(err)
	}
	if c.base != "https://jira.example.com" {
		t.Errorf("base = %q", c.base)
	}
}
