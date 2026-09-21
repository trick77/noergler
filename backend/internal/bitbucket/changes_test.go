package bitbucket

import (
	"context"
	"net/http"
	"strings"
	"testing"
)

// The happy path: paths come back in the order Bitbucket reports them, since
// the too-large log lists them as the PR has them.
func TestFetchPRChanges(t *testing.T) {
	f, c := newFake(t, jsonReply(200, `{"values":[
		{"nodeType":"FILE","type":"MODIFY","path":{"toString":"src/app.go"}},
		{"nodeType":"FILE","type":"ADD","path":{"toString":"fixtures/big.json"}}
	],"isLastPage":true}`))

	got, err := c.FetchPRChanges(context.Background(), "PROJ", "repo", 7)
	if err != nil {
		t.Fatalf("FetchPRChanges: %v", err)
	}
	if len(got) != 2 || got[0] != "src/app.go" || got[1] != "fixtures/big.json" {
		t.Errorf("got %v, want both paths in order", got)
	}
	if path := f.seen[0].Path; !strings.HasSuffix(path, "/pull-requests/7/changes") {
		t.Errorf("requested %q, want the PR's changes endpoint", path)
	}
}

// A directory entry carries no diff, so it must not be reported as a file.
func TestFetchPRChangesSkipsDirectories(t *testing.T) {
	_, c := newFake(t, jsonReply(200, `{"values":[
		{"nodeType":"DIRECTORY","path":{"toString":"src"}},
		{"nodeType":"FILE","path":{"toString":"src/app.go"}}
	],"isLastPage":true}`))

	got, err := c.FetchPRChanges(context.Background(), "PROJ", "repo", 7)
	if err != nil {
		t.Fatalf("FetchPRChanges: %v", err)
	}
	if len(got) != 1 || got[0] != "src/app.go" {
		t.Errorf("got %v, want the directory dropped", got)
	}
}

// The response shape is unverified against a real server, so an entry that
// does not carry a path is skipped rather than failing the call: this runs on
// a failure path and must degrade to a short list, never to a second error.
func TestFetchPRChangesToleratesUnexpectedEntries(t *testing.T) {
	_, c := newFake(t, jsonReply(200, `{"values":[
		{"nodeType":"FILE"},
		{"nodeType":"FILE","path":"not-an-object"},
		{"nodeType":"FILE","path":{"toString":""}},
		{"nodeType":"FILE","path":{"toString":"src/app.go"}}
	],"isLastPage":true}`))

	got, err := c.FetchPRChanges(context.Background(), "PROJ", "repo", 7)
	if err != nil {
		t.Fatalf("FetchPRChanges: %v", err)
	}
	if len(got) != 1 || got[0] != "src/app.go" {
		t.Errorf("got %v, want only the well-formed entry", got)
	}
}

// An entry with no nodeType is a file: older servers omit it, and dropping
// those would silently empty the list.
func TestFetchPRChangesTreatsMissingNodeTypeAsFile(t *testing.T) {
	_, c := newFake(t, jsonReply(200, `{"values":[
		{"path":{"toString":"src/app.go"}}
	],"isLastPage":true}`))

	got, err := c.FetchPRChanges(context.Background(), "PROJ", "repo", 7)
	if err != nil {
		t.Fatalf("FetchPRChanges: %v", err)
	}
	if len(got) != 1 {
		t.Errorf("got %v, want the entry kept", got)
	}
}

// The shape being wrong in every entry must be an ERROR, not an empty list:
// the caller would otherwise report head-only scope as though /changes had
// never been tried, hiding exactly the breakage this call exists to surface.
func TestFetchPRChangesErrorsWhenNothingParses(t *testing.T) {
	_, c := newFake(t, jsonReply(200, `{"values":[
		{"nodeType":"FILE","srcPath":{"toString":"src/app.go"}},
		{"nodeType":"FILE","srcPath":{"toString":"src/other.go"}}
	],"isLastPage":true}`))

	_, err := c.FetchPRChanges(context.Background(), "PROJ", "repo", 7)
	if err == nil {
		t.Fatal("want an error when no entry carries a path")
	}
	if !strings.Contains(err.Error(), "2 entries") {
		t.Errorf("error %q does not say how many entries were seen", err)
	}
}

// An empty PR is an empty list, not an error: nothing was misparsed.
func TestFetchPRChangesAllowsAnEmptyList(t *testing.T) {
	_, c := newFake(t, jsonReply(200, `{"values":[],"isLastPage":true}`))

	got, err := c.FetchPRChanges(context.Background(), "PROJ", "repo", 7)
	if err != nil {
		t.Fatalf("FetchPRChanges: %v", err)
	}
	if len(got) != 0 {
		t.Errorf("got %v, want an empty list", got)
	}
}

// The caller logs the error and carries on with the head alone, so the error
// has to surface rather than be swallowed into an empty list.
func TestFetchPRChangesReportsErrors(t *testing.T) {
	_, c := newFake(t, jsonReply(http.StatusNotFound, `{"errors":[{"message":"no such pull request"}]}`))

	if _, err := c.FetchPRChanges(context.Background(), "PROJ", "repo", 7); err == nil {
		t.Fatal("want an error for a 404")
	}
}
