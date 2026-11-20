package bitbucket

import (
	"context"
	"net/http"
	"testing"
)

// The response shape as a Data Center instance sends it (2026-11-20): lines
// HTML-escaped, highlight markup in empty pairs. One request, the query and
// limit as given, the lines handed back as they stand in the file.
func TestSearchCode(t *testing.T) {
	f, c := newFake(t, jsonReply(http.StatusOK, `{"scope":{"type":"GLOBAL"},"code":{"category":"primary",
		"isLastPage":true,"count":1,"start":0,"values":[{"repository":{"slug":"app","project":{"key":"PROJ"}},
		"file":"src/session/manager.go","hitContexts":[[{"line":41,"text":"\tts := &quot;x&quot;"},
		{"line":42,"text":"\ts.<em></em>Touch(ctx, key, <em>int64</em>(d.Seconds()))"}]]}]}}`))

	hits, err := c.SearchCode(context.Background(), "Touch project:PROJ repo:app", 25)
	if err != nil {
		t.Fatal(err)
	}
	if f.calls() != 1 || f.last().Method != http.MethodPost || f.last().Path != "/rest/search/latest/search" {
		t.Fatalf("request = %+v", f.last())
	}
	if f.last().Body["query"] != "Touch project:PROJ repo:app" {
		t.Errorf("query = %v", f.last().Body["query"])
	}
	if len(hits) != 1 || hits[0].Project != "PROJ" || hits[0].Repo != "app" || hits[0].File != "src/session/manager.go" {
		t.Fatalf("hits = %+v", hits)
	}
	want := []CodeLine{{41, "\tts := \"x\""}, {42, "\ts.Touch(ctx, key, int64(d.Seconds()))"}}
	if len(hits[0].Lines) != 2 || hits[0].Lines[0] != want[0] || hits[0].Lines[1] != want[1] {
		t.Errorf("lines = %+v, want %+v", hits[0].Lines, want)
	}
}

// A search the instance does not run is an error the caller can fail open on.
func TestSearchCodeUnavailable(t *testing.T) {
	_, c := newFake(t, jsonReply(http.StatusNotFound, `{"errors":[{"message":"search is unavailable"}]}`))
	if _, err := c.SearchCode(context.Background(), "x", 1); err == nil {
		t.Error("want an error")
	}
}
