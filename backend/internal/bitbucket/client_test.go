package bitbucket

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"net/http/httptest"
	"net/url"
	"strings"
	"testing"

	"github.com/trick77/noergler-go/internal/config"
	"github.com/trick77/noergler-go/internal/httpstats"
)

// recorded is one request the fake server saw.
type recorded struct {
	Method string
	Path   string
	Query  url.Values
	Header http.Header
	Body   map[string]any
}

// fake is a Bitbucket stand-in: one handler, every request recorded.
type fake struct {
	t       *testing.T
	srv     *httptest.Server
	seen    []recorded
	handler func(w http.ResponseWriter, r *http.Request)
}

func newFake(t *testing.T, handler func(w http.ResponseWriter, r *http.Request)) (*fake, *Client) {
	t.Helper()
	f := &fake{t: t, handler: handler}
	f.srv = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		rec := recorded{Method: r.Method, Path: r.URL.Path, Query: r.URL.Query(), Header: r.Header.Clone()}
		if raw, _ := io.ReadAll(r.Body); len(raw) > 0 {
			_ = json.Unmarshal(raw, &rec.Body)
		}
		f.seen = append(f.seen, rec)
		f.handler(w, r)
	}))
	t.Cleanup(f.srv.Close)

	c, err := New(config.Bitbucket{
		BaseURL:      f.srv.URL,
		Token:        "test-token",
		Username:     "bot-user",
		MaxDiffBytes: 10 * 1024 * 1024,
		MaxFileBytes: 1024 * 1024,
	}, slog.New(slog.DiscardHandler))
	if err != nil {
		t.Fatal(err)
	}
	return f, c
}

func (f *fake) calls() int { return len(f.seen) }

func (f *fake) last() recorded {
	f.t.Helper()
	if len(f.seen) == 0 {
		f.t.Fatal("no request recorded")
	}
	return f.seen[len(f.seen)-1]
}

// text replies 200 with a plain-text body.
func text(body string) func(http.ResponseWriter, *http.Request) {
	return func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "text/plain")
		_, _ = io.WriteString(w, body)
	}
}

// jsonReply replies with status and a JSON body.
func jsonReply(status int, body string) func(http.ResponseWriter, *http.Request) {
	return func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(status)
		_, _ = io.WriteString(w, body)
	}
}

// --- connectivity ------------------------------------------------------------

func TestCheckConnectivity(t *testing.T) {
	_, c := newFake(t, jsonReply(200, `{"version":"8.19.2","displayName":"Bitbucket"}`))
	if err := c.CheckConnectivity(context.Background()); err != nil {
		t.Fatalf("CheckConnectivity: %v", err)
	}
}

func TestCheckConnectivityFailurePropagates(t *testing.T) {
	_, c := newFake(t, jsonReply(401, `{"errors":[{"message":"unauthorized"}]}`))
	err := c.CheckConnectivity(context.Background())
	if Status(err) != 401 {
		t.Fatalf("err = %v, want a 401 StatusError", err)
	}
}

// --- diffs -------------------------------------------------------------------

func TestFetchPRDiff(t *testing.T) {
	const diff = "diff --git a/file.py b/file.py\n+hello\n"
	f, c := newFake(t, text(diff))

	got, err := c.FetchPRDiff(context.Background(), "PROJ", "my-repo", 1, 0)
	if err != nil {
		t.Fatal(err)
	}
	if got != diff {
		t.Errorf("diff = %q, want %q", got, diff)
	}
	if want := "/rest/api/1.0/projects/PROJ/repos/my-repo/pull-requests/1/diff"; f.last().Path != want {
		t.Errorf("path = %q, want %q", f.last().Path, want)
	}
	if got := f.last().Header.Get("Accept"); got != "text/plain" {
		t.Errorf("Accept = %q, want text/plain", got)
	}
	if got := f.last().Header.Get("Authorization"); got != "Bearer test-token" {
		t.Errorf("Authorization = %q", got)
	}
}

// contextLines is omitted at 0, not sent as "0": Bitbucket treats the presence
// of the parameter as a request for context.
func TestFetchPRDiffOmitsContextLinesByDefault(t *testing.T) {
	f, c := newFake(t, text("d"))
	if _, err := c.FetchPRDiff(context.Background(), "PROJ", "my-repo", 1, 0); err != nil {
		t.Fatal(err)
	}
	if _, ok := f.last().Query["contextLines"]; ok {
		t.Errorf("contextLines present: %v", f.last().Query)
	}
}

func TestFetchPRDiffWithContextLines(t *testing.T) {
	f, c := newFake(t, text("d"))
	if _, err := c.FetchPRDiff(context.Background(), "PROJ", "my-repo", 1, 20); err != nil {
		t.Fatal(err)
	}
	if got := f.last().Query.Get("contextLines"); got != "20" {
		t.Errorf("contextLines = %q, want 20", got)
	}
}

func TestFetchCommitDiffSendsFromAndTo(t *testing.T) {
	f, c := newFake(t, text("compare-diff"))
	got, err := c.FetchCommitDiff(context.Background(), "PROJ", "my-repo", "abc123", "def456")
	if err != nil {
		t.Fatal(err)
	}
	if got != "compare-diff" {
		t.Errorf("diff = %q", got)
	}
	if f.last().Query.Get("from") != "abc123" || f.last().Query.Get("to") != "def456" {
		t.Errorf("query = %v", f.last().Query)
	}
}

// 406 means the two commits have no comparable history, normally a rebase. It
// must be distinguishable so the reviewer falls back to a full review.
func TestFetchCommitDiff406IsTyped(t *testing.T) {
	_, c := newFake(t, jsonReply(406, `{"errors":[{"message":"not acceptable"}]}`))
	_, err := c.FetchCommitDiff(context.Background(), "PROJ", "my-repo", "abc1234567890", "def4567890abc")
	if !errors.Is(err, ErrIncrementalDiffUnavailable) {
		t.Fatalf("err = %v, want ErrIncrementalDiffUnavailable", err)
	}
	// The message carries the shortened SHAs the operator sees in the log.
	if !strings.Contains(err.Error(), "abc1234567") || !strings.Contains(err.Error(), "def4567890") {
		t.Errorf("err = %v, want the short SHAs", err)
	}
}

// Any other status stays a plain status error, not the typed fallback signal.
func TestFetchCommitDiff404IsNotTyped(t *testing.T) {
	_, c := newFake(t, jsonReply(404, `{"errors":[{"message":"gone"}]}`))
	_, err := c.FetchCommitDiff(context.Background(), "PROJ", "my-repo", "abc", "def")
	if errors.Is(err, ErrIncrementalDiffUnavailable) {
		t.Fatalf("404 reported as an incremental-diff failure: %v", err)
	}
	if Status(err) != 404 {
		t.Fatalf("err = %v, want a 404 StatusError", err)
	}
}

func TestFetchFileContent(t *testing.T) {
	f, c := newFake(t, text("print('hi')\n"))
	got, err := c.FetchFileContent(context.Background(), "PROJ", "my-repo", "abc", "src/main.py")
	if err != nil {
		t.Fatal(err)
	}
	if got != "print('hi')\n" {
		t.Errorf("content = %q", got)
	}
	if want := "/rest/api/1.0/projects/PROJ/repos/my-repo/raw/src/main.py"; f.last().Path != want {
		t.Errorf("path = %q, want %q", f.last().Path, want)
	}
	if got := f.last().Query.Get("at"); got != "abc" {
		t.Errorf("at = %q", got)
	}
	// Unlike the diff endpoints this one keeps the default JSON Accept.
	if got := f.last().Header.Get("Accept"); got != "application/json" {
		t.Errorf("Accept = %q, want application/json", got)
	}
}

// Python interpolated the path raw, so a space or '#' produced a broken URL.
// Divergence: the path is escaped, and '#' does not become a fragment.
func TestFetchFileContentEscapesPath(t *testing.T) {
	f, c := newFake(t, text("x"))
	if _, err := c.FetchFileContent(context.Background(), "PROJ", "r", "abc", "src/my file#1.py"); err != nil {
		t.Fatal(err)
	}
	if want := "/rest/api/1.0/projects/PROJ/repos/r/raw/src/my file#1.py"; f.last().Path != want {
		t.Errorf("decoded path = %q, want %q", f.last().Path, want)
	}
}

// --- byte caps ---------------------------------------------------------------

// A declared Content-Length over the cap fails before the body is read, and
// reports the size the server promised.
func TestDiffOverCapByContentLength(t *testing.T) {
	_, c := newFake(t, text(strings.Repeat("x", 100)))
	c.maxDiffBytes = 10

	_, err := c.FetchPRDiff(context.Background(), "PROJ", "my-repo", 1, 0)
	var tooLarge *ContentTooLarge
	if !errors.As(err, &tooLarge) {
		t.Fatalf("err = %v, want ContentTooLarge", err)
	}
	if tooLarge.Size == nil || *tooLarge.Size != 100 {
		t.Errorf("Size = %v, want 100", tooLarge.Size)
	}
	if tooLarge.What != "PROJ/my-repo#1 diff" {
		t.Errorf("What = %q", tooLarge.What)
	}
}

// Without a declared length the cap trips mid-stream and the size is unknown.
func TestDiffOverCapWhileStreaming(t *testing.T) {
	_, c := newFake(t, func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "text/plain")
		w.(http.Flusher).Flush() // chunked: no Content-Length
		for i := 0; i < 5; i++ {
			_, _ = io.WriteString(w, "xxxx")
			w.(http.Flusher).Flush()
		}
	})
	c.maxDiffBytes = 10

	_, err := c.FetchPRDiff(context.Background(), "PROJ", "my-repo", 1, 0)
	var tooLarge *ContentTooLarge
	if !errors.As(err, &tooLarge) {
		t.Fatalf("err = %v, want ContentTooLarge", err)
	}
	if tooLarge.Size != nil {
		t.Errorf("Size = %v, want nil when the length was never declared", *tooLarge.Size)
	}
}

// Exactly at the cap is allowed: the comparison is strictly greater than.
func TestDiffExactlyAtCapPasses(t *testing.T) {
	_, c := newFake(t, text(strings.Repeat("x", 10)))
	c.maxDiffBytes = 10

	got, err := c.FetchPRDiff(context.Background(), "PROJ", "my-repo", 1, 0)
	if err != nil {
		t.Fatalf("a body of exactly the cap was rejected: %v", err)
	}
	if got != strings.Repeat("x", 10) {
		t.Errorf("body = %q", got)
	}
}

// The file cap is separate from the diff cap and names the bare path.
func TestFileOverCapNamesThePath(t *testing.T) {
	_, c := newFake(t, text(strings.Repeat("y", 11)))
	c.maxFileBytes = 10

	_, err := c.FetchFileContent(context.Background(), "PROJ", "my-repo", "abc", "big.bin")
	var tooLarge *ContentTooLarge
	if !errors.As(err, &tooLarge) {
		t.Fatalf("err = %v, want ContentTooLarge", err)
	}
	if tooLarge.What != "big.bin" {
		t.Errorf("What = %q, want big.bin", tooLarge.What)
	}
}

// An error status wins over the cap: a huge error page is a failed request,
// not an oversized diff.
func TestErrorStatusWinsOverCap(t *testing.T) {
	_, c := newFake(t, jsonReply(500, strings.Repeat("e", 100)))
	c.maxDiffBytes = 10

	_, err := c.FetchPRDiff(context.Background(), "PROJ", "my-repo", 1, 0)
	var tooLarge *ContentTooLarge
	if errors.As(err, &tooLarge) {
		t.Fatalf("reported as ContentTooLarge, want the status error: %v", err)
	}
	if Status(err) != 500 {
		t.Fatalf("err = %v, want a 500 StatusError", err)
	}
}

// Invalid UTF-8 is replaced, not passed through: the text goes into a prompt
// and a JSON payload later.
func TestInvalidUTF8IsReplaced(t *testing.T) {
	_, c := newFake(t, func(w http.ResponseWriter, _ *http.Request) {
		_, _ = w.Write([]byte{'o', 'k', 0xff, 0xfe})
	})
	got, err := c.FetchPRDiff(context.Background(), "PROJ", "r", 1, 0)
	if err != nil {
		t.Fatal(err)
	}
	if !strings.HasPrefix(got, "ok") || strings.ContainsRune(got, 0xff) {
		t.Errorf("body = %q, want the invalid bytes replaced", got)
	}
}

// --- comments ----------------------------------------------------------------

func TestPostInlineCommentAnchor(t *testing.T) {
	f, c := newFake(t, jsonReply(201, `{"id":7}`))
	id, err := c.PostInlineComment(context.Background(), "PROJ", "my-repo", 1, "src/main.py", 10, "**Issue:** Bug here")
	if err != nil {
		t.Fatal(err)
	}
	if id != 7 {
		t.Errorf("id = %d, want 7", id)
	}
	anchor, _ := f.last().Body["anchor"].(map[string]any)
	if anchor["path"] != "src/main.py" || anchor["fileType"] != "TO" || anchor["lineType"] != "ADDED" {
		t.Errorf("anchor = %v", anchor)
	}
	if anchor["line"] != float64(10) {
		t.Errorf("line = %v, want 10", anchor["line"])
	}
	if f.last().Body["text"] != "**Issue:** Bug here" {
		t.Errorf("text = %v; the client must post the body verbatim", f.last().Body["text"])
	}
}

// A git diff path keeps its a/ or b/ prefix; the anchor needs the repo path.
func TestPostInlineCommentStripsDiffPrefix(t *testing.T) {
	f, c := newFake(t, jsonReply(201, `{"id":1}`))
	if _, err := c.PostInlineComment(context.Background(), "PROJ", "r", 1, "b/src/main.py", 3, "x"); err != nil {
		t.Fatal(err)
	}
	anchor, _ := f.last().Body["anchor"].(map[string]any)
	if anchor["path"] != "src/main.py" {
		t.Errorf("path = %v, want src/main.py", anchor["path"])
	}
}

// Bitbucket rejects an ADDED anchor on a context line; retry once as CONTEXT.
func TestPostInlineCommentFallsBackToContext(t *testing.T) {
	var n int
	f, c := newFake(t, func(w http.ResponseWriter, _ *http.Request) {
		n++
		if n == 1 {
			jsonReply(400, `{"errors":[{"message":"invalid line"}]}`)(w, nil)
			return
		}
		jsonReply(201, `{"id":9}`)(w, nil)
	})

	id, err := c.PostInlineComment(context.Background(), "PROJ", "r", 1, "src/main.py", 10, "x")
	if err != nil {
		t.Fatal(err)
	}
	if id != 9 {
		t.Errorf("id = %d, want 9", id)
	}
	if f.calls() != 2 {
		t.Fatalf("made %d requests, want 2", f.calls())
	}
	first, _ := f.seen[0].Body["anchor"].(map[string]any)
	second, _ := f.seen[1].Body["anchor"].(map[string]any)
	if first["lineType"] != "ADDED" || second["lineType"] != "CONTEXT" {
		t.Errorf("lineTypes = %v then %v", first["lineType"], second["lineType"])
	}
}

// Both anchors rejected: the error surfaces rather than being swallowed.
func TestPostInlineCommentBothAnchorsRejected(t *testing.T) {
	f, c := newFake(t, jsonReply(400, `{"errors":[{"message":"invalid line"}]}`))
	_, err := c.PostInlineComment(context.Background(), "PROJ", "r", 1, "src/main.py", 10, "x")
	if Status(err) != 400 {
		t.Fatalf("err = %v, want a 400 StatusError", err)
	}
	if f.calls() != 2 {
		t.Errorf("made %d requests, want 2", f.calls())
	}
}

// Only a 400 retries: another status is a real failure and must not be retried.
func TestPostInlineCommentDoesNotRetryNon400(t *testing.T) {
	f, c := newFake(t, jsonReply(500, `boom`))
	_, err := c.PostInlineComment(context.Background(), "PROJ", "r", 1, "src/main.py", 10, "x")
	if Status(err) != 500 {
		t.Fatalf("err = %v, want a 500 StatusError", err)
	}
	if f.calls() != 1 {
		t.Errorf("made %d requests, want 1: a 500 must not retry", f.calls())
	}
}

func TestPostPRComment(t *testing.T) {
	f, c := newFake(t, jsonReply(201, `{"id":2,"version":3}`))
	id, version, err := c.PostPRComment(context.Background(), "PROJ", "r", 1, "Review summary")
	if err != nil {
		t.Fatal(err)
	}
	if id != 2 || version != 3 {
		t.Errorf("id, version = %d, %d, want 2, 3", id, version)
	}
	if f.last().Body["text"] != "Review summary" {
		t.Errorf("body = %v", f.last().Body)
	}
	if _, ok := f.last().Body["anchor"]; ok {
		t.Error("a summary comment must carry no anchor")
	}
}

// A fresh comment usually comes back without a version, which means 0.
func TestPostPRCommentVersionDefaultsToZero(t *testing.T) {
	_, c := newFake(t, jsonReply(201, `{"id":2}`))
	_, version, err := c.PostPRComment(context.Background(), "PROJ", "r", 1, "s")
	if err != nil {
		t.Fatal(err)
	}
	if version != 0 {
		t.Errorf("version = %d, want 0", version)
	}
}

func TestReplyToComment(t *testing.T) {
	f, c := newFake(t, jsonReply(201, `{"id":5}`))
	if err := c.ReplyToComment(context.Background(), "PROJ", "r", 1, 42, "Feedback noted"); err != nil {
		t.Fatal(err)
	}
	parent, _ := f.last().Body["parent"].(map[string]any)
	if parent["id"] != float64(42) {
		t.Errorf("parent = %v, want id 42", parent)
	}
	if f.last().Body["text"] != "Feedback noted" {
		t.Errorf("text = %v", f.last().Body["text"])
	}
}

func TestFetchPRComment(t *testing.T) {
	_, c := newFake(t, jsonReply(200, `{"id":42,"version":7,"text":"### Review summary"}`))
	got, err := c.FetchPRComment(context.Background(), "PROJ", "r", 1, 42)
	if err != nil {
		t.Fatal(err)
	}
	if got.Version != 7 || got.Text != "### Review summary" {
		t.Errorf("comment = %+v", got)
	}
}

func TestUpdatePRComment(t *testing.T) {
	f, c := newFake(t, jsonReply(200, `{"id":42,"version":4}`))
	version, err := c.UpdatePRComment(context.Background(), "PROJ", "r", 1, 42, 3, "new text")
	if err != nil {
		t.Fatal(err)
	}
	if version != 4 {
		t.Errorf("version = %d, want the new version 4", version)
	}
	if f.last().Body["version"] != float64(3) {
		t.Errorf("sent version = %v, want the held version 3", f.last().Body["version"])
	}
	if f.last().Method != http.MethodPut {
		t.Errorf("method = %s, want PUT", f.last().Method)
	}
}

// A concurrent edit is not an error the caller should retry: it posts fresh.
func TestUpdatePRComment409IsTyped(t *testing.T) {
	_, c := newFake(t, jsonReply(409, `{"errors":[{"message":"version conflict"}]}`))
	_, err := c.UpdatePRComment(context.Background(), "PROJ", "r", 1, 42, 3, "t")
	if !errors.Is(err, ErrVersionConflict) {
		t.Fatalf("err = %v, want ErrVersionConflict", err)
	}
}

// --- paging ------------------------------------------------------------------

func TestPagedWalksEveryPage(t *testing.T) {
	f, c := newFake(t, func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Query().Get("start") {
		case "0":
			jsonReply(200, `{"values":[{"slug":"a"}],"isLastPage":false,"nextPageStart":100}`)(w, nil)
		default:
			jsonReply(200, `{"values":[{"slug":"b"}],"isLastPage":true}`)(w, nil)
		}
	})

	got, err := c.ListRepos(context.Background(), "PROJ")
	if err != nil {
		t.Fatal(err)
	}
	if len(got) != 2 || got[0]["slug"] != "a" || got[1]["slug"] != "b" {
		t.Fatalf("values = %v", got)
	}
	if f.calls() != 2 {
		t.Errorf("made %d requests, want 2", f.calls())
	}
	if got := f.seen[0].Query.Get("limit"); got != "100" {
		t.Errorf("limit = %q, want 100", got)
	}
	if got := f.seen[1].Query.Get("start"); got != "100" {
		t.Errorf("second start = %q, want 100", got)
	}
}

// A response without isLastPage is the last page: never loop on a silent server.
func TestPagedStopsWhenIsLastPageAbsent(t *testing.T) {
	f, c := newFake(t, jsonReply(200, `{"values":[{"slug":"a"}]}`))
	got, err := c.ListRepos(context.Background(), "PROJ")
	if err != nil {
		t.Fatal(err)
	}
	if len(got) != 1 || f.calls() != 1 {
		t.Errorf("got %d values in %d calls, want 1 in 1", len(got), f.calls())
	}
}

// A server echoing its offset must not spin the loop forever.
func TestPagedStopsWhenNextPageStartDoesNotAdvance(t *testing.T) {
	f, c := newFake(t, jsonReply(200, `{"values":[{"slug":"a"}],"isLastPage":false,"nextPageStart":0}`))
	got, err := c.ListRepos(context.Background(), "PROJ")
	if err != nil {
		t.Fatal(err)
	}
	if f.calls() != 1 {
		t.Errorf("made %d requests, want 1: a non-advancing offset must stop the walk", f.calls())
	}
	if len(got) != 1 {
		t.Errorf("values = %v", got)
	}
}

// Same guard for a missing nextPageStart with isLastPage false.
func TestPagedStopsWhenNextPageStartMissing(t *testing.T) {
	f, c := newFake(t, jsonReply(200, `{"values":[{"slug":"a"}],"isLastPage":false}`))
	if _, err := c.ListRepos(context.Background(), "PROJ"); err != nil {
		t.Fatal(err)
	}
	if f.calls() != 1 {
		t.Errorf("made %d requests, want 1", f.calls())
	}
}

// --- webhooks and permissions ------------------------------------------------

func TestListWebhooksProjectAndRepoPaths(t *testing.T) {
	f, c := newFake(t, jsonReply(200, `{"values":[],"isLastPage":true}`))

	if _, err := c.ListWebhooks(context.Background(), "PROJ", ""); err != nil {
		t.Fatal(err)
	}
	if want := "/rest/api/1.0/projects/PROJ/webhooks"; f.last().Path != want {
		t.Errorf("project path = %q, want %q", f.last().Path, want)
	}

	if _, err := c.ListWebhooks(context.Background(), "PROJ", "my-repo"); err != nil {
		t.Fatal(err)
	}
	if want := "/rest/api/1.0/projects/PROJ/repos/my-repo/webhooks"; f.last().Path != want {
		t.Errorf("repo path = %q, want %q", f.last().Path, want)
	}
}

func TestCreateWebhookBody(t *testing.T) {
	f, c := newFake(t, jsonReply(201, `{"id":11,"name":"noergler"}`))
	body := Webhook{
		Name:                    "noergler",
		URL:                     "https://noergler.example.com/webhook/platform",
		Active:                  true,
		Events:                  []string{"pr:opened", "pr:merged"},
		SSLVerificationRequired: true,
	}
	body.Configuration.Secret = "s3cret"

	got, err := c.CreateWebhook(context.Background(), "PROJ", "", body)
	if err != nil {
		t.Fatal(err)
	}
	if got.ID != 11 {
		t.Errorf("id = %d, want 11", got.ID)
	}
	sent := f.last().Body
	if sent["name"] != "noergler" || sent["active"] != true || sent["sslVerificationRequired"] != true {
		t.Errorf("body = %v", sent)
	}
	cfg, _ := sent["configuration"].(map[string]any)
	if cfg["secret"] != "s3cret" {
		t.Errorf("configuration = %v", cfg)
	}
	if events, _ := sent["events"].([]any); len(events) != 2 {
		t.Errorf("events = %v", sent["events"])
	}
}

func TestUpdateAndDeleteWebhook(t *testing.T) {
	f, c := newFake(t, jsonReply(200, `{"id":11}`))
	if _, err := c.UpdateWebhook(context.Background(), "PROJ", "r", 11, Webhook{Name: "noergler"}); err != nil {
		t.Fatal(err)
	}
	if f.last().Method != http.MethodPut || !strings.HasSuffix(f.last().Path, "/webhooks/11") {
		t.Errorf("%s %s", f.last().Method, f.last().Path)
	}

	if err := c.DeleteWebhook(context.Background(), "PROJ", "r", 11); err != nil {
		t.Fatal(err)
	}
	if f.last().Method != http.MethodDelete || !strings.HasSuffix(f.last().Path, "/webhooks/11") {
		t.Errorf("%s %s", f.last().Method, f.last().Path)
	}
}

// Bitbucket takes the grant as query parameters, not a JSON body.
func TestGrantUserPermissionUsesQueryParams(t *testing.T) {
	f, c := newFake(t, jsonReply(204, ``))
	if err := c.GrantUserPermission(context.Background(), "PROJ", "", "bot-user", "PROJECT_WRITE"); err != nil {
		t.Fatal(err)
	}
	if f.last().Method != http.MethodPut {
		t.Errorf("method = %s, want PUT", f.last().Method)
	}
	if want := "/rest/api/1.0/projects/PROJ/permissions/users"; f.last().Path != want {
		t.Errorf("path = %q, want %q", f.last().Path, want)
	}
	if f.last().Query.Get("name") != "bot-user" || f.last().Query.Get("permission") != "PROJECT_WRITE" {
		t.Errorf("query = %v", f.last().Query)
	}
	if f.last().Body != nil {
		t.Errorf("body = %v, want none", f.last().Body)
	}
}

func TestGetProjectGetRepoListPullRequests(t *testing.T) {
	f, c := newFake(t, jsonReply(200, `{"key":"PROJ","values":[]}`))

	if _, err := c.GetProject(context.Background(), "PROJ"); err != nil {
		t.Fatal(err)
	}
	if want := "/rest/api/1.0/projects/PROJ"; f.last().Path != want {
		t.Errorf("path = %q, want %q", f.last().Path, want)
	}

	if _, err := c.GetRepo(context.Background(), "PROJ", "r"); err != nil {
		t.Fatal(err)
	}
	if want := "/rest/api/1.0/projects/PROJ/repos/r"; f.last().Path != want {
		t.Errorf("path = %q, want %q", f.last().Path, want)
	}

	if _, err := c.ListPullRequests(context.Background(), "PROJ", "r", 1); err != nil {
		t.Fatal(err)
	}
	if got := f.last().Query.Get("limit"); got != "1" {
		t.Errorf("limit = %q, want 1", got)
	}
}

// --- transport behaviour -----------------------------------------------------

// httpx does not follow redirects, so a 3xx is an error. Go would follow it and
// replay the bearer token at whatever host it names.
func TestRedirectsAreNotFollowed(t *testing.T) {
	var hits int
	_, c := newFake(t, func(w http.ResponseWriter, r *http.Request) {
		hits++
		http.Redirect(w, r, "http://example.invalid/elsewhere", http.StatusMovedPermanently)
	})
	err := c.CheckConnectivity(context.Background())
	if Status(err) != http.StatusMovedPermanently {
		t.Fatalf("err = %v, want a 301 StatusError", err)
	}
	if hits != 1 {
		t.Errorf("server saw %d requests, want 1", hits)
	}
}

// WithToken swaps the identity for onboarding writes without a new transport.
func TestWithTokenSwapsAuthorization(t *testing.T) {
	f, c := newFake(t, jsonReply(200, `{}`))
	admin := c.WithToken("admin-token")

	if _, err := admin.GetProject(context.Background(), "PROJ"); err != nil {
		t.Fatal(err)
	}
	if got := f.last().Header.Get("Authorization"); got != "Bearer admin-token" {
		t.Errorf("Authorization = %q, want the admin token", got)
	}
	if _, err := c.GetProject(context.Background(), "PROJ"); err != nil {
		t.Fatal(err)
	}
	if got := f.last().Header.Get("Authorization"); got != "Bearer test-token" {
		t.Errorf("Authorization = %q, want the original token unchanged", got)
	}
	if c.BotUsername() != "bot-user" {
		t.Errorf("BotUsername = %q", c.BotUsername())
	}
}

// Every request that goes out is counted, including a retry and each page.
func TestRequestsAreCounted(t *testing.T) {
	var n int
	_, c := newFake(t, func(w http.ResponseWriter, _ *http.Request) {
		n++
		if n == 1 {
			jsonReply(400, `{}`)(w, nil)
			return
		}
		jsonReply(201, `{"id":1}`)(w, nil)
	})

	ctx, counter := httpstats.WithScope(context.Background())
	if _, err := c.PostInlineComment(ctx, "PROJ", "r", 1, "a.py", 1, "x"); err != nil {
		t.Fatal(err)
	}
	if got := counter.Methods()["bitbucket:POST"]; got != 2 {
		t.Errorf("counted %d POSTs, want 2 (the retry counts)", got)
	}
}

func TestNewRejectsARelativeBaseURL(t *testing.T) {
	_, err := New(config.Bitbucket{BaseURL: "bitbucket.example.com"}, slog.New(slog.DiscardHandler))
	if err == nil {
		t.Fatal("a base URL without a scheme must be refused")
	}
}

func TestContentTooLargeMessage(t *testing.T) {
	size := 100
	withSize := &ContentTooLarge{What: "PROJ/r#1 diff", Limit: 10, Size: &size}
	if got, want := withSize.Error(), "PROJ/r#1 diff: 100 bytes exceeds cap of 10 bytes"; got != want {
		t.Errorf("Error() = %q, want %q", got, want)
	}
	unknown := &ContentTooLarge{What: "big.bin", Limit: 10}
	if got, want := unknown.Error(), "big.bin: > 10 bytes exceeds cap of 10 bytes"; got != want {
		t.Errorf("Error() = %q, want %q", got, want)
	}
}

func TestStatusOfANonStatusError(t *testing.T) {
	if got := Status(fmt.Errorf("dial tcp: refused")); got != 0 {
		t.Errorf("Status(non-status error) = %d, want 0", got)
	}
}
