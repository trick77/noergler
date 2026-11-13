package bitbucket

import (
	"context"
	"io"
	"log/slog"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/trick77/noergler/internal/config"
)

// editServer answers the edit-file endpoint with status and body and keeps
// what it was sent: the multipart fields and the file's bytes.
func editServer(t *testing.T, status int, body string) (*Client, *http.Request, map[string]string) {
	t.Helper()
	var got http.Request
	fields := map[string]string{}
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		got = *r.Clone(context.Background())
		if err := r.ParseMultipartForm(1 << 20); err != nil {
			t.Errorf("not multipart: %v", err)
		}
		for k, v := range r.MultipartForm.Value {
			fields[k] = v[0]
		}
		if fh := r.MultipartForm.File["content"]; len(fh) == 1 {
			f, _ := fh[0].Open()
			b, _ := io.ReadAll(f)
			fields["content"] = string(b)
		}
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(status)
		_, _ = io.WriteString(w, body)
	}))
	t.Cleanup(srv.Close)
	c, err := New(config.Bitbucket{BaseURL: srv.URL, Token: "test-token", Username: "bot-user", MaxFileBytes: 1 << 20},
		slog.New(slog.DiscardHandler))
	if err != nil {
		t.Fatal(err)
	}
	return c, &got, fields
}

func TestEditFile(t *testing.T) {
	c, req, fields := editServer(t, 200, `{"id":"abc123","displayId":"abc123"}`)
	content := "a\r\n\xef\xbb\xbfb — c\n"
	sha, err := c.EditFile(context.Background(), "PROJ", "repo", "feature/x", "src/a b.go", "src1", "fix it\n\nlink", []byte(content))
	if err != nil || sha != "abc123" {
		t.Fatalf("sha %q, err %v", sha, err)
	}
	if req.Method != http.MethodPut || req.URL.Path != "/rest/api/1.0/projects/PROJ/repos/repo/browse/src/a b.go" {
		t.Errorf("%s %s", req.Method, req.URL.Path)
	}
	if req.Header.Get("X-Atlassian-Token") != "no-check" || req.Header.Get("Authorization") != "Bearer test-token" {
		t.Errorf("headers = %v", req.Header)
	}
	want := map[string]string{"branch": "feature/x", "sourceCommitId": "src1", "message": "fix it\n\nlink", "content": content}
	for k, v := range want {
		if fields[k] != v {
			t.Errorf("%s = %q, want %q (byte for byte)", k, fields[k], v)
		}
	}
}

func TestEditFileRefusals(t *testing.T) {
	for _, status := range []int{401, 403, 409, 400} {
		c, _, _ := editServer(t, status, `{"errors":[{"message":"no"}]}`)
		_, err := c.EditFile(context.Background(), "P", "r", "b", "a.go", "s", "m", []byte("x"))
		if Status(err) != status {
			t.Errorf("status %d: err = %v", status, err)
		}
	}
	c, _, _ := editServer(t, 200, `{}`)
	if _, err := c.EditFile(context.Background(), "P", "r", "b", "a.go", "s", "m", []byte("x")); err == nil {
		t.Error("an answer without a commit id must fail")
	}
}

func TestCommitURL(t *testing.T) {
	c, _, _ := editServer(t, 200, `{}`)
	if got := c.CommitURL("PROJ", "my repo", "abc"); got != c.base.String()+"/projects/PROJ/repos/my%20repo/commits/abc" {
		t.Errorf("CommitURL = %q", got)
	}
}
