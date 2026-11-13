package gitrepo

import (
	"context"
	"errors"
	"fmt"
	"net/http"
	"net/http/cgi"
	"net/http/httptest"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/go-git/go-git/v5/plumbing"
)

// The tests run against git's own smart HTTP server (git http-backend as
// CGI): shallow fetches and receive-pack as a real server answers them,
// which go-git's in-process server does not (no shallow). Skipped without
// a git binary.

const (
	user  = "bot"
	token = "secret-token"
)

type fixture struct {
	t      *testing.T
	git    string
	bare   string
	srv    *httptest.Server
	remote Remote
}

func newFixture(t *testing.T) *fixture {
	t.Helper()
	gitBin, err := exec.LookPath("git")
	if err != nil {
		t.Skip("no git binary")
	}
	root := t.TempDir()
	f := &fixture{t: t, git: gitBin, bare: filepath.Join(root, "repo.git")}
	work := filepath.Join(root, "work")
	f.run(root, "init", "-q", "--bare", "-b", "master", f.bare)
	f.run(f.bare, "config", "http.receivepack", "true")
	f.run(root, "init", "-q", "-b", "master", work)
	write := func(name, body string) {
		if err := os.MkdirAll(filepath.Dir(filepath.Join(work, name)), 0o700); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(filepath.Join(work, name), []byte(body), 0o600); err != nil {
			t.Fatal(err)
		}
	}
	commit := func(msg string) {
		f.run(work, "add", "-A")
		f.run(work, "commit", "-q", "-m", msg)
	}
	// A history deeper than one commit on both sides of the base.
	write("list.txt", "a\nz\n")
	write("dir/keep.go", "package dir\n")
	write("dir/sub/deep.txt", "deep\n")
	write("README", "readme\n")
	commit("base")
	for i := range 3 {
		write("old.txt", strings.Repeat("x\n", i+1))
		commit("more base")
	}
	f.run(work, "checkout", "-q", "-b", "feature")
	write("list.txt", "a\ns\nz\n")
	commit("source 1")
	write("feature.txt", "f\n")
	commit("source 2")
	f.run(work, "checkout", "-q", "master")
	write("list.txt", "a\nt\nz\n")
	commit("target 1")
	write("README", "readme 2\n")
	commit("target 2")
	f.run(work, "push", "-q", f.bare, "master", "feature")

	backend := &cgi.Handler{
		Path: gitBin,
		Args: []string{"http-backend"},
		Env:  []string{"GIT_PROJECT_ROOT=" + root, "GIT_HTTP_EXPORT_ALL=1", "REMOTE_USER=" + user},
	}
	f.srv = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		u, p, ok := r.BasicAuth()
		if !ok || u != user || p != token {
			w.Header().Set("WWW-Authenticate", `Basic realm="git"`)
			http.Error(w, "auth", http.StatusUnauthorized)
			return
		}
		backend.ServeHTTP(w, r)
	}))
	t.Cleanup(f.srv.Close)
	f.remote = Remote{URL: f.srv.URL + "/repo.git", Username: user, Token: token,
		HTTP: &http.Client{CheckRedirect: func(*http.Request, []*http.Request) error { return http.ErrUseLastResponse }}}
	return f
}

func (f *fixture) run(dir string, args ...string) string {
	f.t.Helper()
	cmd := exec.Command(f.git, args...)
	cmd.Dir = dir
	cmd.Env = append(os.Environ(), "GIT_AUTHOR_NAME=t", "GIT_AUTHOR_EMAIL=t@example.invalid",
		"GIT_COMMITTER_NAME=t", "GIT_COMMITTER_EMAIL=t@example.invalid", "GIT_CONFIG_GLOBAL=/dev/null")
	out, err := cmd.CombinedOutput()
	if err != nil {
		f.t.Fatalf("git %v: %v\n%s", args, err, out)
	}
	return strings.TrimSpace(string(out))
}

func TestFetchMergeBaseAndPush(t *testing.T) {
	f := newFixture(t)
	ctx := context.Background()

	// One commit deep, the base is out of reach.
	shallow, heads, err := Fetch(ctx, f.remote, 1, "feature", "master")
	if err != nil {
		t.Fatal(err)
	}
	if _, err := shallow.MergeBase(heads[0], heads[1]); !errors.Is(err, ErrIncomplete) {
		t.Fatalf("depth 1: err = %v, want ErrIncomplete", err)
	}

	repo, heads, err := Fetch(ctx, f.remote, 50, "feature", "master")
	if err != nil {
		t.Fatal(err)
	}
	source, target := heads[0], heads[1]
	base, err := repo.MergeBase(source, target)
	if err != nil {
		t.Fatal(err)
	}
	if want := f.run(f.bare, "merge-base", "feature", "master"); base.String() != want {
		t.Fatalf("merge base %s, git says %s", base, want)
	}

	// The flattened tree written back is the same tree, byte for byte:
	// git's entry order, modes and nesting.
	files, err := repo.Files(target)
	if err != nil {
		t.Fatal(err)
	}
	if _, ok := files["dir/sub/deep.txt"]; !ok {
		t.Fatalf("nested path missing: %v", files)
	}
	want := f.run(f.bare, "rev-parse", "master^{tree}")
	got, err := repo.writeTree(files, func(o plumbing.EncodedObject) (plumbing.Hash, error) { return o.Hash(), nil })
	if err != nil || got.String() != want {
		t.Fatalf("rebuilt tree %s (%v), want %s", got, err, want)
	}
	b, err := repo.Blob(files["README"].Hash, 0)
	if err != nil || string(b) != "readme 2\n" {
		t.Fatalf("blob = %q, %v", b, err)
	}

	// The merge: target's tree, source's new file, list.txt resolved.
	srcFiles, err := repo.Files(source)
	if err != nil {
		t.Fatal(err)
	}
	files["feature.txt"] = srcFiles["feature.txt"]
	content := map[string][]byte{"list.txt": []byte("a\nt\ns\nz\n")}
	// Detached, as across the model's call: the fetch's objects are gone,
	// the trees it walked are known.
	light := repo.Detach()
	commit, objects, err := light.Commit(files, content, []plumbing.Hash{source, target},
		Signature{Name: "noergler", Email: "bot@example.invalid"}, "Merge branch 'master' into feature\n", time.Unix(1_700_000_000, 0))
	if err != nil {
		t.Fatal(err)
	}
	// Only what is new goes out: the merged blob, the root tree, the commit.
	if len(objects) != 3 {
		t.Errorf("objects to send = %d, want 3", len(objects))
	}
	if err := Push(ctx, f.remote, light, "feature", source, commit, objects); err != nil {
		t.Fatal(err)
	}
	if got := f.run(f.bare, "rev-parse", "feature"); got != commit.String() {
		t.Fatalf("feature at %s, want %s", got, commit)
	}
	if got := f.run(f.bare, "rev-parse", "feature^1", "feature^2"); got != source.String()+"\n"+target.String() {
		t.Fatalf("parents %q", got)
	}
	if got := f.run(f.bare, "show", "feature:list.txt"); got != "a\nt\ns\nz" {
		t.Fatalf("list.txt = %q", got)
	}
	f.run(f.bare, "fsck", "--strict")

	// The same push again: the branch is no longer at source.
	if err := Push(ctx, f.remote, light, "feature", source, commit, objects); !errors.Is(err, ErrMoved) {
		t.Fatalf("second push: err = %v, want ErrMoved", err)
	}
}

func TestFetchStopsAtMaxBytes(t *testing.T) {
	f := newFixture(t)
	r := f.remote
	r.MaxBytes = 200
	if _, _, err := Fetch(context.Background(), r, 50, "feature", "master"); !errors.Is(err, ErrTooLarge) {
		t.Fatalf("err = %v, want ErrTooLarge", err)
	}
}

func TestPushWithoutWritePermission(t *testing.T) {
	f := newFixture(t)
	repo, heads, err := Fetch(context.Background(), f.remote, 50, "feature")
	if err != nil {
		t.Fatal(err)
	}
	// A read-only account: receive-pack answers 403.
	ro := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if strings.Contains(r.URL.RawQuery, "git-receive-pack") || strings.HasSuffix(r.URL.Path, "git-receive-pack") {
			http.Error(w, "forbidden", http.StatusForbidden)
			return
		}
		f.srv.Config.Handler.ServeHTTP(w, r)
	}))
	defer ro.Close()
	r := f.remote
	r.URL = ro.URL + "/repo.git"
	err = Push(context.Background(), r, repo, "feature", heads[0], heads[0], nil)
	if !errors.Is(err, ErrNoWrite) {
		t.Fatalf("err = %v, want ErrNoWrite", err)
	}
}

func TestRedirectIsNotFollowed(t *testing.T) {
	f := newFixture(t)
	var leaked bool
	other := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _, leaked = r.BasicAuth()
		f.srv.Config.Handler.ServeHTTP(w, r)
	}))
	defer other.Close()
	redirect := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		http.Redirect(w, r, other.URL+r.URL.RequestURI(), http.StatusFound)
	}))
	defer redirect.Close()
	r := f.remote
	r.URL = redirect.URL + "/repo.git"
	if _, _, err := Fetch(context.Background(), r, 1, "master"); err == nil {
		t.Fatal("a redirect was followed")
	}
	if leaked {
		t.Fatal("the token reached the redirect's target")
	}
}

// The target merged an old long-lived branch after the PR branched off:
// that branch's history lies past the shallow edge, on the target's side
// only. The source's walk ends without an edge, so the base found stands.
func TestMergeBasePastAOneSidedEdge(t *testing.T) {
	gitBin, err := exec.LookPath("git")
	if err != nil {
		t.Skip("no git binary")
	}
	root := t.TempDir()
	f := &fixture{t: t, git: gitBin, bare: filepath.Join(root, "repo.git")}
	work := filepath.Join(root, "work")
	f.run(root, "init", "-q", "--bare", "-b", "master", f.bare)
	f.run(root, "init", "-q", "-b", "master", work)
	commit := func(name, msg string) {
		if err := os.WriteFile(filepath.Join(work, name), []byte(msg+"\n"), 0o600); err != nil {
			t.Fatal(err)
		}
		f.run(work, "add", "-A")
		f.run(work, "commit", "-q", "-m", msg)
	}
	commit("root.txt", "root")
	f.run(work, "checkout", "-q", "-b", "old")
	for i := range 8 {
		commit("old.txt", fmt.Sprintf("old %d", i))
	}
	f.run(work, "checkout", "-q", "master")
	for i := range 8 {
		commit("main.txt", fmt.Sprintf("main %d", i))
	}
	f.run(work, "checkout", "-q", "-b", "feature")
	commit("feature.txt", "feature")
	f.run(work, "checkout", "-q", "master")
	f.run(work, "merge", "-q", "--no-edit", "old")
	f.run(work, "push", "-q", f.bare, "master", "feature")
	backend := &cgi.Handler{Path: gitBin, Args: []string{"http-backend"},
		Env: []string{"GIT_PROJECT_ROOT=" + root, "GIT_HTTP_EXPORT_ALL=1"}}
	srv := httptest.NewServer(backend)
	defer srv.Close()
	repo, heads, err := Fetch(context.Background(), Remote{URL: srv.URL + "/repo.git"}, 3, "feature", "master")
	if err != nil {
		t.Fatal(err)
	}
	base, err := repo.MergeBase(heads[0], heads[1])
	if err != nil {
		t.Fatal(err)
	}
	if want := f.run(f.bare, "merge-base", "feature", "master"); base.String() != want {
		t.Fatalf("merge base %s, git says %s", base, want)
	}
}

func TestFetchStopsAtMaxInflated(t *testing.T) {
	f := newFixture(t)
	r := f.remote
	r.MaxInflated = 100
	if _, _, err := Fetch(context.Background(), r, 50, "feature", "master"); !errors.Is(err, ErrTooLarge) {
		t.Fatalf("err = %v, want ErrTooLarge", err)
	}
}
