package main

import (
	"bytes"
	"encoding/json"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"

	"github.com/trick77/noergler/internal/evals"
	"github.com/trick77/noergler/internal/merge"
)

// repo is a scratch git repository driven by the git binary.
type repo struct {
	t   *testing.T
	dir string
}

func (r repo) git(args ...string) string {
	r.t.Helper()
	cmd := exec.Command("git", append([]string{"-C", r.dir}, args...)...)
	cmd.Env = append(os.Environ(), "GIT_CONFIG_GLOBAL=/dev/null", "GIT_CONFIG_NOSYSTEM=1",
		"GIT_AUTHOR_NAME=t", "GIT_AUTHOR_EMAIL=t@example.com", "GIT_COMMITTER_NAME=t", "GIT_COMMITTER_EMAIL=t@example.com")
	out, err := cmd.CombinedOutput()
	if err != nil && args[0] != "merge" {
		r.t.Fatalf("git %v: %v\n%s", args, err, out)
	}
	return strings.TrimSpace(string(out))
}

func (r repo) write(name, content string) {
	r.t.Helper()
	if err := os.WriteFile(filepath.Join(r.dir, name), []byte(content), 0o600); err != nil {
		r.t.Fatal(err)
	}
	r.git("add", name)
}

// conflict branches topic off master, changes name on both, merges topic
// into master and commits resolved as the merge's file.
func (r repo) conflict(topic, name, onMaster, onTopic, resolved string) string {
	r.git("checkout", "-q", "-b", topic)
	r.write(name, onTopic)
	r.git("commit", "-q", "-m", topic)
	r.git("checkout", "-q", "master")
	r.write(name, onMaster)
	r.git("commit", "-q", "-m", "master "+topic)
	r.git("merge", "-q", "--no-ff", topic)
	r.write(name, resolved)
	r.git("commit", "-q", "--no-edit")
	return r.git("rev-parse", "HEAD")
}

func TestMine(t *testing.T) {
	if _, err := exec.LookPath("git"); err != nil {
		t.Skip("no git")
	}
	r := repo{t, t.TempDir()}
	r.git("init", "-q", "-b", "master")
	r.write("conf.go", "a\nx = 1\ny = 2\nb\n")
	r.write("yarn.lock", "l1\nl2\nl3\nl4\n")
	r.write("nl.txt", "a\nb")
	r.git("commit", "-q", "-m", "base")
	solve := r.conflict("t1", "conf.go", "a\nx = 10\ny = 2\nb\n", "a\nx = 1\ny = 20\nb\n", "a\nx = 10\ny = 20\nb\n")
	control := r.conflict("t2", "conf.go", "a\nx = 10\ny = 20\nb master\n", "a\nx = 10\ny = 20\nb topic\n",
		"a\nx = 10\ny = 20\nb both\n")
	r.conflict("t3", "conf.go", "a\nx = 10\ny = 20\nb both\nm\n", "a\nx = 10\ny = 20\nb both\nt\n",
		"a\nx = 10\ny = 20\nb both\nm\nt\n")
	evil := r.conflict("t4", "conf.go", "a\nx = 11\ny = 20\nb both\nm\nt\n", "a\nx = 12\ny = 20\nb both\nm\nt\n",
		"a changed\nx = 12\ny = 20\nb both\nm\nt\n")
	// A lockfile both branches changed, merged cleanly by git: production
	// refuses the whole request, so the miner does too.
	r.git("checkout", "-q", "-b", "t5")
	r.write("yarn.lock", "L1\nl2\nl3\nl4\n")
	r.write("conf.go", "a changed\nx = 14\ny = 20\nb both\nm\nt\n")
	r.git("commit", "-q", "-m", "t5")
	r.git("checkout", "-q", "master")
	r.write("yarn.lock", "l1\nl2\nl3\nL4\n")
	r.write("conf.go", "a changed\nx = 13\ny = 20\nb both\nm\nt\n")
	r.git("commit", "-q", "-m", "master t5")
	r.git("merge", "-q", "--no-ff", "t5")
	r.write("yarn.lock", "L1\nl2\nl3\nL4\n")
	r.write("conf.go", "a changed\nx = 14\ny = 20\nb both\nm\nt\n")
	r.git("commit", "-q", "--no-edit")
	// A conflict on a last line without a newline.
	r.conflict("t7", "nl.txt", "a\nm", "a\nt", "a\nt")
	out := t.TempDir()
	var stdout, stderr bytes.Buffer
	if code := run([]string{"-repo", r.dir, "-url", "https://example.com/o/r", "-rev", "master", "-scan", "10", "-out", out},
		&stdout, &stderr); code != 0 {
		t.Fatalf("exit %d: %s", code, stderr.String())
	}
	got := stdout.String()
	for _, want := range []string{"6 merges, 6 conflicted, 2 cases written", "conflicts in bounds: 3, code resolves 1, the model is asked 2",
		"code against the author: 0 same, 1 order only, 0 blank lines (order aside), 0 content",
		"skipped evil_merge", "skipped trivial_only", "skipped lockfile", "skipped no_final_newline"} {
		if !strings.Contains(got, want) {
			t.Errorf("stdout lacks %q:\n%s", want, got)
		}
	}
	if !strings.Contains(stderr.String(), "--- author\nm\nt") {
		t.Errorf("trivial difference not shown:\n%s", stderr.String())
	}
	read := func(sha string) evals.ConflictCase {
		t.Helper()
		raw, err := os.ReadFile(filepath.Join(out, "mined-r-"+sha[:10]+".json"))
		if err != nil {
			t.Fatal(err)
		}
		var c evals.ConflictCase
		if err := json.Unmarshal(raw, &c); err != nil {
			t.Fatal(err)
		}
		return c
	}
	s := read(solve)
	if s.Control || len(s.Accept) != 1 || s.Accept[0].Texts[0] != "x = 10\ny = 20\n" || s.TargetBranch != "t1" || s.SourceBranch != "master" {
		t.Errorf("solve case: %+v", s)
	}
	if c := read(control); !c.Control || len(c.Accept) != 0 {
		t.Errorf("control case: %+v", c)
	}
	if _, err := os.Stat(filepath.Join(out, "mined-r-"+evil[:10]+".json")); err == nil {
		t.Error("an evil merge became a case")
	}
	// A short hash names the case by the full one.
	stdout.Reset()
	if code := run([]string{"-repo", r.dir, "-url", "https://example.com/o/r", "-merges", solve[:7], "-out", t.TempDir()},
		&stdout, &stderr); code != 0 || !strings.Contains(stdout.String(), "mined-r-"+solve[:10]) {
		t.Errorf("short hash: exit %d\n%s", code, stdout.String())
	}
	if code := run([]string{"-repo", r.dir}, &stdout, &stderr); code != 2 {
		t.Errorf("missing flags: exit %d", code)
	}
}

func TestAuthorRegions(t *testing.T) {
	chunks := []merge.Chunk{
		{Lines: []string{"a\n"}},
		{Conflict: &merge.Conflict{Target: []string{"t\n"}, Source: []string{"s\n"}}},
		{Lines: []string{"z\n"}},
		{Conflict: &merge.Conflict{Target: []string{"u\n"}, Source: []string{"v\n"}}},
	}
	got, ok := authorRegions(chunks, []string{"a\n", "s\n", "t\n", "z\n", "v\n"})
	if !ok || len(got) != 2 || strings.Join(got[0], "") != "s\nt\n" || strings.Join(got[1], "") != "v\n" {
		t.Fatalf("regions %q, %v", got, ok)
	}
	// The author's region holds the next settled chunk's text: the split is
	// the one that fits the whole file, not the first z.
	tail := []merge.Chunk{chunks[0], chunks[1], chunks[2], {Lines: []string{"e\n"}}}
	got, ok = authorRegions(tail, []string{"a\n", "s\n", "z\n", "t\n", "z\n", "e\n"})
	if !ok || len(got) != 1 || strings.Join(got[0], "") != "s\nz\nt\n" {
		t.Fatalf("regions %q, %v", got, ok)
	}
	for _, merged := range [][]string{
		{"b\n", "s\n", "z\n", "v\n"},                      // a settled line changed
		{"a\n", "s\n", "v\n"},                             // the next settled chunk gone
		{"a\n", "x\n", "z\n", "y\n", "z\n", "w\n", "v\n"}, // two splits fit: unclear
	} {
		if _, ok := authorRegions(chunks, merged); ok {
			t.Errorf("%q: regions found", merged)
		}
	}
}

func TestCompare(t *testing.T) {
	for _, tc := range []struct {
		got, want []string
		kind      string
	}{
		{[]string{"a", "b"}, []string{"a", "b"}, kindSame},
		{[]string{"b", "a"}, []string{"a", "b"}, kindOrder},
		{[]string{"b", "a"}, []string{"a", "", "b"}, kindBlank},
		{[]string{"f () {", "f() {"}, []string{"f() {"}, kindContent},
	} {
		if got := compare(tc.got, tc.want); got != tc.kind {
			t.Errorf("compare(%q, %q) = %s, want %s", tc.got, tc.want, got, tc.kind)
		}
	}
}

func TestDiff3Crops(t *testing.T) {
	long := func(p string) []string {
		var ls []string
		for i := range 8 {
			ls = append(ls, p+string(rune('0'+i))+"\n")
		}
		return ls
	}
	c := &merge.Conflict{Base: []string{"b\n"}, Target: []string{"t\n"}, Source: []string{"s"}}
	got := diff3([]merge.Chunk{{Lines: long("p")}, {Conflict: c}, {Lines: long("m")}, {Conflict: c}, {Lines: long("q")}})
	want := "p5\np6\np7\n<<<<<<< source\ns\n||||||| base\nb\n=======\nt\n>>>>>>> target\n" +
		"m0\nm1\nm2\nm5\nm6\nm7\n<<<<<<< source\ns\n||||||| base\nb\n=======\nt\n>>>>>>> target\nq0\nq1\nq2\n"
	if got != want {
		t.Errorf("diff3:\n%s\nwant\n%s", got, want)
	}
}
