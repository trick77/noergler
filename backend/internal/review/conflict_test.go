package review

import (
	"context"
	"net/http"
	"net/http/cgi"
	"net/http/httptest"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"

	"github.com/trick77/noergler/internal/bitbucket"
	"github.com/trick77/noergler/internal/gitrepo"
	"github.com/trick77/noergler/internal/inference"
	"github.com/trick77/noergler/internal/store"
)

func TestConflictRequest(t *testing.T) {
	for _, c := range []struct {
		text string
		want bool
	}{
		{"behebe den Merge-Konflikt!", true},
		{"Behebe den Merge-Konflikt bitte", true},
		{"löse den Konflikt auf", true},
		{"bitte die Merge-Konflikte lösen", true},
		{"resolve the merge conflict", true},
		{"fix the conflicts please", true},
		{"fix merge conflicts", true},
		{"how do I resolve the merge conflict?", false},
		{"don't fix the conflict", false},
		{"bitte den Konflikt nicht lösen", false},
		{"review", false},
		{"fix it", false},
		{"is there a conflict between the config and the docs in this PR, and should I worry about it", false},
		{"fix `conflict()`", false},
		// The commenter doing it, or later: no order to the bot.
		{"I'll resolve the conflict myself, just review", false},
		{"Ich löse den Konflikt selbst", false},
		{"will fix the conflict later, please review", false},
		{"we fix the merge conflict tomorrow", false},
		{"can't resolve the conflict", false},
		{"fix the merge conflict for me", true},
		{"kannst du den Merge-Konflikt beheben", true},
	} {
		if got := conflictRequest(c.text); got != c.want {
			t.Errorf("conflictRequest(%q) = %t, want %t", c.text, got, c.want)
		}
	}
}

// conflictRepo is a bare repository behind git's own smart HTTP server,
// with a base, a master and a feature/thing branch the test shapes.
type conflictRepo struct {
	t         *testing.T
	git, bare string
	work      string
	remote    gitrepo.Remote
	srv       *httptest.Server
	readOnly  bool
}

func newConflictRepo(t *testing.T, base, target, source map[string]string) *conflictRepo {
	t.Helper()
	gitBin, err := exec.LookPath("git")
	if err != nil {
		t.Skip("no git binary")
	}
	root := t.TempDir()
	c := &conflictRepo{t: t, git: gitBin, bare: filepath.Join(root, "repo.git"), work: filepath.Join(root, "work")}
	c.run(root, "init", "-q", "--bare", "-b", "master", c.bare)
	c.run(c.bare, "config", "http.receivepack", "true")
	c.run(root, "init", "-q", "-b", "master", c.work)
	c.commit(base, "base")
	c.run(c.work, "checkout", "-q", "-b", "feature/thing")
	c.commit(source, "source")
	c.run(c.work, "checkout", "-q", "master")
	c.commit(target, "target")
	c.run(c.work, "push", "-q", c.bare, "master", "feature/thing")
	backend := &cgi.Handler{Path: gitBin, Args: []string{"http-backend"},
		Env: []string{"GIT_PROJECT_ROOT=" + root, "GIT_HTTP_EXPORT_ALL=1", "REMOTE_USER=noergler"}}
	c.srv = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if c.readOnly && (strings.Contains(r.URL.RawQuery, "receive-pack") || strings.HasSuffix(r.URL.Path, "receive-pack")) {
			http.Error(w, "forbidden", http.StatusForbidden)
			return
		}
		backend.ServeHTTP(w, r)
	}))
	t.Cleanup(c.srv.Close)
	c.remote = gitrepo.Remote{URL: c.srv.URL + "/repo.git", Username: "noergler", Token: "t",
		HTTP: &http.Client{CheckRedirect: func(*http.Request, []*http.Request) error { return http.ErrUseLastResponse }}}
	return c
}

func (c *conflictRepo) commit(files map[string]string, msg string) {
	c.t.Helper()
	for name, body := range files {
		p := filepath.Join(c.work, name)
		if err := os.MkdirAll(filepath.Dir(p), 0o700); err != nil {
			c.t.Fatal(err)
		}
		if err := os.WriteFile(p, []byte(body), 0o600); err != nil {
			c.t.Fatal(err)
		}
	}
	c.run(c.work, "add", "-A")
	c.run(c.work, "commit", "-q", "--allow-empty", "-m", msg)
}

func (c *conflictRepo) run(dir string, args ...string) string {
	c.t.Helper()
	cmd := exec.Command(c.git, args...)
	cmd.Dir = dir
	cmd.Env = append(os.Environ(), "GIT_AUTHOR_NAME=t", "GIT_AUTHOR_EMAIL=t@example.invalid",
		"GIT_COMMITTER_NAME=t", "GIT_COMMITTER_EMAIL=t@example.invalid", "GIT_CONFIG_GLOBAL=/dev/null")
	out, err := cmd.CombinedOutput()
	if err != nil {
		c.t.Fatalf("git %v: %v\n%s", args, err, out)
	}
	return strings.TrimSpace(string(out))
}

func (c *conflictRepo) show(rev string) string { return c.run(c.bare, "show", rev) }

func conflictHarness(t *testing.T, repo *conflictRepo) *harness {
	t.Helper()
	h := newHarness(t, nil)
	h.bb.mergeStatus = bitbucket.MergeStatus{Conflicted: true}
	if repo != nil {
		h.bb.remote = repo.remote
	}
	return h
}

func (h *harness) askConflict(t *testing.T, author string) {
	t.Helper()
	h.r.HandleMention(context.Background(), mentionPayload("@noergler behebe den Merge-Konflikt!", author), "platform", &inlineScheduler{})
}

// Both branches only added lines at one spot: code resolves it, target's
// then source's, no model, no cost, and the merge is pushed.
func TestConflictResolvedByCode(t *testing.T) {
	repo := newConflictRepo(t,
		map[string]string{"list.txt": "a\nz\n", "other.txt": "1\n2\n3\n4\n5\n"},
		map[string]string{"list.txt": "a\nt\nz\n", "other.txt": "1\n2\n3\n4\nfive\n"},
		map[string]string{"list.txt": "a\ns\nz\n", "other.txt": "one\n2\n3\n4\n5\n"})
	h := conflictHarness(t, repo)

	h.askConflict(t, "alice")

	if len(h.llm.Conflicts) != 0 {
		t.Fatal("a trivial conflict called the model")
	}
	if got := repo.show("feature/thing:list.txt"); got != "a\nt\ns\nz" {
		t.Fatalf("list.txt = %q", got)
	}
	if got := repo.show("feature/thing:other.txt"); got != "one\n2\n3\n4\nfive" {
		t.Fatalf("other.txt merged cleanly = %q", got)
	}
	parents := repo.run(repo.bare, "rev-parse", "feature/thing^1", "feature/thing^2")
	if want := repo.run(repo.work, "rev-parse", "feature/thing", "master"); parents != want {
		t.Fatalf("parents %q, want %q", parents, want)
	}
	if msg := repo.run(repo.bare, "log", "-1", "--format=%an <%ae>%n%B", "feature/thing"); !strings.Contains(msg, "Merge branch 'master' into feature/thing") ||
		!strings.Contains(msg, "list.txt:2 (rule)") || !strings.HasPrefix(msg, "noergler bot <noergler@example.invalid>") {
		t.Errorf("commit:\n%s", msg)
	}
	if len(h.st.Fixes) != 1 || h.st.Fixes[0].Outcome != store.FixCommitted || h.st.Fixes[0].ConflictCommentID != 77 {
		t.Fatalf("fixes = %+v", h.st.Fixes)
	}
	reply := h.bb.Replies[len(h.bb.Replies)-1].Text
	for _, want := range []string{"Merge conflict resolved in", "`list.txt:2` (rule: both sides kept)", "Merged without a conflict: `other.txt`", "build and test"} {
		if !strings.Contains(reply, want) {
			t.Errorf("reply lacks %q:\n%s", want, reply)
		}
	}
	repo.run(repo.bare, "fsck", "--strict")
}

// Both branches changed the same line: the model picks, its dropped line
// is quoted, its call is billed as a fix call.
func TestConflictResolvedByTheModel(t *testing.T) {
	repo := newConflictRepo(t,
		map[string]string{"a.go": "func f() {\n\treturn 1\n}\n"},
		map[string]string{"a.go": "func f() {\n\treturn 2\n}\n"},
		map[string]string{"a.go": "func f() {\n\treturn 2 // fixed\n}\n"})
	h := conflictHarness(t, repo)
	h.llm.conflict = inference.ConflictResult{Outcome: inference.OutcomeOK,
		Resolution: inference.ConflictResolution{Feasible: true, Lines: map[int][]string{1: {"S1"}}},
		Cost:       inference.CallCost{Answered: true, NanoUSD: ptri(1000)}}

	h.askConflict(t, "alice")

	if len(h.llm.Conflicts) != 1 || !strings.Contains(h.llm.Conflicts[0].Prompt, "T1: \treturn 2") {
		t.Fatalf("model requests = %+v", h.llm.Conflicts)
	}
	if got := repo.show("feature/thing:a.go"); got != "func f() {\n\treturn 2 // fixed\n}" {
		t.Fatalf("a.go = %q", got)
	}
	reply := h.bb.Replies[len(h.bb.Replies)-1].Text
	if !strings.Contains(reply, "(model)") || !strings.Contains(reply, "Dropped at `a.go:2` from `master`") || !strings.Contains(reply, "\treturn 2\n") {
		t.Errorf("reply:\n%s", reply)
	}
	if len(h.st.CommentCalls) != 1 || h.st.CommentCalls[0].Kind != store.CallFix {
		t.Errorf("calls = %+v, want one fix call", h.st.CommentCalls)
	}
}

// A model answer that names a line no side has is refused: nothing pushed.
func TestConflictModelAnswerValidated(t *testing.T) {
	repo := newConflictRepo(t,
		map[string]string{"a.go": "x := 1\n"},
		map[string]string{"a.go": "x := 2\n"},
		map[string]string{"a.go": "x := 3\n"})
	before := repo.run(repo.bare, "rev-parse", "feature/thing")
	for name, lines := range map[string][]string{"base line": {"B1"}, "unknown": {"S2"}, "twice": {"S1", "S1"}} {
		t.Run(name, func(t *testing.T) {
			h := conflictHarness(t, repo)
			h.llm.conflict = inference.ConflictResult{Outcome: inference.OutcomeOK,
				Resolution: inference.ConflictResolution{Feasible: true, Lines: map[int][]string{1: lines}}}
			h.askConflict(t, "alice")
			if len(h.st.Fixes) != 1 || h.st.Fixes[0].Outcome != store.FixRejected {
				t.Fatalf("fixes = %+v", h.st.Fixes)
			}
			if got := repo.run(repo.bare, "rev-parse", "feature/thing"); got != before {
				t.Fatal("a rejected resolution was pushed")
			}
		})
	}
}

// The model says it is no choice of lines: the author is told why.
func TestConflictModelSaysNotFeasible(t *testing.T) {
	repo := newConflictRepo(t, map[string]string{"a.go": "x := 1\n"}, map[string]string{"a.go": "x := 2\n"}, map[string]string{"a.go": "x := 3\n"})
	h := conflictHarness(t, repo)
	h.llm.conflict = inference.ConflictResult{Outcome: inference.OutcomeOK,
		Resolution: inference.ConflictResolution{Reason: "both changed the value; only you know which is right."}}
	h.askConflict(t, "alice")
	if len(h.st.Fixes) != 1 || h.st.Fixes[0].Outcome != store.FixNotFeasible ||
		!strings.Contains(h.bb.Replies[0].Text, "only you know which is right") {
		t.Fatalf("fixes = %+v, replies = %+v", h.st.Fixes, h.bb.Replies)
	}
}

func TestConflictOnlyForThePRAuthor(t *testing.T) {
	h := conflictHarness(t, nil)
	h.askConflict(t, "bob")
	if len(h.st.Fixes) != 0 || len(h.bb.Replies) != 1 || !strings.Contains(h.bb.Replies[0].Text, "only when the PR's author asks") {
		t.Fatalf("fixes = %+v, replies = %+v", h.st.Fixes, h.bb.Replies)
	}
}

func TestConflictNoneReported(t *testing.T) {
	h := conflictHarness(t, nil)
	h.bb.mergeStatus = bitbucket.MergeStatus{CanMerge: true}
	h.askConflict(t, "alice")
	if len(h.st.Fixes) != 1 || h.st.Fixes[0].Outcome != store.FixNotFeasible || !strings.Contains(h.bb.Replies[0].Text, "reports no merge conflict") {
		t.Fatalf("fixes = %+v, replies = %+v", h.st.Fixes, h.bb.Replies)
	}
}

func TestConflictFromAForkIsRefused(t *testing.T) {
	h := conflictHarness(t, nil)
	p := mentionPayload("@noergler resolve the merge conflict", "alice")
	p.PullRequest.FromRef.Repository.Project.Key = "~ALICE"
	h.r.HandleMention(context.Background(), p, "platform", &inlineScheduler{})
	if len(h.st.Fixes) != 1 || h.st.Fixes[0].Outcome != store.FixNotFeasible || !strings.Contains(h.bb.Replies[0].Text, "fork") {
		t.Fatalf("fixes = %+v, replies = %+v", h.st.Fixes, h.bb.Replies)
	}
}

// At the cost cap a conflict code resolves still is; one the model must
// answer is refused with an answer, never silently.
func TestConflictAtTheCap(t *testing.T) {
	t.Run("code", func(t *testing.T) {
		repo := newConflictRepo(t, map[string]string{"l": "a\n"}, map[string]string{"l": "a\nt\n"}, map[string]string{"l": "a\ns\n"})
		h := conflictHarness(t, repo)
		h.st.prCost = ptri(6_000_000_000)
		h.askConflict(t, "alice")
		if len(h.st.Fixes) != 1 || h.st.Fixes[0].Outcome != store.FixCommitted {
			t.Fatalf("fixes = %+v", h.st.Fixes)
		}
	})
	t.Run("model", func(t *testing.T) {
		repo := newConflictRepo(t, map[string]string{"l": "a\n"}, map[string]string{"l": "b\n"}, map[string]string{"l": "c\n"})
		h := conflictHarness(t, repo)
		h.st.prCost = ptri(6_000_000_000)
		h.askConflict(t, "alice")
		if len(h.llm.Conflicts) != 0 || len(h.st.Fixes) != 1 || h.st.Fixes[0].Outcome != store.FixNotFeasible ||
			!strings.Contains(h.bb.Replies[0].Text, "cost limit") {
			t.Fatalf("calls = %d, fixes = %+v, replies = %+v", len(h.llm.Conflicts), h.st.Fixes, h.bb.Replies)
		}
	})
}

// Past the bounds nothing is asked or pushed: a lockfile, too many
// conflicts.
func TestConflictPastTheBounds(t *testing.T) {
	t.Run("lockfile", func(t *testing.T) {
		repo := newConflictRepo(t, map[string]string{"go.sum": "a\n"}, map[string]string{"go.sum": "b\n"}, map[string]string{"go.sum": "c\n"})
		h := conflictHarness(t, repo)
		h.askConflict(t, "alice")
		if len(h.st.Fixes) != 1 || h.st.Fixes[0].Outcome != store.FixNotFeasible || !strings.Contains(h.bb.Replies[0].Text, "`go.sum`") {
			t.Fatalf("fixes = %+v, replies = %+v", h.st.Fixes, h.bb.Replies)
		}
	})
	t.Run("too many conflicts", func(t *testing.T) {
		var b, tg, s strings.Builder
		for i := range 6 {
			b.WriteString("keep\nx\n")
			tg.WriteString("keep\nt" + string(rune('0'+i)) + "\n")
			s.WriteString("keep\ns" + string(rune('0'+i)) + "\n")
		}
		repo := newConflictRepo(t, map[string]string{"f": b.String()}, map[string]string{"f": tg.String()}, map[string]string{"f": s.String()})
		h := conflictHarness(t, repo)
		h.askConflict(t, "alice")
		if len(h.llm.Conflicts) != 0 || len(h.st.Fixes) != 1 || !strings.Contains(h.bb.Replies[0].Text, "6 conflicts in 1 files") {
			t.Fatalf("fixes = %+v, replies = %+v", h.st.Fixes, h.bb.Replies)
		}
	})
}

func TestConflictPushWithoutWritePermission(t *testing.T) {
	repo := newConflictRepo(t, map[string]string{"l": "a\n"}, map[string]string{"l": "a\nt\n"}, map[string]string{"l": "a\ns\n"})
	repo.readOnly = true
	h := conflictHarness(t, repo)
	h.askConflict(t, "alice")
	if len(h.st.Fixes) != 1 || h.st.Fixes[0].Outcome != store.FixNoWrite {
		t.Fatalf("fixes = %+v", h.st.Fixes)
	}
}

// The branch moves while the model answers: the push is refused, nothing
// is overwritten.
func TestConflictBranchMovedDuringTheCall(t *testing.T) {
	repo := newConflictRepo(t, map[string]string{"a.go": "x := 1\n"}, map[string]string{"a.go": "x := 2\n"}, map[string]string{"a.go": "x := 3\n"})
	h := conflictHarness(t, repo)
	h.llm.conflict = inference.ConflictResult{Outcome: inference.OutcomeOK,
		Resolution: inference.ConflictResolution{Feasible: true, Lines: map[int][]string{1: {"S1"}}}}
	sched := &deferredScheduler{}
	h.r.HandleMention(context.Background(), mentionPayload("@noergler fix the merge conflict", "alice"), "platform", sched)
	repo.run(repo.work, "checkout", "-q", "feature/thing")
	repo.commit(map[string]string{"new.txt": "n\n"}, "a push meanwhile")
	repo.run(repo.work, "push", "-q", repo.bare, "feature/thing")
	moved := repo.run(repo.bare, "rev-parse", "feature/thing")
	sched.finish()
	if len(h.st.Fixes) != 1 || h.st.Fixes[0].Outcome != store.FixConflict {
		t.Fatalf("fixes = %+v", h.st.Fixes)
	}
	if got := repo.run(repo.bare, "rev-parse", "feature/thing"); got != moved {
		t.Fatal("the push overwrote the branch")
	}
}

// Under a finding, "@bot behebe den Merge-Konflikt" is the PR's conflict:
// the finding route hands it on, the model never reads it as a fix.
func TestConflictRequestUnderAFindingGoesToTheMentionPath(t *testing.T) {
	h := conflictHarness(t, nil)
	h.st.byComment = map[int]*store.Finding{900: {ID: 5, FilePath: "a.go", LineNumber: 1}}
	p := mentionPayload("@noergler behebe den Merge-Konflikt", "alice")
	p.CommentParentID = 900
	handled, _ := h.r.HandleFindingReply(context.Background(), p, "platform", &inlineScheduler{})
	if handled || len(h.llm.Replies) != 0 {
		t.Fatalf("handled = %t, replies = %d", handled, len(h.llm.Replies))
	}
}

// One branch added a file where the other added a directory: git's
// file/directory conflict, which no tree holds. Refused, nothing pushed.
func TestConflictFileAgainstDirectory(t *testing.T) {
	repo := newConflictRepo(t,
		map[string]string{"a.go": "x := 1\n"},
		map[string]string{"a.go": "x := 2\n", "lib/util.go": "package lib\n"},
		map[string]string{"a.go": "x := 3\n", "lib": "a file\n"})
	h := conflictHarness(t, repo)
	h.askConflict(t, "alice")
	if len(h.llm.Conflicts) != 0 || len(h.st.Fixes) != 1 || h.st.Fixes[0].Outcome != store.FixNotFeasible ||
		!strings.Contains(h.bb.Replies[0].Text, "is a file on one branch and a directory on the other") {
		t.Fatalf("fixes = %+v, replies = %+v", h.st.Fixes, h.bb.Replies)
	}
}

// Dropped lines too long to quote in full: refused before the push, never
// a cut quote.
func TestConflictAnswerTooLongToQuote(t *testing.T) {
	long := strings.Repeat("x", 13_000)
	repo := newConflictRepo(t,
		map[string]string{"a.json": "{\"k\": 1}\n"},
		map[string]string{"a.json": "{\"k\": \"" + long + "\"}\n"},
		map[string]string{"a.json": "{\"k\": 2}\n"})
	before := repo.run(repo.bare, "rev-parse", "feature/thing")
	h := conflictHarness(t, repo)
	h.llm.conflict = inference.ConflictResult{Outcome: inference.OutcomeOK,
		Resolution: inference.ConflictResolution{Feasible: true, Lines: map[int][]string{1: {"S1"}}}}
	h.askConflict(t, "alice")
	if len(h.st.Fixes) != 1 || h.st.Fixes[0].Outcome != store.FixNotFeasible || !strings.Contains(h.bb.Replies[0].Text, "too long") {
		t.Fatalf("fixes = %+v, replies = %+v", h.st.Fixes, h.bb.Replies)
	}
	if got := repo.run(repo.bare, "rev-parse", "feature/thing"); got != before {
		t.Fatal("pushed")
	}
}

// The author's plain reply under a finding, no mention: the PR's conflict
// all the same, never the finding's fix.
func TestConflictPlainReplyUnderAFinding(t *testing.T) {
	repo := newConflictRepo(t, map[string]string{"l": "a\n"}, map[string]string{"l": "a\nt\n"}, map[string]string{"l": "a\ns\n"})
	h := conflictHarness(t, repo)
	h.st.byComment = map[int]*store.Finding{900: {ID: 5, FilePath: "a.go", LineNumber: 1}}
	p := mentionPayload("bitte behebe den Merge-Konflikt", "alice")
	p.CommentParentID = 900
	handled, _ := h.r.HandleFindingReply(context.Background(), p, "platform", &inlineScheduler{})
	if !handled || len(h.llm.Replies) != 0 || len(h.st.Fixes) != 1 || h.st.Fixes[0].Outcome != store.FixCommitted {
		t.Fatalf("handled = %t, replies = %d, fixes = %+v", handled, len(h.llm.Replies), h.st.Fixes)
	}
}

// A bot account without an email commits nothing.
func TestConflictWithoutBotEmail(t *testing.T) {
	repo := newConflictRepo(t, map[string]string{"l": "a\n"}, map[string]string{"l": "a\nt\n"}, map[string]string{"l": "a\ns\n"})
	before := repo.run(repo.bare, "rev-parse", "feature/thing")
	h := conflictHarness(t, repo)
	h.bb.noBotEmail = true
	h.askConflict(t, "alice")
	if len(h.st.Fixes) != 1 || h.st.Fixes[0].Outcome != store.FixNotFeasible || !strings.Contains(h.bb.Replies[0].Text, "no email address") {
		t.Fatalf("fixes = %+v, replies = %+v", h.st.Fixes, h.bb.Replies)
	}
	if got := repo.run(repo.bare, "rev-parse", "feature/thing"); got != before {
		t.Fatal("pushed")
	}
}
