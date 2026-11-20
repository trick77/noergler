package review

import (
	"context"
	"errors"
	"strings"
	"testing"

	"github.com/trick77/noergler/internal/bitbucket"
	"github.com/trick77/noergler/internal/webhook"
)

const touchDiff = `diff --git a/store.go b/store.go
--- a/store.go
+++ b/store.go
@@ -1,2 +1,2 @@
 package a
-func (s *Store) Touch(key string, ttlSeconds int64) {}
+func (s *Store) Touch(key string, ttlMillis int64) {}
`

func touchHarness(t *testing.T) *harness {
	t.Helper()
	return newHarness(t, func(h *harness) {
		h.bb.prDiff = touchDiff
		h.bb.files["src1:store.go"] = "package a\nfunc (s *Store) Touch(key string, ttlMillis int64) {}\n"
		h.bb.searchHits = []bitbucket.CodeHit{
			{Project: "PROJ", Repo: "my-repo", File: "manager.go", Lines: []bitbucket.CodeLine{
				{Line: 12, Text: "\ts.Touch(key, int64(d.Seconds()))"}}},
			// The PR's own file: the default branch holds its old version.
			{Project: "PROJ", Repo: "my-repo", File: "store.go", Lines: []bitbucket.CodeLine{
				{Line: 2, Text: "func (s *Store) Touch(key string, ttlSeconds int64) {}"}}},
			// Another repository: never this PR's code.
			{Project: "OTHER", Repo: "x", File: "y.go", Lines: []bitbucket.CodeLine{{Line: 1, Text: "Touch(k, 1)"}}},
		}
	})
}

// One search for the changed method, scoped to the repo; the unchanged
// caller reaches the prompt, the PR's own file and another repo do not.
func TestReviewSearchesUnchangedCallersOnce(t *testing.T) {
	h := touchHarness(t)
	h.r.reviewInline(context.Background(), prPayload(webhook.EventOpened), false)

	if len(h.bb.Searches) != 1 || h.bb.Searches[0] != "Touch project:PROJ repo:my-repo" {
		t.Fatalf("searches = %q, want exactly one for Touch", h.bb.Searches)
	}
	p := h.llm.Reviews[0].Prompt
	if !strings.Contains(p, "## References in unchanged files") || !strings.Contains(p, "`manager.go:12`") {
		t.Errorf("the unchanged caller must reach the prompt:\n%s", p)
	}
	if strings.Contains(p, "`store.go:2`") || strings.Contains(p, "y.go") {
		t.Errorf("the PR's own file and another repo must stay out:\n%s", p)
	}
}

// A PR into another branch than the default: the search reads the default
// branch only, so there is none. A failed search is no block, never a
// failed review.
func TestReviewSkipsTheSearchOffTheDefaultBranch(t *testing.T) {
	h := touchHarness(t)
	h.bb.defaultBranch = "refs/heads/main"
	h.r.reviewInline(context.Background(), prPayload(webhook.EventOpened), false)
	if len(h.bb.Searches) != 0 {
		t.Errorf("searches = %q, want none for a PR into master when main is the default", h.bb.Searches)
	}

	f := touchHarness(t)
	f.bb.searchErr = errors.New("search is unavailable")
	f.r.reviewInline(context.Background(), prPayload(webhook.EventOpened), false)
	if len(f.llm.Reviews) != 1 || strings.Contains(f.llm.Reviews[0].Prompt, "References in unchanged files") {
		t.Errorf("a failed search must leave the review as it was")
	}
}

// An incremental review whose PR diff could not be read does not know the
// PR's earlier files: an earlier push's file would read as unchanged, so
// there is no search.
func TestReviewSkipsTheSearchWithoutThePRsFiles(t *testing.T) {
	h := touchHarness(t)
	h.st.lastCommit, h.st.hasLast = "older", true
	h.bb.commitDiff = touchDiff
	h.bb.prDiffErr = errors.New("diff unavailable")
	h.r.reviewInline(context.Background(), prPayload(webhook.EventFromRefUpdated), false)
	if len(h.bb.Searches) != 0 {
		t.Errorf("searches = %q, want none while the PR's files are unknown", h.bb.Searches)
	}
}

// A deleted file's definitions are searched: its callers break first, and
// compression would drop the file before the review sees it.
func TestReviewSearchesADeletedFilesNames(t *testing.T) {
	h := touchHarness(t)
	h.bb.prDiff = touchDiff + "diff --git a/legacy.go b/legacy.go\ndeleted file mode 100644\n--- a/legacy.go\n+++ /dev/null\n" +
		"@@ -1,2 +0,0 @@\n-package a\n-func FetchAccount(id string) {}\n"
	h.r.reviewInline(context.Background(), prPayload(webhook.EventOpened), false)
	if len(h.bb.Searches) != 1 || h.bb.Searches[0] != "(Touch OR FetchAccount) project:PROJ repo:my-repo" {
		t.Errorf("searches = %q, want one for Touch and FetchAccount", h.bb.Searches)
	}
}

// A moved file's old path is the PR's too: the default branch's copy there
// never reads as an unchanged file.
func TestReviewLeavesOutAMovedFilesOldPath(t *testing.T) {
	h := touchHarness(t)
	h.bb.prDiff = "diff --git a/old/store.go b/store.go\nrename from old/store.go\nrename to store.go\n" +
		strings.TrimPrefix(touchDiff, "diff --git a/store.go b/store.go\n")
	h.bb.searchHits = append(h.bb.searchHits, bitbucket.CodeHit{Project: "PROJ", Repo: "my-repo", File: "old/store.go",
		Lines: []bitbucket.CodeLine{{Line: 2, Text: "func (s *Store) Touch(key string, ttlSeconds int64) {}"}}})
	h.r.reviewInline(context.Background(), prPayload(webhook.EventOpened), false)
	if len(h.llm.Reviews) != 1 {
		t.Fatalf("reviews = %d, want 1", len(h.llm.Reviews))
	}
	if p := h.llm.Reviews[0].Prompt; strings.Contains(p, "`old/store.go:2`") || !strings.Contains(p, "`manager.go:12`") {
		t.Errorf("the moved file's old path must stay out, the caller in:\n%s", p)
	}
}
