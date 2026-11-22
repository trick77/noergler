package review

import (
	"context"
	"fmt"
	"slices"
	"strings"
	"testing"

	"github.com/trick77/noergler/internal/bitbucket"
	"github.com/trick77/noergler/internal/webhook"
)

// Candidates are every ancestor directory of a changed file below the root,
// at any depth: shallow first, then by coverage, then by path; root files
// probe nothing.
func TestNestedCandidatesEveryAncestor(t *testing.T) {
	got, cut := nestedCandidates([]string{
		"README.md",
		"services/pay/api/handler.go",
		"services/pay/api/routes.go",
		"services/pay/a/b/c/d/deep.go",
		"web/app.ts",
	})
	want := []string{
		"services/", "web/", "services/pay/", "services/pay/api/", "services/pay/a/",
		"services/pay/a/b/", "services/pay/a/b/c/", "services/pay/a/b/c/d/",
	}
	if !slices.Equal(got, want) || cut != 0 {
		t.Errorf("candidates = %v (cut %d), want %v", got, cut, want)
	}
}

// A module root with one changed file outranks deep packages with many: a
// cut never drops it.
func TestNestedCandidatesCutKeepsAThinModule(t *testing.T) {
	var paths []string
	for i := range 50 {
		pkg := fmt.Sprintf("services/a/src/main/java/com/acme/pkg%d/", i)
		paths = append(paths, pkg+"A.java", pkg+"B.java")
	}
	paths = append(paths, "services/b/x.go")
	got, cut := nestedCandidates(paths)
	if cut == 0 || !slices.Contains(got, "services/b/") {
		t.Errorf("services/b/ must be probed (cut %d): %v", cut, got)
	}
}

// Over the probe cap, the deep package directories of a Java monorepo go
// first: the parent and every module root are probed.
func TestNestedCandidatesCutKeepsModuleRoots(t *testing.T) {
	var paths []string
	modules := []string{"a", "b", "c", "d", "e", "f", "g", "h", "i"}
	for _, m := range modules {
		paths = append(paths, "services/"+m+"/src/main/java/com/acme/X.java")
	}
	got, cut := nestedCandidates(paths)
	// services/ + 9 modules x 6 directories = 55, five over the cap.
	if len(got) != nestedAgentsProbes || cut != 55-nestedAgentsProbes {
		t.Fatalf("probes %d, cut %d", len(got), cut)
	}
	if !slices.Contains(got, "services/") {
		t.Errorf("the shared parent must be probed: %v", got)
	}
	for _, m := range modules {
		if !slices.Contains(got, "services/"+m+"/") {
			t.Errorf("module root services/%s/ not probed", m)
		}
	}
	for _, m := range modules[4:] {
		if slices.Contains(got, "services/"+m+"/src/main/java/com/acme/") {
			t.Errorf("the deepest directories are the ones cut, got services/%s/.../acme/", m)
		}
	}
}

// Over the token limit the nested files go, last first; the root stays.
func TestWithNestedDropsOverTheLimit(t *testing.T) {
	nested := []nestedAgentsMD{{Dir: "a/", Content: "aaaa"}, {Dir: "b/", Content: "bbbb"}}
	count := func(s string) int { return len(s) }
	all, dropped := withNested("ROOT", nested, 0, count)
	if !strings.Contains(all, "### a/AGENTS.md (applies to files under `a/`; where it differs from the guidelines above, it wins for them)\n\naaaa") || len(dropped) != 0 {
		t.Fatalf("no limit: %q, dropped %v", all, dropped)
	}
	// Room for the root and the first file, not the second.
	limit := count("ROOT") + count(nestedBlock(nested[0])) + 1
	limited, dropped := withNested("ROOT", nested, limit, count)
	if !strings.HasPrefix(limited, "ROOT") || strings.Contains(limited, "bbbb") || !slices.Equal(dropped, []string{"b/AGENTS.md"}) {
		t.Errorf("limited = %q, dropped %v", limited, dropped)
	}
	if only, _ := withNested("", nested[:1], 0, count); strings.HasPrefix(only, "\n") || !strings.HasPrefix(only, "### a/") {
		t.Errorf("no root: %q", only)
	}
}

// With the limit off, the root is not counted but the nested files alone
// stay under nestedUnlimitedTokens.
func TestWithNestedUnlimitedStillCapsNested(t *testing.T) {
	big := strings.Repeat("x", nestedUnlimitedTokens/2)
	nested := []nestedAgentsMD{{Dir: "a/", Content: big}, {Dir: "b/", Content: big}, {Dir: "c/", Content: big}}
	count := func(s string) int { return len(s) }
	root := strings.Repeat("r", 3*nestedUnlimitedTokens)
	all, dropped := withNested(root, nested, 0, count)
	if !strings.Contains(all, "### a/AGENTS.md") || !slices.Equal(dropped, []string{"b/AGENTS.md", "c/AGENTS.md"}) {
		t.Errorf("dropped %v, want b/ and c/ (a/ fits whatever the root's size)", dropped)
	}
}

// The memo starts over past its byte cap.
func TestNestedMemoByteCap(t *testing.T) {
	var c nestedMemo
	half := nestedProbe{content: strings.Repeat("x", nestedMemoMaxBytes/2+1)}
	c.put("k1", "a/", half)
	c.put("k2", "a/", half)
	if _, ok := c.get("k1", "a/"); ok {
		t.Errorf("k1 must be gone past the byte cap")
	}
	if _, ok := c.get("k2", "a/"); !ok || c.bytes != len(half.content) {
		t.Errorf("k2 must be held alone, bytes %d", c.bytes)
	}
}

// The footer counts the root file alone: nested files neither replace it
// nor bloat it.
func TestRootTokensCountsTheRootAlone(t *testing.T) {
	count := func(s string) int { return len(s) }
	if got := rootTokens("ROOT", "ROOT", 99, count); got != 99 {
		t.Errorf("unchanged prompt text: %d, want its own count", got)
	}
	if got := rootTokens("ROOT", "ROOT\n\n### a/AGENTS.md", 99, count); got != 4 {
		t.Errorf("with nested: %d, want the root's 4", got)
	}
	if got := rootTokens("", "### a/AGENTS.md", 99, count); got != 0 {
		t.Errorf("no root: %d, want 0", got)
	}
}

const nestedDiff = `diff --git a/services/pay/a.go b/services/pay/a.go
--- a/services/pay/a.go
+++ b/services/pay/a.go
@@ -1,1 +1,2 @@
 package a
+func new() {}
`

// A review carries the whole chain below the root: every AGENTS.md above
// the changed file, read at the PR head, parent before child.
func TestReviewCarriesTheNestedChain(t *testing.T) {
	h := newHarness(t, func(h *harness) {
		h.bb.prDiff = nestedDiff
		h.bb.files["src1:services/pay/a.go"] = "package a\nfunc new() {}\n"
		h.bb.files["src1:services/pay/AGENTS.md"] = "Money is integer cents."
		h.bb.files["src1:services/AGENTS.md"] = "Every service logs with team=."
		h.bb.files["src1:web/AGENTS.md"] = "Never used: no changed file below."
	})

	h.r.reviewInline(context.Background(), prPayload(webhook.EventOpened), false)

	if len(h.llm.Reviews) != 1 {
		t.Fatalf("reviews = %d, want 1", len(h.llm.Reviews))
	}
	p := h.llm.Reviews[0].Prompt
	root := strings.Index(p, "Be terse.")
	parent := strings.Index(p, "### services/AGENTS.md")
	child := strings.Index(p, "### services/pay/AGENTS.md")
	if root < 0 || parent < root || child < parent {
		t.Errorf("want root, then services/, then services/pay/:\n%s", p)
	}
	if !strings.Contains(p, "Money is integer cents.") || !strings.Contains(p, "Every service logs with team=.") {
		t.Errorf("both nested files must be carried:\n%s", p)
	}
	if strings.Contains(p, "Never used") {
		t.Errorf("a file above no changed file must not be carried:\n%s", p)
	}
}

// No depth limit: a file five levels down is found.
func TestFetchNestedFindsAnyDepth(t *testing.T) {
	h := newHarness(t, func(h *harness) {
		h.bb.files["c1:a/b/c/d/e/AGENTS.md"] = "deep rules"
	})
	got := h.r.fetchNestedAgentsMD(context.Background(), "P", "R", "c1", "t", []string{"a/b/c/d/e/f/x.go"})
	if len(got) != 1 || got[0].Dir != "a/b/c/d/e/" {
		t.Errorf("found %v, want a/b/c/d/e/", got)
	}
}

// A second lookup at the same commit reads nothing again; another commit,
// one that adds or removes a file, is probed afresh.
func TestFetchNestedMemoByCommit(t *testing.T) {
	h := newHarness(t, nil)
	ctx := context.Background()
	paths := []string{"services/pay/a.go"}
	look := func(commit string) []nestedAgentsMD {
		return h.r.fetchNestedAgentsMD(ctx, "P", "R", commit, "t", paths)
	}
	h.bb.files["c1:services/pay/AGENTS.md"] = "pay rules"
	if got := look("c1"); len(got) != 1 {
		t.Fatalf("c1: %v", got)
	}
	look("c1")
	for _, f := range []string{"c1:services/AGENTS.md", "c1:services/pay/AGENTS.md"} {
		if n := h.bb.fileReads[f]; n != 1 {
			t.Errorf("%s read %d times, want 1 (the second lookup from the memo)", f, n)
		}
	}
	// c2 adds services/AGENTS.md, c3 removes both.
	h.bb.files["c2:services/AGENTS.md"] = "shared"
	h.bb.files["c2:services/pay/AGENTS.md"] = "pay rules"
	if got := look("c2"); len(got) != 2 {
		t.Errorf("c2: %v, want both", got)
	}
	if got := look("c3"); len(got) != 0 {
		t.Errorf("c3: %v, want none", got)
	}
}

// A failed read is not a fact about the commit: never remembered, so the
// next lookup at the same commit reads again and finds the file.
func TestFetchNestedErrorNotRemembered(t *testing.T) {
	h := newHarness(t, nil)
	ctx := context.Background()
	key := "c1:services/AGENTS.md"
	h.bb.fileErr[key] = &bitbucket.StatusError{Method: "GET", Path: "/raw/services/AGENTS.md", Status: 500}
	if got := h.r.fetchNestedAgentsMD(ctx, "P", "R", "c1", "t", []string{"services/a.go"}); len(got) != 0 {
		t.Fatalf("first lookup: %v, want none", got)
	}
	delete(h.bb.fileErr, key)
	h.bb.files[key] = "shared"
	if got := h.r.fetchNestedAgentsMD(ctx, "P", "R", "c1", "t", []string{"services/a.go"}); len(got) != 1 {
		t.Errorf("second lookup: %v, want the file", got)
	}
}

// The first failed read ends the lookup: the deeper directories are not
// probed against a Bitbucket that is failing.
func TestFetchNestedStopsOnFirstError(t *testing.T) {
	h := newHarness(t, nil)
	h.bb.fileErr["c1:services/AGENTS.md"] = &bitbucket.StatusError{Method: "GET", Path: "/raw/services/AGENTS.md", Status: 503}
	h.bb.files["c1:services/pay/AGENTS.md"] = "pay rules"
	got := h.r.fetchNestedAgentsMD(context.Background(), "P", "R", "c1", "t", []string{"services/pay/api/a.go"})
	if len(got) != 0 {
		t.Errorf("found %v after a failed read", got)
	}
	for _, f := range []string{"c1:services/pay/AGENTS.md", "c1:services/pay/api/AGENTS.md"} {
		if n := h.bb.fileReads[f]; n != 0 {
			t.Errorf("%s read %d times after the lookup failed", f, n)
		}
	}
}

// A file over the byte cap is left out, remembered as such, and logged on
// every lookup.
func TestFetchNestedOversizeLeftOut(t *testing.T) {
	h := newHarness(t, nil)
	h.bb.files["c1:services/AGENTS.md"] = strings.Repeat("x", nestedAgentsMaxBytes+1)
	for range 2 {
		if got := h.r.fetchNestedAgentsMD(context.Background(), "P", "R", "c1", "t", []string{"services/a.go"}); len(got) != 0 {
			t.Errorf("oversize kept: %v", got)
		}
	}
	if n := h.bb.fileReads["c1:services/AGENTS.md"]; n != 1 {
		t.Errorf("read %d times, want 1", n)
	}
}
