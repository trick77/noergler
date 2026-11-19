package review

import (
	"context"
	"slices"
	"strings"
	"testing"

	"github.com/trick77/noergler/internal/webhook"
)

// Candidates are the changed files' directories below the root, down to
// three levels: deepest first (the nearest file applies, and a parent
// covers at least its children's files), then by coverage, then by path;
// root files probe nothing.
func TestNestedCandidatesDeepestFirst(t *testing.T) {
	got, cut := nestedCandidates([]string{
		"README.md",
		"services/pay/api/handler.go",
		"services/pay/api/routes.go",
		"services/pay/a/b/c/deep.go",
		"web/app.ts",
	})
	want := []string{"services/pay/api/", "services/pay/a/", "services/pay/", "services/", "web/"}
	if !slices.Equal(got, want) || cut != 0 {
		t.Errorf("candidates = %v (cut %d), want %v", got, cut, want)
	}
	// Nine services under one parent: every service directory is probed
	// before the parent, which a coverage-first order would not do.
	var many []string
	for _, d := range []string{"a", "b", "c", "d", "e", "f", "g", "h", "i"} {
		many = append(many, "services/"+d+"/x.go")
	}
	got, cut = nestedCandidates(many)
	if len(got) != nestedAgentsProbes || cut != 2 || slices.Contains(got, "services/") {
		t.Errorf("probes = %v, cut %d: want %d service dirs and the rest cut", got, cut, nestedAgentsProbes)
	}
}

// A found file is kept only where it is some changed file's nearest one;
// the most-covering win the cap; the kept ones go out in path order.
func TestPickNestedKeepsTheNearest(t *testing.T) {
	found := []nestedAgentsMD{{Dir: "services/"}, {Dir: "services/pay/"}, {Dir: "web/"}, {Dir: "lib/"}, {Dir: "tools/"}}
	paths := []string{"services/pay/a.go", "services/pay/b.go", "web/x.ts", "lib/l.go", "tools/t.go", "tools/u.go"}
	var dirs []string
	kept, left := pickNested(found, paths)
	for _, n := range kept {
		dirs = append(dirs, n.Dir)
	}
	// services/ is nobody's nearest (services/pay/ is); of the four left,
	// the two covering two files and then lib/ before web/ by path.
	if want := []string{"lib/", "services/pay/", "tools/"}; !slices.Equal(dirs, want) {
		t.Errorf("kept = %v, want %v", dirs, want)
	}
	if want := []string{"services/AGENTS.md (not the nearest)", "web/AGENTS.md (over the cap)"}; !slices.Equal(left, want) {
		t.Errorf("left = %v, want %v (each named, for the log)", left, want)
	}
}

// Over the token limit the nested files go, last first; the root stays.
func TestWithNestedDropsOverTheLimit(t *testing.T) {
	nested := []nestedAgentsMD{{Dir: "a/", Content: "aaaa"}, {Dir: "b/", Content: "bbbb"}}
	count := func(s string) int { return len(s) }
	all, dropped := withNested("ROOT", nested, 0, count)
	if !strings.Contains(all, "### a/AGENTS.md (applies to files under `a/`)\n\naaaa") || len(dropped) != 0 {
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

// A review carries the nested AGENTS.md of the changed file's directory
// after the root one, read at the PR head.
func TestReviewCarriesTheNestedAgentsMD(t *testing.T) {
	h := newHarness(t, func(h *harness) {
		h.bb.prDiff = nestedDiff
		h.bb.files["src1:services/pay/a.go"] = "package a\nfunc new() {}\n"
		h.bb.files["src1:services/pay/AGENTS.md"] = "Money is integer cents."
		h.bb.files["src1:services/AGENTS.md"] = "Never used: not the nearest."
	})

	h.r.reviewInline(context.Background(), prPayload(webhook.EventOpened), false)

	if len(h.llm.Reviews) != 1 {
		t.Fatalf("reviews = %d, want 1", len(h.llm.Reviews))
	}
	p := h.llm.Reviews[0].Prompt
	root, nested := strings.Index(p, "Be terse."), strings.Index(p, "### services/pay/AGENTS.md")
	if root < 0 || nested < root || !strings.Contains(p, "Money is integer cents.") {
		t.Errorf("the prompt must carry the root rules, then the nested file:\n%s", p)
	}
	if strings.Contains(p, "Never used") {
		t.Errorf("a file nearer to every changed file wins:\n%s", p)
	}
}
