package review

import (
	"context"
	"fmt"
	"strings"
	"testing"

	"github.com/trick77/noergler/internal/inference"
	"github.com/trick77/noergler/internal/store"
	"github.com/trick77/noergler/internal/webhook"
)

// incrementalHarness puts the reviewer on the incremental path, which is the
// only one that fetches a cumulative diff for cross-file context.
func incrementalHarness(t *testing.T, tweak func(*harness)) *harness {
	t.Helper()
	return newHarness(t, func(h *harness) {
		h.st.lastCommit, h.st.hasLast = "old1", true
		h.bb.commitDiff = sampleDiff
		if tweak != nil {
			tweak(h)
		}
	})
}

// trimPreviouslyPosted drops the OLDEST findings in chunks until the rendered
// block fits, so what survives is a tail of the input. No test ever exceeded
// the budget.
func TestTrimPreviouslyPostedDropsOldestUntilItFits(t *testing.T) {
	budget := 628_000
	limit := inference.PreviouslyPostedBudget(budget)

	// One over the count cap, so the cap itself fires before the token loop
	// runs. At exactly maxPreviouslyPostedFindings the slice expression is
	// the identity and deleting the cap from production changes nothing.
	const overCap = maxPreviouslyPostedFindings + 1

	// Each finding renders to well over a token, so enough of them must
	// overflow the limit and force the loop to run too.
	var existing []store.Finding
	for i := 0; i < overCap; i++ {
		existing = append(existing, store.Finding{
			FilePath:    fmt.Sprintf("file%02d.go", i),
			LineNumber:  i + 1,
			Severity:    "issue",
			CommentText: strings.Repeat("a long finding body. ", 200),
		})
	}

	h := newHarness(t, nil)
	got := h.r.trimPreviouslyPosted(context.Background(), "PROJ/my-repo#42", existing, budget)

	if len(got) == 0 {
		t.Fatal("trim dropped everything")
	}
	if len(got) >= len(existing) {
		t.Fatalf("trim kept %d of %d findings; the over-budget branch never ran", len(got), len(existing))
	}
	// The count cap alone would leave maxPreviouslyPostedFindings; anything
	// fewer means the token loop ran on top of it.
	if len(got) > maxPreviouslyPostedFindings {
		t.Errorf("trim kept %d findings, over the %d count cap", len(got), maxPreviouslyPostedFindings)
	}
	if rendered := h.tok.Count(inference.RenderPreviouslyPostedFindings(got)); rendered > limit {
		t.Errorf("trimmed block is %d tokens, over the %d limit", rendered, limit)
	}

	// The survivors are the most recent: the store returns oldest first.
	wantFirst := existing[len(existing)-len(got)].FilePath
	if got[0].FilePath != wantFirst {
		t.Errorf("survivors start at %s, want %s: the trim is not a tail", got[0].FilePath, wantFirst)
	}
	for i, p := range got {
		if want := existing[len(existing)-len(got)+i].FilePath; p.FilePath != want {
			t.Fatalf("survivor %d is %s, want %s", i, p.FilePath, want)
		}
	}
}

// The count cap is the first of two stages and must be pinned on its own:
// with findings long enough to also blow the token budget, the token loop
// trims below the cap anyway and deleting the cap from production changes
// nothing observable. These findings are short enough that the token loop
// never runs, so only the cap can drop anything.
func TestTrimPreviouslyPostedCapsTheCount(t *testing.T) {
	const overCap = maxPreviouslyPostedFindings + 1

	var existing []store.Finding
	for i := 0; i < overCap; i++ {
		existing = append(existing, store.Finding{
			FilePath: fmt.Sprintf("file%02d.go", i), LineNumber: i + 1,
			Severity: "issue", CommentText: "short",
		})
	}

	budget := 628_000
	h := newHarness(t, nil)
	got := h.r.trimPreviouslyPosted(context.Background(), "PROJ/my-repo#42", existing, budget)

	// Proves the token loop stayed out of it: the whole block fits.
	if rendered := h.tok.Count(inference.RenderPreviouslyPostedFindings(got)); rendered > inference.PreviouslyPostedBudget(budget) {
		t.Fatalf("block is %d tokens; this test must isolate the COUNT cap", rendered)
	}
	if len(got) != maxPreviouslyPostedFindings {
		t.Fatalf("kept %d of %d findings, want the cap of %d",
			len(got), overCap, maxPreviouslyPostedFindings)
	}
	// The cap drops the oldest, so the survivors start one in.
	if got[0].FilePath != existing[1].FilePath {
		t.Errorf("survivors start at %s, want %s: the cap is not a tail",
			got[0].FilePath, existing[1].FilePath)
	}
}

// Under budget nothing is dropped.
func TestTrimPreviouslyPostedKeepsEverythingUnderBudget(t *testing.T) {
	existing := []store.Finding{
		{FilePath: "a.go", LineNumber: 1, Severity: "issue", CommentText: "short"},
		{FilePath: "b.go", LineNumber: 2, Severity: "suggestion", CommentText: "also short"},
	}

	h := newHarness(t, nil)
	got := h.r.trimPreviouslyPosted(context.Background(), "PROJ/my-repo#42", existing, 628_000)

	if len(got) != 2 {
		t.Fatalf("kept %d of 2 findings under budget", len(got))
	}
	if got[0].FilePath != "a.go" || got[1].FilePath != "b.go" {
		t.Errorf("order changed: %s, %s", got[0].FilePath, got[1].FilePath)
	}
}

// Previously-posted findings reach the prompt. Nothing asserted this: the
// forwarding at review.go was executed by the dedupe test but never read.
func TestPreviouslyPostedFindingsReachThePrompt(t *testing.T) {
	h := incrementalHarness(t, func(h *harness) {
		h.st.existing = []store.Finding{{
			FilePath: "a.go", LineNumber: 2, Severity: "issue",
			CommentText: "this was already said",
		}}
	})

	h.r.ReviewPullRequest(context.Background(), prPayload(webhook.EventFromRefUpdated), false)

	if len(h.llm.Reviews) != 1 {
		t.Fatalf("expected one LLM call, got %d", len(h.llm.Reviews))
	}
	prompt := h.llm.Reviews[0].Prompt
	for _, want := range []string{"a.go", "this was already said"} {
		if !strings.Contains(prompt, want) {
			t.Errorf("prompt is missing the previously-posted %q", want)
		}
	}
}
