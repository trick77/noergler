package review

import (
	"bytes"
	"context"
	"fmt"
	"log/slog"
	"strings"
	"testing"

	"github.com/trick77/noergler/internal/inference"
	"github.com/trick77/noergler/internal/webhook"
)

// addedFileDiff is a one-hunk diff of path adding body lines.
func addedFileDiff(path string, body ...string) string {
	var b strings.Builder
	fmt.Fprintf(&b, "diff --git a/%s b/%s\n--- a/%s\n+++ b/%s\n@@ -1,1 +1,%d @@\n", path, path, path, path, len(body))
	for _, l := range body {
		b.WriteString("+" + l + "\n")
	}
	return b.String()
}

// cumulativeBlock is the part of the prompt inside the cumulative tag: the
// focused files precede it and carry the same paths.
func cumulativeBlock(t *testing.T, prompt string) string {
	t.Helper()
	_, after, ok := strings.Cut(prompt, "<cumulative_pr_diff>")
	if !ok {
		t.Fatalf("prompt has no cumulative block:\n%s", prompt)
	}
	return after
}

// runIncremental reviews one push against prDiff and returns the prompt and
// the log.
func runIncremental(t *testing.T, prDiff string) (prompt, logs string) {
	t.Helper()
	h := incrementalHarness(t, func(h *harness) { h.bb.prDiff = prDiff })
	var buf bytes.Buffer
	h.r.log = slog.New(slog.NewTextHandler(&buf, nil))
	h.r.ReviewPullRequest(context.Background(), prPayload(webhook.EventFromRefUpdated), false)
	if len(h.llm.Reviews) != 1 {
		t.Fatalf("expected one LLM call, got %d", len(h.llm.Reviews))
	}
	return h.llm.Reviews[0].Prompt, buf.String()
}

// The push in these tests is sampleDiff, a few dozen tokens, so the scaled
// budget sits on its floor.
const floorBudget = 4_000

// A part hopelessly over budget by byte count alone is left out without ever
// being tokenized: tokenizing expands the text in RAM, and the pod has 2 Gi.
// The skip is only observable as an absence, so the fake tokenizer records
// what it was asked to count.
func TestOversizedCumulativePartIsOmittedUntokenized(t *testing.T) {
	huge := strings.Repeat("x", floorBudget*bytesPerTokenCeiling+1)

	h := incrementalHarness(t, func(h *harness) { h.bb.prDiff = huge })
	h.r.ReviewPullRequest(context.Background(), prPayload(webhook.EventFromRefUpdated), false)

	if h.tok.counted(len(huge)) {
		t.Error("an over-byte-budget cumulative part was tokenized; the byte pre-check did not fire")
	}
	if len(h.llm.Reviews) != 1 {
		t.Fatalf("expected one LLM call, got %d", len(h.llm.Reviews))
	}
	if strings.Contains(h.llm.Reviews[0].Prompt, huge[:1000]) {
		t.Error("the omitted cumulative part reached the prompt")
	}
}

// Under the byte ceiling the part IS tokenized, which is what makes the test
// above a pre-check test and not a "large diffs are dropped" test.
func TestCumulativePartUnderByteCeilingIsTokenized(t *testing.T) {
	// At 4 bytes per token in the fake, half the budget.
	sized := strings.Repeat("x", floorBudget*2)

	h := incrementalHarness(t, func(h *harness) { h.bb.prDiff = sized })
	h.r.ReviewPullRequest(context.Background(), prPayload(webhook.EventFromRefUpdated), false)

	if !h.tok.counted(len(sized)) {
		t.Error("a part under the byte ceiling should have been tokenized")
	}
	if !strings.Contains(h.llm.Reviews[0].Prompt, sized) {
		t.Error("a part within budget did not reach the prompt")
	}
}

// A lockfile is in the PR diff and is never reviewed, so it is not context
// either: unfiltered, it took the cumulative budget from real code.
func TestCumulativeDiffDropsNonReviewableFiles(t *testing.T) {
	prompt, logs := runIncremental(t, sampleDiff+addedFileDiff("go.sum", "example.com/mod v1.0.0 h1:lockfilehash"))

	if strings.Contains(prompt, "lockfilehash") {
		t.Error("a lockfile hunk reached the cumulative block")
	}
	if !strings.Contains(logs, "1 filtered as non-reviewable") {
		t.Errorf("the selection line does not report the filtered file:\n%s", logs)
	}
	if !strings.Contains(cumulativeBlock(t, prompt), "+func new() {}") {
		t.Error("the reviewable file is missing from the cumulative block")
	}
	// The lockfile is gone from the block, so the block is not the whole PR.
	if strings.Contains(prompt, "**entire PR**") {
		t.Error("a block with a file filtered out must not claim to be the entire PR")
	}
}

// When no file fits, the model still learns which files the PR changes. The
// largest PRs are where nothing fits, and they used to get no context at all.
func TestCumulativeDiffNamesFilesWhenNothingFits(t *testing.T) {
	prompt, logs := runIncremental(t, addedFileDiff("big.go", strings.Repeat("y", floorBudget*5)))

	if strings.Contains(prompt, "<cumulative_pr_diff>") || strings.Contains(prompt, "yyyy") {
		t.Error("an over-budget file reached the prompt")
	}
	if !strings.Contains(prompt, "diff not shown") || !strings.Contains(prompt, "\n- big.go") {
		t.Errorf("the PR's files are not named to the model:\n%s", prompt)
	}
	if !strings.Contains(logs, "kept 0 of 1 file(s)") {
		t.Errorf("the selection line does not report it:\n%s", logs)
	}
}

// A diff part whose header does not parse has no path. Keyed on "", one in
// the push would rank every unparsed part of the PR as touched by it.
func TestUnparsedPartsAreNotRankedAsTouched(t *testing.T) {
	h := incrementalHarness(t, func(h *harness) {
		h.bb.commitDiff = "preamble with no header\n" + sampleDiff
		h.bb.prDiff = "another unparsed part\n" + sampleDiff
	})
	var buf bytes.Buffer
	h.r.log = slog.New(slog.NewTextHandler(&buf, nil))
	h.r.ReviewPullRequest(context.Background(), prPayload(webhook.EventFromRefUpdated), false)

	if want := "(1 touched by the push, 0 related, 1 other)"; !strings.Contains(buf.String(), want) {
		t.Errorf("log lacks %q:\n%s", want, buf.String())
	}
}

// The push defines new(), so a file calling it outranks one that does not,
// and the file the push touched outranks both. Path order alone would put
// b.go before caller.go.
func TestCumulativeDiffRanksTouchedThenRelatedThenRest(t *testing.T) {
	prDiff := addedFileDiff("b.go", "var unrelated = 1") +
		addedFileDiff("caller.go", "var x = new()") +
		sampleDiff
	prompt, logs := runIncremental(t, prDiff)
	block := cumulativeBlock(t, prompt)

	touched := strings.Index(block, "+func new() {}")
	related := strings.Index(block, "var x = new()")
	other := strings.Index(block, "var unrelated = 1")
	if touched < 0 || related < 0 || other < 0 {
		t.Fatalf("a file is missing from the cumulative block:\n%s", block)
	}
	if !(touched < related && related < other) {
		t.Errorf("order is touched=%d related=%d other=%d, want ascending", touched, related, other)
	}
	if !strings.Contains(logs, "(1 touched by the push, 1 related, 1 other)") {
		t.Errorf("the selection line does not report the tiers:\n%s", logs)
	}
	if !strings.Contains(prompt, "**entire PR**") {
		t.Error("with nothing omitted the block is the whole PR and must say so")
	}
}

// A file that does not fit is left out whole and named; the files after it
// that do fit are still kept. Dropping the block instead left the largest
// PRs with no context at all.
func TestCumulativeDiffOmitsWhatDoesNotFitAndKeepsTheRest(t *testing.T) {
	// 5,000 fake tokens: over the 4,000 budget, under the byte pre-check.
	big := addedFileDiff("big.go", strings.Repeat("y", floorBudget*5))
	prompt, logs := runIncremental(t, big+addedFileDiff("z.go", "var small = 1")+sampleDiff)
	block := cumulativeBlock(t, prompt)

	if strings.Contains(block, "yyyy") {
		t.Error("the over-budget file reached the cumulative block")
	}
	if !strings.Contains(block, "var small = 1") {
		t.Error("a file that fits was dropped because an earlier one did not")
	}
	if !strings.Contains(prompt, "**part of the PR**") || strings.Contains(prompt, "**entire PR**") {
		t.Error("a partial block must not claim to be the entire PR")
	}
	if !strings.Contains(block, "diff not shown") || !strings.Contains(block, "\n- big.go") {
		t.Errorf("the omitted file is not named to the model:\n%s", block)
	}
	for _, want := range []string{"budget 4000 (floor)", "kept 2 of 3 file(s)", "1 omitted for budget (largest: big.go "} {
		if !strings.Contains(logs, want) {
			t.Errorf("log lacks %q:\n%s", want, logs)
		}
	}
}

// A push large enough lifts the budget off the floor, up to the ceiling.
func TestCumulativeBudgetFollowsThePush(t *testing.T) {
	h := incrementalHarness(t, func(h *harness) {
		// 16,000 bytes is ~4,000 tokens of push: 20x reaches the ceiling.
		h.bb.commitDiff = addedFileDiff("a.go", strings.Repeat("p", 16_000))
	})
	var buf bytes.Buffer
	h.r.log = slog.New(slog.NewTextHandler(&buf, nil))
	h.r.ReviewPullRequest(context.Background(), prPayload(webhook.EventFromRefUpdated), false)

	want := fmt.Sprintf("budget %d (ceiling)", inference.CumulativeDiffBudget(h.llm.budget))
	if !strings.Contains(buf.String(), want) {
		t.Errorf("log lacks %q:\n%s", want, buf.String())
	}
}

// What a run sends and why it was a full or an incremental one must be
// answerable from the log: the cost follows the prompt, not the diff.
func TestPromptCompositionAndReviewModeAreLogged(t *testing.T) {
	t.Run("incremental", func(t *testing.T) {
		_, logs := runIncremental(t, sampleDiff)
		for _, want := range []string{"(incremental review) - files ", "cumulative PR diff ", "AGENTS.md ", "template and rest "} {
			if !strings.Contains(logs, want) {
				t.Errorf("log lacks %q:\n%s", want, logs)
			}
		}
		if strings.Contains(logs, "cumulative PR diff 0,") {
			t.Errorf("the cumulative block was sent but counted as 0:\n%s", logs)
		}
	})

	t.Run("a push with no prior reviewed commit says why it is full", func(t *testing.T) {
		h := newHarness(t, nil)
		var buf bytes.Buffer
		h.r.log = slog.New(slog.NewTextHandler(&buf, nil))
		h.r.ReviewPullRequest(context.Background(), prPayload(webhook.EventFromRefUpdated), false)
		for _, want := range []string{"full review (no prior reviewed commit)", "(full review) - files ", "cumulative PR diff 0,"} {
			if !strings.Contains(buf.String(), want) {
				t.Errorf("log lacks %q:\n%s", want, buf.String())
			}
		}
	})

	t.Run("an opened PR names its event", func(t *testing.T) {
		h := newHarness(t, nil)
		var buf bytes.Buffer
		h.r.log = slog.New(slog.NewTextHandler(&buf, nil))
		h.r.ReviewPullRequest(context.Background(), prPayload(webhook.EventOpened), false)
		if want := "full review (event pr:opened is not a push)"; !strings.Contains(buf.String(), want) {
			t.Errorf("log lacks %q:\n%s", want, buf.String())
		}
	})
}
