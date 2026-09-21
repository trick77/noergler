package review

import (
	"context"
	"encoding/json"
	"os"
	"strings"
	"testing"
	"time"

	"github.com/trick77/noergler/internal/config"
	"github.com/trick77/noergler/internal/inference"
	"github.com/trick77/noergler/internal/store"
	"github.com/trick77/noergler/internal/webhook"
)

const sampleDiff = `diff --git a/a.go b/a.go
index 1111111..2222222 100644
--- a/a.go
+++ b/a.go
@@ -1,3 +1,3 @@
 package a
-func old() {}
+func new() {}
`

type harness struct {
	r   *Reviewer
	bb  *fakeBitbucket
	st  *fakeStore
	llm *fakeLLM
	jr  *fakeJira
	rt  *fakeRiptide
	tok *fakeTokens
}

// newHarness builds a Reviewer whose every dependency is a fake, with the
// defaults a plain auto review needs: AGENTS.md present, a diff, a model that
// answers OK.
func newHarness(t *testing.T, tweak func(*harness)) *harness {
	t.Helper()

	h := &harness{
		bb:  newFakeBitbucket(),
		st:  newFakeStore(),
		llm: newFakeLLM(),
		rt:  &fakeRiptide{enabled: true},
		tok: &fakeTokens{},
	}
	h.bb.prDiff = sampleDiff
	h.bb.files["src1:AGENTS.md"] = "# Rules\nBe terse.\n"
	h.bb.files["src1:a.go"] = "package a\nfunc new() {}\n"

	cfg := config.Review{
		MaxComments:           25,
		MaxFileLines:          1000,
		RequireAgentsMD:       true,
		AgentsMDMaxTokens:     7000,
		AgentsMDWarnTokens:    4000,
		MaxPRCostUSD:          5.0,
		TicketComplianceCheck: true,
	}

	if tweak != nil {
		tweak(h)
	}

	var jira JiraClient
	if h.jr != nil {
		jira = h.jr
	}
	var rt RiptideEmitter
	if h.rt != nil {
		rt = h.rt
	}

	h.r = New(Options{
		TeamSlug:        "payments",
		Bitbucket:       h.bb,
		LLM:             h.llm,
		Store:           h.st,
		Jira:            jira,
		Riptide:         rt,
		Tokens:          h.tok,
		Config:          cfg,
		Template:        "REVIEW {repo_instructions} {files} {cumulative_pr_diff} {previously_posted_findings} {ticket_context} {compliance_instructions}",
		MentionTemplate: "ASK {question} {repo_instructions} {ticket_context} {diff}",
		Log:             quietLogger(),
	})
	return h
}

func prPayload(event string) *webhook.Payload {
	p := &webhook.Payload{EventKey: event}
	p.PullRequest.ID = 42
	p.PullRequest.Title = "Add new feature"
	p.PullRequest.State = "OPEN"
	p.PullRequest.Author.User.Name = "alice"
	p.PullRequest.FromRef = webhook.Ref{
		ID: "refs/heads/feature", DisplayID: "feature/thing", LatestCommit: "src1",
		Repository: &webhook.Repository{Slug: "my-repo", Project: webhook.Project{Key: "PROJ"}},
	}
	p.PullRequest.ToRef = webhook.Ref{
		ID: "refs/heads/master", DisplayID: "master", LatestCommit: "tgt1",
		Repository: &webhook.Repository{Slug: "my-repo", Project: webhook.Project{Key: "PROJ"}},
	}
	return p
}

func ptrs(s string) *string { return &s }
func ptri(i int64) *int64   { return &i }

func okResultWith(findings ...inference.ReviewFinding) inference.ReviewResult {
	return inference.ReviewResult{
		Outcome: inference.OutcomeOK,
		Review: inference.ParsedReview{
			Findings:               findings,
			Summary:                inference.NewReviewSummary(),
			ComplianceRequirements: []inference.ComplianceRequirement{},
		},
		Cost: inference.CallCost{NanoUSD: ptri(120_000_000), PromptTokens: 1000, CompletionTokens: 200},
	}
}

func finding(file string, line int, severity, comment string) inference.ReviewFinding {
	return inference.ReviewFinding{File: file, Line: line, Severity: severity, Comment: comment}
}

// --- Guard order -----------------------------------------------------------

func TestReviewRunsForAllowedAuthor(t *testing.T) {
	h := newHarness(t, nil)
	h.llm.review = okResultWith(finding("a.go", 2, "issue", "bad"))

	h.r.ReviewPullRequest(context.Background(), prPayload(webhook.EventOpened), false)

	if len(h.llm.Reviews) != 1 {
		t.Fatalf("expected one LLM call, got %d", len(h.llm.Reviews))
	}
	if len(h.bb.Inline) != 1 {
		t.Errorf("expected one inline comment, got %d", len(h.bb.Inline))
	}
	if len(h.bb.Posted) != 1 {
		t.Errorf("expected one summary comment, got %d", len(h.bb.Posted))
	}
	if len(h.st.Runs) != 1 {
		t.Errorf("expected one run row, got %d", len(h.st.Runs))
	}
	if len(h.st.Findings) != 1 {
		t.Errorf("expected one finding row, got %d", len(h.st.Findings))
	}
}

// The model string a reader sees carries the reasoning effort, and the same
// string is stored on the run row. Both come from the one model label built
// from the profile id and the effort.
//
// This drives the pipeline rather than the renderer: render's summary golden
// takes ModelLabel as an input, so it pins the formatting but says nothing
// about what the reviewer passes in. Phase 8 found the reviewer passing the
// bare profile id, which that golden could not catch.
func TestSummaryAndRunRowCarryTheEffortInTheModelLabel(t *testing.T) {
	h := newHarness(t, nil)
	h.llm.review = okResultWith(finding("a.go", 2, "issue", "bad"))

	h.r.ReviewPullRequest(context.Background(), prPayload(webhook.EventOpened), false)

	// The fake is built with model gpt-5.5 at effort high.
	const want = "gpt-5.5-high"

	if len(h.st.Runs) != 1 {
		t.Fatalf("expected one run row, got %d", len(h.st.Runs))
	}
	if got := h.st.Runs[0].ModelLabel; got != want {
		t.Errorf("run row model label = %q, want %q", got, want)
	}

	if len(h.bb.Posted) != 1 {
		t.Fatalf("expected one summary comment, got %d", len(h.bb.Posted))
	}
	// The summary renders it as "Model: `<label>`".
	if body := h.bb.Posted[0].Text; !strings.Contains(body, "`"+want+"`") {
		t.Errorf("summary does not show model %q; footnote was:\n%s", want, body)
	}
}

// Guard 1: a payload with no repository on either ref names no PR, so there
// is nothing to review and nothing to write.
func TestReviewRequiresAProjectAndRepo(t *testing.T) {
	h := newHarness(t, nil)
	p := prPayload(webhook.EventOpened)
	p.PullRequest.FromRef.Repository = nil
	p.PullRequest.ToRef.Repository = nil

	h.r.ReviewPullRequest(context.Background(), p, false)

	if len(h.llm.Reviews) != 0 || len(h.st.Upserts) != 0 || len(h.bb.Posted) != 0 {
		t.Error("a payload without a repository must be dropped before anything happens")
	}
}

func TestSkipsDisallowedAuthorAndIgnoredActor(t *testing.T) {
	t.Run("author not in allow list", func(t *testing.T) {
		h := newHarness(t, nil)
		h.r.SetAuthorLists([]string{"bob"}, nil)
		h.r.ReviewPullRequest(context.Background(), prPayload(webhook.EventOpened), false)
		if len(h.llm.Reviews) != 0 {
			t.Error("review ran for a disallowed author")
		}
	})

	t.Run("push by an ignored actor", func(t *testing.T) {
		h := newHarness(t, nil)
		h.r.SetAuthorLists(nil, []string{"ci-bot"})
		p := prPayload(webhook.EventFromRefUpdated)
		p.Actor = &webhook.User{Name: "ci-bot"}
		h.r.ReviewPullRequest(context.Background(), p, false)
		if len(h.llm.Reviews) != 0 {
			t.Error("review ran on a push by an ignored actor")
		}
	})

	t.Run("a mention bypasses both gates", func(t *testing.T) {
		h := newHarness(t, nil)
		h.r.SetAuthorLists([]string{"bob"}, []string{"alice"})
		h.r.ReviewPullRequest(context.Background(), prPayload(webhook.EventOpened), true)
		if len(h.llm.Reviews) != 1 {
			t.Error("a mention must bypass the author gates")
		}
	})
}

func TestIgnoredPRIsSkippedWithoutAnyAPICall(t *testing.T) {
	h := newHarness(t, nil)
	now := time.Now()
	h.st.skipState = &store.SkipState{IgnoredAt: &now}

	h.r.ReviewPullRequest(context.Background(), prPayload(webhook.EventOpened), false)

	if len(h.llm.Reviews) != 0 || len(h.bb.Posted) != 0 {
		t.Error("an ignored PR must not be reviewed or commented on")
	}
}

// A 404 on the tracked summary means the user deleted it. Anything else must
// NOT silence a live PR.
func TestDeletedSummaryIgnoresButTransientErrorProceeds(t *testing.T) {
	t.Run("404 marks the PR ignored", func(t *testing.T) {
		h := newHarness(t, nil)
		h.st.skipState = &store.SkipState{Summary: &store.SummaryComment{ID: 999, Version: 1}}
		// 999 is not in the fake's comment map, so the fetch 404s.
		h.r.ReviewPullRequest(context.Background(), prPayload(webhook.EventOpened), false)

		if h.st.Ignored != 1 {
			t.Errorf("MarkIgnored called %d times, want 1", h.st.Ignored)
		}
		if len(h.llm.Reviews) != 0 {
			t.Error("review ran after the summary was found deleted")
		}
	})

	t.Run("a present summary proceeds", func(t *testing.T) {
		h := newHarness(t, nil)
		h.bb.setComment(555, "### Overview\nold", 3)
		h.st.skipState = &store.SkipState{Summary: &store.SummaryComment{ID: 555, Version: 3}}

		h.r.ReviewPullRequest(context.Background(), prPayload(webhook.EventOpened), false)

		if h.st.Ignored != 0 {
			t.Error("a live summary must not mark the PR ignored")
		}
		if len(h.llm.Reviews) != 1 {
			t.Error("review should have run")
		}
	})
}

func TestOptOutBranchKeyword(t *testing.T) {
	t.Run("matching branch skips and posts the summary", func(t *testing.T) {
		h := newHarness(t, nil)
		h.r.cfg.OptOutBranchKeyword = "noergloff"
		p := prPayload(webhook.EventOpened)
		p.PullRequest.FromRef.DisplayID = "feature/NOERGLOFF-thing" // case-insensitive

		h.r.ReviewPullRequest(context.Background(), p, false)

		if len(h.llm.Reviews) != 0 {
			t.Error("review ran on an opt-out branch")
		}
		if len(h.bb.Posted) != 1 || !strings.Contains(h.bb.Posted[0].Text, "opt-out keyword") {
			t.Errorf("expected the opt-out summary, got %+v", h.bb.Posted)
		}
		// The pointer advances: an opt-out branch is a stable state.
		if len(h.st.Upserts) != 1 || h.st.Upserts[0].LastReviewedCommit == nil ||
			*h.st.Upserts[0].LastReviewedCommit != "src1" {
			t.Errorf("opt-out must advance the pointer, got %+v", h.st.Upserts)
		}
	})

	t.Run("empty keyword disables the feature", func(t *testing.T) {
		h := newHarness(t, nil)
		h.r.cfg.OptOutBranchKeyword = ""
		h.r.ReviewPullRequest(context.Background(), prPayload(webhook.EventOpened), false)
		if len(h.llm.Reviews) != 1 {
			t.Error("an empty keyword must not skip anything")
		}
	})

	t.Run("non-matching branch is unaffected", func(t *testing.T) {
		h := newHarness(t, nil)
		h.r.cfg.OptOutBranchKeyword = "noergloff"
		h.r.ReviewPullRequest(context.Background(), prPayload(webhook.EventOpened), false)
		if len(h.llm.Reviews) != 1 {
			t.Error("a non-matching branch must be reviewed")
		}
	})
}

func TestAgentsMDGates(t *testing.T) {
	t.Run("missing and required skips", func(t *testing.T) {
		h := newHarness(t, func(h *harness) { delete(h.bb.files, "src1:AGENTS.md") })
		h.r.ReviewPullRequest(context.Background(), prPayload(webhook.EventOpened), false)

		if len(h.llm.Reviews) != 0 {
			t.Error("review ran without a required AGENTS.md")
		}
		if len(h.bb.Posted) != 1 || !strings.Contains(h.bb.Posted[0].Text, "no `AGENTS.md` found") {
			t.Errorf("expected the AGENTS.md summary, got %+v", h.bb.Posted)
		}
	})

	t.Run("missing and not required proceeds", func(t *testing.T) {
		h := newHarness(t, func(h *harness) { delete(h.bb.files, "src1:AGENTS.md") })
		h.r.cfg.RequireAgentsMD = false
		h.r.ReviewPullRequest(context.Background(), prPayload(webhook.EventOpened), false)
		if len(h.llm.Reviews) != 1 {
			t.Error("review should run when AGENTS.md is not required")
		}
	})

	t.Run("over the token limit skips", func(t *testing.T) {
		h := newHarness(t, nil)
		h.r.cfg.AgentsMDMaxTokens = 2 // the fake counts one token per 4 bytes
		h.r.ReviewPullRequest(context.Background(), prPayload(webhook.EventOpened), false)

		if len(h.llm.Reviews) != 0 {
			t.Error("review ran with an oversized AGENTS.md")
		}
		if len(h.bb.Posted) != 1 || !strings.Contains(h.bb.Posted[0].Text, "too large") {
			t.Errorf("expected the too-large summary, got %+v", h.bb.Posted)
		}
	})

	t.Run("zero disables the hard limit", func(t *testing.T) {
		h := newHarness(t, nil)
		h.r.cfg.AgentsMDMaxTokens = 0
		h.r.ReviewPullRequest(context.Background(), prPayload(webhook.EventOpened), false)
		if len(h.llm.Reviews) != 1 {
			t.Error("a zero limit must disable the gate")
		}
	})
}

// A skip path still posts its summary when the upsert fails, because every
// store call is wrapped and a nil pr id only means the comment is not tracked.
func TestSkipPathPostsSummaryWhenUpsertFails(t *testing.T) {
	h := newHarness(t, func(h *harness) { delete(h.bb.files, "src1:AGENTS.md") })
	h.st.upsertErr = errUpsert

	h.r.ReviewPullRequest(context.Background(), prPayload(webhook.EventOpened), false)

	if len(h.bb.Posted) != 1 {
		t.Errorf("summary must still be posted when the upsert fails, got %d comments", len(h.bb.Posted))
	}
}

// --- Cost cap --------------------------------------------------------------

func TestCostCap(t *testing.T) {
	t.Run("over the limit blocks the auto review and posts the banner", func(t *testing.T) {
		h := newHarness(t, nil)
		h.st.prCost = ptri(6_000_000_000) // $6 against a $5 cap

		h.r.ReviewPullRequest(context.Background(), prPayload(webhook.EventOpened), false)

		if len(h.llm.Reviews) != 0 {
			t.Error("review ran while over the cost cap")
		}
		if len(h.bb.Posted) != 1 || !strings.Contains(h.bb.Posted[0].Text, "Cost limit exceeded") {
			t.Errorf("expected the cost banner, got %+v", h.bb.Posted)
		}
		if !strings.Contains(h.bb.Posted[0].Text, "not** reviewed automatically") {
			t.Error("a blocked push must say so")
		}
	})

	t.Run("a mention bypasses the cap", func(t *testing.T) {
		h := newHarness(t, nil)
		h.st.prCost = ptri(6_000_000_000)
		h.llm.review = okResultWith()

		h.r.ReviewPullRequest(context.Background(), prPayload(webhook.EventCommentAdded), true)

		if len(h.llm.Reviews) != 1 {
			t.Error("a mention must run even over the cap")
		}
	})

	t.Run("a nil cost never blocks", func(t *testing.T) {
		h := newHarness(t, nil)
		h.st.prCost = nil
		h.r.ReviewPullRequest(context.Background(), prPayload(webhook.EventOpened), false)
		if len(h.llm.Reviews) != 1 {
			t.Error("an unpriced PR must not be blocked: cost fails open")
		}
	})

	t.Run("a completed run at the cap carries the banner", func(t *testing.T) {
		h := newHarness(t, nil)
		// $4.95 before the run and $0.12 for it: the auto gate lets it
		// through, and the finished run lands over the $5 cap.
		h.st.prCost = ptri(4_950_000_000)
		h.llm.review = okResultWith()

		h.r.ReviewPullRequest(context.Background(), prPayload(webhook.EventOpened), false)

		if len(h.bb.Posted) != 1 {
			t.Fatalf("expected one summary, got %d", len(h.bb.Posted))
		}
		if !strings.Contains(h.bb.Posted[0].Text, "Cost limit exceeded") {
			t.Error("a run that ends at or over the cap must carry the banner")
		}
		if strings.Contains(h.bb.Posted[0].Text, "not** reviewed automatically") {
			t.Error("a completed run must not claim it was skipped")
		}
	})

	// The blocked path preserves the prior commit so raising the limit later
	// re-reviews the accumulated range.
	t.Run("blocking preserves the prior commit", func(t *testing.T) {
		h := newHarness(t, nil)
		h.st.prCost = ptri(6_000_000_000)
		h.st.lastCommit, h.st.hasLast = "older", true

		h.r.ReviewPullRequest(context.Background(), prPayload(webhook.EventOpened), false)

		if len(h.st.Upserts) != 1 {
			t.Fatalf("expected one upsert, got %d", len(h.st.Upserts))
		}
		if got := h.st.Upserts[0].LastReviewedCommit; got == nil || *got != "older" {
			t.Errorf("pointer = %v, want the prior commit", got)
		}
	})
}

// --- Terminal outcomes -----------------------------------------------------

// Every non-ok outcome but OutcomeError posts a notice, preserves the prior
// commit and writes no run row. OutcomeError writes and posts nothing at
// all; it is only logged.
func TestTerminalOutcomes(t *testing.T) {
	cases := []struct {
		name        string
		outcome     inference.Outcome
		wantNotice  string
		wantPosting bool
		wantUpsert  bool
	}{
		{"timed out", inference.OutcomeTimedOut, "no response from the model", true, true},
		{"unparseable", inference.OutcomeUnparseable, "could not be processed", true, true},
		{"too large", inference.OutcomeTooLarge, "too large to review", true, true},
		{"error", inference.OutcomeError, "", false, false},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			h := newHarness(t, nil)
			h.st.lastCommit, h.st.hasLast = "older", true
			h.llm.review = inference.ReviewResult{
				Outcome: c.outcome,
				Review:  inference.ParsedReview{Summary: inference.NewReviewSummary()},
			}

			h.r.ReviewPullRequest(context.Background(), prPayload(webhook.EventOpened), false)

			if len(h.st.Runs) != 0 {
				t.Errorf("a non-ok outcome must write no run row, got %d", len(h.st.Runs))
			}
			if len(h.bb.Inline) != 0 {
				t.Errorf("a non-ok outcome must post no inline comments, got %d", len(h.bb.Inline))
			}

			if !c.wantPosting {
				if len(h.bb.Posted) != 0 || len(h.bb.Updates) != 0 {
					t.Errorf("OutcomeError must post nothing, got %d posts and %d updates",
						len(h.bb.Posted), len(h.bb.Updates))
				}
				if len(h.st.Upserts) != 0 {
					t.Errorf("OutcomeError must not upsert, got %d", len(h.st.Upserts))
				}
				return
			}

			if len(h.bb.Posted) != 1 {
				t.Fatalf("expected one notice, got %d", len(h.bb.Posted))
			}
			if !strings.Contains(h.bb.Posted[0].Text, c.wantNotice) {
				t.Errorf("notice = %q, want it to mention %q", h.bb.Posted[0].Text, c.wantNotice)
			}
			if len(h.st.Upserts) != 1 {
				t.Fatalf("expected one upsert, got %d", len(h.st.Upserts))
			}
			if got := h.st.Upserts[0].LastReviewedCommit; got == nil || *got != "older" {
				t.Errorf("pointer = %v, want the prior commit preserved", got)
			}
		})
	}
}

// With a prior summary tracked, a failure prepends a banner instead of
// posting a second comment, and repeated failures must not stack banners.
func TestFailureBannerDoesNotStack(t *testing.T) {
	h := newHarness(t, nil)
	h.bb.setComment(555, "### Overview\nThe original body.", 3)
	h.st.summary = &store.SummaryComment{ID: 555, Version: 3}
	h.llm.review = inference.ReviewResult{
		Outcome: inference.OutcomeTimedOut,
		Review:  inference.ParsedReview{Summary: inference.NewReviewSummary()},
	}

	h.r.ReviewPullRequest(context.Background(), prPayload(webhook.EventOpened), false)
	h.r.ReviewPullRequest(context.Background(), prPayload(webhook.EventOpened), false)

	if len(h.bb.Posted) != 0 {
		t.Errorf("a tracked summary must be updated, not re-posted: %d posts", len(h.bb.Posted))
	}
	if len(h.bb.Updates) != 2 {
		t.Fatalf("expected two updates, got %d", len(h.bb.Updates))
	}
	body := h.bb.Updates[1].Text
	if n := strings.Count(body, "No response from the model within"); n != 1 {
		t.Errorf("banner appears %d times, want 1 (they must not stack)", n)
	}
	if !strings.Contains(body, "The original body.") {
		t.Error("the original body must be preserved under the banner")
	}
}

// --- Dedup, sort, limit ----------------------------------------------------

func TestDedupeAgainstExistingFindings(t *testing.T) {
	h := newHarness(t, nil)
	h.st.existing = []store.Finding{
		{FilePath: "a.go", LineNumber: 2, Severity: "issue", CommentText: "already raised"},
	}
	h.llm.review = okResultWith(
		finding("a.go", 2, "issue", "already raised"), // same key: dropped
		finding("a.go", 3, "issue", "new one"),        // different line: kept
		finding("a.go", 2, "suggestion", "other"),     // different severity: kept
	)

	h.r.ReviewPullRequest(context.Background(), prPayload(webhook.EventOpened), false)

	if len(h.bb.Inline) != 2 {
		t.Fatalf("expected 2 inline comments after dedup, got %d", len(h.bb.Inline))
	}
	for _, c := range h.bb.Inline {
		if strings.Contains(c.Text, "already raised") {
			t.Error("a finding already posted must not be posted again")
		}
	}
}

func TestSortAndLimit(t *testing.T) {
	t.Run("issues sort before suggestions, stably", func(t *testing.T) {
		in := []inference.ReviewFinding{
			{File: "s1.go", Severity: "suggestion", Comment: "s1"},
			{File: "i1.go", Severity: "issue", Comment: "i1"},
			{File: "s2.go", Severity: "suggestion", Comment: "s2"},
			{File: "i2.go", Severity: "issue", Comment: "i2"},
		}
		got, truncated := sortAndLimit(in, 25)
		if truncated {
			t.Error("must not truncate under the limit")
		}
		want := []string{"i1.go", "i2.go", "s1.go", "s2.go"}
		for i, w := range want {
			if got[i].File != w {
				t.Errorf("position %d = %s, want %s (the sort is stable)", i, got[i].File, w)
			}
		}
	})

	t.Run("over the limit truncates and reports it", func(t *testing.T) {
		in := make([]inference.ReviewFinding, 5)
		for i := range in {
			in[i] = inference.ReviewFinding{Severity: "issue"}
		}
		got, truncated := sortAndLimit(in, 3)
		if !truncated || len(got) != 3 {
			t.Errorf("got %d findings, truncated=%v; want 3, true", len(got), truncated)
		}
	})

	t.Run("an unknown severity sorts last", func(t *testing.T) {
		in := []inference.ReviewFinding{
			{File: "x.go", Severity: "nonsense"},
			{File: "i.go", Severity: "issue"},
		}
		got, _ := sortAndLimit(in, 25)
		if got[0].File != "i.go" {
			t.Errorf("first = %s, want i.go", got[0].File)
		}
	})
}

func TestFindingsLimitedInReview(t *testing.T) {
	h := newHarness(t, nil)
	h.r.cfg.MaxComments = 2
	h.llm.review = okResultWith(
		finding("a.go", 1, "issue", "one"),
		finding("a.go", 2, "issue", "two"),
		finding("a.go", 3, "issue", "three"),
	)

	h.r.ReviewPullRequest(context.Background(), prPayload(webhook.EventOpened), false)

	if len(h.bb.Inline) != 2 {
		t.Errorf("posted %d comments, want the max_comments cap of 2", len(h.bb.Inline))
	}
	if len(h.bb.Posted) != 1 || !strings.Contains(h.bb.Posted[0].Text, "Showing top 2 findings") {
		t.Error("the summary must report the truncation")
	}
}

// An inline comment that fails to post is counted but never stored: a finding
// row exists only for a comment that is actually on the PR.
func TestFailedInlineCommentIsNotStored(t *testing.T) {
	h := newHarness(t, nil)
	h.bb.inlineErr = errPost
	h.llm.review = okResultWith(finding("a.go", 2, "issue", "bad"))

	h.r.ReviewPullRequest(context.Background(), prPayload(webhook.EventOpened), false)

	if len(h.st.Findings) != 0 {
		t.Errorf("stored %d findings for comments that never posted", len(h.st.Findings))
	}
	if len(h.st.Runs) != 1 {
		t.Fatalf("expected a run row, got %d", len(h.st.Runs))
	}
	if h.st.Runs[0].FindingsPosted != 0 {
		t.Errorf("FindingsPosted = %d, want 0", h.st.Runs[0].FindingsPosted)
	}
}

// --- Incremental review ----------------------------------------------------

func TestIncrementalReview(t *testing.T) {
	t.Run("an unchanged HEAD skips", func(t *testing.T) {
		h := newHarness(t, nil)
		h.st.lastCommit, h.st.hasLast = "src1", true

		h.r.ReviewPullRequest(context.Background(), prPayload(webhook.EventFromRefUpdated), false)

		if len(h.llm.Reviews) != 0 {
			t.Error("an unchanged HEAD must not be re-reviewed")
		}
	})

	t.Run("a moved HEAD reviews the incremental diff", func(t *testing.T) {
		h := newHarness(t, nil)
		h.st.lastCommit, h.st.hasLast = "older", true
		h.bb.commitDiff = sampleDiff

		h.r.ReviewPullRequest(context.Background(), prPayload(webhook.EventFromRefUpdated), false)

		if len(h.llm.Reviews) != 1 {
			t.Fatal("expected one review")
		}
		if len(h.st.Runs) != 1 || !h.st.Runs[0].Incremental {
			t.Error("the run row must be marked incremental")
		}
		if h.st.Runs[0].FromCommit == nil || *h.st.Runs[0].FromCommit != "older" {
			t.Error("the run row must record the incremental base")
		}
		if len(h.bb.Posted) != 1 || !strings.Contains(h.bb.Posted[0].Text, "incremental update") {
			t.Error("the summary must carry the incremental header")
		}
	})

	t.Run("an unavailable incremental diff falls back to the full review", func(t *testing.T) {
		h := newHarness(t, nil)
		h.st.lastCommit, h.st.hasLast = "older", true
		h.bb.commitErr = errIncrementalUnavailable

		h.r.ReviewPullRequest(context.Background(), prPayload(webhook.EventFromRefUpdated), false)

		if len(h.llm.Reviews) != 1 {
			t.Fatal("a rebase must fall back to a full review, not skip it")
		}
		if h.st.Runs[0].Incremental {
			t.Error("the fallback run is not incremental")
		}
	})

	t.Run("an empty incremental diff falls back rather than skipping", func(t *testing.T) {
		h := newHarness(t, nil)
		h.st.lastCommit, h.st.hasLast = "older", true
		h.bb.commitDiff = "   "

		h.r.ReviewPullRequest(context.Background(), prPayload(webhook.EventFromRefUpdated), false)

		if len(h.llm.Reviews) != 1 {
			t.Error("an empty compare must fall back to the full review, never silently skip")
		}
	})
}

// The cumulative diff is cross-file context only, so losing it must degrade
// the incremental review rather than fail it.
func TestCumulativeDiffFailureDoesNotBlockTheIncrementalReview(t *testing.T) {
	h := newHarness(t, nil)
	h.st.lastCommit, h.st.hasLast = "older", true
	h.bb.commitDiff = sampleDiff
	h.bb.prDiffErr = errDiffTooLarge // the cumulative fetch fails

	h.r.ReviewPullRequest(context.Background(), prPayload(webhook.EventFromRefUpdated), false)

	if len(h.llm.Reviews) != 1 {
		t.Fatal("the incremental review must still run without the cumulative diff")
	}
	if len(h.st.Runs) != 1 || !h.st.Runs[0].Incremental {
		t.Error("the run must still be recorded as incremental")
	}
}

func TestEmptyDiffSkips(t *testing.T) {
	h := newHarness(t, func(h *harness) { h.bb.prDiff = "   " })
	h.r.ReviewPullRequest(context.Background(), prPayload(webhook.EventOpened), false)
	if len(h.llm.Reviews) != 0 {
		t.Error("an empty diff must not reach the model")
	}
}

func TestDiffTooLargePostsSummaryAndPreservesPointer(t *testing.T) {
	h := newHarness(t, nil)
	h.st.lastCommit, h.st.hasLast = "older", true
	h.bb.prDiffErr = errDiffTooLarge

	h.r.ReviewPullRequest(context.Background(), prPayload(webhook.EventOpened), false)

	if len(h.llm.Reviews) != 0 {
		t.Error("an oversized diff must not reach the model")
	}
	if len(h.bb.Posted) != 1 || !strings.Contains(h.bb.Posted[0].Text, "diff too large") {
		t.Errorf("expected the too-large summary, got %+v", h.bb.Posted)
	}
	if got := h.st.Upserts[0].LastReviewedCommit; got == nil || *got != "older" {
		t.Errorf("pointer = %v, want the prior commit preserved", got)
	}
}

// --- Cost reporting --------------------------------------------------------

// The PR total is only read when this run is priced. An unpriced run shows
// no total at all.
func TestUnpricedRunShowsNoCostLine(t *testing.T) {
	h := newHarness(t, nil)
	h.st.prCost = ptri(3_000_000_000)
	h.llm.review = inference.ReviewResult{
		Outcome: inference.OutcomeOK,
		Review:  inference.ParsedReview{Summary: inference.NewReviewSummary()},
		Cost:    inference.CallCost{PromptTokens: 100, CompletionTokens: 10}, // NanoUSD nil
	}

	h.r.ReviewPullRequest(context.Background(), prPayload(webhook.EventOpened), false)

	if len(h.bb.Posted) != 1 {
		t.Fatalf("expected one summary, got %d", len(h.bb.Posted))
	}
	if strings.Contains(h.bb.Posted[0].Text, "PR total") {
		t.Error("an unpriced run must not print a PR total")
	}
	if len(h.st.Runs) != 1 || h.st.Runs[0].CostNanoUSD != nil {
		t.Error("an unpriced run must be stored with a NULL cost")
	}
}

// --- The DB falling over ---------------------------------------------------

// Every store call goes through safeDB, so a dead database degrades the
// review rather than failing it.
func TestReviewSurvivesADeadDatabase(t *testing.T) {
	h := newHarness(t, nil)
	h.st.failAll = true
	h.llm.review = okResultWith(finding("a.go", 2, "issue", "bad"))

	h.r.ReviewPullRequest(context.Background(), prPayload(webhook.EventOpened), false)

	if len(h.llm.Reviews) != 1 {
		t.Error("the review must still run with the DB down")
	}
	if len(h.bb.Inline) != 1 {
		t.Error("inline comments must still be posted with the DB down")
	}
	if len(h.bb.Posted) != 1 {
		t.Error("the summary must still be posted with the DB down")
	}
}

// --- extractQuestion -------------------------------------------------------

func TestExtractQuestionIsPinned(t *testing.T) {
	blob, err := os.ReadFile("testdata/question_golden.json")
	if err != nil {
		t.Fatalf("read question golden: %v", err)
	}
	var cases map[string]string
	if err := json.Unmarshal(blob, &cases); err != nil {
		t.Fatalf("decode question golden: %v", err)
	}
	for in, want := range cases {
		if got := extractQuestion(in, "noergler"); got != want {
			t.Errorf("extractQuestion(%q) = %q, want %q", in, got, want)
		}
	}
}

// The boundary is Unicode-aware, so a trigger followed by a letter is not a
// mention.
func TestExtractQuestionUnicodeBoundary(t *testing.T) {
	if got := extractQuestion("@noerglerü frage", "noergler"); got != "@noerglerü frage" {
		t.Errorf("= %q: a trailing letter means no boundary, so nothing is stripped", got)
	}
	if got := extractQuestion("@noergler-bot something", "noergler"); got != "-bot something" {
		t.Errorf("= %q: a hyphen IS a boundary", got)
	}
}
