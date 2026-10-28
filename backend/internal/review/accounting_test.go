package review

import (
	"context"
	"errors"
	"fmt"
	"slices"
	"strings"
	"testing"
	"time"

	"github.com/trick77/noergler/internal/bitbucket"
	"github.com/trick77/noergler/internal/inference"
	"github.com/trick77/noergler/internal/store"
	"github.com/trick77/noergler/internal/webhook"
)

// earlier stores a finding posted by an earlier run, its thread on the PR
// and open.
func earlier(h *harness, id int64, severity, headline string) store.Finding {
	cid := int(id) + 700
	h.bb.comments[cid] = &bitbucket.Comment{ID: cid}
	f := store.Finding{ID: id, FilePath: "a.go", LineNumber: 2, Severity: severity,
		CommentText: "old", BitbucketCommentID: &cid}
	if headline != "" {
		f.Headline = ptrs(headline)
	}
	return f
}

// incremental puts h on a push after an earlier run.
func incremental(h *harness) {
	h.st.lastCommit, h.st.hasLast = "older", true
	h.bb.commitDiff = sampleDiff
}

// The PR summary lists the earlier finding still open, once: its repeat is
// the open one, not a second item. The dropped one is counted.
func TestSummaryListsEarlierOpenFindingOnce(t *testing.T) {
	h := newHarness(t, nil)
	incremental(h)
	h.st.existing = []store.Finding{earlier(h, 11, "issue", "")}
	invented := inference.ReviewFinding{File: "a.go", Line: 2, Severity: "suggestion", Comment: "x",
		Evidence: []string{"never in the diff"}}
	h.llm.review = okResultWith(finding("a.go", 2, "issue", "same again"), invented)

	h.r.ReviewPullRequest(context.Background(), prPayload(webhook.EventFromRefUpdated), false)

	if len(h.bb.Posted) != 1 {
		t.Fatalf("summaries = %d", len(h.bb.Posted))
	}
	s := h.bb.Posted[0].Text
	for _, want := range []string{"### Issues and suggestions\n1. old\n", "_1 finding dropped by validation._"} {
		if !strings.Contains(s, want) {
			t.Errorf("summary lacks %q:\n%s", want, s)
		}
	}
	if strings.Contains(s, "same again") || strings.Contains(s, "2. ") {
		t.Errorf("the repeat must not be listed beside the open finding:\n%s", s)
	}
}

// A push that adds nothing does not approve a PR whose earlier issue is
// still open: the issue stays listed and the verdict is raised to match.
func TestSummaryOfAQuietPushKeepsThePR(t *testing.T) {
	h := newHarness(t, nil)
	incremental(h)
	h.st.existing = []store.Finding{earlier(h, 11, "issue", "Nil map write"), earlier(h, 12, "suggestion", "Name it")}
	h.llm.review = okResultWith()

	h.r.ReviewPullRequest(context.Background(), prPayload(webhook.EventFromRefUpdated), false)

	s := h.bb.Posted[0].Text
	for _, want := range []string{
		"1. Nil map write\n2. Name it\n",
		"**Request changes 🛑** — Verdict raised: 1 open finding demands it.",
	} {
		if !strings.Contains(s, want) {
			t.Errorf("summary lacks %q:\n%s", want, s)
		}
	}
	if v := decodeDetail(t, h).Verdict; v == nil || v.Model != "approve" || v.Posted != "request_changes" {
		t.Errorf("run page verdict = %+v, want approve -> request_changes", v)
	}
}

// A thread the author resolved or deleted is closed on the PR, so it is
// neither listed nor holding up the verdict.
func TestSummaryLeavesOutClosedThreads(t *testing.T) {
	h := newHarness(t, nil)
	incremental(h)
	resolved := earlier(h, 11, "issue", "Resolved one")
	h.bb.comments[*resolved.BitbucketCommentID].ThreadResolved = true
	deleted := earlier(h, 12, "issue", "Deleted one")
	delete(h.bb.comments, *deleted.BitbucketCommentID)
	h.st.existing = []store.Finding{resolved, deleted}
	h.llm.review = okResultWith()

	h.r.ReviewPullRequest(context.Background(), prPayload(webhook.EventFromRefUpdated), false)

	s := h.bb.Posted[0].Text
	if !strings.Contains(s, "### Issues and suggestions\nNone.") || !strings.Contains(s, "**Approve ✅**") {
		t.Errorf("closed threads must not be listed or raise the verdict:\n%s", s)
	}
}

// At most maxThreadChecks threads are read per review, issues first; the rest
// count as open unread, so they stay listed even when resolved.
func TestThreadChecksAreCapped(t *testing.T) {
	h := newHarness(t, nil)
	incremental(h)
	for i := range maxThreadChecks + 1 {
		f := earlier(h, int64(i+1), "suggestion", fmt.Sprintf("Old %d", i))
		h.bb.comments[*f.BitbucketCommentID].ThreadResolved = true
		h.st.existing = append(h.st.existing, f)
	}
	issue := earlier(h, 500, "issue", "Last-posted issue")
	h.bb.comments[*issue.BitbucketCommentID].ThreadResolved = true
	h.st.existing = append(h.st.existing, issue)
	h.llm.review = okResultWith()

	h.r.ReviewPullRequest(context.Background(), prPayload(webhook.EventFromRefUpdated), false)

	read := 0
	for _, id := range h.bb.FetchedC {
		if id > 700 {
			read++
		}
	}
	if read != maxThreadChecks {
		t.Errorf("threads read = %d, want %d", read, maxThreadChecks)
	}
	s := h.bb.Posted[0].Text
	if strings.Contains(s, "Last-posted issue") {
		t.Errorf("the issue is checked first, and its thread is resolved:\n%s", s)
	}
	if !strings.Contains(s, "1. Old 49\n2. Old 50\n") {
		t.Errorf("the two unread suggestions must stay listed:\n%s", s)
	}
}

// A thread that cannot be read counts as open: fail toward showing it.
func TestSummaryKeepsAnUnreadableThread(t *testing.T) {
	h := newHarness(t, nil)
	incremental(h)
	f := earlier(h, 11, "issue", "Unreadable one")
	h.bb.fetchErr = map[int]error{*f.BitbucketCommentID: errors.New("boom")}
	h.st.existing = []store.Finding{f}
	h.llm.review = okResultWith()

	h.r.ReviewPullRequest(context.Background(), prPayload(webhook.EventFromRefUpdated), false)

	if s := h.bb.Posted[0].Text; !strings.Contains(s, "1. Unreadable one") {
		t.Errorf("an unreadable thread must stay listed:\n%s", s)
	}
}

// Earlier findings unreadable from the DB: the summary lists this run's and
// says so, never reading as a PR with none.
func TestSummarySaysEarlierFindingsUnreadable(t *testing.T) {
	h := newHarness(t, nil)
	incremental(h)
	h.st.existingErr = errors.New("db down")
	h.llm.review = okResultWith()

	h.r.ReviewPullRequest(context.Background(), prPayload(webhook.EventFromRefUpdated), false)

	want := "_Earlier findings unreadable: the list shows this run's only._"
	if s := h.bb.Posted[0].Text; !strings.Contains(s, want) {
		t.Errorf("summary lacks %q:\n%s", want, s)
	}
}

// A drop beside an open earlier issue keeps the verdict, but not the
// model's rationale: it may name the dropped finding, which is on no list.
func TestDropBesideAnOpenIssueKeepsTheVerdict(t *testing.T) {
	h := newHarness(t, nil)
	incremental(h)
	h.st.existing = []store.Finding{earlier(h, 11, "issue", "Old issue")}
	res := okResultWith(inference.ReviewFinding{File: "a.go", Line: 2, Severity: "issue", Comment: "x",
		Evidence: []string{"never in the diff"}})
	res.Review.Summary.VerdictDecision = "request_changes"
	res.Review.Summary.VerdictRationale = "Request changes: nil deref in Foo."
	h.llm.review = res

	h.r.ReviewPullRequest(context.Background(), prPayload(webhook.EventFromRefUpdated), false)

	want := "**Request changes 🛑** — Verdict: 1 open finding demands it."
	if s := h.bb.Posted[0].Text; !strings.Contains(s, want) {
		t.Errorf("summary lacks %q:\n%s", want, s)
	}
	if d := decodeDetail(t, h).Verdict; d == nil || d.Model != d.Posted {
		t.Errorf("run page verdict = %+v, want unchanged", d)
	}
}

// A thread the author resolved leaves nothing to request: the model's
// request_changes, argued from the open block it saw, is lowered.
func TestClosedThreadLowersAStaleVerdict(t *testing.T) {
	h := newHarness(t, nil)
	incremental(h)
	f := earlier(h, 11, "issue", "Resolved one")
	h.bb.comments[*f.BitbucketCommentID].ThreadResolved = true
	h.st.existing = []store.Finding{f}
	res := okResultWith()
	res.Review.Summary.VerdictDecision = "request_changes"
	h.llm.review = res

	h.r.ReviewPullRequest(context.Background(), prPayload(webhook.EventFromRefUpdated), false)

	want := "**Approve ✅** — Verdict lowered: the PR's open findings do not support it."
	if s := h.bb.Posted[0].Text; !strings.Contains(s, want) {
		t.Errorf("summary lacks %q:\n%s", want, s)
	}
}

// A resolved thread is settled (user, 2026-10-28): the model reporting the
// bug again posts nothing, and it holds nothing up. Stamped, so the next
// review does not read the thread again.
func TestResolvedThreadIsSettled(t *testing.T) {
	h := newHarness(t, nil)
	incremental(h)
	f := earlier(h, 11, "issue", "Resolved one")
	h.bb.comments[*f.BitbucketCommentID].ThreadResolved = true
	h.st.existing = []store.Finding{f}
	res := okResultWith(finding("a.go", 2, "issue", "still broken"))
	res.Review.Summary.VerdictDecision = "request_changes"
	h.llm.review = res

	h.r.ReviewPullRequest(context.Background(), prPayload(webhook.EventFromRefUpdated), false)

	if len(h.bb.Inline) != 0 {
		t.Errorf("inline = %d, want the repeat held back", len(h.bb.Inline))
	}
	if s := h.bb.Posted[0].Text; !strings.Contains(s, "### Issues and suggestions\nNone.") || !strings.Contains(s, "**Approve ✅**") {
		t.Errorf("a settled finding must not be listed or hold the verdict:\n%s", s)
	}
	if !slices.Equal(h.st.Closed, []int64{11}) {
		t.Errorf("closed = %v, want 11 stamped", h.st.Closed)
	}
}

// A 404 is closed for this review only, never stamped: a passing Bitbucket
// fault must not settle a finding for good.
func TestDeletedThreadIsNotStamped(t *testing.T) {
	h := newHarness(t, nil)
	incremental(h)
	f := earlier(h, 11, "issue", "Gone one")
	delete(h.bb.comments, *f.BitbucketCommentID)
	h.st.existing = []store.Finding{f}
	h.llm.review = okResultWith()

	h.r.ReviewPullRequest(context.Background(), prPayload(webhook.EventFromRefUpdated), false)

	if len(h.st.Closed) != 0 {
		t.Errorf("closed = %v, want no stamp on a 404", h.st.Closed)
	}
	if s := h.bb.Posted[0].Text; strings.Contains(s, "Gone one") {
		t.Errorf("a deleted thread is closed for this review:\n%s", s)
	}
}

// Over the prompt's cap, settled findings go before open ones: an open one
// keeps its id, so it can still be claimed fixed.
func TestTrimDropsClosedFindingsFirst(t *testing.T) {
	h := newHarness(t, nil)
	incremental(h)
	open := earlier(h, 1, "issue", "Oldest open")
	h.st.existing = []store.Finding{open}
	for i := range maxPreviouslyPostedFindings {
		f := earlier(h, int64(i+2), "suggestion", "")
		at := time.Now()
		f.ClosedAt = &at
		h.st.existing = append(h.st.existing, f)
	}
	h.llm.review = okResultWith()

	h.r.ReviewPullRequest(context.Background(), prPayload(webhook.EventFromRefUpdated), false)

	if p := h.llm.Reviews[0].Prompt; !strings.Contains(p, "[F1]") {
		t.Errorf("the open finding was trimmed for a closed one")
	}
}

// A stamped finding's thread is never read again.
func TestSettledThreadIsNotReadAgain(t *testing.T) {
	h := newHarness(t, nil)
	incremental(h)
	f := earlier(h, 11, "issue", "Settled one")
	at := time.Now()
	f.ClosedAt = &at
	h.st.existing = []store.Finding{f}
	h.llm.review = okResultWith()

	h.r.ReviewPullRequest(context.Background(), prPayload(webhook.EventFromRefUpdated), false)

	if slices.Contains(h.bb.FetchedC, *f.BitbucketCommentID) {
		t.Errorf("the settled thread was read again: %v", h.bb.FetchedC)
	}
	if s := h.bb.Posted[0].Text; strings.Contains(s, "Settled one") {
		t.Errorf("a settled finding must not be listed:\n%s", s)
	}
}

// Earlier findings unreadable: an unread list argues for nothing, so the
// model's request_changes stands.
func TestUnreadableEarlierFindingsNeverLowerTheVerdict(t *testing.T) {
	h := newHarness(t, nil)
	incremental(h)
	h.st.existingErr = errors.New("db down")
	res := okResultWith()
	res.Review.Summary.VerdictDecision = "request_changes"
	res.Review.Summary.VerdictRationale = "The old retry bug is still there."
	h.llm.review = res

	h.r.ReviewPullRequest(context.Background(), prPayload(webhook.EventFromRefUpdated), false)

	if s := h.bb.Posted[0].Text; !strings.Contains(s, "**Request changes 🛑** — The old retry bug is still there.") {
		t.Errorf("the verdict must stand over an unread list:\n%s", s)
	}
}

// A finding over the comment cap is still a finding: max_comments 0 posts
// none and must not approve over it.
func TestCappedFindingsStillHoldTheVerdict(t *testing.T) {
	h := newHarness(t, nil)
	h.r.cfg.MaxComments = 0
	res := okResultWith(finding("a.go", 2, "issue", "real"))
	res.Review.Summary.VerdictDecision = "request_changes"
	h.llm.review = res

	h.r.ReviewPullRequest(context.Background(), prPayload(webhook.EventOpened), false)

	s := h.bb.Posted[0].Text
	for _, want := range []string{"**Request changes 🛑**", "- 1 more new finding over the comment cap not posted."} {
		if !strings.Contains(s, want) {
			t.Errorf("summary lacks %q:\n%s", want, s)
		}
	}
}

// A settled finding stays in the prompt, marked closed and with no id: the
// model is told not to raise it again and cannot claim it fixed.
func TestClosedThreadIsMarkedInThePrompt(t *testing.T) {
	h := newHarness(t, nil)
	incremental(h)
	f := earlier(h, 11, "issue", "Resolved one")
	h.bb.comments[*f.BitbucketCommentID].ThreadResolved = true
	open := earlier(h, 12, "suggestion", "Open one")
	h.st.existing = []store.Finding{f, open}
	h.llm.review = okResultWith()

	h.r.ReviewPullRequest(context.Background(), prPayload(webhook.EventFromRefUpdated), false)

	p := h.llm.Reviews[0].Prompt
	if strings.Contains(p, "[F11]") || !strings.Contains(p, "[F12]") ||
		!strings.Contains(p, "- a.go:2 [issue] old (closed on the PR)") {
		t.Errorf("the prompt must mark the closed one, with no id:\n%s", p)
	}
}

// A legacy row with no comment id has no thread on the PR, so nobody could
// resolve it: closed, or it would hold the verdict for the PR's life. Its
// repeat is still held back.
func TestFindingWithoutThreadIsClosed(t *testing.T) {
	h := newHarness(t, nil)
	incremental(h)
	h.st.existing = []store.Finding{{ID: 11, FilePath: "a.go", LineNumber: 2, Severity: "issue",
		Headline: ptrs("Legacy row"), CommentText: "old"}}
	h.llm.review = okResultWith(finding("a.go", 2, "issue", "same again"))

	h.r.ReviewPullRequest(context.Background(), prPayload(webhook.EventFromRefUpdated), false)

	if s := h.bb.Posted[0].Text; strings.Contains(s, "Legacy row") || !strings.Contains(s, "**Approve ✅**") {
		t.Errorf("a row with no thread must not be listed or hold the verdict:\n%s", s)
	}
	if len(h.bb.Inline) != 0 {
		t.Errorf("inline = %d, want its repeat held back", len(h.bb.Inline))
	}
}

// A new issue under an approve: the list holds it, so the verdict follows.
func TestNewIssueRaisesAnApprove(t *testing.T) {
	h := newHarness(t, nil)
	h.llm.review = okResultWith(finding("a.go", 2, "issue", "new"))

	h.r.ReviewPullRequest(context.Background(), prPayload(webhook.EventOpened), false)

	want := "**Request changes 🛑** — Verdict raised: 1 open finding demands it."
	if s := h.bb.Posted[0].Text; !strings.Contains(s, want) {
		t.Errorf("summary lacks %q:\n%s", want, s)
	}
}

// A retired finding is counted as fixed and leaves the list and the verdict.
func TestSummaryCountsARetiredFinding(t *testing.T) {
	h := fixedHarness(t, "func new() {}")

	h.r.ReviewPullRequest(context.Background(), prPayload(webhook.EventFromRefUpdated), false)

	s := h.bb.Posted[0].Text
	for _, want := range []string{"### Issues and suggestions\nNone.", "_1 earlier finding fixed by this push._", "**Approve ✅**"} {
		if !strings.Contains(s, want) {
			t.Errorf("summary lacks %q:\n%s", want, s)
		}
	}
}

// A full review retires nothing, but the PR's open findings are still the
// PR's: listed, and holding up the verdict.
func TestFullReviewSummaryListsEarlierOpenFindings(t *testing.T) {
	h := newHarness(t, nil)
	h.st.existing = []store.Finding{earlier(h, 11, "suggestion", "")}
	h.llm.review = okResultWith(finding("a.go", 1, "issue", "new"))

	h.r.ReviewPullRequest(context.Background(), prPayload(webhook.EventOpened), false)

	s := h.bb.Posted[0].Text
	if !strings.Contains(s, "1. new _(new)_\n2. old") {
		t.Errorf("the full review must list the earlier open finding after the new issue:\n%s", s)
	}
	if strings.Contains(s, "fixed by this push") {
		t.Errorf("a full review retires nothing:\n%s", s)
	}
}

// A first review has nothing to account for: no line at all.
func TestFirstReviewSummaryHasNoAccounting(t *testing.T) {
	h := newHarness(t, nil)
	h.llm.review = okResultWith(finding("a.go", 2, "issue", "new"))

	h.r.ReviewPullRequest(context.Background(), prPayload(webhook.EventOpened), false)

	s := h.bb.Posted[0].Text
	if strings.Contains(s, "fixed by this push") || strings.Contains(s, "dropped by validation") || strings.Contains(s, "_(new)_") {
		t.Errorf("a first review must not carry the accounting line or new markers:\n%s", s)
	}
}
