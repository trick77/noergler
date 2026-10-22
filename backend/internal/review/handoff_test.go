package review

import (
	"bytes"
	"errors"
	"log/slog"
	"strings"
	"testing"
	"time"

	"github.com/trick77/noergler/internal/bitbucket"
	"github.com/trick77/noergler/internal/lang"
	"github.com/trick77/noergler/internal/render"
	"github.com/trick77/noergler/internal/store"
	"github.com/trick77/noergler/internal/webhook"
)

const handoffSummaryID = 900

// handoffHarness is a reviewed PR whose summary is comment 900 and whose
// findings sit on the given comment ids, every thread open.
func handoffHarness(t *testing.T, commentIDs ...int) *harness {
	t.Helper()
	h := newHarness(t, nil)
	h.st.skipState = &store.SkipState{ID: 7, Summary: &store.SummaryComment{ID: handoffSummaryID, Version: 3}}
	h.bb.comments[handoffSummaryID] = &bitbucket.Comment{ID: handoffSummaryID, Version: 3, Text: "## Summary\n\nbody"}
	for i, id := range commentIDs {
		cid := id
		h.st.existing = append(h.st.existing, store.Finding{
			ID: int64(i + 1), FilePath: "src/a.go", LineNumber: 10 + i, BitbucketCommentID: &cid,
		})
		h.bb.comments[id] = &bitbucket.Comment{ID: id, Text: "**Issue:** x"}
	}
	return h
}

func readyPayload() *webhook.Payload {
	p := prPayload(webhook.EventModified)
	was := true
	p.PreviousDraft = &was
	return p
}

func summaryText(h *harness) string { return h.bb.comments[handoffSummaryID].Text }

func TestReadyForReviewListsOpenThreadsOnTheSummary(t *testing.T) {
	h := handoffHarness(t, 501, 502, 503)
	h.bb.comments[502].ThreadResolved = true // resolved by a human

	h.r.HandleReadyForReview(t.Context(), readyPayload())

	got := summaryText(h)
	if !strings.HasPrefix(got, render.HandoffBannerSentinel+"\n🔎 **Ready for review** — 2 noergler threads still open: ") {
		t.Fatalf("summary does not open with the handoff banner:\n%s", got)
	}
	for _, want := range []string{
		"[`src/a.go:10`](https://bb.test/projects/PROJ/repos/my-repo/pull-requests/42/overview?commentId=501)",
		"[`src/a.go:12`](https://bb.test/projects/PROJ/repos/my-repo/pull-requests/42/overview?commentId=503)",
		"## Summary\n\nbody",
	} {
		if !strings.Contains(got, want) {
			t.Errorf("summary lacks %q:\n%s", want, got)
		}
	}
	if strings.Contains(got, "commentId=502") {
		t.Errorf("a resolved thread is listed:\n%s", got)
	}
	if v := h.bb.UpdateVersions; len(v) != 1 || v[0] != 3 {
		t.Errorf("update versions = %v, want the live 3", v)
	}
	if len(h.st.Summaries) != 1 || h.st.Summaries[0].ID != handoffSummaryID {
		t.Errorf("summary version not stored: %+v", h.st.Summaries)
	}
	if len(h.llm.Mentions) != 0 || len(h.llm.Reviews) != 0 || len(h.st.Runs) != 0 {
		t.Error("the handoff must not call the model or write a run")
	}
}

func TestReadyForReviewWritesNothingWhenEveryThreadIsClosed(t *testing.T) {
	h := handoffHarness(t, 501, 502)
	var logs bytes.Buffer
	h.r.log = slog.New(slog.NewTextHandler(&logs, nil))
	h.bb.comments[501].ThreadResolved = true
	delete(h.bb.comments, 502) // deleted thread: closed

	h.r.HandleReadyForReview(t.Context(), readyPayload())

	if len(h.bb.Updates) != 0 {
		t.Errorf("updated the summary with nothing open: %+v", h.bb.Updates)
	}
	if !strings.Contains(logs.String(), "draft -> ready, 0 of 2 noergler threads open") {
		t.Errorf("missing the decision log line:\n%s", logs.String())
	}
}

// ready -> draft -> ready with the threads closed in between: the old banner
// goes, the rest of the summary stays.
func TestReadyForReviewRemovesAnEarlierBannerOnceAllIsClosed(t *testing.T) {
	h := handoffHarness(t, 501)
	h.bb.comments[501].ThreadResolved = true
	old := render.HandoffBanner(lang.EN, []render.HandoffThread{{File: "src/a.go", Line: 10, URL: "u"}})
	h.bb.comments[handoffSummaryID].Text = old + "\n\n## Summary\n\nbody"

	h.r.HandleReadyForReview(t.Context(), readyPayload())

	if got := summaryText(h); got != "## Summary\n\nbody" {
		t.Errorf("summary = %q, want the banner gone and the body kept", got)
	}
}

func TestReadyForReviewCountsAnUnreadableThreadAsOpen(t *testing.T) {
	h := handoffHarness(t, 501)
	h.bb.fetchErr = map[int]error{501: &bitbucket.StatusError{Method: "GET", Path: "/comments", Status: 500}}

	h.r.HandleReadyForReview(t.Context(), readyPayload())

	if !strings.Contains(summaryText(h), "1 noergler thread still open") {
		t.Errorf("an unreadable thread must count as open:\n%s", summaryText(h))
	}
}

func TestReadyForReviewSpeaksTheAuthorsLanguage(t *testing.T) {
	h := handoffHarness(t, 501)
	h.st.languages = map[string]lang.Lang{"alice": lang.DE}

	h.r.HandleReadyForReview(t.Context(), readyPayload())

	if !strings.Contains(summaryText(h), "🔎 **Bereit für Reviewer** — noch 1 offener noergler-Thread: ") {
		t.Errorf("not in the author's German:\n%s", summaryText(h))
	}
}

func TestReadyForReviewLeavesAlonePRsWithNothingToAnnotate(t *testing.T) {
	cases := map[string]func(h *harness, p *webhook.Payload){
		"merged":     func(_ *harness, p *webhook.Payload) { p.PullRequest.State = "MERGED" },
		"never seen": func(h *harness, _ *webhook.Payload) { h.st.skipState = nil },
		"no summary": func(h *harness, _ *webhook.Payload) { h.st.skipState.Summary = nil },
		"ignored": func(h *harness, _ *webhook.Payload) {
			at := time.Now()
			h.st.skipState.IgnoredAt = &at
		},
		"store broken": func(h *harness, _ *webhook.Payload) { h.st.skipStateErr = errors.New("db down") },
	}
	for name, tweak := range cases {
		t.Run(name, func(t *testing.T) {
			h := handoffHarness(t, 501)
			p := readyPayload()
			tweak(h, p)
			h.r.HandleReadyForReview(t.Context(), p)
			if len(h.bb.Updates) != 0 || len(h.bb.Posted) != 0 {
				t.Errorf("wrote on a PR with nothing to annotate: updates %+v posted %+v", h.bb.Updates, h.bb.Posted)
			}
		})
	}
}

// A cost banner on top stays on top: its stripper only looks there.
func TestReadyForReviewPutsTheBannerBelowACostBanner(t *testing.T) {
	h := handoffHarness(t, 501)
	cost := render.CostLimitBanner(lang.EN, 6, 5, false)
	h.bb.comments[handoffSummaryID].Text = cost + "\n\n## Summary\n\nbody"

	h.r.HandleReadyForReview(t.Context(), readyPayload())

	got := summaryText(h)
	if !strings.HasPrefix(got, cost+"\n\n"+render.HandoffBannerSentinel) {
		t.Errorf("want cost banner, then handoff banner:\n%s", got)
	}
	if render.StripCostBanner(got) == got {
		t.Error("the cost banner is no longer strippable")
	}
}
