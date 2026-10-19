package review

import (
	"context"
	"encoding/json"
	"strings"
	"testing"

	"github.com/trick77/noergler/internal/inference"
	"github.com/trick77/noergler/internal/store"
	"github.com/trick77/noergler/internal/webhook"
)

func decodeDetail(t *testing.T, h *harness) store.Pipeline {
	t.Helper()
	blob, ok := h.st.Details[1]
	if !ok {
		t.Fatalf("no detail record for attempt 1: %v", h.st.Details)
	}
	var d store.Pipeline
	if err := json.Unmarshal(blob, &d); err != nil {
		t.Fatal(err)
	}
	return d
}

// A successful run writes its detail record under its attempt: every stage
// in order, the prompt's parts, how each file went, and each proposed
// finding that was not posted, with why.
func TestRunDetailRecord(t *testing.T) {
	h := newHarness(t, nil)
	invented := finding("a.go", 2, "issue", "SECRET COMMENT BODY")
	invented.Evidence = []string{"func missing() {}"}
	headline := "Invented thing"
	invented.Headline = &headline
	moved := finding("a.go", 40, "issue", "real, wrong line")
	h.llm.review = okResultWith(invented, moved)

	h.r.ReviewPullRequest(context.Background(), prPayload(webhook.EventOpened), false)

	d := decodeDetail(t, h)
	var names []string
	for _, s := range d.Stages {
		names = append(names, s.Name)
	}
	want := []string{stageDiff, stageFiles, stageContext, stagePrompt, stageQueued, stageGateway,
		stageToPost, stageValidate, stageComments, stageRecordSum}
	if strings.Join(names, ",") != strings.Join(want, ",") {
		t.Errorf("stages = %v, want %v", names, want)
	}
	if d.Prompt.Total == 0 || d.Prompt.Budget == 0 {
		t.Errorf("prompt = %+v", d.Prompt)
	}
	if len(d.Files) != 1 || d.Files[0].Path != "a.go" || d.Files[0].Tokens == 0 {
		t.Errorf("files = %+v", d.Files)
	}
	if len(d.Decisions) != 1 || d.Decisions[0].Decision != "dropped" ||
		d.Decisions[0].Reason != string(inference.DropEvidenceNotFound) || d.Decisions[0].Headline != headline {
		t.Errorf("decisions = %+v", d.Decisions)
	}
	if len(d.Moved) != 1 || d.Moved[0].From != 40 || d.Moved[0].To != 2 {
		t.Errorf("moved = %+v", d.Moved)
	}
	if d.Verdict == nil || d.Verdict.Posted == "" {
		t.Errorf("verdict = %+v", d.Verdict)
	}
	// The fakes bypass the HTTP counter, so the counts are empty here; the
	// record still carries the list, never null.
	if d.HTTP == nil {
		t.Error("http counts missing")
	}
	// An unposted finding keeps its text: the page is its only trace.
	if d.Decisions[0].Comment == "" || len(d.Decisions[0].Evidence) == 0 {
		t.Errorf("decision lost its text: %+v", d.Decisions[0])
	}
}

// A failed call still gets a record: its stages up to the gateway and what
// the prompt was are exactly what someone opens the page for.
func TestRunDetailOnFailure(t *testing.T) {
	h := newHarness(t, nil)
	h.llm.review = inference.ReviewResult{
		Outcome: inference.OutcomeUnparseable,
		Review:  inference.ParsedReview{Summary: inference.NewReviewSummary()},
	}
	h.r.ReviewPullRequest(context.Background(), prPayload(webhook.EventOpened), false)

	d := decodeDetail(t, h)
	last := d.Stages[len(d.Stages)-1].Name
	if len(d.Stages) < 7 || last != stageRecordSum {
		t.Errorf("stages = %+v", d.Stages)
	}
	if d.Prompt.Total == 0 || len(d.Decisions) != 0 || d.Verdict != nil {
		t.Errorf("detail = %+v", d)
	}
}

// Only a posted finding's move is shown, read off the finding: one moved and
// then not posted leaves nothing, even when another finding was posted on
// the line it moved to.
func TestPostedMoves(t *testing.T) {
	at := func(n int) *int { return &n }
	moved := finding("a.go", 2, "issue", "x")
	moved.MovedFrom = at(0) // line 0 is a line the model can cite
	notPostedMove := finding("a.go", 2, "issue", "y")
	notPostedMove.MovedFrom = at(9)
	unmoved := finding("a.go", 2, "issue", "z")
	got := postedMoves([]inference.ReviewFinding{moved, notPostedMove, unmoved}, []int{101, 0, 102})
	if len(got) != 1 || got[0].From != 0 || got[0].To != 2 {
		t.Errorf("moves = %+v, want only the posted, moved one", got)
	}
}

// Upstreams come out by name, so the page's order never depends on a map.
func TestHTTPCountsSorted(t *testing.T) {
	got := httpCounts(map[string]int{"jira": 1, "bitbucket": 6, "inference": 1})
	if len(got) != 3 || got[0].Upstream != "bitbucket" || got[2].Upstream != "jira" {
		t.Errorf("counts = %+v", got)
	}
	if httpCounts(nil) == nil {
		t.Error("no counts must be an empty list, not null")
	}
}

// A bug in code the PR did not change goes on the run page, never the PR:
// recorded with the reviewed commit for its link, no inline comment, no
// effect on the verdict. One quoting the PR's own added line is dropped.
func TestPreExistingBugsGoToThePageOnly(t *testing.T) {
	h := newHarness(t, nil)
	res := okResultWith()
	res.Review.PreExisting = []inference.PreExistingBug{
		{File: "a.go", Line: 1, Headline: "Package name shadows the stdlib", Explanation: "Imports break.",
			Evidence: []string{"package a"}},
		{File: "a.go", Line: 2, Headline: "really the PR's", Explanation: "x",
			Evidence: []string{"func new() {}"}},
	}
	h.llm.review = res

	h.r.ReviewPullRequest(context.Background(), prPayload(webhook.EventOpened), false)

	d := decodeDetail(t, h)
	if len(d.PreExisting) != 1 || d.PreExisting[0].Headline != "Package name shadows the stdlib" ||
		d.PreExisting[0].Explanation != "Imports break." || d.PreExisting[0].Line != 1 {
		t.Errorf("pre_existing = %+v, want the one in unchanged code", d.PreExisting)
	}
	if d.Commit == "" {
		t.Error("no commit recorded for the link")
	}
	if len(h.bb.Inline) != 0 {
		t.Errorf("inline comments = %d: a pre-existing bug is never posted", len(h.bb.Inline))
	}
	if d.Verdict == nil || d.Verdict.Posted != "approve" {
		t.Errorf("verdict = %+v: pre-existing bugs never move it", d.Verdict)
	}
}
