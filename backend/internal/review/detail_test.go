package review

import (
	"context"
	"encoding/json"
	"strings"
	"testing"

	"github.com/trick77/noergler/internal/inference"
	"github.com/trick77/noergler/internal/webhook"
)

func decodeDetail(t *testing.T, h *harness) runDetail {
	t.Helper()
	blob, ok := h.st.Details[1]
	if !ok {
		t.Fatalf("no detail record for attempt 1: %v", h.st.Details)
	}
	var d runDetail
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
	// record still carries the map, never null.
	if d.HTTP == nil {
		t.Error("http counts missing")
	}
	// Headlines only: the comment body never reaches the record.
	if strings.Contains(string(h.st.Details[1]), "SECRET COMMENT BODY") {
		t.Error("the detail record carries a finding's comment text")
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
