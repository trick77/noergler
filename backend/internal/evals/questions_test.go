package evals

import (
	"context"
	"encoding/json"
	"strings"
	"testing"

	"github.com/trick77/noergler/internal/inference"
)

// questionsReviewer answers with canned findings and a raw answer carrying
// questions, and keeps the request it was sent.
type questionsReviewer struct {
	findings []inference.ReviewFinding
	raw      string
	req      *inference.ReviewRequest
}

func (q questionsReviewer) Review(_ context.Context, req inference.ReviewRequest) inference.ReviewResult {
	if q.req != nil {
		*q.req = req
	}
	return inference.ReviewResult{Raw: q.raw, Review: inference.ParsedReview{Findings: q.findings,
		Summary: inference.NewReviewSummary()}}
}

const touchLine = "func (s *Store) Touch(ctx context.Context, id string, ttlMillis int64) error {"

func questionsRaw(t *testing.T, qs ...Question) string {
	t.Helper()
	b, err := json.Marshal(map[string]any{"questions": qs})
	if err != nil {
		t.Fatal(err)
	}
	return string(b)
}

func askTouch(confidence int) Question {
	return Question{File: "internal/session/store.go", Line: 20, Evidence: []string{touchLine},
		Question: "Do the callers still pass seconds?", Confidence: confidence}
}

func questionsRun(t *testing.T, name string, r questionsReviewer) Result {
	t.Helper()
	c := WithQuestions([]Case{corpusCase(t, name)}, "\n## Questions brief\n")[0]
	return runOne(c, r).Results[0]
}

// The brief rides the prompt and the schema asks for questions; the
// production schema stays without them.
func TestQuestionsRequest(t *testing.T) {
	var req inference.ReviewRequest
	questionsRun(t, "unchanged-caller-units", questionsReviewer{raw: "{}", req: &req})
	if !strings.HasSuffix(req.Prompt, "## Questions brief\n") {
		t.Errorf("prompt does not end in the brief")
	}
	if _, ok := req.ResponseSchema.Schema["properties"].(map[string]any)["questions"]; !ok {
		t.Errorf("schema has no questions key")
	}
	if _, ok := inference.ReviewResponseFormat().Schema["properties"].(map[string]any)["questions"]; ok {
		t.Errorf("the production schema gained a questions key")
	}
	plain := runOne(corpusCase(t, "unchanged-caller-units"), questionsReviewer{raw: questionsRaw(t, askTouch(60))})
	if len(plain.Results[0].Questions) != 0 {
		t.Errorf("a plain run scored questions: %+v", plain.Results[0].Questions)
	}
}

// A question in a seeded window with no finding there is a rescued miss;
// beside a finding it is redundant; one failing validation is dropped.
func TestQuestionsOnASeededCase(t *testing.T) {
	res := questionsRun(t, "unchanged-caller-units", questionsReviewer{raw: questionsRaw(t, askTouch(60),
		Question{File: "internal/session/store.go", Line: 20, Evidence: []string{"no such line"}, Question: "?", Confidence: 90})})
	if len(res.Questions) != 2 || res.Questions[0].Verdict != QuestionRescued || !res.Questions[0].Keyword ||
		res.Questions[0].Expected != "caller-passes-seconds" || res.Questions[1].Verdict != QuestionDropped {
		t.Fatalf("questions = %+v, want rescued with keyword, then dropped", res.Questions)
	}
	if res.QuestionsOutOfBand != 1 {
		t.Errorf("out of band = %d, want 1 (confidence 90)", res.QuestionsOutOfBand)
	}

	finding := inference.ReviewFinding{File: "internal/session/store.go", Line: 20, Severity: "issue",
		Comment: "the caller still passes a second count", Evidence: []string{touchLine}}
	res = questionsRun(t, "unchanged-caller-units", questionsReviewer{findings: []inference.ReviewFinding{finding},
		raw: questionsRaw(t, askTouch(60), askTouch(55), askTouch(70))})
	if res.Questions[0].Verdict != QuestionRedundant || res.QuestionsOverCap != 1 {
		t.Errorf("questions = %+v, over cap %d: want redundant, 1 over the cap", res.Questions, res.QuestionsOverCap)
	}

	// A finding in the window that names none of the bug's keywords did not
	// catch it: the question beside it still rescues the miss.
	off := finding
	off.Comment = "rename this parameter"
	res = questionsRun(t, "unchanged-caller-units", questionsReviewer{findings: []inference.ReviewFinding{off},
		raw: questionsRaw(t, askTouch(60))})
	if res.Questions[0].Verdict != QuestionRescued {
		t.Errorf("questions = %+v, want rescued beside a finding that missed the bug", res.Questions)
	}
}

// A case that expects a repeat, a fix or a screenshot line is no control.
func TestQuestionsOnACaseWithOtherExpectations(t *testing.T) {
	c := corpusCase(t, "incremental-moved-not-fixed")
	if isControl(c) || len(c.Expected) != 0 {
		t.Fatalf("incremental-moved-not-fixed: want repeats and no expected finding")
	}
	if !isControl(corpusCase(t, "clean-unchanged-caller")) {
		t.Errorf("clean-unchanged-caller must be a control")
	}
}

// On a clean case every kept question costs the author a reply.
func TestQuestionsOnAControl(t *testing.T) {
	res := questionsRun(t, "clean-unchanged-caller", questionsReviewer{raw: questionsRaw(t, askTouch(60))})
	if len(res.Questions) != 1 || res.Questions[0].Verdict != QuestionOnControl {
		t.Fatalf("questions = %+v, want one on the control", res.Questions)
	}
	tally := TallyQuestions([]Result{res})
	if tally.Total != 1 || tally.ByVerdict[QuestionOnControl] != 1 || !strings.Contains(tally.String(), "1 on a control") {
		t.Errorf("tally = %+v", tally)
	}
}

// An answer that does not parse carries no questions; a fenced one does,
// whatever its info string; an item that does not decode is counted.
func TestParseQuestions(t *testing.T) {
	if got, n := parseQuestions("not json"); got != nil || n != 0 {
		t.Errorf("questions = %+v, %d unparsed, want none", got, n)
	}
	if got, _ := parseQuestions("``` JSON\n" + questionsRaw(t, askTouch(60)) + "\n```"); len(got) != 1 || got[0].Line != 20 {
		t.Errorf("fenced questions = %+v", got)
	}
	raw := `{"questions": [{"file": "a.go", "line": "26", "evidence": ["x"], "question": "?", "confidence": 60},` +
		`{"file": "a.go", "line": 26, "evidence": ["x"], "question": "?", "confidence": 60}]}`
	if got, n := parseQuestions(raw); len(got) != 1 || n != 1 {
		t.Errorf("questions = %+v, %d unparsed, want 1 and 1", got, n)
	}
}
