package evals

import (
	"encoding/json"
	"fmt"
	"strings"

	"github.com/trick77/noergler/internal/diff"
	"github.com/trick77/noergler/internal/inference"
	"github.com/trick77/toolkit/llmwire"
)

// The questions experiment (EVALS.md): may a review ask the author about a
// suspected defect it withholds (confidence under 80, the failing path in
// code it cannot see) instead of staying silent? Eval only. A brief from
// briefs/ is appended to the assembled prompt and the strict schema gains a
// "questions" key; production's schema, parser and goldens are untouched.
// Questions are read off the raw answer, validated like a finding and
// scored by file and line window, never by a judge.

// maxQuestions is the brief's cap, held by the schema; Result.QuestionsOverCap
// counts what a model sends past it anyway.
const maxQuestions = 2

// Question band the brief asks for: below a finding's 80, nothing more.
const (
	questionMinConfidence = 0
	questionMaxConfidence = 79
)

// Question is one question to the author, as the model wrote it.
type Question struct {
	File       string   `json:"file"`
	Line       int      `json:"line"`
	Evidence   []string `json:"evidence"`
	Question   string   `json:"question"`
	Confidence int      `json:"confidence"`
}

// Question verdicts.
const (
	// QuestionDropped: fails the validation a finding would fail.
	QuestionDropped = "dropped"
	// QuestionOnControl: a question on a clean case.
	QuestionOnControl = "control"
	// QuestionRescued: in a seeded bug's window where the same review has
	// no finding at all, the miss a question would have turned into a lead.
	QuestionRescued = "rescued"
	// QuestionRedundant: in a seeded bug's window beside a finding there.
	QuestionRedundant = "redundant"
	// QuestionElsewhere: on a seeded case, in no seeded window.
	QuestionElsewhere = "elsewhere"
)

// ScoredQuestion is a question and what the scorer made of it.
type ScoredQuestion struct {
	Question
	Verdict string
	// Expected is the seeded bug whose window it is in, "" for none.
	Expected string `json:",omitempty"`
	// Keyword: it also names one of that bug's keywords, as a finding must.
	Keyword bool   `json:",omitempty"`
	Reason  string `json:",omitempty"`
}

// WithQuestions sets the questions brief on every case: its reviews carry
// the brief and the extended schema, and their questions are scored.
func WithQuestions(cases []Case, brief string) []Case {
	out := make([]Case, len(cases))
	for i, c := range cases {
		c.questions = brief
		out[i] = c
	}
	return out
}

// questionsResponseFormat is the review schema plus the "questions" key.
// ReviewResponseSchema builds a fresh map each call, so extending it here
// changes nothing production sends.
func questionsResponseFormat() *llmwire.ResponseFormat {
	f := inference.ReviewResponseFormat()
	s := f.Schema
	s["required"] = append(s["required"].([]any), "questions")
	s["properties"].(map[string]any)["questions"] = questionsProperty()
	return f
}

// questionsSchemaText is the schema's added part as sent, for the
// pre-flight count: production's count covers its own schema only.
func questionsSchemaText() string {
	b, err := json.Marshal(map[string]any{"questions": questionsProperty()})
	if err != nil {
		return ""
	}
	return string(b)
}

// questionsProperty is the "questions" key's schema.
func questionsProperty() map[string]any {
	return map[string]any{
		"type": "array", "maxItems": maxQuestions,
		"items": map[string]any{
			"type":                 "object",
			"additionalProperties": false,
			"required":             []any{"file", "line", "evidence", "question", "confidence"},
			"properties": map[string]any{
				"file": map[string]any{"type": "string"},
				"line": map[string]any{"type": "integer"},
				"evidence": map[string]any{
					"type": "array", "minItems": 1, "maxItems": 5,
					"items": map[string]any{"type": "string"},
				},
				"question":   map[string]any{"type": "string", "minLength": 1},
				"confidence": map[string]any{"type": "integer", "minimum": 0, "maximum": 100},
			},
		},
	}
}

// parseQuestions reads the questions off a raw answer, item by item:
// unparsed counts the items that did not decode, so a question never
// vanishes unseen. A fence is stripped as ParseReview does: the opening
// line whole, info string and all, the closing line when it is a bare
// fence. An answer that does not parse at all has no questions (the
// review's own parse reports that).
func parseQuestions(raw string) (qs []Question, unparsed int) {
	raw = strings.TrimSpace(raw)
	if strings.HasPrefix(raw, "```") {
		lines := strings.Split(raw, "\n")[1:]
		if n := len(lines); n > 0 && strings.TrimSpace(lines[n-1]) == "```" {
			lines = lines[:n-1]
		}
		raw = strings.Join(lines, "\n")
	}
	var out struct {
		Questions []json.RawMessage `json:"questions"`
	}
	if json.Unmarshal([]byte(raw), &out) != nil {
		return nil, 0
	}
	for _, item := range out.Questions {
		var q Question
		if json.Unmarshal(item, &q) != nil {
			unparsed++
			continue
		}
		qs = append(qs, q)
	}
	return qs, unparsed
}

// isControl: a case that expects nothing at all, no finding, fix,
// screenshot line or repeat. A question on it is a question on clean code.
func isControl(c Case) bool {
	return len(c.Expected) == 0 && len(c.ExpectedFixed) == 0 && len(c.ExpectedVisual) == 0 && len(c.Repeats) == 0
}

// scoreQuestions validates each question as production would a finding and
// places it: on a control, in a seeded window (rescued or redundant by
// whether the same review has a finding there), or elsewhere. Text that
// carries a secret-like token is not written down: results are committed.
func scoreQuestions(c Case, qs []Question, idx diff.AnchorIndex, matches []Match) (scored []ScoredQuestion, overCap, outOfBand int) {
	if len(qs) > maxQuestions {
		overCap = len(qs) - maxQuestions
	}
	for _, q := range qs {
		if q.Confidence < questionMinConfidence || q.Confidence > questionMaxConfidence {
			outOfBand++
		}
		sq := ScoredQuestion{Question: q}
		v := inference.ValidateFindings([]inference.ReviewFinding{{
			File: q.File, Line: q.Line, Severity: "suggestion", Comment: q.Question, Evidence: q.Evidence,
		}}, idx)
		switch {
		case len(v.Kept) == 0:
			sq.Verdict, sq.Reason = QuestionDropped, string(v.Dropped[0].Reason)
		case isControl(c):
			sq.File, sq.Line = v.Kept[0].File, v.Kept[0].Line
			sq.Verdict = QuestionOnControl
		default:
			sq.File, sq.Line = v.Kept[0].File, v.Kept[0].Line
			sq.Verdict = QuestionElsewhere
			placeQuestion(&sq, matches)
		}
		if inference.AnySecret(append([]string{sq.Question.Question}, sq.Evidence...)...) {
			sq.Question.Question, sq.Evidence = "", nil
		}
		scored = append(scored, sq)
	}
	return scored, overCap, outOfBand
}

// placeQuestion puts a kept question on the seeded bug whose window holds
// it: rescued when the review missed that bug (no matching finding, a
// finding nearby that names none of its keywords included), redundant when
// it caught it. Of overlapping windows a missed bug wins: that is the one
// the question could have saved.
func placeQuestion(sq *ScoredQuestion, matches []Match) {
	for _, missed := range []bool{true, false} {
		for _, m := range matches {
			if m.Found == missed || !inWindow(sq.File, sq.Line, m.Expected) {
				continue
			}
			sq.Expected = m.Expected.ID
			sq.Keyword = mentions(inference.ReviewFinding{Comment: sq.Question.Question}, m.Expected.Keywords)
			sq.Verdict = QuestionRedundant
			if missed {
				sq.Verdict = QuestionRescued
			}
			return
		}
	}
}

// inWindow: a file and line inside a seeded bug's window.
func inWindow(file string, line int, e Expected) bool {
	return samePath(file, e.File) && line >= e.Lines[0] && line <= e.Lines[1]
}

// QuestionTally sums a run's questions by verdict.
type QuestionTally struct {
	// Reviews counts completed reviews only.
	Reviews, Total, Unparsed, OverCap, OutOfBand int
	ByVerdict                                    map[string]int
	// RescuedKeyword: rescued questions that also name the bug's keyword.
	RescuedKeyword int
}

// TallyQuestions sums the questions of every review given.
func TallyQuestions(results []Result) QuestionTally {
	t := QuestionTally{ByVerdict: map[string]int{}}
	for _, r := range results {
		// A review that never completed could not have asked anything.
		if completed(r) {
			t.Reviews++
		}
		t.Unparsed += r.QuestionsUnparsed
		t.OverCap += r.QuestionsOverCap
		t.OutOfBand += r.QuestionsOutOfBand
		for _, q := range r.Questions {
			t.Total++
			t.ByVerdict[q.Verdict]++
			if q.Verdict == QuestionRescued && q.Keyword {
				t.RescuedKeyword++
			}
		}
	}
	return t
}

// String is the report's questions line.
func (t QuestionTally) String() string {
	return fmt.Sprintf("questions: %d in %d review(s): %d on a control, %d rescued (%d naming a keyword), "+
		"%d redundant, %d elsewhere, %d dropped, %d unparsed; %d over the cap, %d outside confidence %d-%d",
		t.Total, t.Reviews, t.ByVerdict[QuestionOnControl], t.ByVerdict[QuestionRescued], t.RescuedKeyword,
		t.ByVerdict[QuestionRedundant], t.ByVerdict[QuestionElsewhere], t.ByVerdict[QuestionDropped], t.Unparsed,
		t.OverCap, t.OutOfBand, questionMinConfidence, questionMaxConfidence)
}

// SampledResults are every sample's result, in case order.
func (s Sampled) SampledResults() []Result {
	var out []Result
	for _, c := range s.Cases {
		for _, sm := range c.Samples {
			out = append(out, sm.Result)
		}
	}
	return out
}
