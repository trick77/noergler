package evals

// The ticket-check eval scores what the summary shows for a ticket: a
// requirement appears only when ConfirmCompliance finds the line the model
// quoted as proof, and never as unmet. A case is a review as production sends
// it with a ticket: the compiled-in review prompt, the ticket context, the
// compliance instructions. The answer goes through the same parse and the
// same ConfirmCompliance as `post`; findings are not scored here.
//
// Two numbers matter. A requirement the code visibly implements should be
// shown (Shown out of Provable): too many drops and the section is empty
// everywhere, the check off in all but name. A requirement the shown code
// does not implement must never be shown (FalseShown): that is the wrong
// comment, the other way round. Unmet counts what the model still called
// unmet against the prompt; the summary never shows it.
//
// The corpus lives in compliance/, never in corpus/: LoadCorpus embeds
// corpus/*.json, and a ticket case there would join every review series run.

import (
	"context"
	"embed"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"io/fs"
	"sort"
	"strings"
	"sync"
	"time"

	"github.com/trick77/noergler/internal/diff"
	"github.com/trick77/noergler/internal/inference"
	"github.com/trick77/noergler/internal/lang"
	"github.com/trick77/noergler/internal/prompts"
)

//go:embed compliance/*.json
var complianceFS embed.FS

// ComplianceCase is a corpus entry as stored.
type ComplianceCase struct {
	Name string `json:"name"`
	// Why says what the case tests.
	Why string `json:"why"`
	// TicketContext is the {ticket_context} block as the review renders a
	// Jira ticket (review.renderTicketContext), stored rendered so the case
	// reads like the prompt.
	TicketContext string                  `json:"ticket_context"`
	Files         []ComplianceFile        `json:"files"`
	Expect        []ComplianceExpectation `json:"expect"`
}

// ComplianceFile is one changed file as the review series stores it.
type ComplianceFile struct {
	Path           string `json:"path"`
	Diff           string `json:"diff"`
	Content        string `json:"content"`
	ContentFetched bool   `json:"content_fetched"`
}

// reviewFiles is the case's files as the prompt and the anchor index take
// them.
func (c ComplianceCase) reviewFiles() []diff.FileReviewData {
	out := make([]diff.FileReviewData, len(c.Files))
	for i, f := range c.Files {
		out[i] = diff.FileReviewData{Path: f.Path, Diff: f.Diff, Content: f.Content, ContentFetched: f.ContentFetched}
	}
	return out
}

// ComplianceExpectation is one criterion of the ticket and whether the
// summary should show it.
type ComplianceExpectation struct {
	// Label is the criterion's label in the ticket (AK-1). A requirement
	// matches when its text carries the label, else any keyword.
	Label    string   `json:"label"`
	Keywords []string `json:"keywords"`
	// Shown: the shown code visibly implements it, so a quote can prove
	// it. False: implemented outside the shown code, or not at all; showing
	// it is the wrong comment.
	Shown bool   `json:"shown"`
	Why   string `json:"why"`
}

// LoadComplianceCorpus reads every embedded ticket case, sorted by name.
func LoadComplianceCorpus() ([]ComplianceCase, error) {
	entries, err := fs.Glob(complianceFS, "compliance/*.json")
	if err != nil {
		return nil, err
	}
	sort.Strings(entries)
	out := make([]ComplianceCase, 0, len(entries))
	for _, name := range entries {
		raw, err := complianceFS.ReadFile(name)
		if err != nil {
			return nil, err
		}
		var c ComplianceCase
		dec := json.NewDecoder(strings.NewReader(string(raw)))
		dec.DisallowUnknownFields()
		if err := dec.Decode(&c); err != nil {
			return nil, fmt.Errorf("%s: %w", name, err)
		}
		if c.Name == "" || c.TicketContext == "" || len(c.Files) == 0 || len(c.Expect) == 0 {
			return nil, fmt.Errorf("%s: name, ticket_context, files and expect are required", name)
		}
		for _, e := range c.Expect {
			if e.Label == "" && len(e.Keywords) == 0 {
				return nil, fmt.Errorf("%s: an expectation needs a label or keywords", name)
			}
		}
		out = append(out, c)
	}
	return out, nil
}

// Assemble is the review prompt production sends for the case's PR and
// ticket, in English like the review series.
func (c ComplianceCase) Assemble(template string, count inference.CountFunc) inference.AssembledPrompt {
	if template == "" {
		template = prompts.Review
	}
	return inference.AssembleReviewPrompt(inference.AssembleRequest{
		Template:              template,
		Files:                 c.reviewFiles(),
		TicketContext:         c.TicketContext,
		TicketComplianceCheck: true,
		Language:              lang.EN,
	}, count)
}

// Verdicts of one expectation in one answer.
const (
	ComplianceShownRight = "shown"       // provable and shown
	ComplianceDropped    = "dropped"     // provable, listed, left out by ConfirmCompliance
	ComplianceUnlisted   = "unlisted"    // provable, never listed as met
	ComplianceHeldBack   = "held_back"   // not provable and not shown: right
	ComplianceFalseShown = "false_shown" // not provable and shown: the wrong comment
)

// ComplianceRequirementResult is one requirement the model listed, with
// what ConfirmCompliance made of it.
type ComplianceRequirementResult struct {
	Requirement string `json:"requirement"`
	Met         bool   `json:"met"`
	File        string `json:"file,omitempty"`
	Quote       string `json:"quote,omitempty"`
	Drop        string `json:"drop,omitempty"`
}

// ComplianceExpectResult is one expectation's verdict in one answer.
type ComplianceExpectResult struct {
	Label   string `json:"label"`
	Shown   bool   `json:"expected_shown"`
	Verdict string `json:"verdict"`
	// Drop is ConfirmCompliance's reason, on a dropped one.
	Drop string `json:"drop,omitempty"`
}

// ComplianceAnswer is one review of one case.
type ComplianceAnswer struct {
	Outcome      string                        `json:"outcome"`
	Err          string                        `json:"error,omitempty"`
	Redraws      int                           `json:"redraws,omitempty"`
	Seconds      float64                       `json:"seconds"`
	Requirements []ComplianceRequirementResult `json:"requirements"`
	Expect       []ComplianceExpectResult      `json:"expect"`
	// Unmet is how many requirements the model listed as not met, against
	// the prompt. Never shown; counted because a model that keeps doing it
	// may also say so in prose, which nothing checks.
	Unmet         int   `json:"unmet"`
	PromptTokens  int64 `json:"prompt_tokens"`
	OutputTokens  int64 `json:"output_tokens"`
	completedCall bool
}

// ComplianceCaseResult is one case over every sample.
type ComplianceCaseResult struct {
	Case    string             `json:"case"`
	Answers []ComplianceAnswer `json:"answers"`
}

// ComplianceRun is a whole run, written as the JSON report.
type ComplianceRun struct {
	Model   string                 `json:"model"`
	Effort  string                 `json:"effort"`
	Samples int                    `json:"samples"`
	Seconds float64                `json:"seconds"`
	Cases   []ComplianceCaseResult `json:"cases"`
	Totals  ComplianceTotals       `json:"totals"`
}

// ComplianceTotals sums the verdicts over every completed answer.
type ComplianceTotals struct {
	Provable   int            `json:"provable"`
	Shown      int            `json:"shown"`
	Dropped    map[string]int `json:"dropped"`
	Unlisted   int            `json:"unlisted"`
	Unprovable int            `json:"unprovable"`
	FalseShown int            `json:"false_shown"`
	Unmet      int            `json:"unmet"`
	Incomplete int            `json:"incomplete"`
}

// RunCompliance reviews every case samples times, parallel calls in flight.
func RunCompliance(ctx context.Context, client Reviewer, cases []ComplianceCase, template string,
	count inference.CountFunc, samples, parallel int, progress io.Writer) ComplianceRun {
	start := time.Now()
	out := ComplianceRun{Samples: samples, Cases: make([]ComplianceCaseResult, len(cases))}
	prompts := make([]inference.AssembledPrompt, len(cases))
	for i, c := range cases {
		prompts[i] = c.Assemble(template, count)
		out.Cases[i] = ComplianceCaseResult{Case: c.Name, Answers: make([]ComplianceAnswer, samples)}
	}
	var mu sync.Mutex
	var wg sync.WaitGroup
	slots := make(chan struct{}, max(1, parallel))
	for s := range samples {
		for i := range cases {
			wg.Add(1)
			slots <- struct{}{}
			go func() {
				defer wg.Done()
				defer func() { <-slots }()
				a := drawCompliance(ctx, client, cases[i], prompts[i])
				mu.Lock()
				defer mu.Unlock()
				out.Cases[i].Answers[s] = a
				if progress != nil {
					_, _ = fmt.Fprintf(progress, "%-36s sample %d: %s (%.0fs)\n", cases[i].Name, s+1, a.summary(), a.Seconds)
				}
			}()
		}
	}
	wg.Wait()
	out.Seconds = time.Since(start).Seconds()
	out.Totals = out.total()
	return out
}

// drawCompliance is one answer, drawn again while the call does not
// complete, up to maxRedraws times, as the review series does.
func drawCompliance(ctx context.Context, client Reviewer, c ComplianceCase, p inference.AssembledPrompt) ComplianceAnswer {
	var a ComplianceAnswer
	for attempt := 0; ; attempt++ {
		a = complianceAnswer(ctx, client, c, p)
		a.Redraws = attempt
		if a.completedCall || attempt == maxRedraws || ctx.Err() != nil {
			return a
		}
	}
}

func complianceAnswer(ctx context.Context, client Reviewer, c ComplianceCase, p inference.AssembledPrompt) ComplianceAnswer {
	started := time.Now()
	res := client.Review(ctx, inference.ReviewRequest{
		Prompt:         p.Prompt,
		PromptTokens:   p.PromptTokens,
		ResponseSchema: inference.ReviewResponseFormat(),
	})
	a := ComplianceAnswer{Outcome: res.Outcome.String(), Seconds: time.Since(started).Seconds()}
	if n := res.Cost.Usage.Input.Total; n != nil {
		a.PromptTokens = *n
	}
	if n := res.Cost.Usage.Output.Total; n != nil {
		a.OutputTokens = *n
	}
	if res.Err != nil {
		a.Err = res.Err.Error()
	}
	if res.Outcome != inference.OutcomeOK || res.Review.ComplianceRequirements == nil {
		return a
	}
	a.completedCall = true
	_, checks := inference.ConfirmCompliance(res.Review.ComplianceRequirements, diff.BuildAnchorIndex(c.reviewFiles()), nil)
	for _, ch := range checks {
		r := ch.Requirement
		if !r.Met {
			a.Unmet++
		}
		a.Requirements = append(a.Requirements, ComplianceRequirementResult{
			Requirement: r.Requirement, Met: r.Met, File: r.File, Quote: r.Quote, Drop: string(ch.Drop)})
	}
	a.Expect = scoreCompliance(c.Expect, checks)
	return a
}

// scoreCompliance gives each expectation its verdict. The first listed
// requirement that matches an expectation is the one scored for it; a
// requirement matching none is not scored.
func scoreCompliance(expect []ComplianceExpectation, checks []inference.ComplianceCheck) []ComplianceExpectResult {
	out := make([]ComplianceExpectResult, 0, len(expect))
	for _, e := range expect {
		r := ComplianceExpectResult{Label: e.Label, Shown: e.Shown}
		hit := matchExpectation(e, expect, checks)
		shown := hit != nil && hit.Drop == ""
		switch {
		case e.Shown && shown:
			r.Verdict = ComplianceShownRight
		case e.Shown && hit != nil && hit.Drop != inference.ComplianceNotMet:
			r.Verdict, r.Drop = ComplianceDropped, string(hit.Drop)
		case e.Shown:
			r.Verdict = ComplianceUnlisted
		case shown:
			r.Verdict = ComplianceFalseShown
		default:
			r.Verdict = ComplianceHeldBack
		}
		out = append(out, r)
	}
	return out
}

// matchExpectation is the requirement scored for e. A requirement carrying
// e's label wins; failing that, one matching a keyword, but never one that
// carries another criterion's label: "AC-1 ... 429" is AC-1's, even though
// AC-2's keywords name 429.
func matchExpectation(e ComplianceExpectation, all []ComplianceExpectation, checks []inference.ComplianceCheck) *inference.ComplianceCheck {
	if e.Label != "" {
		for i := range checks {
			if hasLabel(checks[i].Requirement.Requirement, e.Label) {
				return &checks[i]
			}
		}
	}
	for i := range checks {
		text := checks[i].Requirement.Requirement
		other := false
		for _, o := range all {
			if o.Label != "" && o.Label != e.Label && hasLabel(text, o.Label) {
				other = true
				break
			}
		}
		if other {
			continue
		}
		lower := strings.ToLower(text)
		for _, k := range e.Keywords {
			if strings.Contains(lower, strings.ToLower(k)) {
				return &checks[i]
			}
		}
	}
	return nil
}

// hasLabel reports whether text carries label as a whole token: AK-1, but
// not the AK-1 inside AK-10.
func hasLabel(text, label string) bool {
	text, label = strings.ToLower(text), strings.ToLower(label)
	for from := 0; ; {
		i := strings.Index(text[from:], label)
		if i < 0 {
			return false
		}
		start, end := from+i, from+i+len(label)
		if (start == 0 || !isLabelRune(text[start-1])) && (end == len(text) || !isLabelRune(text[end])) {
			return true
		}
		from = start + 1
	}
}

func isLabelRune(b byte) bool {
	return b >= '0' && b <= '9' || b >= 'a' && b <= 'z'
}

func (a ComplianceAnswer) summary() string {
	if !a.completedCall {
		return "incomplete: " + a.Outcome
	}
	var parts []string
	for _, e := range a.Expect {
		v := e.Label + " " + e.Verdict
		if e.Drop != "" {
			v += " (" + e.Drop + ")"
		}
		parts = append(parts, v)
	}
	return strings.Join(parts, ", ")
}

func (r ComplianceRun) total() ComplianceTotals {
	t := ComplianceTotals{Dropped: map[string]int{}}
	for _, c := range r.Cases {
		for _, a := range c.Answers {
			if !a.completedCall {
				t.Incomplete++
				continue
			}
			t.Unmet += a.Unmet
			for _, e := range a.Expect {
				if e.Shown {
					t.Provable++
				} else {
					t.Unprovable++
				}
				switch e.Verdict {
				case ComplianceShownRight:
					t.Shown++
				case ComplianceDropped:
					t.Dropped[e.Drop]++
				case ComplianceUnlisted:
					t.Unlisted++
				case ComplianceFalseShown:
					t.FalseShown++
				}
			}
		}
	}
	return t
}

// Report prints the totals.
func (r ComplianceRun) Report(w io.Writer) {
	t := r.Totals
	_, _ = fmt.Fprintf(w, "\nprovable requirements shown: %d of %d (unlisted %d, dropped %v)\n",
		t.Shown, t.Provable, t.Unlisted, t.Dropped)
	_, _ = fmt.Fprintf(w, "unprovable requirements shown (wrong): %d of %d\n", t.FalseShown, t.Unprovable)
	_, _ = fmt.Fprintf(w, "requirements the model still called unmet (never shown): %d\n", t.Unmet)
	if t.Incomplete > 0 {
		_, _ = fmt.Fprintf(w, "incomplete answers: %d\n", t.Incomplete)
	}
	_, _ = fmt.Fprintf(w, "%.0fs\n", r.Seconds)
}

// ErrWrong reports a requirement shown that the shown code does not
// implement: the wrong comment.
func (r ComplianceRun) ErrWrong() error {
	if r.Totals.FalseShown > 0 {
		return fmt.Errorf("%d requirement(s) shown that the shown code does not implement", r.Totals.FalseShown)
	}
	return nil
}

// ErrIncomplete reports answers that never came.
func (r ComplianceRun) ErrIncomplete() error {
	if r.Totals.Incomplete > 0 {
		return errors.New(fmt.Sprint(r.Totals.Incomplete, " answer(s) did not complete"))
	}
	return nil
}
