package evals

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"strings"

	"github.com/trick77/noergler/internal/inference"
)

// A confirm run scores the shape the sampled runs point at: several reviews
// unioned, then ONE more call that merges the duplicates (briefs/merge.txt).
// briefs/judge.txt is the variant that may also drop what the code refutes;
// it lost a real bug in 4 of 9 groups and is kept only so that result can
// be reproduced. The sampled runs showed why each half is needed. A vote
// discards the bug a model finds two times in five; a union keeps it but
// posts the same defect twice, because two reviews anchor it on different
// lines and no file-and-line rule can tell.
//
// The consolidation prompt is the review prompt, byte for byte, with the
// consolidation brief and the candidates appended. Appended, never
// substituted: the candidates are model output, and the review prompt is
// then a prefix the endpoint has already cached for the reviews before it.
//
// The answer comes back in the review schema and passes ValidateFindings
// like any review, so a consolidated finding still has to quote the diff.

// Confirm view names, in report order.
const (
	ViewConfirm3 = "confirm3"
)

var confirmNames = []string{ViewSingle, ViewUnion3, ViewConfirm3}

// candidate is one finding as the consolidation pass is shown it.
type candidate struct {
	ID         string   `json:"id"`
	File       string   `json:"file"`
	Line       int      `json:"line"`
	Severity   string   `json:"severity"`
	Headline   string   `json:"headline,omitempty"`
	Comment    string   `json:"comment"`
	Evidence   []string `json:"evidence"`
	Suggestion string   `json:"suggestion,omitempty"`
}

// renderCandidates lists every finding of every review in the group, not a
// merged set: merging is the consolidation pass's job, and pre-merging by
// line would hide exactly the duplicates it is there to resolve.
func renderCandidates(group [][]inference.ReviewFinding) (string, int) {
	var out []candidate
	for r, findings := range group {
		for i, f := range findings {
			c := candidate{
				ID:   fmt.Sprintf("r%d-%d", r+1, i+1),
				File: f.File, Line: f.Line, Severity: f.Severity,
				Comment: f.Comment, Evidence: f.Evidence,
			}
			if f.Headline != nil {
				c.Headline = *f.Headline
			}
			if f.Suggestion != nil {
				c.Suggestion = *f.Suggestion
			}
			out = append(out, c)
		}
	}
	if len(out) == 0 {
		return "", 0
	}
	// MarshalIndent of plain strings and ints cannot fail.
	blob, _ := json.MarshalIndent(out, "", "  ")
	return string(blob), len(out)
}

// ConfirmGroup is one group of reviews and what consolidating them gave.
type ConfirmGroup struct {
	// Samples are the 1-based samples of the sampled run that were unioned.
	Samples    []int
	Candidates int
	// Result is the consolidated review. With fewer than two candidates
	// there is no call: Outcome is "skipped" and the findings are the
	// group's own.
	Result
	Cost    SampleCost
	Redraws int `json:",omitempty"`
}

// ConfirmedCase is one case's groups and the views over them.
type ConfirmedCase struct {
	Case   string
	Seeded int
	Groups []ConfirmGroup
	Views  []CaseView
}

// Confirmed is a whole confirm run.
type Confirmed struct {
	// Finder is the pair that produced the samples, Confirmer the pair that
	// consolidated them. They may differ.
	Finder, Confirmer string
	// Groups is how many groups of three were consolidated per case.
	Groups int
	Seeded int
	Cases  []ConfirmedCase
	Views  []View
	// Tokens is the mean accounting of a consolidation call.
	Tokens struct{ Prompt, Cached, Completion float64 }
}

const outcomeSkipped = "skipped"

// spread picks k of the given groups, evenly spaced, so the choice does not
// lean on the first samples.
func spread(groups [][]int, k int) [][]int {
	if k >= len(groups) {
		return groups
	}
	out := make([][]int, k)
	for i := range out {
		out[i] = groups[i*len(groups)/k]
	}
	return out
}

// RunConfirm consolidates groups of three reviews from a finished sampled
// run and scores the result next to the single review and the plain union
// over the same groups.
//
// suffix is a brief from briefs/. A group with fewer than two candidates
// makes no call: there is nothing to merge, which is what a clean change, or
// one with a single finding, costs.
func RunConfirm(ctx context.Context, client Reviewer, template, suffix string, cases []Case, sampled Sampled, count inference.CountFunc, groups int, progress io.Writer) (Confirmed, error) {
	if err := sampled.ErrIncomplete(); err != nil {
		return Confirmed{}, err
	}
	out := Confirmed{Finder: sampled.label(), Groups: groups}
	var aggs []caseAgg
	var calls float64
	for _, c := range cases {
		sc, ok := sampled.findCase(c.Name)
		if !ok {
			return Confirmed{}, fmt.Errorf("case %s is not in the sampled run", c.Name)
		}
		samples := sampleFindings(sc)
		assembled := assemble(template, c, count)
		cc := ConfirmedCase{Case: c.Name, Seeded: len(c.Expected)}
		single, union, confirm := tally{expected: c.Expected}, tally{expected: c.Expected}, tally{expected: c.Expected}
		for _, s := range samples {
			single.add(s, 1)
		}
		for _, idx := range spread(subsets(len(samples), 3), groups) {
			group := [][]inference.ReviewFinding{samples[idx[0]], samples[idx[1]], samples[idx[2]]}
			union.add(mergeFindings(group, 1), 3)

			g := ConfirmGroup{Samples: []int{idx[0] + 1, idx[1] + 1, idx[2] + 1}}
			rendered, n := renderCandidates(group)
			g.Candidates = n
			// Fewer than two candidates leaves nothing to merge, so there is
			// no call and the group's own findings stand. This is also the
			// lone finding's protection: asked to consolidate a single
			// candidate, a model can only repeat it or lose it, and it lost
			// a real one in the runs that led here.
			if n < 2 {
				kept := mergeFindings(group, 1)
				g.Result = Result{Case: c.Name, Outcome: outcomeSkipped, Findings: kept}
				g.Matches, g.Extra, _ = match(c.Expected, kept)
				confirm.add(kept, 3)
				cc.Groups = append(cc.Groups, g)
				continue
			}
			prompt := assembled
			prompt.Prompt += suffix + rendered + "\n"
			prompt.PromptTokens += count(suffix + rendered)
			res, cost := reviewCase(ctx, client, c, prompt)
			for !completed(res) && g.Redraws < maxRedraws && ctx.Err() == nil {
				g.Redraws++
				res, cost = reviewCase(ctx, client, c, prompt)
			}
			g.Result = res
			g.Cost = SampleCost{
				PromptTokens: cost.PromptTokens, CachedTokens: cost.CachedTokens,
				CompletionTokens: cost.CompletionTokens, NanoUSD: cost.NanoUSD,
			}
			out.Tokens.Prompt += float64(cost.PromptTokens)
			out.Tokens.Cached += float64(cost.CachedTokens)
			out.Tokens.Completion += float64(cost.CompletionTokens)
			calls++
			confirm.add(res.Findings, 4)
			cc.Groups = append(cc.Groups, g)
		}
		cc.Views = []CaseView{single.view(ViewSingle), union.view(ViewUnion3), confirm.view(ViewConfirm3)}
		out.Cases = append(out.Cases, cc)
		out.Seeded += cc.Seeded
		// Cost is left unpriced: the consolidation call and the reviews it
		// merges may be on different models, and the samples were drawn at
		// another time, so no single multiple describes the group.
		aggs = append(aggs, caseAgg{seeded: cc.Seeded, views: cc.Views})
		if progress != nil {
			v := cc.Views[2]
			_, _ = fmt.Fprintf(progress, "%-26s %d group(s), caught %.2f, extra %.2f\n", c.Name, len(cc.Groups), v.Caught, v.Extra)
		}
	}
	if calls > 0 {
		out.Tokens.Prompt, out.Tokens.Cached, out.Tokens.Completion = out.Tokens.Prompt/calls, out.Tokens.Cached/calls, out.Tokens.Completion/calls
	}
	out.Views = aggregate(confirmNames, aggs)
	return out, nil
}

// ErrIncomplete reports consolidation calls that did not produce a review.
func (c Confirmed) ErrIncomplete() error {
	var broken []string
	for _, cc := range c.Cases {
		for _, g := range cc.Groups {
			if g.Outcome == outcomeSkipped || completed(g.Result) {
				continue
			}
			why := g.ErrMsg
			if why == "" {
				why = "outcome " + g.Outcome
			}
			broken = append(broken, fmt.Sprintf("%s%v: %s", cc.Case, g.Samples, why))
		}
	}
	if len(broken) == 0 {
		return nil
	}
	return fmt.Errorf("%d consolidation call(s) did not complete: %s", len(broken), strings.Join(broken, "; "))
}

// Report renders a confirm run for a terminal. Write errors are dropped, as
// in Score.Report.
func (c Confirmed) Report(w io.Writer) {
	p := func(format string, args ...any) { _, _ = fmt.Fprintf(w, format, args...) }
	p("reviews by %s, consolidated by %s, %d group(s) of 3 per case\n", c.Finder, c.Confirmer, c.Groups)
	rows := make([]viewRow, len(c.Cases))
	for i, cc := range c.Cases {
		rows[i] = viewRow{name: cc.Case, views: cc.Views}
	}
	reportViews(w, rows, c.Views, c.Seeded)
	p("\ntokens per consolidation call: prompt %.0f (cached %.0f) completion %.0f\n",
		c.Tokens.Prompt, c.Tokens.Cached, c.Tokens.Completion)
}
