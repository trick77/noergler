package evals

import (
	"fmt"
	"io"

	"github.com/trick77/noergler/internal/inference"
)

// A mix scores a vote whose members are not the same configuration: one
// review from a primary run and the rest from a secondary one, typically the
// production level once and a cheaper level or model for the other votes. It
// makes no calls. Two finished sampled runs already hold every finding, so
// the mixed groups are formed from their samples.

// Mixed view names, in report order.
const (
	ViewPrimary     = "primary"
	ViewMixVote     = "mix-vote3"
	ViewMixAdaptive = "mix-adaptive"
)

var mixNames = []string{ViewPrimary, ViewMixVote, ViewMixAdaptive}

// MixedCase is one case's mixed views.
type MixedCase struct {
	Case   string
	Seeded int
	Views  []CaseView
}

// Mixed is two sampled runs scored as one mixed vote.
type Mixed struct {
	// Primary and Secondary name the two runs' pairs.
	Primary, Secondary string
	Seeded             int
	Cases              []MixedCase
	Views              []View
}

// Mix scores one primary review voted with secondary ones, over every way to
// draw them from the two runs:
//
//   - primary: the primary review alone, the reference every cost is against
//   - mix-vote3: one primary and two secondary reviews, kept by 2 of 3
//   - mix-adaptive: one primary and one secondary; a second secondary only
//     when those two disagree
//
// Cost is the primary's cold call plus the secondary's calls. On the same
// model those are priced warm: reasoning level is not part of the prompt, so
// the primary's call has written the prefix cache they read. That is an
// assumption the data cannot check, since no run here made a mixed call. On
// another model the first secondary call is cold.
//
// Both runs must be complete and cover the corpus given: a mix over unequal
// groups, or over a case one run never scored, means nothing.
func Mix(cases []Case, primary, secondary Sampled) (Mixed, error) {
	for _, r := range []Sampled{primary, secondary} {
		if err := r.ErrIncomplete(); err != nil {
			return Mixed{}, fmt.Errorf("%s: %w", r.label(), err)
		}
	}
	sameModel := primary.Model != "" && primary.Model == secondary.Model
	out := Mixed{Primary: primary.label(), Secondary: secondary.label()}
	var aggs []caseAgg
	for _, c := range cases {
		p, okP := primary.findCase(c.Name)
		s, okS := secondary.findCase(c.Name)
		if !okP || !okS {
			return Mixed{}, fmt.Errorf("case %s is not in both runs", c.Name)
		}
		if len(s.Samples) < 2 {
			return Mixed{}, fmt.Errorf("case %s: the secondary run needs at least 2 samples", c.Name)
		}
		views := mixViews(c.Expected, sampleFindings(p), sampleFindings(s))
		out.Cases = append(out.Cases, MixedCase{Case: c.Name, Seeded: len(c.Expected), Views: views})
		out.Seeded += len(c.Expected)

		coldP, _, pricedP := caseRates(p.Samples)
		coldS, warmS, pricedS := caseRates(s.Samples)
		aggs = append(aggs, caseAgg{
			seeded: len(c.Expected), views: views, priced: pricedP && pricedS, base: coldP,
			cost: func(calls float64) float64 {
				switch {
				case calls <= 1:
					return coldP
				case sameModel:
					return coldP + (calls-1)*warmS
				default:
					return coldP + coldS + (calls-2)*warmS
				}
			},
		})
	}
	out.Views = aggregate(mixNames, aggs)
	return out, nil
}

func (s Sampled) findCase(name string) (SampledCase, bool) {
	for _, c := range s.Cases {
		if c.Case == name {
			return c, true
		}
	}
	return SampledCase{}, false
}

func sampleFindings(c SampledCase) [][]inference.ReviewFinding {
	out := make([][]inference.ReviewFinding, len(c.Samples))
	for i, s := range c.Samples {
		out[i] = s.Findings
	}
	return out
}

// mixViews scores the mixed views on one case, in mixNames order.
func mixViews(expected []Expected, primary, secondary [][]inference.ReviewFinding) []CaseView {
	alone, vote, adapt := tally{expected: expected}, tally{expected: expected}, tally{expected: expected}
	for _, p := range primary {
		alone.add(p, 1)
		for _, pair := range subsets(len(secondary), 2) {
			a, b := secondary[pair[0]], secondary[pair[1]]
			vote.add(mergeFindings([][]inference.ReviewFinding{p, a, b}, 2), 3)
			// Which secondary review comes first decides whether a third is
			// needed, so both orders are groups.
			for _, o := range [][2][]inference.ReviewFinding{{a, b}, {b, a}} {
				merged, calls := adaptive(p, o[0], o[1])
				adapt.add(merged, calls)
			}
		}
	}
	return []CaseView{alone.view(ViewPrimary), vote.view(ViewMixVote), adapt.view(ViewMixAdaptive)}
}

// Report renders a mix for a terminal. Write errors are dropped, as in
// Score.Report.
func (m Mixed) Report(w io.Writer) {
	_, _ = fmt.Fprintf(w, "primary %s, secondary %s, no calls made\n", m.Primary, m.Secondary)
	rows := make([]viewRow, len(m.Cases))
	for i, c := range m.Cases {
		rows[i] = viewRow{name: c.Case, views: c.Views}
	}
	reportViews(w, rows, m.Views, m.Seeded)
}
