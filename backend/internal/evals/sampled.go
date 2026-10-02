package evals

import (
	"context"
	"fmt"
	"io"
	"sort"
	"strings"
	"sync"

	"github.com/trick77/noergler/internal/inference"
)

// A sampled run reviews every case several times with the identical prompt
// and scores what merging those reviews would have posted. It answers one
// question the series cannot: the model's misses and inventions vary from run
// to run on an unchanged prompt, so would N reviews merged by a fixed rule
// beat one, and at what price.
//
// Nothing here is a series data point. A sampled report has its own shape and
// its own results directory; the single-run series stays as it is.

// mergeSlack is how far apart two samples may anchor the same finding. It is
// the validator's anchorSlack: findings arrive here already validated, so two
// reports of one bug sit within that distance of the same evidence.
const mergeSlack = 2

// maxRedraws is how often a review that did not complete is drawn again, in
// a sampled run and in a series run alike. A sampled run makes five times
// the calls of a series run, so one transport failure would void nearly
// every run; on the 17-case corpus a series run rarely got through either,
// and each void one is a committed row that scores nothing. Production never
// retries; this is the eval tool filling a hole in its own data, and it has
// a cost: a call cut off for thinking too long is replaced by one that
// thought less, so redrawn results lean towards the quick answer. Each
// result records its count.
const maxRedraws = 2

// MinSamples is the fewest samples a sampled run takes: the vote needs three.
const MinSamples = 3

// View names, in report order.
const (
	ViewSingle   = "single"
	ViewAgree2   = "agree2"
	ViewEither2  = "either2"
	ViewVote3    = "vote3"
	ViewTiered3  = "tiered3"
	ViewUnion3   = "union3"
	ViewAdaptive = "adaptive"
)

var viewNames = []string{ViewSingle, ViewAgree2, ViewEither2, ViewVote3, ViewTiered3, ViewUnion3, ViewAdaptive}

// cluster is one finding as several samples reported it.
type cluster struct {
	rep   inference.ReviewFinding
	votes int
	last  int
}

// clusterFindings groups findings across samples. Two findings are the same
// when they are on the same file within mergeSlack lines: file and line only,
// no text comparison and no model, blunt like the scorer. The cost is known:
// two different bugs within mergeSlack lines collapse into one.
//
// A sample votes once per cluster, so a model repeating itself inside one
// review does not out-vote the others.
func clusterFindings(samples [][]inference.ReviewFinding) []cluster {
	var out []cluster
	for s, findings := range samples {
		sorted := append([]inference.ReviewFinding(nil), findings...)
		sort.SliceStable(sorted, func(i, j int) bool {
			if sorted[i].File != sorted[j].File {
				return sorted[i].File < sorted[j].File
			}
			return sorted[i].Line < sorted[j].Line
		})
	next:
		for _, f := range sorted {
			for i := range out {
				c := &out[i]
				d := c.rep.Line - f.Line
				if samePath(c.rep.File, f.File) && d >= -mergeSlack && d <= mergeSlack {
					if c.last != s {
						c.votes, c.last = c.votes+1, s
					}
					continue next
				}
			}
			out = append(out, cluster{rep: f, votes: 1, last: s})
		}
	}
	return out
}

// mergeFindings keeps what at least quorum samples reported, represented by
// the earliest sample's wording. Quorum 1 is the union.
func mergeFindings(samples [][]inference.ReviewFinding, quorum int) []inference.ReviewFinding {
	var out []inference.ReviewFinding
	for _, c := range clusterFindings(samples) {
		if c.votes >= quorum {
			out = append(out, c.rep)
		}
	}
	return out
}

// mergeTiered is the vote with one exception: a finding only one sample
// reported is kept when that sample rated it an issue. A vote alone discards
// the bug a model finds one time in three, which is the subtle one; this
// keeps it at the price of trusting the model's own severity.
func mergeTiered(samples [][]inference.ReviewFinding) []inference.ReviewFinding {
	var out []inference.ReviewFinding
	for _, c := range clusterFindings(samples) {
		if c.votes >= 2 || c.rep.Severity == "issue" {
			out = append(out, c.rep)
		}
	}
	return out
}

// adaptive is two samples and a third only as a tie-breaker: when the two
// agree on everything, both empty included, their findings stand; otherwise
// the third decides by 2-of-3. It reports how many calls that took.
func adaptive(a, b, c []inference.ReviewFinding) ([]inference.ReviewFinding, int) {
	pair := [][]inference.ReviewFinding{a, b}
	for _, cl := range clusterFindings(pair) {
		if cl.votes < 2 {
			return mergeFindings(append(pair, c), 2), 3
		}
	}
	return mergeFindings(pair, 2), 2
}

// subsets is every k-subset of 0..n-1, in lexical order.
func subsets(n, k int) [][]int {
	var out [][]int
	idx := make([]int, k)
	var rec func(start, depth int)
	rec = func(start, depth int) {
		if depth == k {
			out = append(out, append([]int(nil), idx...))
			return
		}
		for i := start; i < n; i++ {
			idx[depth] = i
			rec(i+1, depth+1)
		}
	}
	rec(0, 0)
	return out
}

// SampleCost is one call's accounting. NanoUSD is nil when the gateway did
// not price the call; it must never read as zero.
type SampleCost struct {
	PromptTokens     int64
	CachedTokens     int64
	CompletionTokens int64
	NanoUSD          *int64
}

// Sample is one review of one case.
type Sample struct {
	Result
	Cost SampleCost
}

// CaseView is one view's score on one case, averaged over every group of
// samples the view can be formed from.
type CaseView struct {
	Name string
	// Caught and Extra are means per group. Full is the share of groups that
	// caught every seeded bug, Clean the share that reported nothing extra.
	Caught, Extra float64
	Full, Clean   float64
	// Calls is the mean number of reviews a group used.
	Calls float64
}

// Agreement counts findings by how many of a case's samples reported them,
// over the whole corpus. It is the evidence for or against a vote: Real in
// the Votes 1 row is what a vote would discard, Invented there is what a
// union would post.
type Agreement struct {
	// Votes is how many samples reported the finding.
	Votes int
	// Real findings match a seeded bug, Invented ones match none. The Issue
	// counts are the subset the model rated severity issue.
	Real, RealIssue         int
	Invented, InventedIssue int
}

// agreement tallies one case's clusters into rows indexed by votes-1.
func agreement(rows []Agreement, expected []Expected, samples [][]inference.ReviewFinding) {
	for _, c := range clusterFindings(samples) {
		r := &rows[c.votes-1]
		issue := c.rep.Severity == "issue"
		matches, _, _ := match(expected, []inference.ReviewFinding{c.rep})
		isReal := false
		for _, m := range matches {
			isReal = isReal || m.Found
		}
		switch {
		case isReal:
			r.Real++
			if issue {
				r.RealIssue++
			}
		default:
			r.Invented++
			if issue {
				r.InventedIssue++
			}
		}
	}
}

// SampledCase is one case's samples and what each view makes of them.
type SampledCase struct {
	Case    string
	Seeded  int
	Samples []Sample
	Views   []CaseView
}

// View is one merge rule scored over the whole corpus.
type View struct {
	Name string
	// Calls is the mean reviews per case.
	Calls float64
	// Caught and Extra are what one run is expected to score: per-case means
	// summed over cases.
	Caught, Extra float64
	// AllCaught is the chance one run catches every seeded bug, ControlsClean
	// the chance it reports nothing on any clean control, taking cases as
	// independent. Their product is the chance of exit 0.
	AllCaught, ControlsClean float64
	// CostMultiple is the view's cost against one cold review. Nil when any
	// sample went unpriced: an unpriced run has no cost verdict.
	CostMultiple *float64
}

// TokenMeans is the mean accounting per call. Cold is a case's first sample,
// the one that writes the endpoint's prefix cache; warm is every later one.
// WarmCached next to WarmPrompt shows whether the cache was hit at all.
type TokenMeans struct {
	ColdPrompt, ColdCached, ColdCompletion float64
	WarmPrompt, WarmCached, WarmCompletion float64
}

// Sampled is a whole sampled run.
type Sampled struct {
	// Model and Effort are the pair the run scored, recorded by the caller.
	// Mix reads Model to tell whether two runs share a prefix cache.
	Model   string `json:",omitempty"`
	Effort  string `json:",omitempty"`
	Samples int
	Seeded  int
	Cases   []SampledCase
	Views   []View
	// Agreement has one row per vote count, 1 to Samples.
	Agreement []Agreement
	Tokens    TokenMeans
}

// RunSampled reviews every case samples times and scores every view.
//
// A case's first sample runs alone and completes before the rest start, with
// up to parallel of those in flight. That is the order a production vote
// would use: identical calls sent together all miss the prefix cache, since
// none has written it yet, and the measured cost would be one no deployment
// pays. Cases stay sequential, as in Run.
//
// progress, when not nil, gets one line per finished case: a sampled run is
// long, and silence until the end hides a gateway that has stopped answering.
func RunSampled(ctx context.Context, client Reviewer, template string, cases []Case, count inference.CountFunc, samples, parallel int, progress io.Writer) Sampled {
	out := Sampled{Samples: samples}
	for _, c := range cases {
		assembled := assemble(template, c, count)
		sc := SampledCase{Case: c.Name, Seeded: len(c.Expected), Samples: make([]Sample, samples)}
		draw := func(i int) {
			res, cost := drawCase(ctx, client, c, assembled)
			sc.Samples[i] = Sample{Result: res, Cost: SampleCost{
				PromptTokens:     cost.PromptTokens,
				CachedTokens:     cost.CachedTokens,
				CompletionTokens: cost.CompletionTokens,
				NanoUSD:          cost.NanoUSD,
			}}
		}
		draw(0)
		var wg sync.WaitGroup
		slots := make(chan struct{}, parallel)
		for i := 1; i < samples; i++ {
			wg.Add(1)
			slots <- struct{}{}
			go func() {
				defer wg.Done()
				defer func() { <-slots }()
				draw(i)
			}()
		}
		wg.Wait()

		out.Cases = append(out.Cases, sc)
		if progress != nil {
			ok, redrawn := 0, 0
			for _, sm := range sc.Samples {
				redrawn += sm.Redraws
				if completed(sm.Result) {
					ok++
				}
			}
			_, _ = fmt.Fprintf(progress, "%-26s %d/%d sample(s) ok, %d redraw(s)\n", c.Name, ok, samples, redrawn)
		}
	}
	// The corpus that was just sampled always covers its own cases.
	_ = out.Rescore(cases)
	return out
}

// Rescore recomputes every view from the samples the run holds. The samples
// are the paid-for part; a merge rule added later is scored on them without
// a single new call.
func (s *Sampled) Rescore(cases []Case) error {
	s.Seeded = 0
	s.Agreement = make([]Agreement, s.Samples)
	for i := range s.Agreement {
		s.Agreement[i].Votes = i + 1
	}
	for ci := range s.Cases {
		sc := &s.Cases[ci]
		var expected []Expected
		found := false
		for _, c := range cases {
			if c.Name == sc.Case {
				expected, found = c.Expected, true
			}
		}
		if !found {
			return fmt.Errorf("case %s is not in the corpus", sc.Case)
		}
		if len(sc.Samples) != s.Samples {
			return fmt.Errorf("case %s has %d sample(s), the run says %d", sc.Case, len(sc.Samples), s.Samples)
		}
		findings := sampleFindings(*sc)
		sc.Seeded = len(expected)
		sc.Views = caseViews(expected, findings)
		agreement(s.Agreement, expected, findings)
		s.Seeded += sc.Seeded
	}
	s.summarize()
	return nil
}

// completed reports whether a review produced a usable result.
func completed(r Result) bool {
	return r.ErrMsg == "" && r.Outcome == inference.OutcomeOK.String()
}

// caseViews scores every view on one case's samples, in viewNames order.
func caseViews(expected []Expected, samples [][]inference.ReviewFinding) []CaseView {
	n := len(samples)
	pick := func(idx []int) [][]inference.ReviewFinding {
		g := make([][]inference.ReviewFinding, len(idx))
		for i, j := range idx {
			g[i] = samples[j]
		}
		return g
	}
	fixed := []struct {
		name         string
		size, quorum int
	}{
		{ViewSingle, 1, 1}, {ViewAgree2, 2, 2}, {ViewEither2, 2, 1},
		{ViewVote3, 3, 2}, {ViewUnion3, 3, 1},
	}
	var views []CaseView
	for _, d := range fixed {
		acc := tally{expected: expected}
		for _, idx := range subsets(n, d.size) {
			acc.add(mergeFindings(pick(idx), d.quorum), d.size)
		}
		views = append(views, acc.view(d.name))
	}
	// Tiered sits after the vote in viewNames, before the union.
	tiered := tally{expected: expected}
	for _, idx := range subsets(n, 3) {
		tiered.add(mergeTiered(pick(idx)), 3)
	}
	views = append(views[:4], append([]CaseView{tiered.view(ViewTiered3)}, views[4:]...)...)

	// Adaptive depends on which two samples come first, so it is scored over
	// every ordered triple, not every subset.
	acc := tally{expected: expected}
	for a := 0; a < n; a++ {
		for b := 0; b < n; b++ {
			for c := 0; c < n; c++ {
				if a == b || a == c || b == c {
					continue
				}
				merged, calls := adaptive(samples[a], samples[b], samples[c])
				acc.add(merged, calls)
			}
		}
	}
	return append(views, acc.view(ViewAdaptive))
}

// tally accumulates one view's groups on one case.
type tally struct {
	expected                           []Expected
	groups, caught, extra, full, clean int
	calls                              int
}

func (t *tally) add(findings []inference.ReviewFinding, calls int) {
	matches, extra, _ := match(t.expected, findings)
	found := 0
	for _, m := range matches {
		if m.Found {
			found++
		}
	}
	t.groups++
	t.caught += found
	t.extra += extra
	t.calls += calls
	if found == len(t.expected) {
		t.full++
	}
	if extra == 0 {
		t.clean++
	}
}

func (t *tally) view(name string) CaseView {
	g := float64(t.groups)
	return CaseView{
		Name:   name,
		Caught: float64(t.caught) / g, Extra: float64(t.extra) / g,
		Full: float64(t.full) / g, Clean: float64(t.clean) / g,
		Calls: float64(t.calls) / g,
	}
}

// caseRates is what one call on a case costs: cold for the first sample,
// which pays full input, warm for the mean of the later ones, which read the
// cache it wrote. Not priced when any sample is not.
func caseRates(samples []Sample) (cold, warm float64, priced bool) {
	for i, sm := range samples {
		if sm.Cost.NanoUSD == nil {
			return 0, 0, false
		}
		if i == 0 {
			cold = float64(*sm.Cost.NanoUSD)
			continue
		}
		warm += float64(*sm.Cost.NanoUSD) / float64(len(samples)-1)
	}
	return cold, warm, len(samples) > 1
}

// caseAgg is one case as aggregate reads it: its views, and what a group of
// that many calls costs against base, the one review it is compared with.
type caseAgg struct {
	seeded int
	views  []CaseView
	priced bool
	base   float64
	cost   func(calls float64) float64
}

// aggregate turns per-case views into corpus-wide ones.
func aggregate(names []string, cases []caseAgg) []View {
	priced := len(cases) > 0
	for _, c := range cases {
		priced = priced && c.priced
	}
	var out []View
	for vi, name := range names {
		v := View{Name: name, AllCaught: 1, ControlsClean: 1}
		var cost, base float64
		for _, c := range cases {
			cv := c.views[vi]
			v.Calls += cv.Calls / float64(len(cases))
			v.Caught += cv.Caught
			v.Extra += cv.Extra
			if c.seeded > 0 {
				v.AllCaught *= cv.Full
			} else {
				v.ControlsClean *= cv.Clean
			}
			if priced {
				cost += c.cost(cv.Calls)
				base += c.base
			}
		}
		if priced && base > 0 {
			m := cost / base
			v.CostMultiple = &m
		}
		out = append(out, v)
	}
	return out
}

// summarize fills Views and Tokens from the per-case numbers.
func (s *Sampled) summarize() {
	var coldN, warmN float64
	t := &s.Tokens
	*t = TokenMeans{}
	var cases []caseAgg
	for _, c := range s.Cases {
		for i, sm := range c.Samples {
			p, ca, co := float64(sm.Cost.PromptTokens), float64(sm.Cost.CachedTokens), float64(sm.Cost.CompletionTokens)
			if i == 0 {
				t.ColdPrompt, t.ColdCached, t.ColdCompletion = t.ColdPrompt+p, t.ColdCached+ca, t.ColdCompletion+co
				coldN++
				continue
			}
			t.WarmPrompt, t.WarmCached, t.WarmCompletion = t.WarmPrompt+p, t.WarmCached+ca, t.WarmCompletion+co
			warmN++
		}
		// A view's cost on a case is one cold call plus (calls-1) warm ones.
		// Summing the members of each group instead would price most groups
		// as all-warm.
		cold, warm, priced := caseRates(c.Samples)
		cases = append(cases, caseAgg{
			seeded: c.Seeded, views: c.Views, priced: priced, base: cold,
			cost: func(calls float64) float64 { return cold + (calls-1)*warm },
		})
	}
	if coldN > 0 {
		t.ColdPrompt, t.ColdCached, t.ColdCompletion = t.ColdPrompt/coldN, t.ColdCached/coldN, t.ColdCompletion/coldN
	}
	if warmN > 0 {
		t.WarmPrompt, t.WarmCached, t.WarmCompletion = t.WarmPrompt/warmN, t.WarmCached/warmN, t.WarmCompletion/warmN
	}
	s.Views = aggregate(viewNames, cases)
}

// view returns the named view. The names are fixed, so a miss is a defect.
func (s Sampled) view(name string) View { return findView(s.Views, name) }

func findView(views []View, name string) View {
	for _, v := range views {
		if v.Name == name {
			return v
		}
	}
	panic("evals: no view " + name)
}

// ErrIncomplete reports samples that did not produce a usable review. One
// missing sample leaves a case's groups unequal, so the views mean nothing.
func (s Sampled) ErrIncomplete() error {
	var broken []string
	for _, c := range s.Cases {
		for i, sm := range c.Samples {
			switch {
			case sm.ErrMsg != "":
				broken = append(broken, fmt.Sprintf("%s#%d: %s", c.Case, i+1, sm.ErrMsg))
			case sm.Outcome != inference.OutcomeOK.String():
				broken = append(broken, fmt.Sprintf("%s#%d: outcome %s", c.Case, i+1, sm.Outcome))
			}
		}
	}
	if len(broken) == 0 {
		return nil
	}
	return fmt.Errorf("%d sample(s) did not complete, so the views mean nothing: %s",
		len(broken), strings.Join(broken, "; "))
}

// ErrRegressed judges the run by the 2-of-3 vote, the view the experiment is
// about: some triple missed a seeded bug, or invented on a clean control.
func (s Sampled) ErrRegressed() error {
	v := s.view(ViewVote3)
	var what []string
	if v.AllCaught < 1 {
		what = append(what, "misses a seeded bug")
	}
	if v.ControlsClean < 1 {
		what = append(what, "invents on a clean control")
	}
	if len(what) == 0 {
		return nil
	}
	return fmt.Errorf("the %s view %s in at least one group", ViewVote3, strings.Join(what, " and "))
}

// Report renders a sampled run for a terminal. Write errors are dropped, as
// in Score.Report.
func (s Sampled) Report(w io.Writer) {
	p := func(format string, args ...any) { _, _ = fmt.Fprintf(w, format, args...) }
	p("%s, %d sample(s) per case\n", s.label(), s.Samples)
	rows := make([]viewRow, len(s.Cases))
	for i, c := range s.Cases {
		ok := 0
		for _, sm := range c.Samples {
			if completed(sm.Result) {
				ok++
			}
		}
		rows[i] = viewRow{name: c.Case, views: c.Views}
		if ok < len(c.Samples) {
			rows[i].note = fmt.Sprintf(" INCOMPLETE %d/%d", ok, len(c.Samples))
		}
	}
	reportViews(w, rows, s.Views, s.Seeded)
	p("\nfindings by how many of %d samples reported them (issue-rated in brackets)\n%-6s %12s %12s\n",
		s.Samples, "votes", "real", "invented")
	for _, a := range s.Agreement {
		p("%-6d %12s %12s\n", a.Votes,
			fmt.Sprintf("%d (%d)", a.Real, a.RealIssue), fmt.Sprintf("%d (%d)", a.Invented, a.InventedIssue))
	}
	t := s.Tokens
	p("\ntokens per call: cold prompt %.0f (cached %.0f) completion %.0f; warm prompt %.0f (cached %.0f) completion %.0f\n",
		t.ColdPrompt, t.ColdCached, t.ColdCompletion, t.WarmPrompt, t.WarmCached, t.WarmCompletion)
}

// label names the pair the run scored, when the caller recorded one.
func (s Sampled) label() string {
	if s.Model == "" {
		return "unlabelled run"
	}
	return s.Model + " (" + s.Effort + ")"
}

type viewRow struct {
	name  string
	views []CaseView
	note  string
}

// reportViews prints the per-case table and the corpus-wide one.
func reportViews(w io.Writer, rows []viewRow, views []View, seeded int) {
	p := func(format string, args ...any) { _, _ = fmt.Fprintf(w, format, args...) }
	p("mean caught/extra per group\n\n%-26s", "case")
	for _, v := range views {
		p(" %-12s", v.Name)
	}
	p("\n")
	for _, r := range rows {
		p("%-26s", r.name)
		for _, v := range r.views {
			p(" %-12s", fmt.Sprintf("%.2f/%.2f", v.Caught, v.Extra))
		}
		p("%s\n", r.note)
	}
	p("\n%-12s %6s %12s %7s %11s %15s %8s\n",
		"view", "calls", "caught", "extra", "all-caught", "controls-clean", "cost")
	for _, v := range views {
		cost := "unpriced"
		if v.CostMultiple != nil {
			cost = fmt.Sprintf("%.2fx", *v.CostMultiple)
		}
		p("%-12s %6.2f %12s %7.2f %11.2f %15.2f %8s\n", v.Name, v.Calls,
			fmt.Sprintf("%.2f of %d", v.Caught, seeded), v.Extra, v.AllCaught, v.ControlsClean, cost)
	}
}
