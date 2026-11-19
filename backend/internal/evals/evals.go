// Package evals scores the review prompt against a corpus of diffs with
// known bugs.
//
// The unit tests pin prompt ASSEMBLY: that placeholders substitute in one
// pass, that file order is content-independent, that a hostile file cannot
// rewrite a section. None of them can tell whether the prompt actually
// produces a good review, because none of them calls a model. That is what
// this package measures.
//
// A case is a diff plus the findings a competent reviewer must report.
// Scoring is deliberately blunt: a seeded bug counts as found when a finding
// lands on the right file, within a line window, mentioning one of the
// keywords. Anything else on a case is noise. Blunt beats a judge model here
// - the score is a regression signal for prompt edits, not a grade, and a
// judge would add a second non-deterministic thing to explain when the number
// moves.
//
// The corpus carries a case with NO expected findings on purpose. Without it,
// "find every seeded bug" is trivially satisfied by reporting everything.
package evals

import (
	"context"
	"embed"
	"encoding/json"
	"fmt"
	"io"
	"io/fs"
	"slices"
	"sort"
	"strconv"
	"strings"
	"time"

	"go.yaml.in/yaml/v3"

	"github.com/trick77/noergler/internal/config"
	"github.com/trick77/noergler/internal/diff"
	"github.com/trick77/noergler/internal/inference"
	"github.com/trick77/noergler/internal/lang"
)

//go:embed corpus/*.json
var corpusFS embed.FS

// seriesYAML is embedded on its own: a missing file fails the build, so
// there is no read error to handle.
//
//go:embed corpus/series.yaml
var seriesYAML []byte

// Series is the model and level a run scores at when EVAL_MODEL and -effort
// leave them unset. Data beside the corpus, not code: no model id lives in Go,
// and changing the pair is a visible edit that starts a new series.
type Series struct {
	Model  string `yaml:"model"`
	Effort string `yaml:"effort"`
}

// DefaultSeries is corpus/series.yaml. The file is embedded and pinned by
// TestDefaultSeries_NamesBothFields, so a malformed one is a build defect,
// not a run-time condition: it panics rather than threading an error no
// operator could act on through every caller.
func DefaultSeries() Series { return mustSeries(seriesYAML) }

func mustSeries(raw []byte) Series {
	s, err := parseSeries(raw)
	if err != nil {
		panic(err)
	}
	return s
}

// parseSeries requires both fields: a half-filled file would default the
// model but not its level, and a run at the wrong level reads as a prompt
// regression.
func parseSeries(raw []byte) (Series, error) {
	var s Series
	if err := yaml.Unmarshal(raw, &s); err != nil {
		return Series{}, fmt.Errorf("corpus/series.yaml: %w", err)
	}
	s.Model, s.Effort = strings.TrimSpace(s.Model), strings.TrimSpace(s.Effort)
	if s.Model == "" || s.Effort == "" {
		return Series{}, fmt.Errorf("corpus/series.yaml: model and effort are both required")
	}
	return s, nil
}

// EffortFor is the level a run on model uses. A passed -effort always wins,
// empty included (the model's balanced level). Unset, the series level
// applies to the series model only: another model's levels are its own.
func (s Series) EffortFor(model, flagValue string, flagSet bool) string {
	if flagSet || model != s.Model {
		return flagValue
	}
	return s.Effort
}

// Expected is one bug a reviewer must report.
type Expected struct {
	ID   string `json:"id"`
	File string `json:"file"`
	// Lines is the inclusive window the finding must land in. A reviewer may
	// reasonably anchor a bug a line or two either side of its cause, so the
	// window is the case author's call, not a single line.
	Lines [2]int `json:"lines"`
	// Keywords: the finding must mention at least one, case-insensitively.
	// This is what stops an unrelated remark on the right line from scoring.
	Keywords []string `json:"keywords"`
	Why      string   `json:"why"`
	// Blocking is the rule id (B<n>, from the case's agents_md) the
	// finding must carry and production must accept as a task
	// (inference.CheckBlocking). Empty: the finding may be anything but a
	// task, which counts under FalseTasks.
	Blocking string `json:"blocking,omitempty"`
}

// CorpusFile is the on-disk shape of one case.
type CorpusFile struct {
	Name        string `json:"name"`
	Description string `json:"description"`
	// AgentsMD is the repo's AGENTS.md, empty for none: its [blocking]
	// rules go into the prompt as production puts them.
	AgentsMD string `json:"agents_md"`
	Files    []struct {
		Path           string `json:"path"`
		Diff           string `json:"diff"`
		Content        string `json:"content"`
		ContentFetched bool   `json:"content_fetched"`
	} `json:"files"`
	Expected []Expected `json:"expected"`
	// Screenshots are the test baselines the case changes, sent as
	// production sends them; ExpectedVisual the lines they must get.
	Screenshots    []corpusScreenshot `json:"screenshots"`
	ExpectedVisual []ExpectedVisual   `json:"expected_visual"`
	// Incremental makes the case a later push on a PR an earlier review
	// already saw: Files are that push, the block holds the rest.
	Incremental *corpusIncremental `json:"incremental"`
	// ExpectedFixed are the earlier findings the push fixes: each must be
	// claimed under `fixed` and kept by production's ValidateFixed.
	ExpectedFixed []ExpectedFixed `json:"expected_fixed"`
	// Repeats are earlier findings the push must not raise again, each as
	// a window and keywords on the push's version of the file.
	Repeats []Expected `json:"repeats"`
}

// corpusIncremental is what production adds to the prompt on a push after
// the first review: the PR's cumulative diff, the findings already posted
// and the overview the last review wrote.
type corpusIncremental struct {
	// CumulativeDiff is the whole PR as it stands, a multi-file diff with
	// `diff --git` headers, as production fetches it.
	CumulativeDiff   string         `json:"cumulative_diff"`
	PreviousOverview string         `json:"previous_overview"`
	PreviousFindings []corpusPosted `json:"previous_findings"`
}

// corpusPosted is one earlier finding. Line is where it was posted, on the
// version the earlier review saw: a push that shifts code does not move it.
type corpusPosted struct {
	ID       int64  `json:"id"`
	File     string `json:"file"`
	Line     int    `json:"line"`
	Severity string `json:"severity"`
	Comment  string `json:"comment"`
	// Dispute is the PR author's reply disputing it; Closed its thread
	// resolved on the PR.
	Dispute string `json:"dispute"`
	Closed  bool   `json:"closed"`
}

// ExpectedFixed is an earlier finding, by its prompt id (`F<n>`), the push
// must be found to fix.
type ExpectedFixed struct {
	ID  string `json:"id"`
	Why string `json:"why"`
}

// Case is a loaded corpus entry.
type Case struct {
	Name           string
	Description    string
	Files          []diff.FileReviewData
	Expected       []Expected
	ExpectedVisual []ExpectedVisual
	// AgentsMD and its blocking rules, read once at load.
	AgentsMD      string
	BlockingRules []inference.BlockingRule
	// shots are the case's screenshots, cut once at load.
	shots caseImages
	// Incremental: Files are one push; the rest is the PR around it.
	Incremental      bool
	CumulativeDiff   string
	PreviouslyPosted []inference.PostedFinding
	PreviousOverview string
	ExpectedFixed    []ExpectedFixed
	Repeats          []Expected
}

// asPosted is one earlier finding as production shows it (review.asPosted):
// a closed one has no id, so it can never be claimed fixed.
func (p corpusPosted) asPosted() inference.PostedFinding {
	line := p.Line
	out := inference.PostedFinding{ID: p.ID, FilePath: p.File, LineNumber: &line, Severity: p.Severity,
		CommentText: p.Comment, Dispute: p.Dispute}
	if p.Closed {
		out.ID, out.Closed = 0, true
	}
	return out
}

// LoadCorpus reads every embedded case, sorted by name so a run is ordered.
//
// Embedded rather than read from disk: the corpus ships with the binary, so
// an eval run does not depend on the working directory or on the repo being
// checked out next to it.
func LoadCorpus() ([]Case, error) {
	entries, err := fs.Glob(corpusFS, "corpus/*.json")
	if err != nil {
		return nil, err
	}
	sort.Strings(entries)
	var out []Case
	for _, name := range entries {
		raw, err := corpusFS.ReadFile(name)
		if err != nil {
			return nil, err
		}
		var cf CorpusFile
		if err := json.Unmarshal(raw, &cf); err != nil {
			return nil, fmt.Errorf("%s: %w", name, err)
		}
		if cf.Name == "" {
			return nil, fmt.Errorf("%s: case has no name", name)
		}
		c := Case{Name: cf.Name, Description: cf.Description, Expected: cf.Expected, ExpectedVisual: cf.ExpectedVisual,
			AgentsMD: cf.AgentsMD, ExpectedFixed: cf.ExpectedFixed, Repeats: cf.Repeats}
		if inc := cf.Incremental; inc != nil {
			c.Incremental, c.CumulativeDiff, c.PreviousOverview = true, inc.CumulativeDiff, inc.PreviousOverview
			for _, p := range inc.PreviousFindings {
				c.PreviouslyPosted = append(c.PreviouslyPosted, p.asPosted())
			}
			// Closed first, as production lists them (trimPreviouslyPosted):
			// the eval must send the prompt production sends.
			slices.SortStableFunc(c.PreviouslyPosted, func(a, b inference.PostedFinding) int {
				switch {
				case a.Closed == b.Closed:
					return 0
				case a.Closed:
					return -1
				}
				return 1
			})
		} else if len(cf.ExpectedFixed) > 0 || len(cf.Repeats) > 0 {
			// Only an incremental review lists ids; a full one could never
			// catch it, and the run would read as a regression.
			return nil, fmt.Errorf("%s: expected_fixed needs an incremental block", name)
		}
		c.BlockingRules, _ = inference.ExtractBlockingRules(cf.AgentsMD)
		if c.shots, err = loadImages(cf.Screenshots); err != nil {
			return nil, fmt.Errorf("%s: %w", name, err)
		}
		for _, f := range cf.Files {
			c.Files = append(c.Files, diff.FileReviewData{
				Path:           f.Path,
				Diff:           f.Diff,
				Content:        f.Content,
				ContentFetched: f.ContentFetched,
			})
		}
		out = append(out, c)
	}
	return out, nil
}

// Match records whether one seeded bug was found, and by which finding.
type Match struct {
	Expected Expected
	Found    bool
	// Finding is the index into the run's findings that matched, -1 when none.
	Finding int
	// Why names the first check that rejected the closest candidate, so a miss
	// is actionable rather than just a zero.
	Why string
}

// Result is one case's outcome.
type Result struct {
	Case     string
	Outcome  string
	Findings []inference.ReviewFinding
	Matches  []Match
	// Extra is findings that matched no seeded bug. On a clean case every
	// finding is extra, which is the point of having one.
	Extra int
	// Duplicates is how many findings were byte-identical copies of an
	// earlier one. They are excluded from Extra, because a stutter invents
	// nothing, but they are reported: a model repeating itself is worth
	// seeing rather than silently collapsing.
	Duplicates int `json:"Duplicates,omitempty"`
	// Dropped and Moved are what inference.ValidateFindings did before
	// scoring, exactly as production does before posting. A dropped finding
	// is neither scored nor Extra: it would never have reached the PR.
	Dropped []inference.DroppedFinding `json:"Dropped,omitempty"`
	Moved   []inference.Reanchor       `json:"Moved,omitempty"`
	// PreExisting are the bugs in unchanged code the model reported, as
	// validation kept them: never scored (production never posts them), but
	// recorded, so a run shows what the run page would.
	PreExisting        []inference.PreExistingBug     `json:"PreExisting,omitempty"`
	PreExistingDropped []inference.DroppedPreExisting `json:"PreExistingDropped,omitempty"`
	// Redraws is how many attempts that did not complete came before this
	// one. A redrawn result leans towards the quick answer; see maxRedraws.
	Redraws int `json:",omitempty"`
	// VisualMatches are the case's expected screenshot lines, scored;
	// Visual the model's lines by path, as the summary would show them.
	VisualMatches []VisualMatch     `json:",omitempty"`
	Visual        map[string]string `json:",omitempty"`
	// Blocking is every finding's blocking claim as production settles it
	// (inference.CheckBlocking), FalseTasks the tasks no expectation asked
	// for: a task blocks a merge, so one too many is noise that costs.
	Blocking   []BlockingClaim `json:",omitempty"`
	FalseTasks int             `json:",omitempty"`
	// FixedMatches are the case's expected fixes, scored; Fixed every claim
	// ValidateFixed kept and FixedDropped the ones it refused, as production
	// would. FalseFixed counts kept claims no expectation asked for: each
	// resolves a thread on a live bug.
	FixedMatches []FixedMatch             `json:",omitempty"`
	Fixed        []string                 `json:",omitempty"`
	FixedDropped []inference.DroppedFixed `json:",omitempty"`
	FalseFixed   []string                 `json:",omitempty"`
	// Repeated are the case's `repeats` a kept finding raised again.
	Repeated []string `json:",omitempty"`
	// Err is excluded from JSON: encoding/json renders an error as {}, so a
	// committed report would record THAT a case failed but not why, making a
	// 401 and a timeout look identical in history. ErrMsg carries the text.
	Err    error  `json:"-"`
	ErrMsg string `json:"ErrMsg,omitempty"`
}

// Found reports how many seeded bugs this case's review caught, expected
// screenshot lines included.
func (r Result) Found() int {
	n := 0
	for _, m := range r.Matches {
		if m.Found {
			n++
		}
	}
	for _, m := range r.VisualMatches {
		if m.Found {
			n++
		}
	}
	for _, m := range r.FixedMatches {
		if m.Found {
			n++
		}
	}
	return n
}

// FixedMatch records whether one expected fix was claimed and kept.
type FixedMatch struct {
	Expected ExpectedFixed
	Found    bool
	// Why says what happened to the claim when it was not kept.
	Why string `json:",omitempty"`
}

// Score is a whole run.
type Score struct {
	Results []Result
	// Seeded, Caught and Extra are summed across cases. Extra counts noise:
	// on a clean case every finding is noise, on a buggy one it is a finding
	// that pinned nothing the case asked for.
	Seeded, Caught, Extra int
	// FalseTasks is summed across cases: tasks no expectation asked for.
	FalseTasks int `json:",omitempty"`
	// Route is where the calls went (Settings.Route): a gateway's URL or
	// "vendor". Rows from different routes are not one series.
	Route string `json:",omitempty"`
	// FalseFixed is summed across cases: kept fixed claims no case asked for.
	FalseFixed int `json:",omitempty"`
	// Repeated is summed across cases: earlier findings raised again.
	Repeated int `json:",omitempty"`
}

// Reviewer is the inference call an eval run makes. The real
// *inference.Client satisfies it; a test can stub it without a gateway.
type Reviewer interface {
	Review(ctx context.Context, req inference.ReviewRequest) inference.ReviewResult
}

// Run scores every case in the corpus.
//
// Prompt assembly goes through inference.AssembleReviewPrompt, the same
// function the review pipeline calls. Rendering the template here instead
// would measure a prompt the product never sends.
func Run(ctx context.Context, client Reviewer, template string, cases []Case, count inference.CountFunc) Score {
	var score Score
	for _, c := range cases {
		res, _ := drawCase(ctx, client, c, assemble(template, c, count))
		score.Seeded += len(c.Expected) + len(c.ExpectedVisual) + len(c.ExpectedFixed)
		score.Caught += res.Found()
		score.Extra += res.Extra
		score.FalseTasks += res.FalseTasks
		score.FalseFixed += len(res.FalseFixed)
		score.Repeated += len(res.Repeated)
		score.Results = append(score.Results, res)
	}
	return score
}

// assemble renders a case's prompt. PromptTokens carries its screenshots as
// production weighs them, images and the text around them.
func assemble(template string, c Case, count inference.CountFunc) inference.AssembledPrompt {
	a := inference.AssembleReviewPrompt(inference.AssembleRequest{
		Template:         template,
		Files:            c.Files,
		RepoInstructions: c.AgentsMD,
		BlockingRules:    c.BlockingRules,
		CumulativePRDiff: c.CumulativeDiff,
		Incremental:      c.Incremental,
		PreviouslyPosted: c.PreviouslyPosted,
		PreviousOverview: c.PreviousOverview,
		// The scoring matches English corpus keywords.
		Language: lang.EN,
	}, count)
	a.PromptTokens += c.shots.pixels + inference.VisualTextTokens(c.shots.images, count)
	return a
}

// reviewCase is one review of one case, validated and scored. Run and
// RunSampled both go through it, so a sample is scored exactly like a run.
func reviewCase(ctx context.Context, client Reviewer, c Case, assembled inference.AssembledPrompt) (Result, inference.CallCost) {
	res := Result{Case: c.Name}
	out := client.Review(ctx, inference.ReviewRequest{
		Prompt:         assembled.Prompt,
		PromptTokens:   assembled.PromptTokens,
		ResponseSchema: inference.ReviewResponseFormat(),
		Images:         c.shots.images,
	})
	res.VisualMatches, res.Visual = matchVisual(c.ExpectedVisual, c.shots.ids, out.Review.Visual)
	res.Outcome = out.Outcome.String()
	// As production builds it (review.prepare): on an incremental case the
	// cumulative diff is quotable too.
	idx := diff.BuildAnchorIndex(c.Files)
	idx.AddContextDiff(c.CumulativeDiff)
	v := inference.ValidateFindings(out.Review.Findings, idx)
	res.Findings, res.Dropped, res.Moved = v.Kept, v.Dropped, v.Reanchored
	// A full case is one diff: its own added lines are the PR's. An
	// incremental one's are the cumulative diff's.
	var prAdded map[string]map[uint64]bool
	if c.Incremental {
		prAdded = diff.AddedLines(c.CumulativeDiff)
	}
	res.PreExisting, res.PreExistingDropped = inference.ValidatePreExisting(out.Review.PreExisting, idx, prAdded)
	res.FixedMatches, res.Fixed, res.FixedDropped, res.FalseFixed = scoreFixed(c, out.Review.Fixed, idx)
	for i, d := range res.PreExistingDropped {
		// Results are committed: prose carrying a secret-like token
		// is not written down, whatever the claim was dropped for.
		if inference.AnySecret(append([]string{d.Bug.Headline, d.Bug.Explanation}, d.Bug.Evidence...)...) {
			res.PreExistingDropped[i].Bug.Headline, res.PreExistingDropped[i].Bug.Explanation = "", ""
			res.PreExistingDropped[i].Bug.Evidence = nil
		}
	}
	if out.Err != nil {
		res.Err, res.ErrMsg = out.Err, out.Err.Error()
	}
	res.Matches, res.Extra, res.Duplicates = match(c.Expected, res.Findings)
	res.Repeated = scoreRepeats(c, res.Findings)
	res.Blocking, res.FalseTasks = scoreBlocking(res.Matches, res.Findings, c.BlockingRules)
	return res, out.Cost
}

// BlockingClaim is one finding's blocking claim and what production makes
// of it before the task cap and the switches (inference.CheckBlocking).
type BlockingClaim struct {
	Finding int
	Rule    string
	Outcome inference.BlockingOutcome
}

// scoreBlocking holds the blocking half of a case: an expectation naming a
// rule is caught only by a finding production would post as a task for
// that rule (else it turns into a miss, saying why), and every task no
// expectation asked for is a false task. Blunt, like the rest: production's
// own check, never a judge.
func scoreBlocking(matches []Match, findings []inference.ReviewFinding, rules []inference.BlockingRule) ([]BlockingClaim, int) {
	var claims []BlockingClaim
	task := make(map[int]string, len(findings))
	for i, f := range findings {
		if !f.Marked() {
			continue
		}
		rule, outcome, ok := inference.CheckBlocking(f, rules)
		claim := BlockingClaim{Finding: i, Rule: *f.BlockingRule, Outcome: outcome}
		if rule != nil {
			claim.Rule = rule.ID
		}
		claims = append(claims, claim)
		if ok {
			task[i] = rule.ID
		}
	}
	asked := map[int]bool{}
	for mi := range matches {
		m := &matches[mi]
		if !m.Found || m.Expected.Blocking == "" {
			continue
		}
		if task[m.Finding] == m.Expected.Blocking {
			asked[m.Finding] = true
			continue
		}
		why := "no blocking_rule"
		if f := findings[m.Finding]; f.Marked() {
			_, outcome, _ := inference.CheckBlocking(f, rules)
			why = fmt.Sprintf("blocking_rule %q: %s", *f.BlockingRule, outcome)
		}
		m.Found, m.Why = false, fmt.Sprintf("caught on %s:%d, but not a task for %s (%s)",
			findings[m.Finding].File, findings[m.Finding].Line, m.Expected.Blocking, why)
		m.Finding = -1
	}
	falseTasks := 0
	for i := range task {
		if !asked[i] {
			falseTasks++
		}
	}
	return claims, falseTasks
}

// scoreFixed holds a case's `fixed` claims to production's ValidateFixed: an
// expected fix is caught only by a claim it keeps, and every kept claim no
// expectation names is a false fix.
func scoreFixed(c Case, claims []inference.FixedClaim, idx diff.AnchorIndex) (matches []FixedMatch, kept []string, dropped []inference.DroppedFixed, falseIDs []string) {
	v := inference.ValidateFixed(claims, c.PreviouslyPosted, idx)
	keptIDs := map[int64]bool{}
	for _, k := range v.Kept {
		keptIDs[k.Finding.ID] = true
		kept = append(kept, inference.FixedIDPrefix+strconv.FormatInt(k.Finding.ID, 10))
	}
	asked := map[int64]bool{}
	for _, e := range c.ExpectedFixed {
		want, _ := fixedNum(e.ID)
		asked[want] = true
		m := FixedMatch{Expected: e, Found: keptIDs[want]}
		if !m.Found {
			m.Why = "not claimed"
			// The first drop of a claim on it: a later duplicate would
			// hide the reason the real claim failed.
			for _, d := range v.Dropped {
				if n, ok := fixedNum(d.Claim.ID); ok && n == want {
					m.Why = "claimed, dropped: " + string(d.Reason)
					break
				}
			}
		}
		matches = append(matches, m)
	}
	for _, k := range v.Kept {
		if !asked[k.Finding.ID] {
			falseIDs = append(falseIDs, inference.FixedIDPrefix+strconv.FormatInt(k.Finding.ID, 10))
		}
	}
	return matches, kept, v.Dropped, falseIDs
}

// fixedNum reads an id as production's ValidateFixed does: "F101",
// "[F101]", "f101" and a bare "101" are the same finding.
func fixedNum(id string) (int64, bool) {
	raw := strings.TrimSpace(strings.Trim(id, "[] \t"))
	if len(raw) > 0 && strings.EqualFold(raw[:1], inference.FixedIDPrefix) {
		raw = raw[1:]
	}
	n, err := strconv.ParseInt(raw, 10, 64)
	return n, err == nil
}

// scoreRepeats counts the kept findings that raise an earlier finding
// again: one that lands in a case's `repeats` window and names one of its
// keywords. A repeat is the regression the incremental cases exist for, on
// a case with a seeded bug as on a clean one.
func scoreRepeats(c Case, findings []inference.ReviewFinding) []string {
	var out []string
	for _, r := range c.Repeats {
		for _, f := range findings {
			if samePath(f.File, r.File) && f.Line >= r.Lines[0] && f.Line <= r.Lines[1] && mentions(f, r.Keywords) {
				out = append(out, r.ID)
				break
			}
		}
	}
	return out
}

// drawCase is reviewCase, drawn again while the call does not complete, up
// to maxRedraws times. The count is on the result. Run and RunSampled both
// draw through it.
func drawCase(ctx context.Context, client Reviewer, c Case, assembled inference.AssembledPrompt) (Result, inference.CallCost) {
	res, cost := reviewCase(ctx, client, c, assembled)
	redraws := 0
	for !completed(res) && redraws < maxRedraws && ctx.Err() == nil {
		redraws++
		res, cost = reviewCase(ctx, client, c, assembled)
	}
	res.Redraws = redraws
	return res, cost
}

// Settings is what a run needs from the environment.
type Settings struct {
	BaseURL, APIKey, Model string
	// Alias is the name the endpoint serves the model under. Empty means the
	// model id itself, which is what a plain OpenAI-compatible host expects.
	Alias         string
	ContextWindow int
	// CallTimeout is serve's REVIEW_INFERENCE_CALL_TIMEOUT, so a slow
	// profile times out here where it would in production, not sooner.
	CallTimeout time.Duration
	// VendorKeys holds the LLMWIRE_<PROVIDER>_API_KEY variables found when
	// no gateway is set: llmwire then sends the profile to its vendor's host
	// from its profiles, with that vendor's key. Empty on a gateway run.
	VendorKeys map[string]string
}

// Route names where a run's calls go, for the report's header: the
// gateway's URL, or "vendor" on the vendor route.
func (s Settings) Route() string {
	if s.VendorKeys != nil {
		return "vendor"
	}
	return s.BaseURL
}

// VendorKeyNames are the vendor key variables a run may read, the ones
// llmwire's profiles name a provider for.
var VendorKeyNames = []string{"LLMWIRE_MIMO_API_KEY", "LLMWIRE_ZAI_API_KEY", "LLMWIRE_OPENAI_API_KEY", "LLMWIRE_ANTHROPIC_API_KEY"}

// ResolveSettings reads the run's settings, naming what is missing.
//
// getenv rather than os.Getenv so this is testable without touching the
// process environment, which would make the tests order-dependent.
func ResolveSettings(getenv func(string) string, windowFlag int) (Settings, error) {
	s := Settings{
		BaseURL:       strings.TrimSpace(getenv("EVAL_BASE_URL")),
		APIKey:        strings.TrimSpace(getenv("EVAL_API_KEY")),
		Model:         strings.TrimSpace(getenv("EVAL_MODEL")),
		Alias:         strings.TrimSpace(getenv("EVAL_ALIAS")),
		ContextWindow: windowFlag,
	}
	// Unset, the model is the series' (corpus/series.yaml): a run on a model
	// nobody chose cannot be read as a point in the prompt's history, so the
	// default is the one pair the committed series is scored at.
	if s.Model == "" {
		s.Model = DefaultSeries().Model
	}
	// No gateway at all: the vendor route, when a vendor key is set. A half
	// gateway (a URL without a key, or the other way) is still an error: it
	// is a typo, never a request for the vendor.
	if s.BaseURL == "" && s.APIKey == "" {
		for _, name := range VendorKeyNames {
			if v := strings.TrimSpace(getenv(name)); v != "" {
				if s.VendorKeys == nil {
					s.VendorKeys = map[string]string{}
				}
				s.VendorKeys[name] = v
			}
		}
	}
	var missing []string
	if s.BaseURL == "" && s.VendorKeys == nil {
		missing = append(missing, "EVAL_BASE_URL")
	}
	if s.APIKey == "" && s.VendorKeys == nil {
		missing = append(missing, "EVAL_API_KEY")
	}
	switch {
	case len(missing) == 2:
		return s, fmt.Errorf("set %s to point at an OpenAI-compatible endpoint, or a vendor key (%s) to reach the profile's vendor",
			strings.Join(missing, ", "), strings.Join(VendorKeyNames, ", "))
	case len(missing) == 1:
		// A half gateway: a vendor key, if set, is ignored while it stands.
		return s, fmt.Errorf("set %s too, or unset EVAL_BASE_URL and EVAL_API_KEY both to use a vendor key", missing[0])
	}
	if s.VendorKeys != nil && s.Alias != "" {
		return s, fmt.Errorf("EVAL_ALIAS=%q names a gateway alias; the vendor route sends the profile's own model id, unset it", s.Alias)
	}
	// resolveWindow's error tells the operator to set OPENAI_CONTEXT_WINDOW,
	// which only internal/config reads. Honouring it here makes that advice
	// followable instead of a permanent failure.
	if s.ContextWindow == 0 {
		if v := strings.TrimSpace(getenv("OPENAI_CONTEXT_WINDOW")); v != "" {
			n, err := strconv.Atoi(v)
			if err != nil {
				return s, fmt.Errorf("OPENAI_CONTEXT_WINDOW=%q: %w", v, err)
			}
			s.ContextWindow = n
		}
	}
	// Same knob, same default as serve, so an eval row and production time
	// out at the same bound. The floor is config's; here a shorter value is
	// the operator's own experiment.
	s.CallTimeout = config.DefaultCallTimeout
	if v := strings.TrimSpace(getenv("REVIEW_INFERENCE_CALL_TIMEOUT")); v != "" {
		d, err := time.ParseDuration(v)
		if err != nil || d <= 0 {
			return s, fmt.Errorf("REVIEW_INFERENCE_CALL_TIMEOUT=%q: not a positive duration (e.g. 5m)", v)
		}
		s.CallTimeout = d
	}
	return s, nil
}

// GatewayEnv is the environment llmwire reads, routing the profile to the
// name the endpoint actually serves.
//
// inference.New leaves Config.BaseURL empty on purpose, so this is the
// supported way to aim the client elsewhere: exactly what a team's own
// gateway does.
//
// The alias defaults to the profile's own id. An invented one only works on
// a LiteLLM instance configured with that exact name; a plain
// OpenAI-compatible endpoint serves its models under their real names and
// answers Startup with "not listed by the gateway for this key". EVAL_ALIAS
// covers the case where an operator's gateway does rename it.
func GatewayEnv(s Settings) map[string]string {
	// The vendor route: only the vendor keys, no gateway variable, so
	// llmwire takes the profile's own host.
	if s.VendorKeys != nil {
		env := make(map[string]string, len(s.VendorKeys))
		for k, v := range s.VendorKeys {
			env[k] = v
		}
		return env
	}
	alias := s.Alias
	if alias == "" {
		alias = s.Model
	}
	return map[string]string{
		"LLMWIRE_LITELLM_MODELS":   s.Model + "=" + alias,
		"LLMWIRE_LITELLM_BASE_URL": s.BaseURL,
		"LLMWIRE_LITELLM_API_KEY":  s.APIKey,
	}
}

// ErrIncomplete reports cases that did not produce a usable review.
//
// A case that did not complete is not a prompt result. Without this a gateway
// 503 on a seeded case reads as a regression, and the same failure on a case
// with nothing seeded reads as a pass - a green gate on a run that never
// happened.
func (s Score) ErrIncomplete() error {
	var broken []string
	for _, r := range s.Results {
		if why := incomplete(r); why != "" {
			broken = append(broken, r.Case+": "+why)
		}
	}
	if len(broken) == 0 {
		return nil
	}
	return fmt.Errorf("%d case(s) did not complete, so the score means nothing: %s",
		len(broken), strings.Join(broken, "; "))
}

// ErrMissed reports seeded bugs that went unreported. Nil when every one was
// caught.
func (s Score) ErrMissed() error {
	if s.Caught >= s.Seeded {
		return nil
	}
	return fmt.Errorf("missed %d of %d seeded bug(s)", s.Seeded-s.Caught, s.Seeded)
}

// ErrInvented reports findings on a case that seeds no bug. Nil when the
// clean controls stayed clean.
//
// The controls exist so that "caught every seeded bug" cannot be satisfied by
// reporting everything, and until this they proved nothing: Extra reached no
// exit code, so a run inventing a finding on every clean case still exited 0.
// A finding here is a false positive with no qualification available - the
// case seeds nothing, so there is no honest anchor to have found.
func (s Score) ErrInvented() error {
	var noisy []string
	for _, r := range s.Results {
		// len(r.Matches) is how many bugs the case SEEDS, not how many were
		// caught: match builds one Match per expectation either way. Extra
		// rather than len(r.Findings) both here and below, so a control whose
		// only findings are stutters does not fail, and the number in the
		// message is the one the per-case line and the README already use.
		if len(r.Matches) > 0 || r.Extra == 0 {
			continue
		}
		noisy = append(noisy, fmt.Sprintf("%s: %d", r.Case, r.Extra))
	}
	if len(noisy) == 0 {
		return nil
	}
	return fmt.Errorf("invented finding(s) on %d clean control(s): %s",
		len(noisy), strings.Join(noisy, "; "))
}

// ErrFalseTasks reports blocking tasks no expectation asked for, on any
// case: a task blocks a merge, so a false one costs like an invented
// finding on a control. Nil when there was none.
func (s Score) ErrFalseTasks() error {
	var noisy []string
	for _, r := range s.Results {
		if r.FalseTasks > 0 {
			noisy = append(noisy, fmt.Sprintf("%s: %d", r.Case, r.FalseTasks))
		}
	}
	if len(noisy) == 0 {
		return nil
	}
	return fmt.Errorf("blocking task(s) no case asked for: %s", strings.Join(noisy, "; "))
}

// ErrFalseFixed reports kept fixed claims no case asked for: production
// would reply "Fixed" and resolve the thread of a live bug. Nil when there
// was none.
func (s Score) ErrFalseFixed() error {
	var noisy []string
	for _, r := range s.Results {
		if len(r.FalseFixed) > 0 {
			noisy = append(noisy, r.Case+": "+strings.Join(r.FalseFixed, ", "))
		}
	}
	if len(noisy) == 0 {
		return nil
	}
	return fmt.Errorf("fixed claim(s) no case asked for: %s", strings.Join(noisy, "; "))
}

// ErrRepeated reports earlier findings a review raised again: the author
// gets the same comment twice. Nil when there was none.
func (s Score) ErrRepeated() error {
	var noisy []string
	for _, r := range s.Results {
		if len(r.Repeated) > 0 {
			noisy = append(noisy, r.Case+": "+strings.Join(r.Repeated, ", "))
		}
	}
	if len(noisy) == 0 {
		return nil
	}
	return fmt.Errorf("earlier finding(s) raised again: %s", strings.Join(noisy, "; "))
}

// Report renders a run for a terminal.
//
// Write errors are dropped: this goes to stdout, and a report that cannot be
// printed changes nothing about the verdict, which the caller reads from
// ErrIncomplete and ErrMissed.
func (s Score) Report(w io.Writer) {
	p := func(format string, args ...any) { _, _ = fmt.Fprintf(w, format, args...) }
	for _, r := range s.Results {
		status := "ok"
		if r.ErrMsg != "" {
			status = "ERROR: " + r.ErrMsg
		}
		dup := ""
		if r.Duplicates > 0 {
			dup = fmt.Sprintf(", %d duplicate", r.Duplicates)
		}
		if r.Redraws > 0 {
			dup += fmt.Sprintf(", %d redraw(s)", r.Redraws)
		}
		p("%-18s %-10s %d finding(s), %d seeded, %d caught, %d extra%s  %s\n",
			r.Case, r.Outcome, len(r.Findings), len(r.Matches)+len(r.VisualMatches)+len(r.FixedMatches), r.Found(), r.Extra, dup, status)
		for _, m := range r.Matches {
			if m.Found {
				f := r.Findings[m.Finding]
				p("    FOUND  %-22s %s:%d\n", m.Expected.ID, f.File, f.Line)
				continue
			}
			p("    MISS   %-22s %s\n", m.Expected.ID, m.Why)
			p("           why it matters: %s\n", m.Expected.Why)
		}
		for _, m := range r.VisualMatches {
			if m.Found {
				p("    SEEN   %-22s %s\n", m.Expected.ID, m.Change)
				continue
			}
			p("    UNSEEN %-22s %s\n", m.Expected.ID, m.Why)
			if m.Change != "" {
				p("           the line: %s\n", m.Change)
			}
			p("           why it matters: %s\n", m.Expected.Why)
		}
		for _, b := range r.Blocking {
			f := r.Findings[b.Finding]
			p("    BLOCK  %-22s %s:%d %s\n", b.Rule, f.File, f.Line, b.Outcome)
		}
		if r.FalseTasks > 0 {
			p("    FALSE TASK(S) %d\n", r.FalseTasks)
		}
		for _, m := range r.FixedMatches {
			if m.Found {
				p("    FIXED  %-22s\n", m.Expected.ID)
				continue
			}
			p("    UNFIXED %-21s %s\n", m.Expected.ID, m.Why)
			p("           why it matters: %s\n", m.Expected.Why)
		}
		if len(r.FalseFixed) > 0 {
			p("    FALSE FIXED %s\n", strings.Join(r.FalseFixed, ", "))
		}
		if len(r.Repeated) > 0 {
			p("    REPEATED %s\n", strings.Join(r.Repeated, ", "))
		}
	}
	p("\nseeded %d, caught %d, missed %d, extra findings %d, false tasks %d, false fixed %d, repeated %d\n",
		s.Seeded, s.Caught, s.Seeded-s.Caught, s.Extra, s.FalseTasks, s.FalseFixed, s.Repeated)
}

// match pairs seeded bugs with findings. One finding can satisfy at most one
// seeded bug, so two findings are needed to catch two bugs on the same line.
//
// Assignment is least-options-first rather than in corpus order: taking each
// expectation in turn lets a permissive one consume the only finding a
// stricter one could have used, and the run reports a miss on a review that
// actually caught everything. Two expectations on the same lines, one keyed
// on "nil" and one on "error", against findings "nil pointer causes an
// error" and "nil deref", is the case that breaks in order.
func match(expected []Expected, findings []inference.ReviewFinding) ([]Match, int, int) {
	// A model sometimes emits the same finding twice, byte for byte: run2's
	// lock-not-released carried two identical comments on line 22. The copy
	// pins nothing new, so counting it as Extra reads as invention when it is
	// a stutter. Later copies are marked here and are neither claimable nor
	// counted; the caller keeps the raw list, so the JSON still records what
	// the model actually said.
	duplicate := make([]bool, len(findings))
	dupes := 0
	seenFinding := make(map[string]bool, len(findings))
	for i, f := range findings {
		k := findingKey(f)
		if seenFinding[k] {
			duplicate[i], dupes = true, dupes+1
			continue
		}
		seenFinding[k] = true
	}

	// candidates[e] is every finding that satisfies expectation e on its own.
	candidates := make([][]int, len(expected))
	matches := make([]Match, len(expected))
	for ei, e := range expected {
		matches[ei] = Match{Expected: e, Finding: -1, Why: "no finding on " + e.File}
		// Keep the CLOSEST rejection, not the last one seen: a later
		// out-of-window finding would otherwise overwrite "mentions none of
		// [...]" and send the operator hunting for the wrong problem. A
		// finding that reached the keyword check is nearer than one that
		// failed on the line.
		var reachedKeywords bool
		for i, f := range findings {
			if duplicate[i] {
				continue
			}
			if !samePath(f.File, e.File) {
				continue
			}
			if f.Line < e.Lines[0] || f.Line > e.Lines[1] {
				if !reachedKeywords {
					matches[ei].Why = fmt.Sprintf("finding on %s:%d, outside %d-%d",
						f.File, f.Line, e.Lines[0], e.Lines[1])
				}
				continue
			}
			if !mentions(f, e.Keywords) {
				reachedKeywords = true
				matches[ei].Why = fmt.Sprintf("finding on %s:%d mentions none of %v",
					f.File, f.Line, e.Keywords)
				continue
			}
			candidates[ei] = append(candidates[ei], i)
		}
	}

	used := make([]bool, len(findings))
	done := make([]bool, len(expected))
	for range expected {
		// The expectation with the fewest remaining candidates goes first, so
		// a finding only one expectation can use is never taken by another.
		best, bestN := -1, 0
		for ei := range expected {
			if done[ei] {
				continue
			}
			n := 0
			for _, i := range candidates[ei] {
				if !used[i] {
					n++
				}
			}
			if n == 0 {
				continue
			}
			if best == -1 || n < bestN {
				best, bestN = ei, n
			}
		}
		if best == -1 {
			// Everything still unmatched either had no candidate at all, or
			// had its only candidates taken. Say which: "no finding on x.go"
			// is a lie when the finding exists and another expectation is
			// holding it, and sends the operator looking for a comment the
			// model actually wrote.
			for ei := range expected {
				if done[ei] || len(candidates[ei]) == 0 {
					continue
				}
				matches[ei].Why = fmt.Sprintf(
					"%d qualifying finding(s), all claimed by another expectation "+
						"on the same lines; a second finding is needed",
					len(candidates[ei]))
			}
			break
		}
		for _, i := range candidates[best] {
			if !used[i] {
				used[i] = true
				matches[best].Found, matches[best].Finding, matches[best].Why = true, i, ""
				break
			}
		}
		done[best] = true
	}

	extra := 0
	for i, u := range used {
		if !u && !duplicate[i] {
			extra++
		}
	}
	return matches, extra, dupes
}

// findingKey is a byte-identical finding's identity. The three optional
// fields are pointers, so they are dereferenced: comparing the pointers
// would make every finding unique and defeat the whole check.
//
// Every field is length-prefixed rather than separated by a delimiter, and
// absent is a prefix of its own. A separator is a byte the model can also
// send - the comment is free text and JSON can carry a NUL - and a sentinel
// like "nil" is a string it can send too, so either would let two distinct
// findings produce one key and silently collapse.
func findingKey(f inference.ReviewFinding) string {
	var b strings.Builder
	put := func(s string) {
		b.WriteString(strconv.Itoa(len(s)))
		b.WriteByte(':')
		b.WriteString(s)
	}
	opt := func(s *string) {
		if s == nil {
			b.WriteString("-")
			return
		}
		b.WriteString("+")
		put(*s)
	}
	put(f.File)
	put(strconv.Itoa(f.Line))
	put(f.Severity)
	put(f.Comment)
	if f.Confidence == nil {
		b.WriteString("-")
	} else {
		b.WriteString("+")
		put(strconv.Itoa(*f.Confidence))
	}
	opt(f.Headline)
	opt(f.Suggestion)
	put(strconv.Itoa(len(f.Evidence)))
	for _, e := range f.Evidence {
		put(e)
	}
	return b.String()
}

// samePath compares by suffix: a model may answer with the path as it
// appeared in the diff header rather than the exact string it was handed.
func samePath(got, want string) bool {
	got, want = strings.TrimPrefix(got, "./"), strings.TrimPrefix(want, "./")
	return got == want ||
		strings.HasSuffix(got, "/"+want) || strings.HasSuffix(want, "/"+got)
}

func mentions(f inference.ReviewFinding, keywords []string) bool {
	hay := strings.ToLower(f.Comment)
	if f.Headline != nil {
		hay += " " + strings.ToLower(*f.Headline)
	}
	if f.Suggestion != nil {
		hay += " " + strings.ToLower(*f.Suggestion)
	}
	for _, k := range keywords {
		if strings.Contains(hay, strings.ToLower(k)) {
			return true
		}
	}
	return false
}
