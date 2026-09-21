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
	"sort"
	"strconv"
	"strings"

	"github.com/trick77/noergler/internal/diff"
	"github.com/trick77/noergler/internal/inference"
)

//go:embed corpus/*.json
var corpusFS embed.FS

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
}

// CorpusFile is the on-disk shape of one case.
type CorpusFile struct {
	Name        string `json:"name"`
	Description string `json:"description"`
	Files       []struct {
		Path           string `json:"path"`
		Diff           string `json:"diff"`
		Content        string `json:"content"`
		ContentFetched bool   `json:"content_fetched"`
	} `json:"files"`
	Expected []Expected `json:"expected"`
}

// Case is a loaded corpus entry.
type Case struct {
	Name        string
	Description string
	Files       []diff.FileReviewData
	Expected    []Expected
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
		c := Case{Name: cf.Name, Description: cf.Description, Expected: cf.Expected}
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
	// Err is excluded from JSON: encoding/json renders an error as {}, so a
	// committed report would record THAT a case failed but not why, making a
	// 401 and a timeout look identical in history. ErrMsg carries the text.
	Err    error  `json:"-"`
	ErrMsg string `json:"ErrMsg,omitempty"`
}

// Found reports how many seeded bugs this case's review caught.
func (r Result) Found() int {
	n := 0
	for _, m := range r.Matches {
		if m.Found {
			n++
		}
	}
	return n
}

// Score is a whole run.
type Score struct {
	Results []Result
	// Seeded, Caught and Extra are summed across cases. Extra counts noise:
	// on a clean case every finding is noise, on a buggy one it is a finding
	// that pinned nothing the case asked for.
	Seeded, Caught, Extra int
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
		res := Result{Case: c.Name}
		assembled := inference.AssembleReviewPrompt(inference.AssembleRequest{
			Template: template,
			Files:    c.Files,
		}, count)

		out := client.Review(ctx, inference.ReviewRequest{
			Prompt:         assembled.Prompt,
			PromptTokens:   assembled.PromptTokens,
			ResponseSchema: inference.ReviewResponseFormat(),
		})
		res.Outcome = out.Outcome.String()
		res.Findings = out.Review.Findings
		if out.Err != nil {
			res.Err, res.ErrMsg = out.Err, out.Err.Error()
		}
		res.Matches, res.Extra, res.Duplicates = match(c.Expected, res.Findings)
		score.Seeded += len(c.Expected)
		score.Caught += res.Found()
		score.Extra += res.Extra
		score.Results = append(score.Results, res)
	}
	return score
}

// Settings is what a run needs from the environment.
type Settings struct {
	BaseURL, APIKey, Model string
	// Alias is the name the endpoint serves the model under. Empty means the
	// model id itself, which is what a plain OpenAI-compatible host expects.
	Alias         string
	ContextWindow int
}

// DefaultModel is the profile evals are scored on.
//
// Overridable for a deliberate comparison, never by accident: a weaker model
// or less thinking scores worse on the same prompt, so a mixed history cannot
// be read as a prompt history.
const DefaultModel = "mimo-v2.5-pro"

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
	if s.Model == "" {
		s.Model = DefaultModel
	}
	var missing []string
	if s.BaseURL == "" {
		missing = append(missing, "EVAL_BASE_URL")
	}
	if s.APIKey == "" {
		missing = append(missing, "EVAL_API_KEY")
	}
	if len(missing) > 0 {
		return s, fmt.Errorf("set %s to point at an OpenAI-compatible endpoint",
			strings.Join(missing, ", "))
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
		switch {
		case r.ErrMsg != "":
			broken = append(broken, r.Case+": "+r.ErrMsg)
		case r.Outcome != inference.OutcomeOK.String():
			broken = append(broken, r.Case+": outcome "+r.Outcome)
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
		p("%-18s %-10s %d finding(s), %d seeded, %d caught, %d extra%s  %s\n",
			r.Case, r.Outcome, len(r.Findings), len(r.Matches), r.Found(), r.Extra, dup, status)
		for _, m := range r.Matches {
			if m.Found {
				f := r.Findings[m.Finding]
				p("    FOUND  %-22s %s:%d\n", m.Expected.ID, f.File, f.Line)
				continue
			}
			p("    MISS   %-22s %s\n", m.Expected.ID, m.Why)
			p("           why it matters: %s\n", m.Expected.Why)
		}
	}
	p("\nseeded %d, caught %d, missed %d, extra findings %d\n",
		s.Seeded, s.Caught, s.Seeded-s.Caught, s.Extra)
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
