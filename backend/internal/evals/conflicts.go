package evals

// The merge-conflict eval scores the model's part of a conflict
// resolution: given the conflicts code left over, does it pick the lines a
// person would have, and does it decline where picking lines is not
// enough? The code part (the merge, the trivial rule) is tested against git
// elsewhere and never reaches a call here.
//
// A case is one request as production sends it: every conflict of a merge,
// the trivial ones resolved by code first, the rest in one call through
// inference.HunkOf and inference.RenderConflictPrompt, the answer read by
// merge.ReadAnswer. Scoring compares the picked text, never the ids: T2
// and an identical S3 are the same answer.
//
// The corpus lives in conflicts/, never in corpus/: LoadCorpus embeds
// corpus/*.json, and a conflict case there would join every review series
// run.

import (
	"context"
	"embed"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"io/fs"
	"path"
	"slices"
	"sort"
	"strings"
	"sync"
	"time"

	"github.com/trick77/noergler/internal/inference"
	"github.com/trick77/noergler/internal/lang"
	"github.com/trick77/noergler/internal/merge"
	"github.com/trick77/noergler/internal/prompts"
)

//go:embed conflicts/*.json
var conflictFS embed.FS

// Where a conflict case came from. The report splits by it: an invented
// case tests what its author thought of, a mined one what people met.
const (
	OriginMined    = "mined"
	OriginInvented = "invented"
	OriginWeb      = "web"
)

// ConflictCase is a corpus entry as stored.
type ConflictCase struct {
	Name string `json:"name"`
	// Origin is mined, invented or web; Source says where from (a repo
	// and merge commit, a URL, or "hand-written").
	Origin string `json:"origin"`
	Source string `json:"source"`
	// Why is what makes the expected answer right, or the control a
	// control.
	Why          string `json:"why"`
	TargetBranch string `json:"target_branch"`
	SourceBranch string `json:"source_branch"`
	// Control is a case only declining answers right: some conflict needs
	// a line neither side has, or the author's decision. Any answer that
	// passes validation is wrong on it.
	Control bool           `json:"control"`
	Files   []ConflictFile `json:"files"`
	// Accept lists, per conflict the model is asked about, every text a
	// correct answer may assemble. Empty on a control.
	Accept []ConflictAccept `json:"accept"`
}

// ConflictFile is one file of a case in diff3 form, cut to the lines
// around its conflicts: settled lines as they are, each conflict as
//
//	<<<<<<< source
//	... the PR branch's lines
//	||||||| base
//	... the merge base's lines
//	=======
//	... the target branch's lines
//	>>>>>>> target
//
// git's own order (ours, base, theirs), with the PR branch as ours.
type ConflictFile struct {
	Path   string `json:"path"`
	Merged string `json:"merged"`
	// Trimmed lists the conflicts (from 1) whose edges merge.Merge3 cut
	// lines both sides added there: settled lines in the diff3 text, which
	// cannot show it, though it rules code's trivial resolution out.
	Trimmed []int `json:"trimmed,omitempty"`
}

// ConflictAccept is the texts a conflict may resolve to. Conflict counts
// from 1 over the file's conflicts, trivial ones included.
type ConflictAccept struct {
	File     string   `json:"file"`
	Conflict int      `json:"conflict"`
	Texts    []string `json:"texts"`
	// AnyOrder accepts a text's lines in any order (imports, set
	// entries): the lines and how often each appears count, not where.
	AnyOrder bool `json:"any_order"`
}

// asked is one conflict the model answers, in the prompt's order.
type asked struct {
	file     string
	k        int
	conflict *merge.Conflict
	accept   ConflictAccept
}

// LoadedConflictCase is a case ready to send.
type LoadedConflictCase struct {
	ConflictCase
	asked   []asked
	trivial int
	// chunks is each file's parsed diff3 text, by path.
	chunks map[string][]merge.Chunk
}

// LoadConflictCorpus reads and checks every embedded conflict case.
func LoadConflictCorpus() ([]LoadedConflictCase, error) {
	names, err := fs.Glob(conflictFS, "conflicts/*.json")
	if err != nil {
		return nil, err
	}
	sort.Strings(names)
	out := make([]LoadedConflictCase, 0, len(names))
	for _, name := range names {
		raw, err := conflictFS.ReadFile(name)
		if err != nil {
			return nil, err
		}
		var c ConflictCase
		dec := json.NewDecoder(strings.NewReader(string(raw)))
		dec.DisallowUnknownFields()
		if err := dec.Decode(&c); err != nil {
			return nil, fmt.Errorf("%s: %w", name, err)
		}
		if want := strings.TrimSuffix(path.Base(name), ".json"); c.Name != want {
			return nil, fmt.Errorf("%s: name %q, want %q", name, c.Name, want)
		}
		l, err := LoadConflictCase(c)
		if err != nil {
			return nil, fmt.Errorf("%s: %w", name, err)
		}
		out = append(out, l)
	}
	return out, nil
}

// LoadConflictCase checks one case against what production would send:
// the merge bounds, at least one conflict for the model, and an expected
// answer for each of those that ids can express.
func LoadConflictCase(c ConflictCase) (LoadedConflictCase, error) {
	l := LoadedConflictCase{ConflictCase: c, chunks: map[string][]merge.Chunk{}}
	switch {
	case c.Origin != OriginMined && c.Origin != OriginInvented && c.Origin != OriginWeb:
		return l, fmt.Errorf("origin %q: want mined, invented or web", c.Origin)
	case strings.TrimSpace(c.Source) == "" || strings.TrimSpace(c.Why) == "":
		return l, errors.New("source and why are required")
	case c.TargetBranch == "" || c.SourceBranch == "":
		return l, errors.New("target_branch and source_branch are required")
	case c.Control && len(c.Accept) > 0:
		return l, errors.New("a control accepts nothing")
	}
	accept := map[string]ConflictAccept{}
	for _, a := range c.Accept {
		key := fmt.Sprintf("%s#%d", a.File, a.Conflict)
		if _, dup := accept[key]; dup || len(a.Texts) == 0 {
			return l, fmt.Errorf("accept %s: listed twice or without texts", key)
		}
		accept[key] = a
	}
	total, files := 0, 0
	for _, f := range c.Files {
		// Production refuses such a file before any call: never the
		// model's to answer. Only the path and the excerpt are here; a
		// generated marker in a full file's head is cmd/conflictmine's
		// to check, on all three versions, before it writes a case.
		if why := merge.Excluded(f.Path, []byte(f.Merged)); why != "" {
			return l, fmt.Errorf("%s: %s, production excludes it", f.Path, why)
		}
		if _, dup := l.chunks[f.Path]; dup {
			return l, fmt.Errorf("%s: listed twice", f.Path)
		}
		parsed, err := ParseDiff3(f.Merged)
		if err != nil {
			return l, fmt.Errorf("%s: %w", f.Path, err)
		}
		chunks, err := remerge(parsed)
		if err != nil {
			return l, fmt.Errorf("%s: %w", f.Path, err)
		}
		l.chunks[f.Path] = chunks
		cs := merge.Conflicts(chunks)
		if len(cs) == 0 {
			return l, fmt.Errorf("%s: no conflict", f.Path)
		}
		for _, k := range f.Trimmed {
			if k < 1 || k > len(cs) {
				return l, fmt.Errorf("%s: trimmed conflict %d of %d", f.Path, k, len(cs))
			}
			cs[k-1].Trimmed = true
		}
		total += len(cs)
		files++
		for k, cf := range cs {
			key := fmt.Sprintf("%s#%d", f.Path, k+1)
			if cf.Oversized() {
				return l, fmt.Errorf("%s: over %d lines", key, merge.MaxRegionLines)
			}
			if cf.Trivial() {
				if _, ok := accept[key]; ok {
					return l, fmt.Errorf("%s: trivial, code resolves it", key)
				}
				l.trivial++
				continue
			}
			a, ok := accept[key]
			if !ok && !c.Control {
				return l, fmt.Errorf("%s: no accepted answer", key)
			}
			delete(accept, key)
			for _, t := range a.Texts {
				if !Expressible(cf, splitText(t)) {
					return l, fmt.Errorf("%s: %q is no choice of its lines", key, t)
				}
			}
			l.asked = append(l.asked, asked{file: f.Path, k: k, conflict: cf, accept: a})
		}
	}
	switch {
	case len(accept) > 0:
		return l, fmt.Errorf("accept names %d conflict(s) the files do not have", len(accept))
	case total > merge.MaxConflicts || files > merge.MaxConflictFiles:
		return l, fmt.Errorf("%d conflicts in %d files, over the bounds", total, files)
	case len(l.asked) == 0:
		return l, errors.New("code resolves every conflict: nothing for the model")
	}
	return l, nil
}

// ParseDiff3 reads a ConflictFile's text into merge chunks.
func ParseDiff3(text string) ([]merge.Chunk, error) {
	const (
		settled = iota
		inSource
		inBase
		inTarget
	)
	state := settled
	var out []merge.Chunk
	var settledLines []string
	var cur *merge.Conflict
	for _, l := range merge.Lines([]byte(text)) {
		marker := strings.TrimRight(l, "\r\n")
		switch {
		case marker == "<<<<<<< source":
			if state != settled {
				return nil, errors.New("conflict opened inside a conflict")
			}
			if len(settledLines) > 0 {
				out = append(out, merge.Chunk{Lines: settledLines})
				settledLines = nil
			}
			cur, state = &merge.Conflict{}, inSource
		case marker == "||||||| base":
			if state != inSource {
				return nil, errors.New("base marker out of place")
			}
			state = inBase
		case marker == "=======":
			if state != inBase {
				return nil, errors.New("separator out of place (a conflict needs its base section)")
			}
			state = inTarget
		case marker == ">>>>>>> target":
			if state != inTarget {
				return nil, errors.New("closing marker out of place")
			}
			out = append(out, merge.Chunk{Conflict: cur})
			cur, state = nil, settled
		case strings.HasPrefix(marker, "<<<<<<<") || strings.HasPrefix(marker, "|||||||") || strings.HasPrefix(marker, ">>>>>>>"):
			return nil, fmt.Errorf("marker %q: want the source, base and target labels", marker)
		default:
			switch state {
			case settled:
				settledLines = append(settledLines, l)
			case inSource:
				cur.Source = append(cur.Source, l)
			case inBase:
				cur.Base = append(cur.Base, l)
			case inTarget:
				cur.Target = append(cur.Target, l)
			}
		}
	}
	if state != settled {
		return nil, errors.New("conflict not closed")
	}
	if len(settledLines) > 0 {
		out = append(out, merge.Chunk{Lines: settledLines})
	}
	return out, nil
}

// matches reports whether got, lines joined by "\n", is an accepted text.
func (a ConflictAccept) matches(got string) bool {
	norm := func(t string) string {
		t = strings.TrimSuffix(t, "\n")
		if !a.AnyOrder {
			return t
		}
		ls := strings.Split(t, "\n")
		slices.Sort(ls)
		return strings.Join(ls, "\n")
	}
	return slices.ContainsFunc(a.Texts, func(t string) bool { return norm(t) == norm(got) })
}

// remerge rebuilds the three versions a diff3 text holds (settled lines in
// all three) and merges them again with merge.Merge3: a case's conflicts
// are exactly what production would see, or the case does not load. A
// trimmed edge the text cannot carry comes from ConflictFile.Trimmed.
func remerge(parsed []merge.Chunk) ([]merge.Chunk, error) {
	var base, target, source []string
	for _, ch := range parsed {
		if ch.Conflict == nil {
			base, target, source = append(base, ch.Lines...), append(target, ch.Lines...), append(source, ch.Lines...)
			continue
		}
		base = append(base, ch.Conflict.Base...)
		target = append(target, ch.Conflict.Target...)
		source = append(source, ch.Conflict.Source...)
	}
	got, err := merge.Merge3(base, target, source)
	if err != nil {
		return nil, err
	}
	if render(got) != render(parsed) {
		return nil, fmt.Errorf("merge.Merge3 shows other conflicts:\n%s", render(got))
	}
	return got, nil
}

// render is chunks as comparable diff3-like text.
func render(chunks []merge.Chunk) string {
	var b strings.Builder
	for _, ch := range chunks {
		if ch.Conflict == nil {
			b.WriteString(strings.Join(ch.Lines, ""))
			continue
		}
		b.WriteString("<<<\n" + strings.Join(ch.Conflict.Source, "") + "|||\n" + strings.Join(ch.Conflict.Base, "") +
			"===\n" + strings.Join(ch.Conflict.Target, "") + ">>>\n")
	}
	return b.String()
}

// splitText is an accepted text's lines, terminators dropped.
func splitText(t string) []string {
	t = strings.TrimSuffix(t, "\n")
	if t == "" {
		return nil
	}
	return strings.Split(t, "\n")
}

// joinLines is lines as one comparable text, terminators dropped.
func joinLines(ls []string) string {
	return strings.Join(merge.TrimEOLs(ls), "\n")
}

// Expressible reports whether ids can resolve c to want: want interleaves
// an ordered choice of the target's lines with one of the source's.
func Expressible(c *merge.Conflict, want []string) bool {
	t, s := merge.TrimEOLs(c.Target), merge.TrimEOLs(c.Source)
	if len(want) == 0 {
		return len(t) == 0 && len(s) == 0
	}
	// A state is how far into each side the lines of want so far reach:
	// target lines before a and source lines before b are used up.
	type state struct{ a, b int }
	cur := map[state]bool{{}: true}
	for _, w := range want {
		next := map[state]bool{}
		for st := range cur {
			for a := st.a; a < len(t); a++ {
				if t[a] == w {
					next[state{a + 1, st.b}] = true
				}
			}
			for b := st.b; b < len(s); b++ {
				if s[b] == w {
					next[state{st.a, b + 1}] = true
				}
			}
		}
		if len(next) == 0 {
			return false
		}
		cur = next
	}
	return true
}

// ConflictResolver is the call a conflict eval makes. The real
// *inference.Client satisfies it.
type ConflictResolver interface {
	ResolveConflict(ctx context.Context, req inference.ConflictRequest) inference.ConflictResult
}

// Request is the case's call, rendered as production renders it, from
// template (empty: the compiled-in prompt; a draft otherwise).
func (l LoadedConflictCase) Request(template string, count inference.CountFunc) inference.ConflictRequest {
	if template == "" {
		template = prompts.MergeConflict
	}
	hunks := make([]inference.ConflictHunk, 0, len(l.asked))
	for _, a := range l.asked {
		before, after := merge.Context(l.chunks[a.file], a.k)
		hunks = append(hunks, inference.HunkOf(a.file, a.conflict, before, after))
	}
	prompt := inference.RenderConflictPrompt(inference.ConflictPromptRequest{
		Template: template, TargetBranch: l.TargetBranch, SourceBranch: l.SourceBranch,
		Hunks: hunks, Language: lang.EN,
	})
	return inference.ConflictRequest{Prompt: prompt,
		PromptTokens: count(inference.ConflictSystemMessage) + count(prompt)}
}

// The verdicts of one answer.
const (
	// VerdictCorrect: every conflict resolved to an accepted text.
	VerdictCorrect = "correct"
	// VerdictDeclined: feasible false. Right on a control, a harmless
	// miss elsewhere.
	VerdictDeclined = "declined"
	// VerdictRejected: production would drop the answer (unparseable, or
	// merge.ReadAnswer refuses it). Nothing pushed.
	VerdictRejected = "rejected"
	// VerdictWrong: a valid answer production would push, with a text no
	// case accepts, or any valid answer on a control. The number that
	// counts.
	VerdictWrong = "wrong"
	// VerdictIncomplete: the call did not complete, even redrawn.
	VerdictIncomplete = "incomplete"
)

// ConflictAnswer is one sample of one case.
type ConflictAnswer struct {
	Verdict string
	// Why is the verdict's detail: the reject reason, the model's reason
	// for declining, the conflict answered wrongly, the call error.
	Why string `json:",omitempty"`
	// Lines is the ids the model chose per conflict, as answered.
	Lines            map[int][]string `json:",omitempty"`
	Redraws          int              `json:",omitempty"`
	Seconds          float64
	PromptTokens     int64
	CachedTokens     int64
	CompletionTokens int64
	NanoUSD          *int64 `json:",omitempty"`
}

// ScoreConflict judges one answer to a case.
func ScoreConflict(l LoadedConflictCase, res inference.ConflictResult) ConflictAnswer {
	out := ConflictAnswer{Lines: res.Resolution.Lines}
	switch {
	case res.Outcome == inference.OutcomeUnparseable:
		out.Verdict, out.Why = VerdictRejected, "unparseable"
		return out
	case res.Err != nil || res.Outcome != inference.OutcomeOK:
		out.Verdict, out.Why = VerdictIncomplete, "outcome "+res.Outcome.String()
		if res.Err != nil {
			out.Why += ": " + res.Err.Error()
		}
		return out
	case !res.Resolution.Feasible:
		out.Verdict, out.Why = VerdictDeclined, res.Resolution.Reason
		return out
	}
	cs := make([]*merge.Conflict, len(l.asked))
	for i, a := range l.asked {
		cs[i] = a.conflict
	}
	refs, why := merge.ReadAnswer(cs, res.Resolution.Lines)
	if why != "" {
		out.Verdict, out.Why = VerdictRejected, why
		return out
	}
	// Production assembles every file before it commits, and drops the
	// answer when a file will not assemble (a line without a newline
	// landing inside it). Its other refusal, an answer too long to quote,
	// depends on the reply text and is not modelled.
	if why := l.assemble(refs); why != "" {
		out.Verdict, out.Why = VerdictRejected, why
		return out
	}
	if l.Control {
		out.Verdict, out.Why = VerdictWrong, "control answered"
		return out
	}
	for i, a := range l.asked {
		got := joinLines(merge.Pick(a.conflict, refs[i]))
		if !a.accept.matches(got) {
			out.Verdict, out.Why = VerdictWrong, fmt.Sprintf("%s#%d: %q", a.file, a.k+1, got)
			return out
		}
	}
	out.Verdict = VerdictCorrect
	return out
}

// assemble merges each file with code's resolutions and the answer's
// refs (in asked order), as production does before it commits; why is
// empty when every file assembles.
func (l LoadedConflictCase) assemble(refs [][]merge.Ref) (why string) {
	next := 0
	for _, f := range l.Files {
		chunks := l.chunks[f.Path]
		cs := merge.Conflicts(chunks)
		res := make([][]merge.Ref, len(cs))
		for k, c := range cs {
			if next < len(l.asked) && l.asked[next].file == f.Path && l.asked[next].k == k {
				res[k] = refs[next]
				next++
				continue
			}
			res[k] = merge.TrivialResolution(c)
		}
		if _, err := merge.Assemble(chunks, res); err != nil {
			return fmt.Sprintf("%s: %v", f.Path, err)
		}
	}
	return ""
}

// ConflictCaseResult is every sample of one case.
type ConflictCaseResult struct {
	Case    string
	Origin  string
	Control bool
	// Asked is how many conflicts went to the model, Trivial how many
	// code resolved.
	Asked, Trivial int
	Answers        []ConflictAnswer
}

// ConflictRun is a conflict eval's report.
type ConflictRun struct {
	Model, Effort string
	Samples       int
	Cases         []ConflictCaseResult
	Seconds       float64
}

// RunConflicts answers every case samples times, parallel calls in
// flight. The calls share no case prefix worth waiting for: the cases
// differ from the first conflict on. template is Request's.
func RunConflicts(ctx context.Context, client ConflictResolver, cases []LoadedConflictCase, template string,
	count inference.CountFunc, samples, parallel int, progress io.Writer) ConflictRun {
	start := time.Now()
	out := ConflictRun{Samples: samples, Cases: make([]ConflictCaseResult, len(cases))}
	reqs := make([]inference.ConflictRequest, len(cases))
	for i, c := range cases {
		reqs[i] = c.Request(template, count)
		out.Cases[i] = ConflictCaseResult{Case: c.Name, Origin: c.Origin, Control: c.Control,
			Asked: len(c.asked), Trivial: c.trivial, Answers: make([]ConflictAnswer, samples)}
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
				a := drawConflict(ctx, client, cases[i], reqs[i])
				mu.Lock()
				out.Cases[i].Answers[s] = a
				mu.Unlock()
				if progress != nil {
					mu.Lock()
					_, _ = fmt.Fprintf(progress, "%-40s sample %d: %s (%.0fs)\n", cases[i].Name, s+1, a.Verdict, a.Seconds)
					mu.Unlock()
				}
			}()
		}
	}
	wg.Wait()
	out.Seconds = time.Since(start).Seconds()
	return out
}

// drawConflict is one answer, drawn again while the call does not
// complete, up to maxRedraws times. An unparseable answer is an answer
// (production drops it), never redrawn.
func drawConflict(ctx context.Context, client ConflictResolver, c LoadedConflictCase, req inference.ConflictRequest) ConflictAnswer {
	var a ConflictAnswer
	for redraws := 0; ; redraws++ {
		t := time.Now()
		res := client.ResolveConflict(ctx, req)
		a = ScoreConflict(c, res)
		a.Seconds = time.Since(t).Seconds()
		a.PromptTokens, a.CachedTokens, a.CompletionTokens = res.Cost.PromptTokens, res.Cost.CachedTokens, res.Cost.CompletionTokens
		a.NanoUSD = res.Cost.NanoUSD
		a.Redraws = redraws
		if a.Verdict != VerdictIncomplete || redraws == maxRedraws || ctx.Err() != nil {
			return a
		}
	}
}

// verdictTally counts verdicts.
type verdictTally map[string]int

func (t verdictTally) total() int {
	n := 0
	for _, v := range t {
		n += v
	}
	return n
}

// Report writes the run: a line per case, then the totals by kind and
// origin.
func (r ConflictRun) Report(w io.Writer) {
	verdicts := []string{VerdictCorrect, VerdictDeclined, VerdictRejected, VerdictWrong, VerdictIncomplete}
	_, _ = fmt.Fprintf(w, "\n%-40s %-8s %-7s", "case", "origin", "kind")
	for _, v := range verdicts {
		_, _ = fmt.Fprintf(w, " %10s", v)
	}
	_, _ = fmt.Fprintln(w)
	byKind := map[string]verdictTally{}
	var secs, maxSecs float64
	var calls, redraws int
	var prompt, cached, completion, nano int64
	priced := 0
	for _, c := range r.Cases {
		kind := "solve"
		if c.Control {
			kind = "control"
		}
		t := verdictTally{}
		for _, a := range c.Answers {
			t[a.Verdict]++
			for _, key := range []string{kind, c.Origin + "/" + kind} {
				if byKind[key] == nil {
					byKind[key] = verdictTally{}
				}
				byKind[key][a.Verdict]++
			}
			calls++
			redraws += a.Redraws
			secs += a.Seconds
			maxSecs = max(maxSecs, a.Seconds)
			prompt += a.PromptTokens
			cached += a.CachedTokens
			completion += a.CompletionTokens
			if a.NanoUSD != nil {
				nano += *a.NanoUSD
				priced++
			}
		}
		_, _ = fmt.Fprintf(w, "%-40s %-8s %-7s", c.Case, c.Origin, kind)
		for _, v := range verdicts {
			_, _ = fmt.Fprintf(w, " %10d", t[v])
		}
		_, _ = fmt.Fprintln(w)
	}
	keys := make([]string, 0, len(byKind))
	for k := range byKind {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	_, _ = fmt.Fprintln(w)
	for _, k := range keys {
		t := byKind[k]
		_, _ = fmt.Fprintf(w, "%-57s", k)
		for _, v := range verdicts {
			_, _ = fmt.Fprintf(w, " %10d", t[v])
		}
		_, _ = fmt.Fprintf(w, "  of %d\n", t.total())
	}
	if calls > 0 {
		_, _ = fmt.Fprintf(w, "\n%d answers, %d redraws, %.0fs wall; per answer %.1fs mean, %.1fs max\n",
			calls, redraws, r.Seconds, secs/float64(calls), maxSecs)
		_, _ = fmt.Fprintf(w, "tokens per answer: %d prompt (%d cached), %d completion\n",
			prompt/int64(calls), cached/int64(calls), completion/int64(calls))
		if priced > 0 {
			_, _ = fmt.Fprintf(w, "cost: %s USD over %d priced answers\n", inference.FormatUSD6(nano), priced)
		} else {
			_, _ = fmt.Fprintln(w, "cost: unpriced")
		}
	}
}

// ErrWrong reports answers production would have pushed wrong.
func (r ConflictRun) ErrWrong() error {
	var errs []error
	for _, c := range r.Cases {
		for i, a := range c.Answers {
			if a.Verdict == VerdictWrong {
				errs = append(errs, fmt.Errorf("%s sample %d: %s", c.Case, i+1, a.Why))
			}
		}
	}
	return errors.Join(errs...)
}

// ErrIncomplete reports answers that never came.
func (r ConflictRun) ErrIncomplete() error {
	var errs []error
	for _, c := range r.Cases {
		for i, a := range c.Answers {
			if a.Verdict == VerdictIncomplete {
				errs = append(errs, fmt.Errorf("%s sample %d: %s", c.Case, i+1, a.Why))
			}
		}
	}
	return errors.Join(errs...)
}
