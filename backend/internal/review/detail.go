package review

import (
	"context"
	"sort"
	"time"

	"github.com/trick77/noergler/internal/diff"
	"github.com/trick77/noergler/internal/inference"
	"github.com/trick77/noergler/internal/store"
)

// stopwatch times the stages of one review: each lap is the time since the
// previous one. The plan carries it from prepare through infer to post, on
// one goroutine at a time (the queue hands the plan over), so it needs no
// lock.
type stopwatch struct {
	last   time.Time
	stages []store.StageTime
}

func newStopwatch(start time.Time) *stopwatch { return &stopwatch{last: start} }

func (s *stopwatch) lap(name string) {
	if s == nil {
		return
	}
	now := time.Now()
	s.stages = append(s.stages, store.StageTime{Name: name, MS: now.Sub(s.last).Milliseconds()})
	s.last = now
}

// The stage names, as the run page's catalog knows them.
const (
	stageDiff      = "diff"
	stageFiles     = "files"
	stageContext   = "context"
	stagePrompt    = "prompt"
	stageQueued    = "queued"
	stageGateway   = "gateway"
	stageToPost    = "to_post"
	stageValidate  = "validate"
	stageComments  = "comments"
	stageRecordSum = "record"
)

// filesSent classifies every changed file for the page. sent is the prompt's
// final file set; the other lists are the ones the review left out, each
// for its own reason.
func filesSent(prep prepared, sent []diff.FileReviewData, deleted, renamed, overBudget []string,
	count func(string) int) []store.FileSent {
	out := []store.FileSent{}
	seen := map[string]bool{}
	add := func(path, how string, tokens int) {
		if seen[path] {
			return
		}
		seen[path] = true
		lc := prep.lines[path]
		out = append(out, store.FileSent{Path: path, Sent: how, Added: lc.Added, Removed: lc.Removed, Tokens: tokens})
	}
	skipped := map[string]bool{}
	for _, p := range prep.contentSkipped {
		skipped[p] = true
	}
	for _, f := range sent {
		how := "diff"
		switch {
		case diff.IsDeleted(f.Diff):
			how = "deleted"
		case diff.IsRenameOnly(f.Diff):
			how = "renamed"
		case f.ContentFetched:
			how = "full"
		case skipped[f.Path]:
			how = "content_skipped"
		}
		add(f.Path, how, count(inference.FormatFileEntry(f)))
	}
	for _, p := range deleted {
		add(p, "deleted", 0)
	}
	for _, p := range renamed {
		add(p, "renamed", 0)
	}
	for _, p := range overBudget {
		add(p, "over_budget", 0)
	}
	for _, p := range prep.oversized {
		add(p, "oversized", 0)
	}
	for _, p := range prep.notReviewable {
		add(p, "not_reviewable", 0)
	}
	// Biggest first: the table exists to show where the tokens went.
	sort.SliceStable(out, func(i, j int) bool { return out[i].Tokens > out[j].Tokens })
	return out
}

func promptPartsOf(a inference.AssembledPrompt, budget int) store.PromptParts {
	b := a.Breakdown
	return store.PromptParts{
		Template: b.Template, Files: b.Files, CumulativeDiff: b.CumulativeDiff,
		PreviouslyPosted: b.PreviouslyPosted, RepoInstructions: b.RepoInstructions,
		Ticket: b.Ticket, PRDescription: b.PRDescription, Total: a.PromptTokens, Budget: budget,
	}
}

func headlineOf(f inference.ReviewFinding) string {
	if f.Headline == nil {
		return ""
	}
	return *f.Headline
}

func decision(f inference.ReviewFinding, what, reason string) store.FindingDecision {
	return store.FindingDecision{File: f.File, Line: f.Line, Severity: f.Severity,
		Headline: headlineOf(f), Decision: what, Reason: reason}
}

// finishDetail takes the last lap, reads the HTTP counts and writes the
// run's detail record under its attempt. Fails open: a missing record only
// leaves the page's detail tabs empty.
func (r *Reviewer) finishDetail(ctx context.Context, plan *reviewPlan, attemptID int64, last string) {
	d := plan.detail
	if r.store == nil || attemptID == 0 || d == nil {
		return
	}
	plan.sw.lap(last)
	d.Stages = plan.sw.stages
	d.HTTP = httpCounts(plan.counter.Summarize())
	safeDBErr(ctx, r.log, "InsertRunDetail", func() error {
		return r.store.InsertRunDetail(ctx, attemptID, *d)
	})
}

// httpCounts is the counter's per-upstream totals, by name.
func httpCounts(m map[string]int) []store.HTTPCount {
	out := make([]store.HTTPCount, 0, len(m))
	for k, n := range m {
		out = append(out, store.HTTPCount{Upstream: k, Calls: n})
	}
	sort.Slice(out, func(i, j int) bool { return out[i].Upstream < out[j].Upstream })
	return out
}

// postedMoves keeps the moves of findings that were posted: validation moves
// a finding before dedupe, the cap and the post decide whether it is shown,
// and the findings table matches a move by file and posted line.
func postedMoves(moves []inference.Reanchor, findings []inference.ReviewFinding, postedIDs []int) []store.MovedFinding {
	type at struct {
		file string
		line int
	}
	posted := map[at]bool{}
	for i, id := range postedIDs {
		if id != 0 {
			posted[at{findings[i].File, findings[i].Line}] = true
		}
	}
	out := []store.MovedFinding{}
	for _, m := range moves {
		if posted[at{m.File, m.To}] {
			out = append(out, store.MovedFinding{File: m.File, From: m.From, To: m.To})
		}
	}
	return out
}

// memoCount counts each distinct text once: the files-sent log line and the
// page's file table both count every prompt entry, and the second pass would
// otherwise run the tokenizer over every file body again on the worker.
func memoCount(count func(string) int) func(string) int {
	seen := map[string]int{}
	return func(s string) int {
		if n, ok := seen[s]; ok {
			return n
		}
		n := count(s)
		seen[s] = n
		return n
	}
}
