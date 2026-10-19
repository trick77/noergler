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
	tokens map[string]int) []store.FileSent {
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
		add(f.Path, how, tokens[f.Path])
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

// decision is an unposted finding for the page. Its text goes with it, the
// page being its only trace; a repeat keeps its headline only (the posted
// one has the rest, which would be stored again on every push). None of
// what would be stored, headline included, when a secret-like token is in
// any of it.
func decision(f inference.ReviewFinding, what, reason string) store.FindingDecision {
	d := store.FindingDecision{File: f.File, Line: f.Line, Severity: f.Severity, Decision: what, Reason: reason}
	if what == "repeat" {
		if !inference.HasSecret(headlineOf(f)) {
			d.Headline = headlineOf(f)
		}
		return d
	}
	suggestion := ""
	if f.Suggestion != nil {
		suggestion = *f.Suggestion
	}
	if inference.AnySecret(append([]string{headlineOf(f), f.Comment, suggestion}, f.Evidence...)...) {
		return d
	}
	d.Headline, d.Comment, d.Suggestion, d.Evidence = headlineOf(f), f.Comment, suggestion, f.Evidence
	return d
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

// postedMoves is the moves of the findings that were posted, read off each
// finding itself: one moved and then dropped as a repeat, cut or refused
// leaves nothing for the findings table to show.
func postedMoves(findings []inference.ReviewFinding, postedIDs []int) []store.MovedFinding {
	out := []store.MovedFinding{}
	for i, id := range postedIDs {
		if f := findings[i]; id != 0 && f.MovedFrom != nil {
			out = append(out, store.MovedFinding{File: f.File, From: *f.MovedFrom, To: f.Line})
		}
	}
	return out
}
