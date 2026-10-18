package review

import (
	"context"
	"encoding/json"
	"sort"
	"time"

	"github.com/trick77/noergler/internal/diff"
	"github.com/trick77/noergler/internal/inference"
)

// runDetail is what the run detail page shows beyond the run and call rows:
// where the time went, what the prompt was made of, how each file was sent,
// and what happened to every finding the model proposed. Written as one
// JSON record per attempt, fail-open, after the summary is posted.
//
// Headlines only (decided): a finding's comment, suggestion and evidence
// never go in here. The dashboard is unauthenticated.
type runDetail struct {
	Stages           []stageTime       `json:"stages"`
	Prompt           promptParts       `json:"prompt"`
	Files            []fileSent        `json:"files"`
	CrossFile        []string          `json:"cross_file"`
	AgentsMD         agentsMDFacts     `json:"agents_md"`
	Ticket           string            `json:"ticket,omitempty"`
	ParentTicket     string            `json:"parent_ticket,omitempty"`
	PreviouslyPosted previouslyPosted  `json:"previously_posted"`
	Decisions        []findingDecision `json:"decisions"`
	Moved            []movedFinding    `json:"moved"`
	Verdict          *verdictChange    `json:"verdict,omitempty"`
	HTTP             map[string]int    `json:"http"`
}

type stageTime struct {
	Name string `json:"name"`
	MS   int64  `json:"ms"`
}

// promptParts is inference.PromptBreakdown plus the whole and the budget it
// was sized against.
type promptParts struct {
	Template         int `json:"template"`
	Files            int `json:"files"`
	CumulativeDiff   int `json:"cumulative_diff"`
	PreviouslyPosted int `json:"previously_posted"`
	RepoInstructions int `json:"repo_instructions"`
	Ticket           int `json:"ticket"`
	PRDescription    int `json:"pr_description"`
	Total            int `json:"total"`
	Budget           int `json:"budget"`
}

// fileSent is one changed file and how the review treated it. Sent is one
// of full, diff, content_skipped, deleted, renamed, oversized, over_budget,
// not_reviewable. Tokens are the prompt entry's, zero for a file not sent.
type fileSent struct {
	Path    string `json:"path"`
	Sent    string `json:"sent"`
	Added   int    `json:"added"`
	Removed int    `json:"removed"`
	Tokens  int    `json:"tokens"`
}

type agentsMDFacts struct {
	Found      bool `json:"found"`
	Tokens     int  `json:"tokens"`
	WarnTokens int  `json:"warn_tokens"`
}

type previouslyPosted struct {
	Shown int `json:"shown"`
	Total int `json:"total"`
}

// findingDecision is a proposed finding that did NOT end up posted, and
// why: dropped by validation (Reason is the DropReason), a repeat of one
// already posted, cut by the comment cap, or refused by Bitbucket. Posted
// findings are the findings table's.
type findingDecision struct {
	File     string `json:"file"`
	Line     int    `json:"line"`
	Severity string `json:"severity"`
	Headline string `json:"headline"`
	Decision string `json:"decision"`
	Reason   string `json:"reason,omitempty"`
}

type movedFinding struct {
	File string `json:"file"`
	From int    `json:"from"`
	To   int    `json:"to"`
}

type verdictChange struct {
	Model  string `json:"model"`
	Posted string `json:"posted"`
}

// stopwatch times the stages of one review: each lap is the time since the
// previous one. The plan carries it from prepare through infer to post, on
// one goroutine at a time (the queue hands the plan over), so it needs no
// lock.
type stopwatch struct {
	last   time.Time
	stages []stageTime
}

func newStopwatch(start time.Time) *stopwatch { return &stopwatch{last: start} }

func (s *stopwatch) lap(name string) {
	if s == nil {
		return
	}
	now := time.Now()
	s.stages = append(s.stages, stageTime{Name: name, MS: now.Sub(s.last).Milliseconds()})
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
	count func(string) int) []fileSent {
	out := []fileSent{}
	seen := map[string]bool{}
	add := func(path, how string, tokens int) {
		if seen[path] {
			return
		}
		seen[path] = true
		lc := prep.lines[path]
		out = append(out, fileSent{Path: path, Sent: how, Added: lc.Added, Removed: lc.Removed, Tokens: tokens})
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

func promptPartsOf(a inference.AssembledPrompt, budget int) promptParts {
	b := a.Breakdown
	return promptParts{
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

func decision(f inference.ReviewFinding, what, reason string) findingDecision {
	return findingDecision{File: f.File, Line: f.Line, Severity: f.Severity,
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
	d.HTTP = plan.counter.Summarize()
	if d.HTTP == nil {
		d.HTTP = map[string]int{}
	}
	blob, err := json.Marshal(d)
	if err != nil {
		return
	}
	safeDBErr(ctx, r.log, "InsertRunDetail", func() error {
		return r.store.InsertRunDetail(ctx, attemptID, blob)
	})
}
