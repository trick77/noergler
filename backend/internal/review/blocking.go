package review

import (
	"context"
	"fmt"
	"net/http"
	"slices"
	"strings"
	"time"

	"github.com/trick77/noergler/internal/bitbucket"
	"github.com/trick77/noergler/internal/inference"
	"github.com/trick77/noergler/internal/store"
)

// Blocking rules (user, 2026-11-15): AGENTS.md marks a rule `[blocking]`
// (inference.ExtractBlockingRules), the model names the rule a finding
// violates, and code decides: a finding that passes inference.CheckBlocking
// is posted as a Bitbucket task on its line, which blocks the merge where
// the repo's "incomplete tasks" merge check is on. Everything else stays a
// plain comment, never dropped. At most inference.MaxBlockingTasks tasks a
// review; with the instance's or the team's task switch off, none at all
// and no task call (tasksOn).
//
// A task is a finding: deduped like one, so a task the author resolved is
// never posted again; settled once resolved, like a resolved thread
// (closeThreads); resolved by noergler when a fixed claim retires it
// (retireFixed) or its rule is gone from AGENTS.md (retireRemovedRules). A
// dispute in the thread changes nothing: author text is intent, never
// evidence.

// decideBlocking settles the model's blocking claims on the findings this
// run may post, before the sort: a claim CheckBlocking accepts is a task
// for now, so the sort puts it first among its severity and the comment
// cap never cuts it for a plain finding. capTasks then applies the cap and
// the switches to what survived the cut.
func decideBlocking(findings []inference.ReviewFinding, rules []inference.BlockingRule) []inference.ReviewFinding {
	out := slices.Clone(findings)
	for i := range out {
		if !out[i].Marked() {
			continue
		}
		rule, outcome, _ := inference.CheckBlocking(out[i], rules)
		out[i].Blocking = outcome
		if rule != nil {
			out[i].BlockingText = rule.Text
		}
	}
	return out
}

// capTasks downgrades the tasks past inference.MaxBlockingTasks, in post
// order, or every one when tasks are off.
func capTasks(findings []inference.ReviewFinding, tasksOn bool) {
	n := 0
	for i := range findings {
		if !findings[i].IsTask() {
			continue
		}
		switch {
		case !tasksOn:
			findings[i].Blocking = inference.BlockingTasksOff
		case n == inference.MaxBlockingTasks:
			findings[i].Blocking = inference.BlockingCap
		default:
			n++
		}
	}
}

// logBlocking writes one line per finding the model marked that this run
// tried to post (user, 2026-11-15: the threshold is to be tuned on these):
// rule, confidence and outcome. logBlockingNotPosted covers the rest, so
// every claim gets its line.
func (r *Reviewer) logBlocking(ctx context.Context, prTag string, findings []inference.ReviewFinding, ids []int) {
	for i, f := range findings {
		posted := "posted"
		if ids[i] == 0 {
			posted = "not posted (failed)"
		}
		r.logClaim(ctx, prTag, f, posted)
	}
}

// logBlockingNotPosted is logBlocking for marked findings that never reached
// the post: dropped by validation, repeats, cut by the comment cap.
func (r *Reviewer) logBlockingNotPosted(ctx context.Context, prTag string, findings []inference.ReviewFinding, why string) {
	for _, f := range findings {
		r.logClaim(ctx, prTag, f, "not posted ("+why+")")
	}
}

// logClaim is one marked finding's line, after decideBlocking.
func (r *Reviewer) logClaim(ctx context.Context, prTag string, f inference.ReviewFinding, posted string) {
	if !f.Marked() {
		return
	}
	outcome, rule := f.Blocking, f.BlockingText
	if rule == "" {
		rule = "unknown id " + strings.TrimSpace(*f.BlockingRule)
	}
	conf := "none"
	if f.Confidence != nil {
		conf = fmt.Sprint(*f.Confidence)
	}
	r.log.InfoContext(ctx, fmt.Sprintf("%s: blocking claim on %s:%d (%s, confidence %s) %s: %s, rule %q",
		prTag, f.File, f.Line, f.Severity, conf, posted, outcome, rule))
}

// taskDone reports a task (a BLOCKER comment) someone resolved: settled,
// like a resolved thread.
func taskDone(c *bitbucket.Comment) bool {
	return c.Severity == "BLOCKER" && c.State == "RESOLVED"
}

// retireRemovedRules resolves the open tasks whose rule AGENTS.md no longer
// marks: the rule does not block anymore, so neither may its task. The
// finding is settled with it, as when a person resolves the task. Only
// when tasks are on and AGENTS.md was read for sure (known): a fetch that
// failed reads as no rules and would resolve every task.
func (r *Reviewer) retireRemovedRules(ctx context.Context, key store.PRKey, prTag string, existing []store.Finding,
	rules []inference.BlockingRule, known bool) []store.Finding {
	if !r.tasksOn() || !known {
		return existing
	}
	marked := make(map[string]bool, len(rules))
	for _, rule := range rules {
		marked[inference.NormalizeRule(rule.Text)] = true
	}
	out := slices.Clone(existing)
	now := time.Now()
	for i := range out {
		f := &out[i]
		if f.ClosedAt != nil || !f.IsTask() || f.BitbucketCommentID == nil || f.BlockingRule == nil ||
			marked[inference.NormalizeRule(*f.BlockingRule)] {
			continue
		}
		err := r.bitbucket.ResolveTask(ctx, key.Project, key.Repo, key.PRID, *f.BitbucketCommentID)
		switch {
		case bitbucket.Status(err) == http.StatusNotFound:
			// Deleted: closed for this run, never stamped, as closeThreads does.
			f.ClosedAt = &now
			r.log.InfoContext(ctx, fmt.Sprintf("%s: task %d on %s:%d: its rule left AGENTS.md, the task is deleted",
				prTag, *f.BitbucketCommentID, f.FilePath, f.LineNumber))
		case err != nil:
			r.log.WarnContext(ctx, fmt.Sprintf("%s: task %d on %s:%d: its rule left AGENTS.md, resolve failed: %v, tried again next run",
				prTag, *f.BitbucketCommentID, f.FilePath, f.LineNumber, err))
		default:
			// Settled, and marked so the violation, if the model reports it
			// under the rule's new wording, is posted as a task again
			// rather than held back as a repeat (dedupe).
			safeDBErr(ctx, r.log, "MarkRuleRemoved", func() error { return r.store.MarkRuleRemoved(ctx, f.ID) })
			removed := string(inference.BlockingRuleRemoved)
			f.ClosedAt, f.BlockingOutcome = &now, &removed
			r.log.InfoContext(ctx, fmt.Sprintf("%s: task %d on %s:%d resolved: its rule left AGENTS.md (%q)",
				prTag, *f.BitbucketCommentID, f.FilePath, f.LineNumber, *f.BlockingRule))
		}
	}
	return out
}

// targetAgentsMD is the target branch's AGENTS.md, where the blocking rules
// come from: a PR that deletes a marker must not unblock itself, and one
// that adds a rule gets it once merged. known is false on any failure but
// a 404 (no AGENTS.md there): an unread file is no proof a rule left.
// Read on every review, tasks on or off: the rules also go to the prompt.
func (r *Reviewer) targetAgentsMD(ctx context.Context, project, repo, branch string) (string, bool) {
	if branch == "" {
		return "", false
	}
	body, err := r.bitbucket.FetchFileContent(ctx, project, repo, branch, "AGENTS.md")
	switch {
	case bitbucket.Status(err) == http.StatusNotFound:
		return "", true
	case err != nil:
		if ctx.Err() == nil {
			r.log.WarnContext(ctx, fmt.Sprintf("AGENTS.md of %s unreadable: no blocking rules this review, no task resolved for a removed one: %v", branch, err))
		}
		return "", false
	}
	return body, true
}

// blockingRules reads the review's rules off AGENTS.md and logs them.
func (r *Reviewer) blockingRules(ctx context.Context, prTag, agentsMD string) []inference.BlockingRule {
	rules, over := inference.ExtractBlockingRules(agentsMD)
	if len(rules) > 0 {
		r.log.InfoContext(ctx, fmt.Sprintf("%s: %d blocking rule(s) in AGENTS.md", prTag, len(rules)))
	}
	if over > 0 {
		r.log.InfoContext(ctx, fmt.Sprintf("%s: %d more [blocking] rule(s) past the cap of %d left plain guidelines",
			prTag, over, inference.MaxBlockingRules))
	}
	return rules
}
