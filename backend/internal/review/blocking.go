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
// run may post: a claim CheckBlocking accepts is a task for now; capTasks
// then applies the cap and the switches, before the sort, so the sort puts
// only a real task first among its severity and the comment cap never cuts
// it for a plain finding.
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

// capTasks downgrades the tasks past inference.MaxBlockingTasks, or every
// one when tasks are off. In the model's order, which is post order among
// tasks: every task is an `issue`, and the sort is stable.
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
	switch {
	case rule == "":
		rule = "unknown id " + strings.TrimSpace(*f.BlockingRule)
	case outcome == inference.BlockingUnknownRule:
		// A known rule refused as unknown: a nested file's rule that does
		// not cover this file (CheckBlocking).
		rule += " (its rules file does not cover " + f.File + ")"
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
// failed reads as no rules and would resolve every task. rootMarked holds
// the root's marked rule texts, those past the cap too: a rule pushed past
// it is still marked, and its task stays.
//
// A task's rule came from the root or from an AGENTS.md above its file;
// the row keeps the text only. So a text the root does not mark is looked
// for in the files above the task's file at the PR head (stillMarked:
// memoized probes, mostly no call), and only those: the same text in a
// sibling module's file does not cover it. Anything not known for sure
// (a failed read, a file over the byte cap, a cancelled run) keeps the
// task: missing information is no removal.
//
// name is the run's rules file name, the one the target was read under.
func (r *Reviewer) retireRemovedRules(ctx context.Context, key store.PRKey, prTag, commit, name string, existing []store.Finding,
	rootMarked map[string]bool, known bool) []store.Finding {
	if !r.tasksOn() || !known {
		return existing
	}
	out := slices.Clone(existing)
	now := time.Now()
	for i := range out {
		f := &out[i]
		if f.ClosedAt != nil || !f.IsTask() || f.BitbucketCommentID == nil || f.BlockingRule == nil ||
			rootMarked[inference.NormalizeRule(*f.BlockingRule)] {
			continue
		}
		if marked, why := r.stillMarked(ctx, key, prTag, commit, name, f.FilePath, inference.NormalizeRule(*f.BlockingRule)); marked || why != "" {
			if why != "" {
				r.log.InfoContext(ctx, fmt.Sprintf("%s: task %d on %s:%d: rule %q kept, %s",
					prTag, *f.BitbucketCommentID, f.FilePath, f.LineNumber, *f.BlockingRule, why))
			}
			continue
		}
		err := r.bitbucket.ResolveTask(ctx, key.Project, key.Repo, key.PRID, *f.BitbucketCommentID)
		switch {
		case bitbucket.Status(err) == http.StatusNotFound:
			// Deleted: closed for this run, never stamped, as closeThreads does.
			f.ClosedAt = &now
			r.log.InfoContext(ctx, fmt.Sprintf("%s: task %d on %s:%d: its rule left %s, the task is deleted",
				prTag, *f.BitbucketCommentID, f.FilePath, f.LineNumber, name))
		case err != nil:
			r.log.WarnContext(ctx, fmt.Sprintf("%s: task %d on %s:%d: its rule left %s, resolve failed: %v, tried again next run",
				prTag, *f.BitbucketCommentID, f.FilePath, f.LineNumber, name, err))
		default:
			// Settled, and marked so the violation, if the model reports it
			// under the rule's new wording, is posted as a task again
			// rather than held back as a repeat (dedupe).
			safeDBErr(ctx, r.log, "MarkRuleRemoved", func() error { return r.store.MarkRuleRemoved(ctx, f.ID) })
			removed := string(inference.BlockingRuleRemoved)
			f.ClosedAt, f.BlockingOutcome = &now, &removed
			r.log.InfoContext(ctx, fmt.Sprintf("%s: task %d on %s:%d resolved: its rule left %s (%q)",
				prTag, *f.BitbucketCommentID, f.FilePath, f.LineNumber, name, *f.BlockingRule))
		}
	}
	return out
}

// stillMarked reports whether an AGENTS.md below the root that covers file
// marks text at commit. why is non-empty when that is not known for sure:
// then the task stays, whatever marked says. Rows written before paths were
// resolved keep the model's a/ or b/ prefix (inference.ShownPath), and a
// real top-level a/ or b/ directory keeps it too: both readings are
// probed, the rule kept if either marks it.
func (r *Reviewer) stillMarked(ctx context.Context, key store.PRKey, prTag, commit, name, file, text string) (marked bool, why string) {
	if commit == "" {
		return false, "no PR head to read its " + name + " files at"
	}
	files := []string{file}
	if s := inference.StripSidePrefix(file); s != file {
		files = append(files, s)
	}
	nested, left, incomplete := r.lookupNested(ctx, key.Project, key.Repo, commit, prTag, name, files, true)
	switch {
	case ctx.Err() != nil:
		return false, "run cancelled"
	case incomplete != "":
		return false, name + " files above the file not all read (" + incomplete + ")"
	case len(left) > 0:
		return false, left[0].Path + " left out (" + left[0].Reason + ")"
	}
	sources := make([]inference.RuleSource, 0, len(nested))
	for _, n := range nested {
		sources = append(sources, inference.RuleSource{Dir: n.Dir, Text: n.Content})
	}
	rules, over := inference.ExtractBlockingRulesFrom(sources)
	for _, rule := range rules {
		if rule.Text == text {
			return true, ""
		}
	}
	return slices.Contains(over, text), ""
}

// targetAgentsMD is the target branch's AGENTS.md, where the blocking rules
// come from: a PR that deletes a marker must not unblock itself, and one
// that adds a rule gets it once merged. known is false on any failure but
// a 404 (no AGENTS.md there): an unread file is no proof a rule left.
// Read on every review, tasks on or off: the rules also go to the prompt.
//
// name is the run's rules file name (the team's choice): renamed before the
// file is, a 404 here resolves the open blocking tasks as a deleted file
// would (user, 2026-11-23: documented, not prevented).
func (r *Reviewer) targetAgentsMD(ctx context.Context, project, repo, branch, name string) (string, bool) {
	if branch == "" {
		return "", false
	}
	body, err := r.bitbucket.FetchFileContent(ctx, project, repo, branch, name)
	switch {
	case bitbucket.Status(err) == http.StatusNotFound:
		return "", true
	case err != nil:
		if ctx.Err() == nil {
			r.log.WarnContext(ctx, fmt.Sprintf("%s of %s unreadable: no blocking rules this review, no task resolved for a removed one: %v", name, branch, err))
		}
		return "", false
	}
	return body, true
}

// blockingRules reads the review's rules off the target branch's root
// AGENTS.md and the nested ones the PR head carries (in path order), root
// first, and logs them. A nested file's rules are read at the PR head, not
// the target: no Bitbucket call beyond the ones the prompt made, and a PR
// that deletes its own nested rule unblocks itself, accepted (user,
// 2026-11-22: "das ist jetzt wirklich nicht wichtig"). rootMarked is every
// rule text the root marks exactly as BlockingRule.Text holds it
// (normalized and capped, so the stored BlockingRule of a finding
// matches), past the cap too, for retireRemovedRules.
func (r *Reviewer) blockingRules(ctx context.Context, prTag, name, agentsMD string, nested []nestedAgentsMD) (rules []inference.BlockingRule, rootMarked map[string]bool) {
	sources := []inference.RuleSource{{Text: agentsMD}}
	for _, n := range nested {
		sources = append(sources, inference.RuleSource{Dir: n.Dir, Text: n.Content})
	}
	rules, over := inference.ExtractBlockingRulesFrom(sources)
	if len(rules) > 0 {
		root := 0
		for _, rule := range rules {
			if rule.Dir == "" {
				root++
			}
		}
		r.log.InfoContext(ctx, fmt.Sprintf("%s: %d blocking rule(s): %d in the root %s, %d in nested ones",
			prTag, len(rules), root, name, len(rules)-root))
	}
	for _, rule := range rules {
		if rule.Dir != "" {
			r.log.InfoContext(ctx, fmt.Sprintf("%s: blocking rule %s from %s%s covers %s only: %q", prTag, rule.ID, rule.Dir, name, rule.Dir, rule.Text))
		}
	}
	if len(over) > 0 {
		r.log.InfoContext(ctx, fmt.Sprintf("%s: %d more [blocking] rule(s) past the cap of %d left plain guidelines",
			prTag, len(over), inference.MaxBlockingRules))
	}
	// The root's alone: a nested rule's task is judged by the files above
	// its own file (retireRemovedRules).
	rootRules, rootOver := inference.ExtractBlockingRules(agentsMD)
	rootMarked = make(map[string]bool, len(rootRules)+len(rootOver))
	for _, rule := range rootRules {
		rootMarked[rule.Text] = true
	}
	for _, text := range rootOver {
		rootMarked[text] = true
	}
	return rules, rootMarked
}
