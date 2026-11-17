package review

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"log/slog"
	"slices"
	"strings"
	"testing"

	"github.com/trick77/noergler/internal/bitbucket"
	"github.com/trick77/noergler/internal/inference"
	"github.com/trick77/noergler/internal/store"
	"github.com/trick77/noergler/internal/webhook"
)

const blockingAgentsMD = "# Rules\n- [blocking] No SQL in controllers.\n- Be terse.\n"

// blockingHarness is a full review with one blocking rule (B1) in the
// target branch's AGENTS.md (and the PR's) and tasks on for the instance
// and the team.
func blockingHarness(t *testing.T, findings ...inference.ReviewFinding) *harness {
	t.Helper()
	h := newHarness(t, func(h *harness) {
		h.bb.files["src1:AGENTS.md"] = blockingAgentsMD
		h.bb.files["master:AGENTS.md"] = blockingAgentsMD
	})
	h.r.template += " {blocking_rules}"
	h.r.createTasks = true
	h.r.teamTasks.Store(true)
	h.llm.review = okResultWith(findings...)
	return h
}

// marked is a finding the model says violates rule id, at confidence conf.
func marked(id string, conf int, comment string) inference.ReviewFinding {
	f := finding("a.go", 2, "issue", comment)
	f.Confidence, f.BlockingRule = &conf, &id
	return f
}

func TestBlockingFindingIsPostedAsATask(t *testing.T) {
	h := blockingHarness(t, marked("B1", 95, "controller runs SQL"))

	h.r.reviewInline(context.Background(), prPayload(webhook.EventOpened), false)

	if p := h.llm.Reviews[0].Prompt; !strings.Contains(p, "- B1: No SQL in controllers.") || strings.Contains(p, "B2") {
		t.Errorf("the prompt must list the one marked rule as B1:\n%s", p)
	}
	if len(h.bb.InlineTasks) != 1 || len(h.bb.Inline) != 0 {
		t.Fatalf("tasks = %+v, comments = %+v: want one task, no comment", h.bb.InlineTasks, h.bb.Inline)
	}
	if body := h.bb.InlineTasks[0].Text; !strings.Contains(body, "**Blocking rule:** No SQL in controllers.") ||
		!strings.Contains(body, "This task blocks the merge") {
		t.Errorf("task body = %q", body)
	}
	row := h.st.Findings[0]
	if !row.IsTask() || deref(row.BlockingRule) != "No SQL in controllers." {
		t.Errorf("row = outcome %v rule %v, want task and the rule's text", deref(row.BlockingOutcome), deref(row.BlockingRule))
	}
	summary := h.bb.Posted[len(h.bb.Posted)-1].Text
	if !strings.Contains(summary, "1 open blocking task") || !strings.Contains(summary, "_(blocking task)_") {
		t.Errorf("summary must name the task:\n%s", summary)
	}
	if !strings.Contains(summary, "Request changes") && !strings.Contains(summary, "request changes") {
		t.Errorf("an open blocking task must request changes:\n%s", summary)
	}
}

// A rule the PR itself adds to AGENTS.md does not block it: the rules are
// the target branch's.
func TestRuleTheGatedPRAddsDoesNotBlockIt(t *testing.T) {
	h := blockingHarness(t, marked("B1", 95, "controller runs SQL"))
	h.bb.files["master:AGENTS.md"] = "# Rules\n"

	h.r.reviewInline(context.Background(), prPayload(webhook.EventOpened), false)

	if p := h.llm.Reviews[0].Prompt; strings.Contains(p, "Blocking rules") {
		t.Errorf("no rule on the target: no block in the prompt:\n%s", p)
	}
	if len(h.bb.InlineTasks) != 0 || len(h.bb.Inline) != 1 {
		t.Errorf("tasks = %d, comments = %d", len(h.bb.InlineTasks), len(h.bb.Inline))
	}
}

// Every claim code refuses is downgraded to a plain comment, never dropped,
// and the row says why.
func TestWeakBlockingClaimsArePlainComments(t *testing.T) {
	suggestion := marked("B1", 95, "sql, minor")
	suggestion.Severity, suggestion.Line, suggestion.Evidence = "suggestion", 2, []string{"func new() {}"}
	noConf := marked("B1", 0, "no confidence")
	noConf.Confidence = nil
	for name, c := range map[string]struct {
		f    inference.ReviewFinding
		want inference.BlockingOutcome
		rule string
	}{
		"unknown id": {marked("B7", 95, "x"), inference.BlockingUnknownRule, ""},
		"suggestion": {suggestion, inference.BlockingSeverity, "No SQL in controllers."},
		"confidence": {marked("B1", 89, "x"), inference.BlockingConfidence, "No SQL in controllers."},
		"none given": {noConf, inference.BlockingConfidence, "No SQL in controllers."},
	} {
		t.Run(name, func(t *testing.T) {
			h := blockingHarness(t, c.f)
			var logs bytes.Buffer
			h.r.log = slog.New(slog.NewTextHandler(&logs, nil))

			h.r.reviewInline(context.Background(), prPayload(webhook.EventOpened), false)

			if len(h.bb.InlineTasks) != 0 || len(h.bb.Inline) != 1 {
				t.Fatalf("tasks = %d, comments = %d: want a plain comment", len(h.bb.InlineTasks), len(h.bb.Inline))
			}
			if strings.Contains(h.bb.Inline[0].Text, "Blocking rule") {
				t.Errorf("a refused claim must not show as a rule violation: %q", h.bb.Inline[0].Text)
			}
			row := h.st.Findings[0]
			if deref(row.BlockingOutcome) != string(c.want) || deref(row.BlockingRule) != c.rule {
				t.Errorf("row = %q %q, want %q %q", deref(row.BlockingOutcome), deref(row.BlockingRule), c.want, c.rule)
			}
			if !strings.Contains(logs.String(), "blocking claim on a.go:2") || !strings.Contains(logs.String(), string(c.want)) {
				t.Errorf("the claim needs its log line:\n%s", logs.String())
			}
		})
	}
}

// At most MaxBlockingTasks a review; the rest are comments that say why.
func TestBlockingTasksAreCapped(t *testing.T) {
	var fs []inference.ReviewFinding
	for i := range inference.MaxBlockingTasks + 1 {
		fs = append(fs, marked("B1", 95, "sql "+string(rune('a'+i))))
	}
	// One finding a line apart each, all on the one diff line's evidence.
	h := blockingHarness(t, fs...)

	h.r.reviewInline(context.Background(), prPayload(webhook.EventOpened), false)

	if len(h.bb.InlineTasks) != inference.MaxBlockingTasks || len(h.bb.Inline) != 1 {
		t.Fatalf("tasks = %d, comments = %d", len(h.bb.InlineTasks), len(h.bb.Inline))
	}
	if !strings.Contains(h.bb.Inline[0].Text, "no task: this review opened its 3 tasks already") {
		t.Errorf("the capped one must say why: %q", h.bb.Inline[0].Text)
	}
}

// Off is off: no task call at all; the finding is a comment that says so.
func TestBlockingTasksOff(t *testing.T) {
	for name, off := range map[string]func(*Reviewer){
		"team":     func(r *Reviewer) { r.teamTasks.Store(false) },
		"instance": func(r *Reviewer) { r.createTasks = false },
	} {
		t.Run(name, func(t *testing.T) {
			h := blockingHarness(t, marked("B1", 95, "sql"))
			off(h.r)

			h.r.reviewInline(context.Background(), prPayload(webhook.EventOpened), false)

			if len(h.bb.InlineTasks) != 0 || len(h.bb.Inline) != 1 {
				t.Fatalf("tasks = %d, comments = %d", len(h.bb.InlineTasks), len(h.bb.Inline))
			}
			if !strings.Contains(h.bb.Inline[0].Text, "no task: tasks are switched off") {
				t.Errorf("comment = %q", h.bb.Inline[0].Text)
			}
			if got := deref(h.st.Findings[0].BlockingOutcome); got != string(inference.BlockingTasksOff) {
				t.Errorf("outcome = %q", got)
			}
		})
	}
}

// A task Bitbucket refuses (4xx) still reaches the PR, as a comment.
func TestRefusedTaskFallsBackToAComment(t *testing.T) {
	h := blockingHarness(t, marked("B1", 95, "sql"))
	h.bb.inlineTaskErr = &bitbucket.StatusError{Method: "POST", Path: "/blocker-comments", Status: 400}

	h.r.reviewInline(context.Background(), prPayload(webhook.EventOpened), false)

	if len(h.bb.Inline) != 1 || !strings.Contains(h.bb.Inline[0].Text, "no task: Bitbucket refused it") {
		t.Fatalf("comments = %+v", h.bb.Inline)
	}
	if got := deref(h.st.Findings[0].BlockingOutcome); got != string(inference.BlockingTaskFailed) {
		t.Errorf("outcome = %q", got)
	}
	if run := h.st.Runs[0]; run.FindingsPosted != 1 || run.TasksPosted != 0 {
		t.Errorf("run = %d findings, %d tasks: a task posted as a comment is a finding", run.FindingsPosted, run.TasksPosted)
	}
}

// A post is a finding or a task, never both (user, 2026-11-17): the run
// counts each once, in its own figure.
func TestRunCountsTasksApartFromFindings(t *testing.T) {
	plain := finding("a.go", 2, "issue", "plain")
	h := blockingHarness(t, plain, marked("B1", 95, "controller runs SQL"))

	h.r.reviewInline(context.Background(), prPayload(webhook.EventOpened), false)

	if len(h.bb.InlineTasks) != 1 || len(h.bb.Inline) != 1 {
		t.Fatalf("tasks = %d, comments = %d", len(h.bb.InlineTasks), len(h.bb.Inline))
	}
	if run := h.st.Runs[0]; run.FindingsPosted != 1 || run.TasksPosted != 1 {
		t.Errorf("run = %d findings, %d tasks, want 1 and 1", run.FindingsPosted, run.TasksPosted)
	}
}

// Any other failure may hide a task Bitbucket created: no comment beside
// it, failed like a comment, no row.
func TestTaskFailingOtherwiseIsNotDoubled(t *testing.T) {
	for name, err := range map[string]error{
		"5xx":       &bitbucket.StatusError{Method: "POST", Path: "/blocker-comments", Status: 502},
		"transport": errors.New("connection reset"),
	} {
		t.Run(name, func(t *testing.T) {
			h := blockingHarness(t, marked("B1", 95, "sql"))
			h.bb.inlineTaskErr = err

			h.r.reviewInline(context.Background(), prPayload(webhook.EventOpened), false)

			if len(h.bb.Inline) != 0 || len(h.st.Findings) != 0 {
				t.Errorf("comments = %+v, rows = %+v: want neither", h.bb.Inline, h.st.Findings)
			}
			if summary := h.bb.Posted[len(h.bb.Posted)-1].Text; strings.Contains(summary, "blocking task") {
				t.Errorf("a task whose post failed is no open task on the PR:\n%s", summary)
			}
		})
	}
}

// With tasks off a claim is a plain comment before the sort: it takes no
// comment-cap slot ahead of an equal finding the model listed first.
func TestClaimWithTasksOffSortsAsPlain(t *testing.T) {
	plain := finding("a.go", 2, "issue", "nil map write")
	h := blockingHarness(t, plain, marked("B1", 95, "controller runs SQL"))
	h.r.teamTasks.Store(false)
	h.r.cfg.MaxComments = 1

	h.r.reviewInline(context.Background(), prPayload(webhook.EventOpened), false)

	if len(h.bb.Inline) != 1 || !strings.Contains(h.bb.Inline[0].Text, "nil map write") {
		t.Errorf("comments = %+v: want the model's first finding", h.bb.Inline)
	}
}

// A noergler-committed fix leaves its thread open (user, 2026-11-13) but
// resolves its task: the stamped row leaves every later review, so an
// open task would hold the merge unseen.
func TestBotFixedTaskIsResolved(t *testing.T) {
	h := blockingHarness(t)
	h.st.lastCommit, h.st.hasLast = "older", true
	h.bb.commitDiff = sampleDiff
	earlierTask(h, "No SQL in controllers.")
	h.st.existing[0].BotFixCommit = ptrs("c0ffee1234567890")
	h.llm.review.Review.Fixed = []inference.FixedClaim{{ID: "F11", Evidence: []string{"func new() {}"}}}

	h.r.reviewInline(context.Background(), prPayload(webhook.EventFromRefUpdated), false)

	if !slices.Equal(h.bb.ResolvedTasks, []int{501}) || len(h.st.Fixed) != 1 || len(h.bb.Replies) != 0 {
		t.Errorf("resolved tasks = %v, fixed = %v, replies = %v: want stamped, task resolved, no reply",
			h.bb.ResolvedTasks, h.st.Fixed, h.bb.Replies)
	}
}

// A task resolved for its rule's sake holds back a plain repeat; whether
// the claim is a task is settled after the switches and the cap, so a
// claim tasks-off makes plain is held back too.
func TestRetiredTaskHoldsBackAClaimTasksOffMakesPlain(t *testing.T) {
	h := blockingHarness(t, marked("B1", 95, "controller runs SQL"))
	earlierTask(h, "No SQL in the controller layer.")
	removed := string(inference.BlockingRuleRemoved)
	h.st.existing[0].BlockingOutcome = &removed
	h.r.teamTasks.Store(false)

	h.r.reviewInline(context.Background(), prPayload(webhook.EventOpened), false)

	if len(h.bb.Inline) != 0 || len(h.bb.InlineTasks) != 0 {
		t.Errorf("comments = %+v, tasks = %+v: a plain repeat must stay held back", h.bb.Inline, h.bb.InlineTasks)
	}
}

// A task on the PR's earlier finding: the fake's open BLOCKER comment 501.
func earlierTask(h *harness, rule string) {
	cid := 501
	c := &bitbucket.Comment{ID: cid, Severity: "BLOCKER", State: "OPEN"}
	h.bb.comments[cid] = c
	h.bb.nextCommentID = 600
	task := string(inference.BlockingTask)
	h.st.existing = []store.Finding{{
		ID: 11, FilePath: "a.go", LineNumber: 2, Severity: "issue", CommentText: "controller runs SQL",
		BitbucketCommentID: &cid, BlockingRule: &rule, BlockingOutcome: &task,
	}}
}

// A fixed claim on a task resolves the task: that is what lifts the merge
// check. The reply comes first, as for any thread.
func TestFixedTaskIsResolvedAsATask(t *testing.T) {
	h := blockingHarness(t)
	h.st.lastCommit, h.st.hasLast = "older", true
	h.bb.commitDiff = sampleDiff
	earlierTask(h, "No SQL in controllers.")
	h.llm.review.Review.Fixed = []inference.FixedClaim{{ID: "F11", Evidence: []string{"func new() {}"}}}

	h.r.reviewInline(context.Background(), prPayload(webhook.EventFromRefUpdated), false)

	if !slices.Equal(h.bb.ResolvedTasks, []int{501}) || slices.Contains(h.bb.Calls, "resolve:501") {
		t.Errorf("resolved tasks = %v, calls = %v: want the task resolved, not the thread", h.bb.ResolvedTasks, h.bb.Calls)
	}
	if len(h.bb.Replies) != 1 {
		t.Errorf("replies = %+v, want the Fixed in reply first", h.bb.Replies)
	}
}

// The row is stamped fixed whatever the reply did, and no review reads it
// again: a task left open would hold the merge for good. Resolved anyway.
func TestFixedTaskIsResolvedWhenTheReplyFails(t *testing.T) {
	h := blockingHarness(t)
	h.st.lastCommit, h.st.hasLast = "older", true
	h.bb.commitDiff = sampleDiff
	earlierTask(h, "No SQL in controllers.")
	h.bb.replyErr = errors.New("connection reset")
	h.llm.review.Review.Fixed = []inference.FixedClaim{{ID: "F11", Evidence: []string{"func new() {}"}}}

	h.r.reviewInline(context.Background(), prPayload(webhook.EventFromRefUpdated), false)

	if !slices.Equal(h.bb.ResolvedTasks, []int{501}) || len(h.st.Fixed) != 1 {
		t.Errorf("resolved tasks = %v, fixed = %v", h.bb.ResolvedTasks, h.st.Fixed)
	}
}

// A task the author resolved is settled like a resolved thread: out of the
// summary and the verdict, and its repeat is never posted again.
func TestResolvedTaskIsSettled(t *testing.T) {
	h := blockingHarness(t, marked("B1", 95, "controller runs SQL"))
	earlierTask(h, "No SQL in controllers.")
	h.bb.comments[501].State = "RESOLVED"

	h.r.reviewInline(context.Background(), prPayload(webhook.EventOpened), false)

	if !slices.Equal(h.st.Closed, []int64{11}) {
		t.Errorf("closed = %v, want the finding stamped settled", h.st.Closed)
	}
	if len(h.bb.InlineTasks) != 0 || len(h.bb.Inline) != 0 {
		t.Errorf("tasks = %+v, comments = %+v: a resolved task is never posted again", h.bb.InlineTasks, h.bb.Inline)
	}
	if summary := h.bb.Posted[len(h.bb.Posted)-1].Text; strings.Contains(summary, "blocking task") {
		t.Errorf("a resolved task is no open task:\n%s", summary)
	}
}

// A rule AGENTS.md no longer marks blocks nothing: its open task is
// resolved and the finding settled.
func TestTaskOfARemovedRuleIsResolved(t *testing.T) {
	h := blockingHarness(t)
	earlierTask(h, "No DB writes in tests.")

	h.r.reviewInline(context.Background(), prPayload(webhook.EventOpened), false)

	if !slices.Equal(h.bb.ResolvedTasks, []int{501}) || !slices.Equal(h.st.RuleRemoved, []int64{11}) {
		t.Errorf("resolved = %v, rule removed = %v", h.bb.ResolvedTasks, h.st.RuleRemoved)
	}
}

// A reworded rule resolves the old task, and the violation the model still
// reports under the new wording is a task again, not a held-back repeat. A
// plain repeat there stays held back; every claim not posted is logged.
func TestRewordedRuleBlocksAgain(t *testing.T) {
	h := blockingHarness(t, marked("B1", 95, "controller runs SQL"))
	earlierTask(h, "No SQL in the controller layer.")
	var logs bytes.Buffer
	h.r.log = slog.New(slog.NewTextHandler(&logs, nil))

	h.r.reviewInline(context.Background(), prPayload(webhook.EventOpened), false)

	if !slices.Equal(h.bb.ResolvedTasks, []int{501}) || len(h.bb.InlineTasks) != 1 {
		t.Fatalf("resolved = %v, new tasks = %+v", h.bb.ResolvedTasks, h.bb.InlineTasks)
	}

	plain := blockingHarness(t, marked("B1", 85, "controller runs SQL"))
	earlierTask(plain, "No SQL in the controller layer.")
	plain.r.log = slog.New(slog.NewTextHandler(&logs, nil))
	plain.r.reviewInline(context.Background(), prPayload(webhook.EventOpened), false)
	if len(plain.bb.Inline) != 0 || len(plain.bb.InlineTasks) != 0 {
		t.Errorf("a plain repeat must stay held back: %+v %+v", plain.bb.Inline, plain.bb.InlineTasks)
	}
	if !strings.Contains(logs.String(), "not posted (a repeat of an earlier finding): downgraded_confidence") {
		t.Errorf("the held-back claim needs its line:\n%s", logs.String())
	}
}

// The same rule re-wrapped in AGENTS.md is the same rule; an AGENTS.md that
// could not be read is no proof the rule left; tasks off is no task call;
// the rules are the target branch's, so the PR cannot drop its own.
func TestTaskOfAKeptRuleStays(t *testing.T) {
	for name, tweak := range map[string]func(*harness){
		"rewrapped": func(h *harness) {
			h.bb.files["master:AGENTS.md"] = "- [blocking]   No SQL\n  in controllers.\n"
		},
		"unreadable": func(h *harness) {
			delete(h.bb.files, "master:AGENTS.md")
			h.bb.fileErr["master:AGENTS.md"] = errors.New("connection reset")
		},
		"tasks off": func(h *harness) {
			h.bb.files["master:AGENTS.md"] = "# Rules\n"
			h.r.teamTasks.Store(false)
		},
		// The PR's own AGENTS.md drops the rule: the target's still has it.
		"the PR removes it": func(h *harness) {
			h.bb.files["src1:AGENTS.md"] = "# Rules\n"
		},
		// Ten new rules above it: past the cap, still marked.
		"pushed past the cap": func(h *harness) {
			var b strings.Builder
			for i := range inference.MaxBlockingRules {
				fmt.Fprintf(&b, "- [blocking] Rule %d.\n", i)
			}
			h.bb.files["master:AGENTS.md"] = b.String() + "- [blocking] No SQL in controllers.\n"
		},
	} {
		t.Run(name, func(t *testing.T) {
			h := blockingHarness(t)
			earlierTask(h, "No SQL in controllers.")
			tweak(h)

			h.r.reviewInline(context.Background(), prPayload(webhook.EventOpened), false)

			if len(h.bb.ResolvedTasks) != 0 || len(h.st.Closed) != 0 {
				t.Errorf("resolved = %v, closed = %v: the task must stay", h.bb.ResolvedTasks, h.st.Closed)
			}
		})
	}
}
