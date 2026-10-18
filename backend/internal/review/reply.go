package review

import (
	"context"
	"fmt"
	"strings"

	"github.com/trick77/noergler/internal/diff"
	"github.com/trick77/noergler/internal/httpstats"
	"github.com/trick77/noergler/internal/inference"
	"github.com/trick77/noergler/internal/store"
	"github.com/trick77/noergler/internal/webhook"
)

// Caps on what a finding reply carries and posts. The file diff is context
// for one finding, not a review: a PR-sized file is cut, not refused.
const (
	replyFileDiffRunes = 40_000
	replyTextRunes     = 2_000
	replyPostRunes     = 1_500
	disputeTextRunes   = 500
)

// HandleFindingReply answers a reply on the thread of one of noergler's
// inline findings: one call, one reply, once per finding.
//
// The route sends every reply that names no bot; this decides. Only a reply
// whose parent is a stored, unfixed finding of this PR, written by someone
// other than the bot, with text, on an open PR, and the first such reply on
// that finding, gets an answer. Everything else returns before any
// Bitbucket or gateway call.
//
// A dispute never resolves or withdraws the finding: the author resolves
// the thread. It is stored, and later prompts show it beside the finding so
// the issue is not raised again. The reply is author text: intent, never
// evidence, never a reason to drop a finding.
func (r *Reviewer) HandleFindingReply(ctx context.Context, payload *webhook.Payload) {
	comment := payload.Comment
	if comment == nil || payload.CommentParentID == 0 {
		return
	}
	// Our own replies ("Fixed in ...") come back as comment events too.
	if strings.EqualFold(comment.Author.Name, r.bitbucket.BotUsername()) {
		return
	}
	text := strings.TrimSpace(comment.Text)
	if text == "" {
		return
	}
	pr := payload.PullRequest
	if pr.State != "" && pr.State != "OPEN" {
		return
	}
	ctx, key, prTag, ok := prContext(ctx, payload)
	if !ok {
		r.log.ErrorContext(ctx, "Could not extract project/repo from webhook payload")
		return
	}
	f := safeDB(ctx, r.log, "FindingByComment", func() (*store.Finding, error) {
		return r.store.FindingByComment(ctx, key, payload.CommentParentID)
	})
	if f == nil {
		r.log.DebugContext(ctx, prTag+": reply is not on a finding thread")
		return
	}
	where := fmt.Sprintf("%s:%d", f.FilePath, f.LineNumber)
	if f.RepliedAt != nil {
		r.log.InfoContext(ctx, fmt.Sprintf("%s: reply on finding %s not answered: the thread was answered already", prTag, where))
		return
	}

	ctx, httpCounter := httpstats.WithScope(ctx)
	defer r.logHTTPTotals(ctx, prTag, httpCounter)
	project, repo := key.Project, key.Repo

	// Without the diff the finding still stands on its own text: a failed
	// fetch narrows the context, it does not skip the answer.
	fileDiff := ""
	if raw, err := r.bitbucket.FetchPRDiff(ctx, project, repo, pr.ID, 0); err != nil {
		r.log.WarnContext(ctx, fmt.Sprintf("%s: diff for the finding reply unavailable: %v", prTag, err))
	} else {
		fileDiff = truncateRunes(fileDiffOf(raw, f.FilePath), replyFileDiffRunes)
	}
	agentsMD := r.fetchRepoInstructions(ctx, project, repo, pr.FromRef.LatestCommit, pr.ToRef.DisplayID)

	prompt := inference.RenderFindingReplyPrompt(inference.FindingReplyPromptRequest{
		Template:         r.replyTmpl,
		RepoInstructions: agentsMD,
		Finding:          fmt.Sprintf("%s [%s]\n\n%s", where, f.Severity, f.CommentText),
		FileDiff:         fileDiff,
		Reply:            truncateRunes(text, replyTextRunes),
	})
	res := r.llm.FindingReply(ctx, inference.FindingReplyRequest{
		Prompt:       prompt,
		PromptTokens: r.tokens.Count(inference.MentionSystemMessage) + r.tokens.Count(prompt),
	})
	if res.Outcome == inference.OutcomeOK || res.Outcome == inference.OutcomeUnparseable {
		r.logCost(ctx, prTag, res.Cost)
	}
	// Nothing posted, nothing stamped: a later reply on the thread may try
	// again.
	if res.Outcome != inference.OutcomeOK {
		r.log.WarnContext(ctx, fmt.Sprintf("%s: reply on finding %s not answered: %s %v", prTag, where, res.Outcome, res.Err))
		return
	}

	body := truncateRunes(res.Reply.Reply, replyPostRunes)
	agentsLine := "none"
	if line := res.Reply.AgentsMDLine; line != "" {
		switch reason := inference.CheckAgentsMDLine(line, agentsMD); {
		case res.Reply.Intent != "disputes":
			agentsLine = "dropped (not a dispute)"
		case reason != "":
			agentsLine = "dropped (" + reason + ")"
		default:
			agentsLine = "suggested"
			body += "\n\nIf this holds project-wide, consider adding to `AGENTS.md`:\n\n```markdown\n" + line + "\n```"
		}
	}
	if err := r.bitbucket.ReplyToComment(ctx, project, repo, pr.ID, comment.ID, body); err != nil {
		r.log.ErrorContext(ctx, fmt.Sprintf("%s: reply on finding %s failed to post: %v", prTag, where, err))
		return
	}
	var dispute *string
	if res.Reply.Intent == "disputes" {
		d := truncateRunes(text, disputeTextRunes)
		dispute = &d
	}
	safeDBErr(ctx, r.log, "MarkFindingReplied", func() error { return r.store.MarkFindingReplied(ctx, f.ID, dispute) })
	r.log.InfoContext(ctx, fmt.Sprintf("%s: answered reply on finding %s: intent %s, AGENTS.md line %s",
		prTag, where, res.Reply.Intent, agentsLine))
}

// fileDiffOf is the numbered diff of one file in a PR diff, or "" when the
// PR no longer changes it. A stored path may carry the model's a/ or b/.
func fileDiffOf(raw, path string) string {
	want := strings.TrimPrefix(strings.TrimPrefix(path, "a/"), "b/")
	for _, part := range diff.SplitByFile(raw) {
		if p := diff.ExtractPath(part); p == path || p == want {
			return diff.NumberDiff(part)
		}
	}
	return ""
}
