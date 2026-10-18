package review

import (
	"context"
	"fmt"
	"net/http"
	"strings"

	"github.com/trick77/noergler/internal/bitbucket"
	"github.com/trick77/noergler/internal/httpstats"
	"github.com/trick77/noergler/internal/inference"
	"github.com/trick77/noergler/internal/store"
	"github.com/trick77/noergler/internal/webhook"
)

// Caps on what a finding reply carries and posts. The excerpt is context for
// one finding, not a review.
const (
	replyExcerptLines = 40
	replyExcerptRunes = 20_000
	replyTextRunes    = 2_000
	replyPostRunes    = 1_500
	// disputeTextRunes is what the previously-posted block renders of it.
	disputeTextRunes = 300
)

// HandleFindingReply answers the PR author's reply on the thread of one of
// noergler's inline findings: one call, one reply, once per finding.
//
// Only the author's: the feature is the author's case for or against a
// finding, and a reviewer's reply used up the one answer before the author
// could make it. Anyone else's @mention goes to the Q&A.
//
// handled is false when the reply is not this handler's: the thread is no
// finding's, or an @mention is better
// answered by the mention path (a review keyword, another replier, a closed
// PR, a thread answered already). The route then hands an @mention to
// HandleMention. handedOff reports, like the staged review, that the gateway
// call went to the inference pool and the posting comes back to the worker,
// so the PR's queue hold must outlive this job.
//
// A dispute never resolves or withdraws the finding: the author resolves the
// thread. It is stored, and later prompts show it beside the finding so the
// issue is not raised again. The reply is author text: intent, never
// evidence, never a reason to drop a finding.
func (r *Reviewer) HandleFindingReply(ctx context.Context, payload *webhook.Payload, team string, sched Scheduler) (handled, handedOff bool) {
	comment := payload.Comment
	if comment == nil || payload.CommentParentID == 0 {
		return false, false
	}
	bot := r.bitbucket.BotUsername()
	// Our own replies ("Fixed in ...") come back as comment events too.
	if strings.EqualFold(comment.Author.Name, bot) {
		return true, false
	}
	text := strings.TrimSpace(comment.Text)
	mentioned := strings.Contains(strings.ToLower(text), "@"+strings.ToLower(bot))
	ctx, key, prTag, ok := prContext(ctx, payload)
	if !ok {
		r.log.ErrorContext(ctx, "Could not extract project/repo from webhook payload")
		return !mentioned, false
	}
	// Only a reply directly under a finding's comment is found: Bitbucket
	// sends the immediate parent, and a reply nested under another reply
	// names that reply. Known limit, accepted: walking the thread for every
	// reply costs a fetch per comment.
	f := safeDB(ctx, r.log, "FindingByComment", func() (*store.Finding, error) {
		return r.store.FindingByComment(ctx, key, payload.CommentParentID)
	})
	if f == nil {
		r.log.DebugContext(ctx, prTag+": reply is not on a finding thread")
		return false, false
	}
	where := fmt.Sprintf("%s:%d", f.FilePath, f.LineNumber)
	pr := payload.PullRequest
	author := pr.Author.User.Name
	// Not this handler's. Without a mention the reply goes unanswered, and
	// says why; a mention goes on to the Q&A, which logs for itself.
	notAnswered := func(why string) (bool, bool) {
		if mentioned {
			return false, false
		}
		r.log.InfoContext(ctx, fmt.Sprintf("%s: reply on finding %s not answered: %s", prTag, where, why))
		return true, false
	}
	switch {
	case pr.State != "" && pr.State != "OPEN":
		return notAnswered("PR is " + strings.ToLower(pr.State))
	case author == "" || !strings.EqualFold(comment.Author.Name, author):
		return notAnswered("not the PR author")
	case f.RepliedAt != nil:
		return notAnswered("the thread was answered already")
	}
	if mentioned {
		// "@bot review" on a finding thread is still a review.
		text = extractQuestion(text, bot)
		if text == "" || reviewKeywords[strings.ToLower(text)] {
			return false, false
		}
	}
	if text == "" {
		return true, false
	}
	if mentioned {
		// A mention, here as on the Q&A path: it reactivates an ignored PR
		// and is the explicit override of the cost cap and the author lists.
		r.reactivateIfIgnored(ctx, key, prTag)
	} else if r.isIgnoredAuthor(author) {
		// A plain reply is automatic: an author the team ignores buys none.
		return notAnswered("ignored author")
	} else if state := safeDB(ctx, r.log, "GetSkipState", func() (*store.SkipState, error) {
		return r.store.GetSkipState(ctx, key)
	}); state != nil && state.IgnoredAt != nil {
		// The author removed the summary: the bot leaves the PR alone until
		// a mention brings it back.
		return notAnswered("PR is ignored")
	} else if cost := safeDB(ctx, r.log, "PRCost", func() (*int64, error) {
		return r.store.PRCost(ctx, key)
	}); cost != nil && float64(*cost)/nanoPerUSD >= r.cfg.MaxPRCostUSD {
		// The per-PR cap holds for a plain reply. Fail open: an unpriced PR
		// never blocks.
		return notAnswered(fmt.Sprintf("PR cost at the $%.2f cap", r.cfg.MaxPRCostUSD))
	}

	ctx, counter := httpstats.WithScope(ctx)
	project, repo := key.Project, key.Repo

	// The file as the finding saw it: its line number belongs to the commit
	// it was posted at, and a later push moves the lines. Without the file
	// the finding still stands on its own text: a failed fetch narrows the
	// context, it does not skip the answer.
	at := f.RunCommit
	if at == "" {
		at = pr.FromRef.LatestCommit
	}
	excerpt := ""
	if content, err := r.fetchFindingFile(ctx, project, repo, at, f.FilePath); err != nil {
		r.log.WarnContext(ctx, fmt.Sprintf("%s: file for the finding reply unavailable: %v", prTag, err))
	} else if n := len(fileLines(content)); f.LineNumber > n {
		excerpt = fmt.Sprintf("_Line %d is past the end of the file as reviewed (%d lines)._", f.LineNumber, n)
	} else {
		excerpt = truncateRunes(excerptAround(content, f.LineNumber, replyExcerptLines), replyExcerptRunes)
	}
	agentsMD := r.fetchRepoInstructions(ctx, project, repo, pr.FromRef.LatestCommit, pr.ToRef.DisplayID)
	prompt := inference.RenderFindingReplyPrompt(inference.FindingReplyPromptRequest{
		Template:         r.replyTmpl,
		RepoInstructions: agentsMD,
		Finding:          fmt.Sprintf("%s [%s]\n\n%s", where, f.Severity, f.CommentText),
		FileExcerpt:      excerpt,
		Reply:            truncateRunes(text, replyTextRunes),
	})
	req := inference.FindingReplyRequest{
		Prompt:       prompt,
		PromptTokens: r.tokens.Count(inference.MentionSystemMessage) + r.tokens.Count(prompt),
	}
	job := &replyJob{
		project: project, repo: repo, prID: pr.ID, prTag: prTag, where: where,
		findingID: f.ID, replyTo: comment.ID, text: text,
		agentsMD: agentsMD, counter: counter,
	}
	infer := func(inferCtx context.Context) { job.res = r.llm.FindingReply(inferCtx, req) }
	post := func(postCtx context.Context) { r.postFindingReply(postCtx, job) }
	if sched == nil {
		// A direct caller or a test: inline, as the mention does.
		infer(ctx)
		post(ctx)
		return true, false
	}
	sched.Stage(ctx, key, team, infer, post)
	return true, true
}

// replyJob is what the posting stage needs from the reply's prepare. The
// prompt does not cross: inference holds it.
type replyJob struct {
	project, repo string
	prID          int
	prTag, where  string
	findingID     int64
	replyTo       int
	text          string
	agentsMD      string
	counter       *httpstats.Counter
	res           inference.FindingReplyResult
}

// postFindingReply is the reply's posting stage, on the worker: the answer,
// then the stamp. Nothing posted, nothing stamped, so a later reply on the
// thread may try again.
func (r *Reviewer) postFindingReply(ctx context.Context, j *replyJob) {
	defer r.logHTTPTotals(ctx, j.prTag, j.counter)
	res := j.res
	if res.Outcome == inference.OutcomeOK || res.Outcome == inference.OutcomeUnparseable {
		r.logCost(ctx, j.prTag, res.Cost)
	}
	if res.Outcome != inference.OutcomeOK {
		r.log.WarnContext(ctx, fmt.Sprintf("%s: reply on finding %s not answered: %s %v", j.prTag, j.where, res.Outcome, res.Err))
		return
	}
	dispute := res.Reply.Intent == "disputes"

	body := truncateRunes(res.Reply.Reply, replyPostRunes)
	agentsLine := "none"
	if line := res.Reply.AgentsMDLine; line != "" {
		switch reason := inference.CheckAgentsMDLine(line, j.agentsMD); {
		case !dispute:
			agentsLine = "dropped (not a dispute)"
		case reason != "":
			agentsLine = "dropped (" + reason + ")"
		default:
			agentsLine = "suggested"
			fence := codeFence(line)
			body += "\n\nIf this holds project-wide, consider adding to `AGENTS.md`:\n\n" +
				fence + "markdown\n" + line + "\n" + fence
		}
	}
	if err := r.bitbucket.ReplyToComment(ctx, j.project, j.repo, j.prID, j.replyTo, body); err != nil {
		r.log.ErrorContext(ctx, fmt.Sprintf("%s: reply on finding %s failed to post: %v", j.prTag, j.where, err))
		return
	}
	var stored *string
	if dispute {
		d := truncateRunes(j.text, disputeTextRunes)
		stored = &d
	}
	safeDBErr(ctx, r.log, "MarkFindingReplied", func() error { return r.store.MarkFindingReplied(ctx, j.findingID, stored) })
	r.log.InfoContext(ctx, fmt.Sprintf("%s: answered reply on finding %s: intent %s, AGENTS.md line %s",
		j.prTag, j.where, res.Reply.Intent, agentsLine))
}

// fetchFindingFile reads the finding's file at the PR head. A row stored
// before paths were resolved may carry the model's a/ or b/; a real a/ or
// b/ directory is tried as written first, and only a 404 falls back.
func (r *Reviewer) fetchFindingFile(ctx context.Context, project, repo, commit, path string) (string, error) {
	content, err := r.bitbucket.FetchFileContent(ctx, project, repo, commit, path)
	if stripped := inference.StripSidePrefix(path); stripped != path && bitbucket.Status(err) == http.StatusNotFound {
		return r.bitbucket.FetchFileContent(ctx, project, repo, commit, stripped)
	}
	return content, err
}

// codeFence is a backtick fence longer than any backtick run in s, so model
// text cannot close it early and post markdown as the bot's own.
func codeFence(s string) string {
	longest, run := 0, 0
	for _, c := range s {
		if c == '`' {
			run++
			longest = max(longest, run)
			continue
		}
		run = 0
	}
	return strings.Repeat("`", max(3, longest+1))
}

// fileLines splits a file into its lines; a final newline ends the last
// line, it does not start a phantom one.
func fileLines(content string) []string {
	return strings.Split(strings.TrimSuffix(content, "\n"), "\n")
}

// excerptAround is the file's lines within n of line, numbered like the
// review's diff, so the model reads the line it is asked about.
func excerptAround(content string, line, n int) string {
	lines := fileLines(content)
	from, to := max(1, line-n), min(len(lines), line+n)
	var b strings.Builder
	for i := from; i <= to; i++ {
		fmt.Fprintf(&b, "%5d  %s\n", i, lines[i-1])
	}
	return b.String()
}
