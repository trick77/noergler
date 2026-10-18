package review

import (
	"context"
	"fmt"
	"net/http"
	"strings"

	"github.com/trick77/noergler/internal/bitbucket"
	"github.com/trick77/noergler/internal/httpstats"
	"github.com/trick77/noergler/internal/inference"
	"github.com/trick77/noergler/internal/lang"
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
	pr := payload.PullRequest
	author := pr.Author.User.Name
	// The cheap checks first, before a DB lookup on the single worker. These
	// replies are excluded by design, not refused, so they log at Debug: a
	// reviewer-to-reviewer exchange under a finding is no event.
	if (pr.State != "" && pr.State != "OPEN") || author == "" || !strings.EqualFold(comment.Author.Name, author) {
		r.log.DebugContext(ctx, prTag+": reply not answered: not the open PR's author")
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
	// Refused on a finding thread: without a mention the reply goes
	// unanswered and says why; a mention goes on to the Q&A.
	notAnswered := func(why string) (bool, bool) {
		if mentioned {
			return false, false
		}
		r.log.InfoContext(ctx, fmt.Sprintf("%s: reply on finding %s not answered: %s", prTag, where, why))
		return true, false
	}
	if mentioned {
		text = extractQuestion(text, bot)
	}
	// "english" under a finding sets the author's language, even on a
	// thread answered already: a command, not a reply, so it stamps nothing.
	// Without a mention it obeys what a plain reply obeys: the bot does not
	// post on a PR it was told to leave alone.
	if l, ok := languageKeyword(text); ok {
		if !mentioned {
			if auto, _ := r.autoReviewDecision(author); !auto {
				return notAnswered("language switch from an author not auto-reviewed")
			}
			if state := safeDB(ctx, r.log, "GetSkipState", func() (*store.SkipState, error) {
				return r.store.GetSkipState(ctx, key)
			}); state != nil && state.IgnoredAt != nil {
				return notAnswered("language switch on an ignored PR")
			}
		}
		r.setLanguage(ctx, key.Project, key.Repo, pr.ID, comment.ID, prTag, comment.Author.Name, l)
		return true, false
	}
	if f.RepliedAt != nil {
		return notAnswered("the thread was answered already")
	}
	if mentioned {
		// "@bot review" on a finding thread is still a review.
		if text == "" || reviewKeywords[strings.ToLower(text)] {
			return false, false
		}
	}
	if text == "" {
		return true, false
	}
	if mentioned {
		// A mention, here as on the Q&A path: it reactivates an ignored PR
		// and is the explicit override of the author lists, not of the
		// cost cap.
		r.reactivateIfIgnored(ctx, key, prTag)
	} else if auto, ignored := r.autoReviewDecision(author); !auto {
		// A plain reply is automatic: it obeys the author lists like an
		// auto-review does, from one snapshot.
		if ignored {
			return notAnswered("ignored author")
		}
		return notAnswered("not in auto-review authors")
	} else if state := safeDB(ctx, r.log, "GetSkipState", func() (*store.SkipState, error) {
		return r.store.GetSkipState(ctx, key)
	}); state != nil && state.IgnoredAt != nil {
		// The author removed the summary: the bot leaves the PR alone until
		// a mention brings it back.
		return notAnswered("PR is ignored")
	}
	// The per-PR cap holds for every reply, mention or not: at the cap the
	// PR gets the one banner and no answer. A mention goes on to the Q&A,
	// which refuses it the same way and keeps the banner. Fail open.
	if rowID, cumulative, over := r.overCap(ctx, key); over {
		if !mentioned {
			r.refuseOverCap(ctx, key, prTag, author, "reply on finding "+where, rowID, cumulative)
		}
		return notAnswered(fmt.Sprintf("PR cost at the $%.2f cap", r.cfg.MaxPRCostUSD))
	}

	ctx, counter := httpstats.WithScope(ctx)
	project, repo := key.Project, key.Repo

	// The file as the finding saw it: its line number belongs to the commit
	// it was posted at, and a later push moves the lines. A rebase can take
	// that commit away; the head is then the best there is, and says so.
	// Without any file the finding still stands on its own text: a failed
	// fetch narrows the context, it does not skip the answer.
	at, note := f.RunCommit, ""
	content, err := r.fetchFindingFile(ctx, project, repo, at, f.FilePath)
	if err != nil {
		at, note = pr.FromRef.LatestCommit, "_The reviewed commit is gone; this is the PR head, its lines may have moved._\n\n"
		content, err = r.fetchFindingFile(ctx, project, repo, at, f.FilePath)
	}
	excerpt := ""
	if err != nil {
		r.log.WarnContext(ctx, fmt.Sprintf("%s: file for the finding reply unavailable: %v", prTag, err))
	} else if n := len(fileLines(content)); f.LineNumber > n {
		excerpt = note + fmt.Sprintf("_Line %d is past the end of the file (%d lines)._", f.LineNumber, n)
	} else {
		excerpt = note + truncateRunes(excerptAround(content, f.LineNumber, replyExcerptLines), replyExcerptRunes)
	}
	// AGENTS.md from the same commit, so code and guidelines match.
	agentsMD := r.fetchRepoInstructions(ctx, project, repo, at, pr.ToRef.DisplayID)
	name := firstName(r.displayName, comment.Author.DisplayName)
	l := r.languageFor(ctx, author)
	prompt := inference.RenderFindingReplyPrompt(inference.FindingReplyPromptRequest{
		Language:         l,
		Template:         r.replyTmpl,
		RepoInstructions: agentsMD,
		Finding:          fmt.Sprintf("%s [%s]\n\n%s", where, f.Severity, f.CommentText),
		FileExcerpt:      excerpt,
		Reply:            truncateRunes(text, replyTextRunes),
		AuthorFirstName:  name,
	})
	req := inference.FindingReplyRequest{
		Prompt:       prompt,
		PromptTokens: r.tokens.Count(inference.MentionSystemMessage) + r.tokens.Count(prompt),
	}
	job := &replyJob{
		key: key, project: project, repo: repo, prID: pr.ID, prTag: prTag, where: where,
		findingID: f.ID, replyTo: comment.ID, text: text,
		agentsMD: agentsMD, counter: counter,
		name: name, namesOn: r.displayName != nil, lang: l,
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
	key           store.PRKey
	project, repo string
	prID          int
	prTag, where  string
	findingID     int64
	replyTo       int
	text          string
	agentsMD      string
	// name is the author's first name, "" when unknown; namesOn tells that
	// from names off, so a drifted directory format shows in the log.
	name    string
	namesOn bool
	lang    lang.Lang
	counter *httpstats.Counter
	res     inference.FindingReplyResult
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
	// Billed whether or not it is posted: the PR's cost counts it.
	r.recordCommentCall(ctx, j.key, store.CallReply, j.replyTo, res.Cost)
	if res.Outcome != inference.OutcomeOK {
		r.log.WarnContext(ctx, fmt.Sprintf("%s: reply on finding %s not answered: %s %v", j.prTag, j.where, res.Outcome, res.Err))
		return
	}
	dispute := res.Reply.Intent == "disputes"

	reply, nameState := guardName(res.Reply.Reply, j.name, res.Reply.Intent, j.namesOn)
	body := truncateRunes(sharpS(j.lang, reply), replyPostRunes)
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
			body += j.lang.Pick("\n\nFalls das projektweit gilt, lohnt sich ein Eintrag in `AGENTS.md`:\n\n",
				"\n\nIf this holds project-wide, consider adding to `AGENTS.md`:\n\n") +
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
	r.log.InfoContext(ctx, fmt.Sprintf("%s: answered reply on finding %s: intent %s, AGENTS.md line %s, name %s",
		j.prTag, j.where, res.Reply.Intent, agentsLine, nameState))
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
