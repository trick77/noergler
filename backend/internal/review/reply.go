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
	// Nothing is answered on a closed PR: the free check, before any DB
	// read on the single worker. Excluded by design, not refused, so Debug.
	if pr.State != "" && pr.State != "OPEN" {
		r.log.DebugContext(ctx, fmt.Sprintf("%s: reply under comment %d not answered: PR is %s", prTag, payload.CommentParentID, pr.State))
		return !mentioned, false
	}
	// A reply on the summary, or directly under any reply the bot posted,
	// is addressed to the bot: answered like a mention, from anyone (user,
	// 2026-10-19 and 2026-10-22). A mention there is one already. Two DB
	// reads on the worker for every plain reply, accepted: replies are rare
	// next to reviews. Not on an ignored PR: a plain reply never brings a
	// PR back, only an @mention does (decided), so an unread state answers
	// nothing rather than let HandleMention reactivate.
	var st *store.SkipState
	if !mentioned {
		st = safeDB(ctx, r.log, "GetSkipState", func() (*store.SkipState, error) {
			return r.store.GetSkipState(ctx, key)
		})
		if st != nil && st.Summary != nil && st.Summary.ID == payload.CommentParentID {
			r.log.InfoContext(ctx, prTag+": reply on the summary, answered like a mention")
			return true, r.HandleMention(ctx, payload, team, sched)
		}
		if b := safeDB(ctx, r.log, "BotReplyByComment", func() (*store.BotReply, error) {
			return r.store.BotReplyByComment(ctx, key, payload.CommentParentID)
		}); b != nil {
			if st == nil || st.IgnoredAt != nil {
				r.log.InfoContext(ctx, fmt.Sprintf("%s: reply on noergler's %s answer not answered: PR is ignored or its state unread", prTag, b.Kind))
				return true, false
			}
			r.log.InfoContext(ctx, fmt.Sprintf("%s: reply on noergler's %s answer, answered like a mention", prTag, b.Kind))
			return true, r.HandleMention(ctx, payload, team, sched)
		}
	}
	// A reviewer-to-reviewer exchange under a finding is no event.
	if author == "" || !strings.EqualFold(comment.Author.Name, author) {
		r.log.DebugContext(ctx, fmt.Sprintf("%s: reply under comment %d not answered: not the PR's author", prTag, payload.CommentParentID))
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
		r.log.DebugContext(ctx, fmt.Sprintf("%s: reply under comment %d is not on a finding thread", prTag, payload.CommentParentID))
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
			if st != nil && st.IgnoredAt != nil {
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
	// The per-PR cap holds for every reply, mention or not. A mention over it
	// goes on to the Q&A, which refuses it with the banner, before anything
	// reactivates the PR; a plain reply passes its cheap checks first.
	atCap := fmt.Sprintf("PR cost at the $%.2f cap", r.cfg.MaxPRCostUSD)
	if mentioned {
		if _, _, over := r.overCap(ctx, key); over {
			return notAnswered(atCap)
		}
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
	} else if st != nil && st.IgnoredAt != nil {
		// The author removed the summary: the bot leaves the PR alone until
		// a mention brings it back. st was read above, on this same path.
		return notAnswered("PR is ignored")
	}
	// A plain reply at the cap: the banner, no answer. Fail open.
	if !mentioned {
		if rowID, cumulative, over := r.overCap(ctx, key); over {
			r.refuseOverCap(ctx, key, prTag, author, "reply on finding "+where, rowID, cumulative, false)
			return notAnswered(atCap)
		}
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
		excerpt = note + lang.TruncateRunes(excerptAround(content, f.LineNumber, replyExcerptLines), replyExcerptRunes)
	}
	// AGENTS.md from the same commit, so code and guidelines match.
	agentsMD := r.fetchRepoInstructions(ctx, project, repo, at, pr.ToRef.Branch())
	name := firstName(r.displayName, comment.Author.DisplayName)
	l := r.languageFor(ctx, author)
	// The repo's open dispute signals, read here and nowhere else: a reply
	// on a finding is the only place they are consulted. Intent is the same
	// call's verdict, so the list rides every finding reply; postFindingReply
	// acts on a match for a dispute only. A failed read lists none.
	var signalIDs []int64
	var signalRules []string
	if r.disputeSignalsOn() {
		for _, s := range safeDB(ctx, r.log, "OpenDisputeSignals", func() ([]store.DisputeSignal, error) {
			return r.store.OpenDisputeSignals(ctx, team, project, repo, r.cfg.DisputeSignalWindowDays, disputeSignalsListed)
		}) {
			signalIDs = append(signalIDs, s.ID)
			signalRules = append(signalRules, s.Rule)
		}
	}
	prompt := inference.RenderFindingReplyPrompt(inference.FindingReplyPromptRequest{
		Language:         l,
		Template:         r.replyTmpl,
		RepoInstructions: agentsMD,
		Finding:          fmt.Sprintf("%s [%s]\n\n%s", where, f.Severity, f.CommentText),
		FileExcerpt:      excerpt,
		Reply:            lang.TruncateRunes(text, replyTextRunes),
		AuthorFirstName:  name,
		DisputeSignals:   signalRules,
	})
	req := inference.FindingReplyRequest{
		Prompt:       prompt,
		PromptTokens: r.tokens.Count(inference.MentionSystemMessage) + r.tokens.Count(prompt),
	}
	job := &replyJob{
		key: key, team: team, project: project, repo: repo, prID: pr.ID, prTag: prTag, where: where,
		author:    author,
		findingID: f.ID, replyTo: comment.ID, text: text,
		agentsMD: agentsMD, signalIDs: signalIDs, counter: counter,
		name: name, namesOn: r.displayName != nil, lang: l,
	}
	infer := func(inferCtx context.Context) { job.res = r.llm.FindingReply(inferCtx, req) }
	post := func(postCtx context.Context) { r.postFindingReply(postCtx, job) }
	sched.Stage(ctx, key, team, infer, post)
	return true, true
}

// replyJob is what the posting stage needs from the reply's prepare. The
// prompt does not cross: inference holds it.
type replyJob struct {
	key           store.PRKey
	team          string
	project, repo string
	prID          int
	prTag, where  string
	// author is the PR's author, the one whose reply this is.
	author    string
	findingID int64
	replyTo   int
	text      string
	agentsMD  string
	// signalIDs are the open dispute signals the prompt listed, in order:
	// matches_signal n is signalIDs[n-1].
	signalIDs []int64
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
	body := lang.TruncateRunes(sharpS(j.lang, reply), replyPostRunes)
	// A dispute with a project-wide reason is an occurrence of a dispute
	// signal. The AGENTS.md bullet is shown only once the signal reaches the
	// team's threshold of distinct PRs in its window (user, 2026-10-31); below
	// it the answer carries no bullet at all.
	agentsLine := "none"
	var signal *signalOccurrence
	switch line := res.Reply.AgentsMDLine; {
	case line == "":
		if dispute && res.Reply.MatchesSignal > 0 {
			// The model matched a known rule yet wrote no bullet: AGENTS.md
			// states it and the review raised the finding anyway.
			r.log.InfoContext(ctx, fmt.Sprintf("%s: reply on finding %s matches dispute signal [%d] but proposes no AGENTS.md line: the review raised a finding AGENTS.md may already rule out",
				j.prTag, j.where, res.Reply.MatchesSignal))
		}
	case !dispute:
		agentsLine = "dropped (not a dispute)"
	default:
		if reason := inference.CheckAgentsMDLine(line, j.agentsMD); reason != "" {
			agentsLine = "dropped (" + reason + ")"
			break
		}
		if !r.disputeSignalsOn() {
			agentsLine = "dropped (dispute signals off)"
			break
		}
		signal = r.countDisputeSignal(ctx, j, res.Reply.MatchesSignal, line)
		agentsLine = signal.logState(r.cfg.DisputeSignalThreshold, r.cfg.DisputeSignalWindowDays)
		if signal.prs >= r.cfg.DisputeSignalThreshold {
			body += r.disputeStatement(j, signal, line)
		}
	}
	posted, err := r.bitbucket.ReplyToComment(ctx, j.project, j.repo, j.prID, j.replyTo, body)
	if err != nil {
		// The author resolved the thread between replying and our answer:
		// nothing is wrong, and nothing posted means nothing stamped.
		if bitbucket.ThreadResolved(err) {
			r.log.InfoContext(ctx, fmt.Sprintf("%s: reply on finding %s not posted: thread resolved meanwhile", j.prTag, j.where))
			return
		}
		r.log.ErrorContext(ctx, fmt.Sprintf("%s: reply on finding %s failed to post: %v", j.prTag, j.where, err))
		return
	}
	r.recordBotReply(ctx, j.key, posted, store.BotReplyReply)
	var stored *string
	if dispute {
		d := lang.TruncateRunes(j.text, disputeTextRunes)
		stored = &d
	}
	safeDBErr(ctx, r.log, "MarkFindingReplied", func() error { return r.store.MarkFindingReplied(ctx, j.findingID, stored) })
	// Posted, so recorded: nothing posted records no occurrence.
	if signal != nil {
		safeDBErr(ctx, r.log, "RecordDisputeOccurrence", func() error {
			_, err := r.store.RecordDisputeOccurrence(ctx, j.team, j.key, signal.id, j.findingID, signal.rule)
			return err
		})
	}
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
	return strings.Repeat("`", max(3, lang.BacktickRun(s)+1))
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
