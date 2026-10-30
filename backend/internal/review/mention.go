package review

import (
	"context"
	"errors"
	"fmt"
	"regexp"
	"strings"

	"github.com/trick77/noergler/internal/bitbucket"
	"github.com/trick77/noergler/internal/httpstats"
	"github.com/trick77/noergler/internal/inference"
	"github.com/trick77/noergler/internal/lang"
	"github.com/trick77/noergler/internal/store"
	"github.com/trick77/noergler/internal/webhook"
)

// HandleMention answers an @mention, or re-runs the review when the comment
// asks for one.
//
// THE CALLER OWNS THE TRIGGER CHECK. This function does not verify that the
// comment actually mentions the bot: extractQuestion on an unrelated comment
// returns the text unchanged, which is neither empty nor a review keyword, so
// every comment on the PR would cost a Q&A round trip. The webhook route
// owns the gate: a case-insensitive substring match of the trigger against
// the raw comment text. Wiring this up without that gate turns every comment
// into an inference call.
//
// A mention never goes incremental: the event is pr:comment:added and the
// incremental guard only fires on pr:from_ref_updated.
func (r *Reviewer) HandleMention(ctx context.Context, payload *webhook.Payload, team string, sched Scheduler) (handedOff bool) {
	comment := payload.Comment
	if comment == nil {
		return false
	}
	// Self-loop prevention: our own comments are not mentions.
	if strings.EqualFold(comment.Author.Name, r.bitbucket.BotUsername()) {
		r.log.DebugContext(ctx, "Ignoring own comment (bot)")
		return false
	}
	pr := payload.PullRequest
	if pr.State != "" && pr.State != "OPEN" {
		r.log.InfoContext(ctx, "Ignoring mention on non-open PR (state="+pr.State+")")
		return false
	}

	ctx, key, prTag, ok := prContext(ctx, payload)
	if !ok {
		r.log.ErrorContext(ctx, "Could not extract project/repo from webhook payload")
		return false
	}
	project, repo := key.Project, key.Repo

	// The Q&A path fetches a diff and posts a reply of its own, so it gets the
	// same accounting as a review. A keyword mention stages a review, which
	// opens its own scope and logs its own totals when its posting finishes;
	// this one then reports only what the mention itself spent, which is
	// two units of work reported separately rather than merged.
	ctx, httpCounter := httpstats.WithScope(ctx)
	// A staged Q&A logs its totals when its posting finishes, like a review.
	staged := false
	defer func() {
		if !staged {
			r.logHTTPTotals(ctx, prTag, httpCounter)
		}
	}()

	question := extractQuestion(comment.Text, r.bitbucket.BotUsername())

	// "@bot english" sets the commenter's own language: a keyword, decided
	// by code, never by the model, and no inference call. A setting, not
	// engagement with the PR, so it does not reactivate an ignored one.
	if l, ok := languageKeyword(question); ok {
		r.setLanguage(ctx, project, repo, pr.ID, comment.ID, prTag, comment.Author.Name, l)
		return false
	}

	// At the cap a mention stops like everything else, review or question:
	// the banner, no answer, and the PR stays as it was (an ignored one is
	// not reactivated for nothing). Under the cap, a Q&A on a PR with no row
	// writes one, so its calls are counted and capped from the first on.
	isReview := question == "" || reviewKeywords[strings.ToLower(question)]
	rowID, cumulative, over := r.overCap(ctx, key)
	if over {
		r.refuseOverCap(ctx, key, prTag, pr.Author.User.Name, "mention", rowID, cumulative, isReview)
		if isReview {
			// The dashboard shows a blocked mention review like a blocked
			// push: a skipped attempt with its reason. Not abort: the
			// deferred totals line above already logs this scope.
			r.recordAttempt(ctx, store.Attempt{Key: key, TeamSlug: r.TeamSlug, Kind: store.RunMention,
				Outcome: "skipped", Reason: string(SkipCostCap)})
		}
		return false
	}
	if rowID == 0 && !isReview {
		// The review path writes its own row; a Q&A on a PR never reviewed
		// writes one here, once.
		r.ensurePRRow(ctx, key, pr.Author.User.Name, pr.Title, pr.CreatedDate)
	}

	// Any other mention reactivates a PR ignored after its summary was
	// removed. Clearing the state (and the stale summary id, which
	// Reactivate does) lets the next review post a fresh summary and stops
	// the review guard from re-ignoring the PR.
	state := r.reactivateIfIgnored(ctx, key, prTag)

	// An empty question, or one of the review keywords, means "review this".
	// It is a review like any other, so it takes the staged path: the
	// gateway call goes to the inference pool instead of holding the worker,
	// and the PR's queue hold covers it, so it cannot run alongside a review
	// of the same PR already in flight.
	if isReview {
		r.log.InfoContext(ctx, fmt.Sprintf("Mention triggers full review (question=%q)", question))
		// The caller must keep the PR held: the staged posting is what
		// releases it, so a push or a merge for this PR cannot run
		// alongside the inference.
		return r.ReviewPullRequestStaged(ctx, payload, team, true, sched)
	}
	r.log.InfoContext(ctx, fmt.Sprintf("Handling mention Q&A on %s: %q", prTag, question))
	// The answer is the asker's, so it is in their language, not the
	// PR author's.
	l := r.languageFor(ctx, comment.Author.Name)

	// Q&A runs without the review's gates and without compression.
	rawDiff, err := r.bitbucket.FetchPRDiff(ctx, project, repo, pr.ID, 0)
	var tooLarge *bitbucket.ContentTooLarge
	if errors.As(err, &tooLarge) {
		r.log.WarnContext(ctx, fmt.Sprintf("%s: %v - mention not answered", prTag, err))
		r.reply(ctx, project, repo, pr.ID, comment.ID, fmt.Sprintf(l.Pick(
			"Der Diff dieses PRs ist grösser als %d MiB, zu gross, um Fragen dazu zu beantworten.",
			"This PR's diff exceeds %d MiB, too large to answer questions about."), tooLarge.Limit/(1024*1024)))
		return false
	}
	if err != nil {
		r.log.ErrorContext(ctx, fmt.Sprintf("Mention Q&A on %s failed: %v", prTag, err))
		return false
	}

	files := r.prepareFiles(ctx, project, repo, rawDiff, pr.FromRef.LatestCommit, prTag).files
	if len(files) == 0 {
		r.log.InfoContext(ctx, prTag+": mention not answered by the model - no reviewable files")
		r.reply(ctx, project, repo, pr.ID, comment.ID, l.Pick("Dieser PR enthält keine prüfbaren Dateien.",
			"No reviewable files in this PR."))
		return false
	}

	repoInstructions := r.fetchRepoInstructions(ctx, project, repo, pr.FromRef.LatestCommit, pr.ToRef.Branch())
	ticket, parent := r.fetchTicket(ctx, pr.FromRef.Branch(), pr.Title, prTag)
	name := firstName(r.displayName, comment.Author.DisplayName)
	replied, byOther, parentName := r.parentComment(ctx, key, prTag, payload.CommentParentID, state, comment.Author.Name)
	if byOther {
		// The answer is to someone else's comment: naming the asker would
		// address the wrong person, so it names no one (user, 2026-10-20).
		name = ""
	}

	prompt := inference.RenderMentionPrompt(inference.MentionPromptRequest{
		Template:         r.mentionTmpl,
		Question:         question,
		Files:            files,
		RepoInstructions: repoInstructions,
		TicketContext:    renderTicketContext(ticket, parent),
		PRDescription:    r.prDescription(ctx, prTag, pr),
		ParentComment:    replied,
		AskerFirstName:   name,
		Language:         l,
	})

	job := &mentionJob{
		project: project, repo: repo, prID: pr.ID, replyTo: comment.ID, lang: l, counter: httpCounter,
		name: name, namesOn: r.displayName != nil, nameWithheld: byOther, parentName: parentName,
		ask: &mentionAsk{key: key, prTag: prTag, user: comment.Author.Name, text: question, lang: l,
			from: pr.FromRef.Branch(), to: pr.ToRef.Branch()},
	}
	req := inference.MentionRequest{
		Prompt:       prompt,
		PromptTokens: r.tokens.Count(inference.MentionSystemMessage) + r.tokens.Count(prompt),
		Tools:        r.mentionTools(job.ask),
	}
	infer := func(inferCtx context.Context) { job.res = r.llm.Mention(inferCtx, req) }
	post := func(postCtx context.Context) { r.postMention(postCtx, job) }
	// The gateway call, a tool loop of up to MaxToolRounds+1 calls, goes to
	// the inference pool instead of holding the Bitbucket worker; the tools
	// read the store only, never Bitbucket.
	staged = true
	sched.Stage(ctx, key, team, infer, post)
	return true
}

// mentionJob is what a Q&A's posting stage needs from its prepare. The
// prompt does not cross: inference holds it.
type mentionJob struct {
	project, repo string
	prID, replyTo int
	// lang is the asker's at the start; a switch this turn overrides it.
	lang lang.Lang
	// name is the asker's first name, "" when unknown; namesOn as in
	// replyJob.
	name    string
	namesOn bool
	// nameWithheld: the mention replies to someone else's comment, so the
	// answer names no one; parentName is that comment's author's first name,
	// "" when unknown.
	nameWithheld bool
	parentName   string
	ask          *mentionAsk
	counter      *httpstats.Counter
	res          inference.MentionResult
}

// postMention is the Q&A's posting stage, on the worker.
func (r *Reviewer) postMention(ctx context.Context, j *mentionJob) {
	prTag := j.ask.prTag
	defer r.logHTTPTotals(ctx, prTag, j.counter)
	project, repo, prID, replyTo := j.project, j.repo, j.prID, j.replyTo
	result := j.res
	// The asker's language until a switch is stored, which happens only
	// with an answer to post: a failed loop leaves the language alone.
	l := j.lang
	// One line and one row per call, and only when one happened: the
	// too-large branch decides before any request, and a transport error
	// returns with no cost. A tool loop logs its total too; the rows sum to
	// it, so the total itself is never stored.
	for _, c := range result.Rounds {
		r.logCost(ctx, prTag, c)
		r.recordCommentCall(ctx, j.ask.key, store.CallQuestion, replyTo, c)
	}
	// Which way the tools went: without them a settings question is
	// answered with "cannot see", and prod once ran a model so unnoticed.
	switch {
	case result.ToolsOff != "":
		r.log.InfoContext(ctx, fmt.Sprintf("%s: Q&A tools withheld: %s", prTag, result.ToolsOff))
	case len(result.Rounds) > 1:
		r.log.InfoContext(ctx, fmt.Sprintf("%s: Q&A took %d calls, tools %v (%d refused over the cap), total %s",
			prTag, len(result.Rounds), result.ToolCalls, result.CappedCalls, costTotal(result.Cost)))
	case len(result.Rounds) == 1:
		r.log.InfoContext(ctx, prTag+": Q&A tools offered, none called")
	}
	// A tool loop can run for minutes: the team may have given the repo up
	// meanwhile, as in a review's post.
	if r.owns != nil && !r.owns(project, repo) {
		r.log.InfoContext(ctx, prTag+": repo no longer claimed by the team, Q&A not posted")
		return
	}

	// No review_attempts row on THIS path, deliberately. A mention asking
	// for a review delegates to ReviewPullRequestStaged above and is
	// recorded there, as kind=mention; what reaches here is the Q&A, which
	// answers a question in a comment thread, advances no commit pointer,
	// posts no findings and has no run row to link. Recording both in one
	// feed would make "what was reviewed" unanswerable without a second
	// filter nobody asked for. The Q&A reports itself in the log, as before.
	switch result.Outcome {
	case inference.OutcomeTimedOut:
		r.log.ErrorContext(ctx, fmt.Sprintf("Mention Q&A on %s aborted - no response within %.0fs",
			prTag, r.llm.CallTimeout().Seconds()))
		r.reply(ctx, project, repo, prID, replyTo, fmt.Sprintf(l.Pick(
			"⚠️ Keine Antwort vom Modell innerhalb von %d Minuten. Bitte nochmals versuchen, oder die Frage "+
				"vereinfachen, falls sie viel Kontext braucht.",
			"⚠️ No response from the model within %d minutes. Please try again, or simplify the question "+
				"if it requires a lot of context."), timeoutMinutes(r.llm)))
	case inference.OutcomeTooLarge:
		r.log.WarnContext(ctx, prTag+": mention prompt too large for the model's context window")
		reply := inference.MentionTooLargeReply(l)
		if result.ToolOverflow {
			reply = inference.MentionToolOverflowReply(l)
		}
		r.reply(ctx, project, repo, prID, replyTo, reply)
	case inference.OutcomeOK:
		// An empty answer still gets a reply: a mention is never left
		// silently unanswered.
		// The switch is stored here and confirmed by code, in the new
		// language, whatever the model wrote about it.
		prefix := ""
		if s := j.ask.switched; s != "" {
			if r.storeLanguage(ctx, prTag, j.ask.user, s) == nil {
				l, prefix = s, languageConfirmation(s)+"\n\n"
			} else {
				prefix = l.Pick("Die Sprache konnte gerade nicht gespeichert werden. Bitte später nochmals versuchen.",
					"The language could not be saved right now. Please try again later.") + "\n\n"
			}
		}
		answer := sharpS(l, result.Answer)
		if answer == "" {
			answer = inference.MentionEmptyReply(l)
		}
		var nameState string
		if j.nameWithheld {
			// Nor the parent's author: a leading address to them is cut,
			// as on `other`.
			answer, nameState = guardName(answer, j.parentName, "other", j.namesOn)
			nameState = "withheld-" + nameState
		} else {
			_, nameState = guardName(answer, j.name, "question", j.namesOn)
		}
		answer = prefix + answer
		r.reply(ctx, project, repo, prID, replyTo, answer)
		r.log.InfoContext(ctx, fmt.Sprintf("Posted Q&A reply on %s (name=%s)", prTag, nameState))
	default:
		r.log.ErrorContext(ctx, fmt.Sprintf("Mention Q&A on %s failed: %v", prTag, result.Err))
	}
}

// costTotal is a summed cost for the log: 3 decimals, or "unpriced".
func costTotal(c inference.CallCost) string {
	if c.NanoUSD == nil {
		return "unpriced"
	}
	return "$" + inference.FormatUSD3(*c.NanoUSD)
}

// reactivateIfIgnored clears the ignored state a removed summary left, so a
// mention brings the PR back. Shared by every path an @mention takes.
// It returns the state as read before any reactivation, nil when unread.
func (r *Reviewer) reactivateIfIgnored(ctx context.Context, key store.PRKey, prTag string) *store.SkipState {
	state := safeDB(ctx, r.log, "GetSkipState", func() (*store.SkipState, error) {
		return r.store.GetSkipState(ctx, key)
	})
	if state != nil && state.IgnoredAt != nil {
		safeDBErr(ctx, r.log, "Reactivate", func() error { return r.store.Reactivate(ctx, key) })
		r.log.InfoContext(ctx, prTag+": reactivating ignored PR via @mention")
	}
	return state
}

// maxParentRunes caps the replied-to comment in the prompt: a question is
// short, and a pasted log should not crowd out the diff.
const maxParentRunes = 4096

// parentComment renders the comment a mention replies to, so "answer
// Peter's question" has the question. "" on a top-level mention, under the
// summary (the bot's own, long, and no question) or when the fetch fails:
// the Q&A then runs without it. Bitbucket sends the immediate parent only,
// and only a thread's root carries the file anchor, so a reply nested under
// a reply comes without a location (known limit, as for finding replies).
// byOther is a parent neither the bot nor the asker wrote, otherName its
// author's first name ("" when unknown). state is the PR's skip state as
// HandleMention read it, nil when unread.
func (r *Reviewer) parentComment(ctx context.Context, key store.PRKey, prTag string, parentID int,
	state *store.SkipState, asker string) (text string, byOther bool, otherName string) {
	if parentID == 0 {
		return "", false, ""
	}
	if state != nil && state.Summary != nil && state.Summary.ID == parentID {
		return "", false, ""
	}
	c, err := r.bitbucket.FetchPRComment(ctx, key.Project, key.Repo, key.PRID, parentID)
	if err != nil {
		r.log.WarnContext(ctx, fmt.Sprintf("%s: replied-to comment %d not fetched, answered without it: %v", prTag, parentID, err))
		return "", false, ""
	}
	bot := strings.EqualFold(c.Author.Name, r.bitbucket.BotUsername())
	var b strings.Builder
	switch {
	case bot:
		b.WriteString("Author: noergler (this bot)\n")
	case c.Author.DisplayName != "":
		b.WriteString("Author: " + c.Author.DisplayName + "\n")
	case c.Author.Name != "":
		b.WriteString("Author: " + c.Author.Name + "\n")
	}
	if a := c.Anchor; a != nil && a.Path != "" {
		switch {
		case a.Orphaned:
			// The line it sat on is gone from the current diff: its number
			// would point at other code.
			b.WriteString("Location: " + a.Path + " (a line since changed, no longer in the diff)\n")
		case a.Line <= 0:
			b.WriteString("Location: " + a.Path + "\n")
		case a.FileType == "FROM":
			// The diff's gutter numbers the new file: an old-side line
			// number shown bare would point at other code.
			fmt.Fprintf(&b, "Location: %s, line %d of the old version (before this PR)\n", a.Path, a.Line)
		default:
			fmt.Fprintf(&b, "Location: %s:%d\n", a.Path, a.Line)
		}
	}
	full := strings.TrimSpace(c.Text)
	body := lang.TruncateRunes(full, maxParentRunes)
	cut := body != full
	// A code block left open, by the cut or by its author, would run over
	// the sections below, the question among them.
	if f := lang.OpenFence(body); f != "" {
		body += "\n" + f
	}
	if cut {
		body += "\n[... cut]"
	}
	b.WriteString("\n" + body)
	r.log.InfoContext(ctx, fmt.Sprintf("%s: mention replies to comment %d (bot=%t, anchored=%t)",
		prTag, parentID, bot, c.Anchor != nil && c.Anchor.Path != ""))
	if bot || strings.EqualFold(c.Author.Name, asker) {
		return b.String(), false, ""
	}
	return b.String(), true, firstName(r.displayName, c.Author.DisplayName)
}

// reply posts a threaded reply, logging a failure rather than propagating it:
// a mention that cannot be answered must not fail the webhook. What posted
// is recorded as the bot's, so a plain reply under it is answered.
func (r *Reviewer) reply(ctx context.Context, project, repo string, prID, parentID int, text string) {
	id, err := r.bitbucket.ReplyToComment(ctx, project, repo, prID, parentID, text)
	if err != nil {
		r.log.ErrorContext(ctx, fmt.Sprintf("Failed to post reply on %s/%s#%d: %v", project, repo, prID, err))
		return
	}
	r.recordBotReply(ctx, store.PRKey{Project: project, Repo: repo, PRID: prID}, id, store.BotReplyQuestion)
}

// recordBotReply keeps the id of a reply the bot posted. Fail open: a
// missed row only leaves a later reply under it unanswered.
func (r *Reviewer) recordBotReply(ctx context.Context, key store.PRKey, commentID int, kind string) {
	safeDBErr(ctx, r.log, "InsertBotReply", func() error { return r.store.InsertBotReply(ctx, key, commentID, kind) })
}

// extractQuestion strips the @trigger from a comment and trims the rest.
//
// The trailing word boundary must be Unicode-aware, the same rule the ticket
// pattern needs: RE2's \b is ASCII and sees a boundary before a letter like
// ü, so "@noerglerü frage" would be stripped down to "ü frage" instead of
// being left alone. This one substitutes rather than searches, so the
// trailing boundary character is captured and written back
// (TestExtractQuestionUnicodeBoundary).
func extractQuestion(text, trigger string) string {
	re := regexp.MustCompile(`(?i)@` + regexp.QuoteMeta(trigger) + `(?:$|([^\pL\pN_]))`)
	return strings.TrimSpace(re.ReplaceAllString(text, "$1"))
}
