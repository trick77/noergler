package review

import (
	"context"
	"fmt"
	"strings"

	"github.com/trick77/noergler/internal/bitbucket"
	"github.com/trick77/noergler/internal/inference"
	"github.com/trick77/noergler/internal/lang"
	"github.com/trick77/noergler/internal/store"
	"github.com/trick77/noergler/internal/webhook"
)

// The PR author can have noergler apply the suggestion of its own Q&A
// answer, as with a finding, when the question sat on a line of code (user,
// 2026-11-13: inline only, the PR author only). The Q&A's envelope says
// `apply`; the fix call, ValidateFix, the commit and the answers are the
// finding's (autofix.go).

// parentBotReply carries the bot reply the finding-reply route read for a
// reply's parent into HandleMention, so the same row is not read twice on
// the worker.
type parentBotReplyKey struct{}

type parentBotReply struct {
	parentID int
	row      *store.BotReply
}

func withParentBotReply(ctx context.Context, parentID int, b *store.BotReply) context.Context {
	return context.WithValue(ctx, parentBotReplyKey{}, parentBotReply{parentID, b})
}

// parentBotReplyFrom is the row read for parentID, and whether it was read.
func parentBotReplyFrom(ctx context.Context, parentID int) (*store.BotReply, bool) {
	v, ok := ctx.Value(parentBotReplyKey{}).(parentBotReply)
	if !ok || v.parentID != parentID {
		return nil, false
	}
	return v.row, true
}

// answerFix is a Q&A's request to apply noergler's earlier answer.
type answerFix struct {
	// answerID is noergler's answer, the suggestion applied.
	answerID int
	// gate is why the fix may not be asked for: "" when it may,
	// "not_author" when the commenter is not the PR's author (then no plan).
	gate  string
	plan  *fixPlan
	req   *inference.FindingFixRequest
	done  *string
	where string
	// anchor is the answer's code line, kept with every reply of the fix
	// flow, so "apply it" under any of them applies the answer.
	anchor *store.BotReplyAnchor
}

// usableAnchor is a comment's code line, when it points at one in the
// PR's new version: not an old-side line, not one gone from the diff. A
// line on the PR's diff belongs to head; one on a single commit's or a
// range's diff belongs to that diff's to-commit, and is found again by its
// text at fix time, never taken as head's line number.
func usableAnchor(c *bitbucket.Comment, head string) *store.BotReplyAnchor {
	if c == nil || c.Anchor == nil {
		return nil
	}
	a := c.Anchor
	if a.Path == "" || a.Line <= 0 || a.Orphaned || a.FileType == "FROM" {
		return nil
	}
	// The anchor's own to-commit when Bitbucket names it, on any diff: an
	// EFFECTIVE anchor not yet drifted after a push still counts lines of
	// the commit before it. Without one, the PR's diff is head's.
	commit := a.ToHash
	if commit == "" && (a.DiffType == "" || a.DiffType == "EFFECTIVE") {
		commit = head
	}
	if commit == "" {
		return nil
	}
	return &store.BotReplyAnchor{Path: a.Path, Line: a.Line, Commit: commit}
}

// questionAnchor is the code line a Q&A's question sits on, kept with the
// answer. Bitbucket anchors only a thread's root and sends a reply's
// immediate parent only, so, cheapest first: the parent's own line (the
// question replies to an inline root), the line kept with noergler's
// answer it replies to (a further round in the thread), or, for a question
// that is the root itself, its own line, one fetch. nil when none.
func (r *Reviewer) questionAnchor(ctx context.Context, key store.PRKey, prTag string, commentID, parentID int,
	parent *bitbucket.Comment, parentBot *store.BotReply, head string) *store.BotReplyAnchor {
	if a := usableAnchor(parent, head); a != nil {
		return a
	}
	if parentBot != nil && parentBot.Kind == store.BotReplyAnswer && parentBot.Anchor != nil {
		a := *parentBot.Anchor
		return &a
	}
	if parentID != 0 {
		return nil
	}
	c, err := r.bitbucket.FetchPRComment(ctx, key.Project, key.Repo, key.PRID, commentID)
	if err != nil {
		r.log.WarnContext(ctx, fmt.Sprintf("%s: question %d not fetched, its answer keeps no code line: %v", prTag, commentID, err))
		return nil
	}
	return usableAnchor(c, head)
}

// prepareAnswerFix gets a fix of noergler's answer parent ready, before the
// Q&A's call: the posting stage can stage nothing, so the fix call rides
// the Q&A's pool slot. nil when the parent is no Q&A answer of noergler's,
// nor a reply of the fix flow about one (then its source is applied: "apply
// it" under "only the PR author can" or "please ask again" works). Only the
// PR author may have it applied; anyone else is told so, with no call paid.
func (r *Reviewer) prepareAnswerFix(ctx context.Context, pr webhook.PullRequest, key store.PRKey, prTag, commenter string,
	parentID int, parent *bitbucket.Comment, parentBot *store.BotReply, agentsMD, request string, l lang.Lang) *answerFix {
	if parent == nil || parentBot == nil || parentBot.Kind != store.BotReplyAnswer {
		return nil
	}
	answerID := parentID
	if parentBot.Source != 0 {
		answerID = parentBot.Source
	}
	x := &answerFix{answerID: answerID, where: fmt.Sprintf("answer %d", answerID), anchor: parentBot.Anchor}
	if author := pr.Author.User.Name; author == "" || !strings.EqualFold(commenter, author) {
		x.gate = "not_author"
		return x
	}
	// Fails closed, unlike the reads around it: an unread state could
	// commit an answer a second time.
	sha, err := r.store.AnswerFixCommit(ctx, answerID)
	switch {
	case err != nil:
		r.log.WarnContext(ctx, fmt.Sprintf("%s: whether %s was applied could not be read, no fix: %v", prTag, x.where, err))
		x.plan = &fixPlan{unavailable: "state_unread"}
		return x
	case sha != "":
		x.done = &sha
		return x
	}
	a := parentBot.Anchor
	if a == nil {
		x.plan = &fixPlan{unavailable: "no_anchor"}
		return x
	}
	x.where = fmt.Sprintf("answer %d on %s:%d", answerID, a.Path, a.Line)
	suggestion := parent.Text
	if answerID != parentID {
		// The parent is a reply about the answer: the answer is what is
		// applied, one fetch.
		c, err := r.bitbucket.FetchPRComment(ctx, key.Project, key.Repo, key.PRID, answerID)
		if err != nil {
			r.log.WarnContext(ctx, fmt.Sprintf("%s: %s not fetched for its fix: %v", prTag, x.where, err))
			x.plan = &fixPlan{unavailable: "unreadable"}
			return x
		}
		suggestion = c.Text
	}
	// The line belongs to the head the question was read at; at a later
	// head it is found again by its text, as a finding's is.
	f := &store.Finding{FilePath: a.Path, LineNumber: a.Line, RunCommit: a.Commit}
	run := fetchedFile{at: a.Commit}
	if a.Commit != pr.FromRef.LatestCommit {
		run.content, run.path, run.ok = func() (string, string, bool) {
			content, path, err := r.fetchFindingFile(ctx, key.Project, key.Repo, a.Commit, a.Path)
			return content, path, err == nil
		}()
	}
	x.plan = r.prepareFix(ctx, pr, key.Project, key.Repo, prTag, f, run)
	if x.plan.unavailable != "" {
		return x
	}
	prompt := inference.RenderFindingFixPrompt(inference.FindingFixPromptRequest{
		Template:         r.fixTmpl,
		RepoInstructions: agentsMD,
		Finding: fmt.Sprintf("%s:%d (your answer to a question on this line)\n\n%s",
			a.Path, a.Line, lang.TruncateRunes(strings.TrimSpace(suggestion), replyExcerptRunes/4)),
		FilePath:   x.plan.path,
		Code:       x.plan.code,
		StartLine:  x.plan.from,
		AnchorLine: x.plan.anchor,
		Reply:      lang.TruncateRunes(request, replyTextRunes),
		Language:   l,
	})
	x.req = &inference.FindingFixRequest{
		Prompt:       prompt,
		PromptTokens: r.tokens.Count(inference.FixSystemMessage) + r.tokens.Count(prompt),
	}
	x.plan.code = "" // in the prompt now; the post needs the bytes only
	return x
}

// postAnswerFix answers "apply it" under noergler's Q&A answer, on the
// worker, never silently: not the PR author, or the finding's fix flow.
// prefix is what the same turn confirmed (a language switch, a stop).
func (r *Reviewer) postAnswerFix(ctx context.Context, j *mentionJob, prefix string) {
	x, prTag := j.fix, j.ask.prTag
	if x.gate == "not_author" {
		r.log.InfoContext(ctx, fmt.Sprintf("%s: fix of %s asked by %s, not the PR author: not applied", prTag, x.where, j.ask.user))
		// Kept as a reply about the answer: the author's "apply it" under
		// this notice applies the answer.
		r.replyAnswer(ctx, j.project, j.repo, j.prID, j.replyTo, prefix+j.lang.Pick(
			"Umsetzen lasse ich mich nur vom PR-Autor. Wenn er den Vorschlag will, kann er hier ‚setz das um' schreiben.", //nolint:misspell // German "Autor", not a typo of "Author"
			"Only the PR author can have me apply this. If they want the suggestion, they can reply 'apply it' here."),
			x.anchor, x.answerID)
		return
	}
	// The commit message links to the answer applied.
	r.postFix(ctx, &fixTarget{key: j.ask.key, team: j.team, project: j.project, repo: j.repo, prID: j.prID,
		prTag: prTag, where: x.where, replyTo: j.replyTo, threadID: &x.answerID, lang: j.lang,
		plan: x.plan, res: j.res.Fix, answerID: x.answerID, anchor: x.anchor, done: x.done, prefix: prefix})
}

// replyAnswer is reply for the Q&A's answer (source 0) or a reply about
// answer source: the code line its question sat on is kept with it, so the
// author can have its suggestion applied from under either.
func (r *Reviewer) replyAnswer(ctx context.Context, project, repo string, prID, parentID int, text string,
	anchor *store.BotReplyAnchor, source int) {
	id, ok := r.postReply(ctx, project, repo, prID, parentID, text)
	if !ok {
		return
	}
	key := store.PRKey{Project: project, Repo: repo, PRID: prID}
	safeDBErr(ctx, r.log, "InsertAnswerBotReply", func() error {
		return r.store.InsertAnswerBotReply(ctx, key, id, anchor, source)
	})
}
