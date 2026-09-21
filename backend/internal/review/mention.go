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
	"github.com/trick77/noergler/internal/logging"
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
	if comment.Author.Name == r.bitbucket.BotUsername() {
		r.log.DebugContext(ctx, "Ignoring own comment (bot)")
		return false
	}
	pr := payload.PullRequest
	if pr.State != "" && pr.State != "OPEN" {
		r.log.InfoContext(ctx, "Ignoring mention on non-open PR (state="+pr.State+")")
		return false
	}

	project, repo := payload.ProjectRepo()
	if project == "" || repo == "" {
		r.log.ErrorContext(ctx, "Could not extract project/repo from webhook payload")
		return false
	}
	prTag := fmt.Sprintf("%s/%s#%d", project, repo, pr.ID)
	ctx = logging.With(ctx, "pr_tag", prTag, "repo", project+"/"+repo, "pr_id", pr.ID)

	// The Q&A path fetches a diff and posts a reply of its own, so it gets the
	// same accounting as a review. A keyword mention stages a review, which
	// opens its own scope and logs its own totals when its posting finishes;
	// this one then reports only what the mention itself spent, which is
	// two units of work reported separately rather than merged.
	ctx, httpCounter := httpstats.WithScope(ctx)
	defer r.logHTTPTotals(ctx, prTag, httpCounter)

	key := prKey(project, repo, pr.ID)

	// Any mention reactivates a PR ignored after its summary was removed.
	// Clearing the state (and the stale summary id, which Reactivate does)
	// lets the next review post a fresh summary and stops the review guard
	// from re-ignoring the PR.
	if state := safeDB(ctx, r.log, "GetSkipState", func() (*storeSkipState, error) {
		s, err := r.store.GetSkipState(ctx, key)
		if s == nil {
			return nil, err
		}
		return &storeSkipState{Ignored: s.IgnoredAt != nil}, err
	}); state != nil && state.Ignored {
		safeDBErr(ctx, r.log, "Reactivate", func() error { return r.store.Reactivate(ctx, key) })
		r.log.InfoContext(ctx, prTag+": reactivating ignored PR via @mention")
	}

	question := extractQuestion(comment.Text, r.bitbucket.BotUsername())

	// An empty question, or one of the review keywords, means "review this".
	// It is a review like any other, so it takes the staged path: the
	// gateway call goes to the inference pool instead of holding the worker,
	// and the PR's queue hold covers it, so it cannot run alongside a review
	// of the same PR already in flight.
	if question == "" || reviewKeywords[strings.ToLower(question)] {
		r.log.InfoContext(ctx, fmt.Sprintf("Mention triggers full review (question=%q)", question))
		if sched != nil {
			// The caller must keep the PR held: the staged posting is what
			// releases it, so a push or a merge for this PR cannot run
			// alongside the inference.
			return r.ReviewPullRequestStaged(ctx, payload, team, true, sched)
		}
		// No scheduler (a direct caller, or a test): run it inline.
		r.ReviewPullRequest(ctx, payload, true)
		return false
	}
	r.log.InfoContext(ctx, fmt.Sprintf("Handling mention Q&A on %s: %q", prTag, question))

	// Q&A runs without the review's gates and without compression.
	rawDiff, err := r.bitbucket.FetchPRDiff(ctx, project, repo, pr.ID, 0)
	var tooLarge *bitbucket.ContentTooLarge
	if errors.As(err, &tooLarge) {
		r.log.WarnContext(ctx, fmt.Sprintf("%s: %v - mention not answered", prTag, err))
		r.reply(ctx, project, repo, pr.ID, comment.ID, fmt.Sprintf(
			"This PR's diff exceeds %d MiB, too large to answer questions about.", tooLarge.Limit/(1024*1024)))
		return false
	}
	if err != nil {
		r.log.ErrorContext(ctx, fmt.Sprintf("Mention Q&A on %s failed: %v", prTag, err))
		return false
	}

	files, _ := r.prepareFiles(ctx, project, repo, rawDiff, pr.FromRef.LatestCommit, prTag)
	if len(files) == 0 {
		r.reply(ctx, project, repo, pr.ID, comment.ID, "No reviewable files in this PR.")
		return false
	}

	repoInstructions := r.fetchRepoInstructions(ctx, project, repo, pr.FromRef.LatestCommit, pr.ToRef.DisplayID)
	ticket, parent := r.fetchTicket(ctx, pr.FromRef.DisplayID, pr.Title, prTag)

	prompt := inference.RenderMentionPrompt(inference.MentionPromptRequest{
		Template:         r.mentionTmpl,
		Question:         question,
		Files:            files,
		RepoInstructions: repoInstructions,
		TicketContext:    renderTicketContext(ticket, parent),
	})

	result := r.llm.Mention(ctx, inference.MentionRequest{
		Prompt:       prompt,
		PromptTokens: r.tokens.Count(inference.MentionSystemMessage) + r.tokens.Count(prompt),
	})
	// Only when a call actually happened: the too-large branch decides before
	// any request, and a transport error returns with no cost.
	if result.Outcome == inference.OutcomeOK {
		r.logCost(ctx, prTag, result.Cost)
	}

	switch result.Outcome {
	case inference.OutcomeTimedOut:
		r.log.ErrorContext(ctx, fmt.Sprintf("Mention Q&A on %s aborted - no response within %.0fs",
			prTag, inference.CallTimeout.Seconds()))
		r.reply(ctx, project, repo, pr.ID, comment.ID, fmt.Sprintf(
			"⚠️ No response from the model within %d minutes. Please try again, or simplify the question "+
				"if it requires a lot of context.", timeoutMinutes))
	case inference.OutcomeTooLarge:
		r.log.WarnContext(ctx, prTag+": mention prompt too large for the model's context window")
		r.reply(ctx, project, repo, pr.ID, comment.ID, inference.MentionTooLargeReply)
	case inference.OutcomeOK:
		// An empty answer still gets a reply: a mention is never left
		// silently unanswered.
		answer := result.Answer
		if answer == "" {
			answer = inference.MentionEmptyReply
		}
		r.reply(ctx, project, repo, pr.ID, comment.ID, answer)
		r.log.InfoContext(ctx, "Posted Q&A reply on "+prTag)
	default:
		r.log.ErrorContext(ctx, fmt.Sprintf("Mention Q&A on %s failed: %v", prTag, result.Err))
	}
	return false
}

// storeSkipState is the part of the skip state the mention path reads.
type storeSkipState struct{ Ignored bool }

// reply posts a threaded reply, logging a failure rather than propagating it:
// a mention that cannot be answered must not fail the webhook.
func (r *Reviewer) reply(ctx context.Context, project, repo string, prID, parentID int, text string) {
	if err := r.bitbucket.ReplyToComment(ctx, project, repo, prID, parentID, text); err != nil {
		r.log.ErrorContext(ctx, fmt.Sprintf("Failed to post reply on %s/%s#%d: %v", project, repo, prID, err))
	}
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
