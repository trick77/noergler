// Package review is the PR review pipeline: the guard order, the LLM call,
// the posted comments and the rows they leave behind.
//
// One Reviewer per team. The Bitbucket client and the store are shared by all
// teams; the inference client, review config, Jira client and riptide emitter
// belong to the team.
package review

import (
	"context"
	"errors"
	"fmt"
	"log/slog"
	"time"

	"github.com/trick77/noergler/internal/bitbucket"
	"github.com/trick77/noergler/internal/inference"
	"github.com/trick77/noergler/internal/jira"
	"github.com/trick77/noergler/internal/lang"
	"github.com/trick77/noergler/internal/riptide"
	"github.com/trick77/noergler/internal/store"
	"github.com/trick77/ship/kit/logging"
)

// The interfaces below are declared consumer-side, deliberately: the
// adapters export concrete types and this package states what it needs of
// them. That keeps the fakes in the tests small and the dependency arrow
// pointing one way.

// BitbucketClient is the part of the Bitbucket adapter the pipeline uses.
type BitbucketClient interface {
	BotUsername() string
	FetchPRDiff(ctx context.Context, project, repo string, prID, contextLines int) (string, error)
	FetchPRChanges(ctx context.Context, project, repo string, prID int) ([]string, error)
	FetchCommitDiff(ctx context.Context, project, repo, fromCommit, toCommit string) (string, error)
	FetchFileContent(ctx context.Context, project, repo, commit, path string) (string, error)
	PostInlineComment(ctx context.Context, project, repo string, prID int, file string, line int, body string) (int, error)
	PostPRComment(ctx context.Context, project, repo string, prID int, text string) (id, version int, err error)
	ReplyToComment(ctx context.Context, project, repo string, prID, parentCommentID int, text string) error
	FetchPRComment(ctx context.Context, project, repo string, prID, commentID int) (*bitbucket.Comment, error)
	UpdatePRComment(ctx context.Context, project, repo string, prID, commentID, version int, text string) (int, error)
	ResolveCommentThread(ctx context.Context, project, repo string, prID, commentID int) error
}

// JiraClient is the part of the Jira adapter the pipeline uses.
//
// FetchTicketWithParent is the only entry point, so the ticket is fetched
// once per review and never a second time while rendering the prompt block.
// A test pins this ("Jira fetched once per review").
type JiraClient interface {
	FetchTicketWithParent(ctx context.Context, key string) (ticket, parent *jira.Ticket, err error)
}

// RiptideEmitter is the part of the riptide adapter the pipeline uses.
type RiptideEmitter interface {
	Enabled() bool
	EmitPRCompleted(ctx context.Context, r riptide.Rollup)
}

// InferenceClient is the part of the inference client the pipeline uses.
type InferenceClient interface {
	Model() string
	// Label is Model plus the reasoning effort: what a reader sees in the
	// summary and what a run row stores.
	Label() string
	Ready() bool
	InputTokenBudget() int
	Review(ctx context.Context, req inference.ReviewRequest) inference.ReviewResult
	Mention(ctx context.Context, req inference.MentionRequest) inference.MentionResult
	FindingReply(ctx context.Context, req inference.FindingReplyRequest) inference.FindingReplyResult
}

// Store is the part of the store the pipeline uses. Every call goes through
// safeDB: a DB fault must never fail a review.
type Store interface {
	UpsertPullRequest(ctx context.Context, u store.PRUpsert) (int64, error)
	GetLastReviewedCommit(ctx context.Context, k store.PRKey) (string, bool, error)
	SetSummaryComment(ctx context.Context, prID int64, commentID, version int) error
	GetSummaryComment(ctx context.Context, prID int64) (*store.SummaryComment, error)
	GetSkipState(ctx context.Context, k store.PRKey) (*store.SkipState, error)
	MarkIgnored(ctx context.Context, k store.PRKey) error
	Reactivate(ctx context.Context, k store.PRKey) error
	MarkMerged(ctx context.Context, k store.PRKey) error
	MarkDeclined(ctx context.Context, k store.PRKey) error
	MarkDeleted(ctx context.Context, k store.PRKey) error
	InsertRun(ctx context.Context, r store.Run) (int64, error)
	InsertAttempt(ctx context.Context, a store.Attempt) (int64, error)
	InsertRunDetail(ctx context.Context, attemptID int64, p store.Pipeline) error
	InsertFinding(ctx context.Context, f store.Finding) error
	ExistingFindings(ctx context.Context, k store.PRKey) ([]store.Finding, error)
	MarkFindingFixed(ctx context.Context, findingID, runID int64) error
	FindingByComment(ctx context.Context, k store.PRKey, commentID int) (*store.Finding, error)
	MarkFindingReplied(ctx context.Context, findingID int64, dispute *string) error
	PRCost(ctx context.Context, k store.PRKey) (*int64, error)
	PRCapState(ctx context.Context, k store.PRKey) (int64, *int64, error)
	InsertCommentCall(ctx context.Context, team string, k store.PRKey, kind string, commentID int64, c store.LLMCall) error
	FreezeFinalCost(ctx context.Context, k store.PRKey) (*int64, error)
	ClaimRollup(ctx context.Context, k store.PRKey, final store.RollupFinal) (*store.RollupSnapshot, error)
	PRStatus(ctx context.Context, k store.PRKey, team string) (*store.PRStatus, error)
	UserLanguage(ctx context.Context, userName string) (lang.Lang, error)
	SetUserLanguage(ctx context.Context, userName string, l lang.Lang) error
}

// TokenCounter counts tokens in a string.
type TokenCounter interface {
	Count(text string) int
}

// safeDB runs a store call and swallows its failure.
//
// Every store call in the review path goes through a warn-and-
// fallback wrapper, because a DB fault must never fail a review. The zero
// value of T is the fallback, which for the pointer and slice returns here is
// the same "not found" the callers already handle.
func safeDB[T any](ctx context.Context, log *slog.Logger, what string, fn func() (T, error)) T {
	v, err := fn()
	if err != nil {
		var zero T
		log.WarnContext(ctx, "DB operation failed, using fallback: "+what, logging.Err(err))
		return zero
	}
	return v
}

// safeDBErr is safeDB for a call that returns only an error.
func safeDBErr(ctx context.Context, log *slog.Logger, what string, fn func() error) {
	if err := fn(); err != nil {
		log.WarnContext(ctx, "DB operation failed, using fallback: "+what, logging.Err(err))
	}
}

// recordAttempt writes the dashboard's record of one terminal outcome.
//
// Fails open, like every other store call here: the row is a record of what
// happened, and losing it must never change what happened. A returned error
// would turn a successful review into a failed one.
//
// Returns the attempt id, 0 when nothing was written; the run's dashboard
// link is built from it.
func (r *Reviewer) recordAttempt(ctx context.Context, a store.Attempt) int64 {
	// A Reviewer with no store is a legitimate configuration, not a fault:
	// the author and actor gates decide before anything is wired, and the
	// logging tests drive exactly that shape. Every other store call in this
	// package sits behind a path that has already reached the DB, so this is
	// the only one that has to ask.
	if r.store == nil {
		return 0
	}
	// ErrCallNotRecorded means the attempt (and any billed cost on it) is
	// written and only the call's detail is not: logged as that, with the
	// PR, so nobody reads a lost llm_calls row as a lost attempt.
	// team= rides on ctx (the queue binds it per job).
	id, err := r.store.InsertAttempt(ctx, a)
	switch {
	case errors.Is(err, store.ErrCallNotRecorded):
		r.log.WarnContext(ctx, a.Key.Tag()+": gateway call detail not recorded; the attempt was written", logging.Err(err))
	case err != nil:
		r.log.WarnContext(ctx, "DB operation failed, using fallback: InsertAttempt", logging.Err(err))
		return 0
	}
	return id
}

// runURL is the attempt's run detail page, for the summary's footer. Empty
// (no link) without a public URL or an attempt id: the footer never points
// at a host or a run that does not exist.
func (r *Reviewer) runURL(attemptID int64) string {
	if r.publicURL == "" || attemptID == 0 {
		return ""
	}
	return fmt.Sprintf("%s/runs/%d", r.publicURL, attemptID)
}

// llmCall is the llm_calls row for a review's gateway call, nil when the
// gateway never answered (a timeout, a transport error, a pre-flight
// too-large): there is nothing it reported to keep.
func (r *Reviewer) llmCall(c inference.CallCost) *store.LLMCall {
	if !c.Answered {
		return nil
	}
	ms := func(d time.Duration) *int64 {
		if d <= 0 {
			return nil
		}
		v := d.Milliseconds()
		return &v
	}
	return &store.LLMCall{
		Profile:               r.llm.Label(),
		Model:                 c.Model,
		Deployment:            c.Deployment,
		ReasoningSent:         c.ReasoningSent,
		FinishReason:          c.FinishReason,
		CallID:                c.CallID,
		GatewayVer:            c.GatewayVersion,
		InputTokens:           c.Usage.Input.Total,
		InputUncachedTokens:   c.Usage.Input.NoCache,
		CacheReadTokens:       c.Usage.Input.CacheRead,
		CacheWriteTokens:      c.Usage.Input.CacheWrite,
		OutputTokens:          c.Usage.Output.Total,
		ReasoningTokens:       c.Usage.Output.Reasoning,
		CostNanoUSD:           c.NanoUSD,
		CostInputNanoUSD:      c.Split.InputNanoUSD,
		CostCacheReadNanoUSD:  c.Split.CacheReadNanoUSD,
		CostCacheWriteNanoUSD: c.Split.CacheWriteNanoUSD,
		CostOutputNanoUSD:     c.Split.OutputNanoUSD,
		CostReasoningNanoUSD:  c.Split.ReasoningNanoUSD,
		CostToolUsageNanoUSD:  c.Split.ToolUsageNanoUSD,
		KeySpendNanoUSD:       c.KeySpendNanoUSD,
		HeadersMS:             ms(c.Timing.Headers),
		TotalMS:               ms(c.Timing.Total),
		Warnings:              c.Warnings,
	}
}

// runKind spells the auto/mention choice once. prepare, post and recordRun
// all make it, and a fourth inline copy is how the three drift apart.
func runKind(mention bool) store.RunKind {
	if mention {
		return store.RunMention
	}
	return store.RunAuto
}
