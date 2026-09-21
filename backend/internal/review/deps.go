// Package review is the PR review pipeline: the guard order, the LLM call,
// the posted comments and the rows they leave behind.
//
// One Reviewer per team. The Bitbucket client and the store are shared by all
// teams; the inference client, review config, Jira client and riptide emitter
// belong to the team.
package review

import (
	"context"
	"log/slog"

	"github.com/trick77/noergler-go/internal/bitbucket"
	"github.com/trick77/noergler-go/internal/inference"
	"github.com/trick77/noergler-go/internal/jira"
	"github.com/trick77/noergler-go/internal/riptide"
	"github.com/trick77/noergler-go/internal/store"
)

// The interfaces below are declared consumer-side, as AGENTS.md requires: the
// adapters export concrete types and this package states what it needs of
// them. That keeps the fakes in the tests small and the dependency arrow
// pointing one way.

// BitbucketClient is the part of the Bitbucket adapter the pipeline uses.
type BitbucketClient interface {
	BotUsername() string
	FetchPRDiff(ctx context.Context, project, repo string, prID, contextLines int) (string, error)
	FetchCommitDiff(ctx context.Context, project, repo, fromCommit, toCommit string) (string, error)
	FetchFileContent(ctx context.Context, project, repo, commit, path string) (string, error)
	PostInlineComment(ctx context.Context, project, repo string, prID int, file string, line int, body string) (int, error)
	PostPRComment(ctx context.Context, project, repo string, prID int, text string) (id, version int, err error)
	ReplyToComment(ctx context.Context, project, repo string, prID, parentCommentID int, text string) error
	FetchPRComment(ctx context.Context, project, repo string, prID, commentID int) (*bitbucket.Comment, error)
	UpdatePRComment(ctx context.Context, project, repo string, prID, commentID, version int, text string) (int, error)
}

// JiraClient is the part of the Jira adapter the pipeline uses.
//
// FetchTicketWithParent is the only entry point: Python also calls
// fetch_ticket a second time inside _fetch_ticket_context, which AGENTS.md
// pins as a deliberate divergence ("Jira fetched once per review").
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
	ContextWindow() int
	InputTokenBudget() int
	Review(ctx context.Context, req inference.ReviewRequest) inference.ReviewResult
	Mention(ctx context.Context, req inference.MentionRequest) inference.MentionResult
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
	InsertFinding(ctx context.Context, f store.Finding) error
	ExistingFindings(ctx context.Context, k store.PRKey) ([]store.Finding, error)
	PRCost(ctx context.Context, k store.PRKey) (*int64, error)
	FreezeFinalCost(ctx context.Context, k store.PRKey) (*int64, error)
	ClaimRollup(ctx context.Context, k store.PRKey, final store.RollupFinal) (*store.RollupSnapshot, error)
}

// TokenCounter counts tokens in a string.
type TokenCounter interface {
	Count(text string) int
}

// safeDB runs a store call and swallows its failure.
//
// AGENTS.md: every store call in the review path goes through a warn-and-
// fallback wrapper, because a DB fault must never fail a review. The zero
// value of T is the fallback, which for the pointer and slice returns here is
// the same "not found" the callers already handle.
func safeDB[T any](ctx context.Context, log *slog.Logger, what string, fn func() (T, error)) T {
	v, err := fn()
	if err != nil {
		var zero T
		log.WarnContext(ctx, "DB operation failed, using fallback: "+what, slog.String("error", err.Error()))
		return zero
	}
	return v
}

// safeDBErr is safeDB for a call that returns only an error.
func safeDBErr(ctx context.Context, log *slog.Logger, what string, fn func() error) {
	if err := fn(); err != nil {
		log.WarnContext(ctx, "DB operation failed, using fallback: "+what, slog.String("error", err.Error()))
	}
}
