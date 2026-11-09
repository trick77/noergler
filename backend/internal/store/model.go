package store

import (
	"encoding/json"
	"fmt"
	"time"
)

// AlmanachNote is one PR's documentation note as a review round leaves it. See
// migrations/0018_almanach.sql: every round replaces the row, ordered by
// RunID; the merge stamps it final; the forwarder marks it emitted.
//
// Table: almanach_notes
type AlmanachNote struct {
	Key      PRKey
	TeamSlug string
	// RunID is the review run whose round wrote the note.
	RunID          int64
	ReviewedCommit string
	PRTitle        string
	TicketKey      string
	JiraIssueType  string
	ChangeKind     string
	// Note is the validated note object; Chars the rune count it was
	// checked at.
	Note       json.RawMessage
	Chars      int
	ModelLabel string
}

// AlmanachReviewTotals is the PR's review aggregate: the runs, their
// tokens, and the PR's cost as prCostSQL counts it (the one figure the cap,
// the frozen cost and the rollup use). Zero and nil when the PR row is gone.
//
// Table: -
type AlmanachReviewTotals struct {
	Runs             int
	PromptTokens     int64
	CachedTokens     int64
	CompletionTokens int64
	ElapsedMS        int64
	CostNanoUSD      *int64
	Models           []string
}

// AlmanachPending is one final note the forwarder sends, with everything
// Almanach gets beside it.
//
// Table: almanach_notes
type AlmanachPending struct {
	AlmanachNote
	MergedAt     time.Time
	MergeCommit  string
	SourceCommit string
	// IssuesPosted and SuggestionsPosted count the findings noergler posted
	// on the PR, by severity, as of finalize.
	IssuesPosted      int
	SuggestionsPosted int
	// NoteCall is the round's own gateway call as recorded, NoteCallFound
	// false when none was. NoteCallOutcome and NoteStatus are that row's.
	NoteCall        LLMCall
	NoteCallFound   bool
	NoteCallOutcome string
	NoteStatus      string
	// Rounds counts every note call of this PR incarnation, with their
	// summed tokens and cost (nil when none was priced).
	Rounds             int
	RoundsInputTokens  int64
	RoundsOutputTokens int64
	RoundsCostNanoUSD  *int64
	Review             AlmanachReviewTotals
}

// AlmanachNoteState is what a caller reads back before deciding to emit.
//
// Table: almanach_notes
type AlmanachNoteState struct {
	Finalized bool
	HasNote   bool
	Emitted   bool
	// PROpen is whether the PR row a round could write to is still there:
	// not declined, not deleted, not purged.
	PROpen bool
}

// Attempt is one terminal outcome, including the ones that write no run row.
// See migrations/0002_review_attempts.sql for why this is its own table.
//
// Table: review_attempts
type Attempt struct {
	Key      PRKey
	TeamSlug string
	Kind     RunKind
	// Outcome is inference.Outcome's String(), or "skipped" for a
	// pre-flight exit that never reached the gateway.
	Outcome string
	// Reason is a review.SkipReason. Empty unless Outcome is "skipped".
	Reason    string
	ElapsedMS *int64
	// RunID is set only when the attempt produced a run row.
	RunID *int64
	// CostNanoUSD is a billed call that wrote no run row (unparseable). A
	// successful attempt leaves it nil: its cost is on the run.
	CostNanoUSD *int64
	// Call is the gateway call this attempt made, written to llm_calls right
	// after the attempt and apart from it, so a missing call row can mean a
	// refused write, not only no call. Nil when the gateway never answered (a
	// pre-flight skip, a timeout); a late repo-unclaimed skip carries the call
	// it paid for.
	Call *LLMCall
}

// AttemptRow is one row of the dashboard's feed: the attempt, plus the run
// figures where the attempt produced a run.
//
// Table: review_attempts
type AttemptRow struct {
	Attempt
	// ID is the attempt's row id. Never its link token: the feed links to
	// no run page.
	ID        int64
	CreatedAt time.Time
	// Findings and CostNanoUSD come from the joined run. Both nil when the
	// attempt produced none, and CostNanoUSD is nil for an unpriced run
	// too: the two cases are distinguished by Outcome, never by a zero.
	Findings    *int
	CostNanoUSD *int64
}

// OutcomeCount is one bucket of the outcome breakdown.
//
// Table: -
type OutcomeCount struct {
	Outcome string
	Reason  string
	Count   int
}

// SealedCredential is one team_credentials row: the token as ciphertext only.
// Sealing and opening are the caller's (internal/teams); the store never sees
// a token in the clear.
//
// Table: team_credentials
type SealedCredential struct {
	Team       string
	Kind       string
	Username   string
	Nonce      []byte
	Ciphertext []byte
}

// Totals is the headline figure set over a time window.
//
// Table: -
type Totals struct {
	Runs             int
	PromptTokens     int64
	CachedTokens     int64
	CompletionTokens int64
	FindingsPosted   int64
	// CostNanoUSD sums the priced runs and billed run-less attempts. Nil
	// when none were priced.
	CostNanoUSD *int64
	// Unpriced is how many runs carried no cost. Shown beside the total,
	// never added to it.
	Unpriced int
}

// TeamTotals is Totals for one team.
//
// Table: -
type TeamTotals struct {
	TeamSlug string
	Totals
}

// Bucket is one day of the timeseries.
//
// Table: -
type Bucket struct {
	Day            time.Time
	TeamSlug       string
	Runs           int
	FindingsPosted int64
	CostNanoUSD    *int64
	// Unpriced is how many of the day's runs carried no cost, as in Totals.
	Unpriced int
}

// AttemptBucket is one day's count for one outcome.
//
// Table: -
type AttemptBucket struct {
	Day     time.Time
	Outcome string
	Count   int
}

// TeamActivity is the per-team roster figure: how many PRs a team has
// touched, when it last produced a review, and when and how its last run of
// any outcome ended.
//
// Table: -
type TeamActivity struct {
	TeamSlug string
	PRs      int
	// LastReviewed is the last SUCCESS: review_runs holds nothing else.
	LastReviewed *time.Time
	// LastRun is the last run of any outcome, skips and failures included.
	// Nil when the team has none.
	LastRun *time.Time
	// LastOutcome is LastRun's outcome.
	LastOutcome string
	// LastReason is LastRun's skip reason.
	LastReason string
}

// DisputeSignal is one open signal of a repo: its id and the rule as the
// newest occurrence in the window worded it.
//
// Table: dispute_signals
type DisputeSignal struct {
	ID   int64
	Rule string
}

// DisputePR is one PR a signal occurred on inside the window: the Bitbucket
// PR number, the disputed finding's comment (nil if never posted) and the PR
// author.
//
// Table: -
type DisputePR struct {
	PRID      int
	CommentID *int
	Author    string
}

// LLMCall is one gateway call as the AI Gateway described it. See
// migrations/0009_llm_calls.sql and 0012: every pointer is nil when the
// gateway did not send that figure, which is not zero. A review call's
// CostNanoUSD is detail beside the run or attempt that carries the cost; a
// reply's or a question's is the only record of it, and the PR cost sums it.
//
// Table: llm_calls
type LLMCall struct {
	// Profile is the label asked for (`<profile>-<effort>`).
	Profile       string
	Model         string
	Deployment    string
	ReasoningSent string
	FinishReason  string
	CallID        string
	GatewayVer    string

	InputTokens         *int64
	InputUncachedTokens *int64
	CacheReadTokens     *int64
	CacheWriteTokens    *int64
	// OutputTokens includes ReasoningTokens.
	OutputTokens    *int64
	ReasoningTokens *int64

	CostNanoUSD           *int64
	CostInputNanoUSD      *int64
	CostCacheReadNanoUSD  *int64
	CostCacheWriteNanoUSD *int64
	CostOutputNanoUSD     *int64
	CostReasoningNanoUSD  *int64
	CostToolUsageNanoUSD  *int64
	// KeySpendNanoUSD is a gauge of the whole key: shown, never summed.
	KeySpendNanoUSD *int64

	// HeadersMS is the wait for the first byte since every call streams
	// (2026-11-06), the whole call before; TotalMS is the model's time.
	HeadersMS *int64
	TotalMS   *int64
	Warnings  []string
}

// CommentCall is one reply or question call on a PR, for the run page's
// list. Nil cost = unpriced.
//
// Table: llm_calls
type CommentCall struct {
	Kind        string
	CommentID   *int64
	Model       string
	CostNanoUSD *int64
	CreatedAt   time.Time
}

// PRKey identifies a pull request across Bitbucket and the store.
//
// Table: pull_requests
type PRKey struct {
	Project string
	Repo    string
	PRID    int
}

// Tag is `PROJECT/repo#id`, the log and riptide spelling.
func (k PRKey) Tag() string { return fmt.Sprintf("%s/%s#%d", k.Project, k.Repo, k.PRID) }

// SkipState is what the review guard needs to decide whether to skip a PR.
//
// Table: pull_requests
type SkipState struct {
	ID        int64
	IgnoredAt *time.Time
	// IgnoreReason is IgnoreSummaryDeleted or IgnoreStopped while IgnoredAt
	// is set, "" otherwise.
	IgnoreReason string
	// StoppedBy is who stopped the PR, "" unless stopped.
	StoppedBy string
	// Summary is nil when no summary comment is stored.
	Summary *SummaryComment
}

// Stopped reports whether the PR is ignored because someone stopped it.
func (st *SkipState) Stopped() bool {
	return st != nil && st.IgnoredAt != nil && st.IgnoreReason == IgnoreStopped
}

// SummaryDeleted reports whether the PR is ignored because its summary was
// deleted: left alone entirely, unlike a stopped one, whose questions and
// replies are still answered.
func (st *SkipState) SummaryDeleted() bool {
	return st != nil && st.IgnoredAt != nil && !st.Stopped()
}

// Finding is one posted inline comment.
//
// Table: findings
type Finding struct {
	ID                 int64
	PullRequestID      int64
	RunID              int64
	FilePath           string
	LineNumber         int
	Severity           string
	Confidence         *int
	Headline           *string
	CommentText        string
	Suggestion         *string
	BitbucketCommentID *int
	// RepliedAt is set once noergler answered a reply on this finding's
	// thread; DisputeText is the author's reply when it disputed the
	// finding. Read back, never written by InsertFinding.
	RepliedAt   *time.Time
	DisputeText *string
	// ClosedAt is set once the finding's thread was found resolved or
	// deleted on the PR: settled, out of the PR summary. Read back by
	// ExistingFindings only, stamped by MarkFindingClosed.
	ClosedAt *time.Time
	// RunCommit is the source commit of the run that posted the finding: its
	// line number belongs to that commit. Read by FindingByComment only.
	RunCommit string
}

// BotReply is one reply noergler posted, by its Bitbucket comment id.
//
// Table: bot_replies
type BotReply struct {
	CommentID int
	Kind      string
}

// RollupSnapshot is what one riptide rollup carries.
//
// Table: -
type RollupSnapshot struct {
	Runs             int
	PromptTokens     int64
	CompletionTokens int64
	ElapsedMS        int64
	Findings         int
	// CostNanoUSD is nil when no run was priced.
	CostNanoUSD   *int64
	Models        []string
	FirstReviewAt time.Time
	SourceCommit  *string
	MergeCommit   *string
	LinesAdded    *int
	LinesRemoved  *int
	FilesChanged  *int
}

// RunDetail is everything the run page shows about one attempt.
//
// Everything, comment text and dispute included (decided): the page is
// reached only through its PR's link token, handed out in the PR's summary
// comment, so its audience is the PR's.
//
// Table: review_attempts
type RunDetail struct {
	ID int64
	// N is the attempt's number on its PR, 1 for the oldest: the page's name
	// for it and the last segment of its path.
	N         int
	Key       PRKey
	TeamSlug  string
	Kind      RunKind
	Outcome   string
	Reason    string
	ElapsedMS *int64
	CreatedAt time.Time
	// AttemptCostNanoUSD is set only for a billed call with no run.
	AttemptCostNanoUSD *int64

	PRTitle string
	// Run is nil unless the attempt produced one.
	Run *RunFacts
	// Call is nil when no gateway call was recorded for the attempt.
	Call *LLMCall
	// Findings are the ones this run posted; Fixed the earlier ones this run
	// marked fixed.
	Findings []RunFinding
	Fixed    []RunFinding
	// Detail is the run_details record; nil when none was written (an
	// attempt from before it existed, or a skip) or it no longer decodes,
	// which DetailErr then says.
	Detail    *Pipeline
	DetailErr error
}

// RunFacts is the review_runs row behind a successful attempt.
//
// Table: review_runs
type RunFacts struct {
	ID               int64
	Incremental      bool
	FromCommit       string
	ToCommit         string
	ModelLabel       string
	PromptTokens     int64
	CachedTokens     int64
	CompletionTokens int64
	CostNanoUSD      *int64
	FindingsPosted   int
	LinesAdded       int
	LinesRemoved     int
	FilesChanged     int
}

// RunFinding is a posted finding as the pages show it: where, how bad, its
// text, the Bitbucket comment it was posted as, and the runs that posted
// and retired it.
//
// Table: findings
type RunFinding struct {
	Severity    string
	Confidence  *int
	FilePath    string
	Line        int
	Headline    string
	CommentText string
	Suggestion  *string
	// DisputeText is the author's reply when it disputed the finding.
	DisputeText *string
	CommentID   *int64
	Replied     bool
	Fixed       bool
	// RunID posted it; FixedRunID retired it, nil while it is open.
	RunID      int64
	FixedRunID *int64
}

// PRTotals add up over every attempt of the PR.
//
// Table: -
type PRTotals struct {
	Runs             int
	ElapsedMS        int64
	PromptTokens     int64
	CachedTokens     int64
	CompletionTokens int64
}

// PRRun is one attempt as the PR page lists it.
//
// Table: review_attempts
type PRRun struct {
	N         int
	Kind      RunKind
	Outcome   string
	Reason    string
	ElapsedMS *int64
	CreatedAt time.Time
	// CostNanoUSD is the run's cost, or the attempt's for a billed call
	// with no run: the same figure prCostSQL counts, never both.
	CostNanoUSD *int64
	// Run is nil unless the attempt produced one; Fixed counts the earlier
	// findings it retired.
	Run   *RunFacts
	Fixed int
	// Detail is the run_details record's stages, files and PR scope (the
	// rest is never read here); nil when there is none or it does not decode.
	Detail *Pipeline
}

// prRow is the PR row behind a page token.
//
// Table: pull_requests
type prRow struct {
	ID        int64
	Key       PRKey
	TeamSlug  string
	Title     string
	CreatedAt time.Time
	Merged    *time.Time
	Declined  *time.Time
}

// PRStatus is what noergler knows about one PR: the answer to "why was my
// PR (not) reviewed?". Read-only, and nothing in it is author text: no
// finding, no dispute.
//
// Table: pull_requests
type PRStatus struct {
	// Known is false when noergler holds no row for the PR yet; the
	// attempts may still say why (a skip decided before any row).
	Known              bool
	LastReviewedCommit *string
	IgnoredAt          *time.Time
	// IgnoreReason and StoppedBy as in SkipState, nil when not ignored.
	IgnoreReason *string
	StoppedBy    *string
	MergedAt     *time.Time
	DeclinedAt   *time.Time
	Runs         int
	LastRun      *RunBrief
	// Attempts are the latest outcomes, newest first.
	Attempts []AttemptBrief
}

// RunBrief is one review run, without its tokens or cost.
//
// Table: review_runs
type RunBrief struct {
	At          time.Time
	Kind        RunKind
	Incremental bool
	Findings    int
	ModelLabel  string
}

// AttemptBrief is one terminal outcome.
//
// Table: review_attempts
type AttemptBrief struct {
	At      time.Time
	Kind    RunKind
	Outcome string
	// Reason is the review.SkipReason value, empty unless Outcome is
	// "skipped".
	Reason string
}

// TeamSettings are the per-team review settings held in the DB.
//
// Table: team_settings
type TeamSettings struct {
	AutoReviewAuthors []string
	IgnoreAuthors     []string
	ExcludeRepos      []string
	// Branch-name globs on the source (fromRef) and target (toRef) branch.
	ExcludeSourceBranches []string
	ExcludeTargetBranches []string
	UsePRDescription      bool
}

// claimRow is one team_claims row as the claim reads take it.
//
// Table: team_claims
type claimRow struct {
	team string
	key  string
	repo *string
}

// VisualPair identifies one screenshot pair by content, migrations/
// 0023_visual_changes.sql. BeforeID is "" for an added screenshot.
//
// Table: visual_changes
type VisualPair struct {
	Path     string
	BeforeID string
	AfterID  string
}
