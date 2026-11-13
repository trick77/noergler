package store

import (
	"context"
	"encoding/json"
	"errors"
	"time"

	"github.com/jackc/pgx/v5"
)

// PRUpsert is what a review writes about the PR itself.
type PRUpsert struct {
	Key PRKey
	// TeamSlug is the team the webhook route authenticated, never a payload
	// value. A repo belongs to exactly one team, so an update from a
	// different slug means the file moved the repo: the row follows.
	TeamSlug           string
	LastReviewedCommit *string
	// KeepPointer leaves an existing row's pointer as it is: the skip path
	// could not read the prior one, and passing "" back would NULL it.
	KeepPointer bool
	Author      *string
	Title       *string
	OpenedAt    *time.Time
}

// UpsertPullRequest inserts or updates the PR row and returns its id. A new
// row draws its page's link token; an update keeps the one it has.
// opened_at is sticky: the first non-NULL value stays. The pointer, author
// and title are written as given, nil included: the skip paths pass the
// prior pointer back on purpose (unless KeepPointer). A declined PR that
// sees a review again was reopened: declined_at is cleared.
func (s *Store) UpsertPullRequest(ctx context.Context, u PRUpsert) (int64, error) {
	var id int64
	err := s.pool.QueryRow(ctx, `
		INSERT INTO pull_requests (project_key, repo_slug, pr_id, team_slug, last_reviewed_commit, author, title, opened_at, link_token)
		VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $10)
		ON CONFLICT (project_key, repo_slug, pr_id) DO UPDATE SET
			team_slug = EXCLUDED.team_slug,
			last_reviewed_commit = CASE WHEN $9 THEN pull_requests.last_reviewed_commit ELSE EXCLUDED.last_reviewed_commit END,
			author = EXCLUDED.author,
			title = EXCLUDED.title,
			opened_at = COALESCE(pull_requests.opened_at, EXCLUDED.opened_at),
			declined_at = NULL,
			updated_at = now()
		RETURNING id`,
		u.Key.Project, u.Key.Repo, u.Key.PRID, u.TeamSlug, u.LastReviewedCommit, u.Author, u.Title, u.OpenedAt, u.KeepPointer,
		NewLinkToken(),
	).Scan(&id)
	return id, err
}

// activeFilter keeps only PRs whose findings are still on Bitbucket: the
// previously-posted block and dedup read them. A declined PR keeps its
// comments, so declined rows stay in (a reopened PR must not get the same
// findings posted twice).
const activeFilter = `merged_at IS NULL AND deleted_at IS NULL`

// GetLastReviewedCommit returns the pointer for an open PR; ok is false when
// there is no row or the PR is closed. Declined counts as closed, so a
// declined-then-reopened PR gets a full review and the upsert that follows
// clears declined_at.
func (s *Store) GetLastReviewedCommit(ctx context.Context, k PRKey) (string, bool, error) {
	var commit *string
	err := s.pool.QueryRow(ctx, `
		SELECT last_reviewed_commit FROM pull_requests
		WHERE project_key = $1 AND repo_slug = $2 AND pr_id = $3 AND `+activeFilter+` AND declined_at IS NULL`,
		k.Project, k.Repo, k.PRID).Scan(&commit)
	if errors.Is(err, pgx.ErrNoRows) {
		return "", false, nil
	}
	if err != nil || commit == nil {
		return "", false, err
	}
	return *commit, true, nil
}

// SetSummaryComment records the summary comment's id and version for the
// optimistic-locking update on the next run.
func (s *Store) SetSummaryComment(ctx context.Context, prID int64, commentID, version int) error {
	_, err := s.pool.Exec(ctx, `
		UPDATE pull_requests SET summary_comment_id = $2, summary_comment_version = $3, updated_at = now()
		WHERE id = $1`, prID, commentID, version)
	return err
}

// PRLinkToken is the PR page's link token, for the summary's link.
func (s *Store) PRLinkToken(ctx context.Context, prID int64) (string, error) {
	var tok string
	err := s.pool.QueryRow(ctx, `SELECT link_token FROM pull_requests WHERE id = $1`, prID).Scan(&tok)
	return tok, err
}

// SummaryComment is the stored comment reference.
type SummaryComment struct {
	ID      int
	Version int
}

// summaryRef is the stored reference from its two nullable columns: nil when
// no comment is stored, version 0 when none was recorded.
func summaryRef(id, version *int) *SummaryComment {
	if id == nil {
		return nil
	}
	return &SummaryComment{ID: *id, Version: deref(version)}
}

// GetSummaryComment returns the reference, or nil when none is stored.
func (s *Store) GetSummaryComment(ctx context.Context, prID int64) (*SummaryComment, error) {
	var id, version *int
	err := s.pool.QueryRow(ctx, `SELECT summary_comment_id, summary_comment_version FROM pull_requests WHERE id = $1`, prID).Scan(&id, &version)
	if errors.Is(err, pgx.ErrNoRows) {
		return nil, nil
	}
	if err != nil {
		return nil, err
	}
	return summaryRef(id, version), nil
}

// Why a PR is ignored (pull_requests.ignore_reason).
const (
	// IgnoreSummaryDeleted: a human deleted the summary comment.
	IgnoreSummaryDeleted = "summary_deleted"
	// IgnoreStopped: someone told noergler to stop reviewing the PR. Its
	// summary is still there.
	IgnoreStopped = "stopped"
)

// GetSkipState returns nil when the PR has no row yet (first review).
func (s *Store) GetSkipState(ctx context.Context, k PRKey) (*SkipState, error) {
	var st SkipState
	var id, version *int
	var reason, by *string
	err := s.pool.QueryRow(ctx, `
		SELECT id, ignored_at, ignore_reason, stopped_by, summary_comment_id, summary_comment_version FROM pull_requests
		WHERE project_key = $1 AND repo_slug = $2 AND pr_id = $3`,
		k.Project, k.Repo, k.PRID).Scan(&st.ID, &st.IgnoredAt, &reason, &by, &id, &version)
	if errors.Is(err, pgx.ErrNoRows) {
		return nil, nil
	}
	if err != nil {
		return nil, err
	}
	// Set together with ignored_at (pull_requests_ignore_reason_set).
	if reason != nil {
		st.IgnoreReason = *reason
	}
	if by != nil {
		st.StoppedBy = *by
	}
	st.Summary = summaryRef(id, version)
	return &st, nil
}

// MarkIgnored flags a PR whose summary comment a human deleted. Idempotent.
// A stopped PR becomes a deleted summary's: its resume must then forget the
// summary. ignored_at keeps its first stamp.
func (s *Store) MarkIgnored(ctx context.Context, k PRKey) error {
	return s.exec(ctx, `UPDATE pull_requests
		SET ignored_at = COALESCE(ignored_at, now()), ignore_reason = 'summary_deleted', stopped_by = NULL,
			updated_at = now()
		WHERE project_key = $1 AND repo_slug = $2 AND pr_id = $3
			AND (ignored_at IS NULL OR ignore_reason IS DISTINCT FROM 'summary_deleted')`, k)
}

// MarkStopped flags a PR someone told noergler to stop reviewing. A PR
// ignored after its summary was deleted becomes stopped too, or the next
// question would bring it back against the stop; its summary is gone, so
// it is forgotten as Reactivate would. Returns the row id when this call
// stopped it, 0 when there is no row or it was stopped already.
func (s *Store) MarkStopped(ctx context.Context, k PRKey, by string) (int64, error) {
	var id int64
	err := s.pool.QueryRow(ctx, `UPDATE pull_requests
		SET summary_comment_id = CASE WHEN ignored_at IS NULL THEN summary_comment_id END,
			summary_comment_version = CASE WHEN ignored_at IS NULL THEN summary_comment_version END,
			last_summary = CASE WHEN ignored_at IS NULL THEN last_summary END,
			ignored_at = now(), ignore_reason = 'stopped', stopped_by = $4, updated_at = now()
		WHERE project_key = $1 AND repo_slug = $2 AND pr_id = $3
			AND (ignored_at IS NULL OR ignore_reason = 'summary_deleted')
		RETURNING id`,
		k.Project, k.Repo, k.PRID, by).Scan(&id)
	if errors.Is(err, pgx.ErrNoRows) {
		return 0, nil
	}
	return id, err
}

// Reactivate clears the ignored flag. After a deleted summary it also
// forgets the old summary comment, so the next review posts a fresh one
// instead of updating the deleted one; its prose goes with it and the next
// overview starts over. A stopped PR keeps both: its summary is still there.
func (s *Store) Reactivate(ctx context.Context, k PRKey) error {
	return s.exec(ctx, `UPDATE pull_requests
		SET summary_comment_id = CASE WHEN ignore_reason = 'stopped' THEN summary_comment_id END,
			summary_comment_version = CASE WHEN ignore_reason = 'stopped' THEN summary_comment_version END,
			last_summary = CASE WHEN ignore_reason = 'stopped' THEN last_summary END,
			ignored_at = NULL, ignore_reason = NULL, stopped_by = NULL, updated_at = now()
		WHERE project_key = $1 AND repo_slug = $2 AND pr_id = $3`, k)
}

// MarkMerged stamps merged_at once. The first merge also clears
// riptide_emitted_at: set before a merge, it is a decline's, and Bitbucket
// cannot merge a declined PR, so the PR was reopened and its merge emits a
// second, cumulative rollup (riptide reads the newest per PR). Hanging this
// on the reopen's review missed every path that writes no PR row. A
// redelivered pr:merged matches nothing and keeps the stamp.
func (s *Store) MarkMerged(ctx context.Context, k PRKey) error {
	return s.exec(ctx, `UPDATE pull_requests SET merged_at = now(), riptide_emitted_at = NULL, updated_at = now()
		WHERE project_key = $1 AND repo_slug = $2 AND pr_id = $3 AND merged_at IS NULL`, k)
}

// MarkDeclined stamps declined_at once.
func (s *Store) MarkDeclined(ctx context.Context, k PRKey) error {
	return s.exec(ctx, `UPDATE pull_requests SET declined_at = now(), updated_at = now()
		WHERE project_key = $1 AND repo_slug = $2 AND pr_id = $3 AND declined_at IS NULL`, k)
}

// MarkDeleted stamps deleted_at once.
func (s *Store) MarkDeleted(ctx context.Context, k PRKey) error {
	return s.exec(ctx, `UPDATE pull_requests SET deleted_at = now(), updated_at = now()
		WHERE project_key = $1 AND repo_slug = $2 AND pr_id = $3 AND deleted_at IS NULL`, k)
}

func (s *Store) exec(ctx context.Context, sql string, k PRKey) error {
	_, err := s.pool.Exec(ctx, sql, k.Project, k.Repo, k.PRID)
	return err
}

// RunKind is auto or mention.
type RunKind string

const (
	// RunAuto is an automatic review triggered by a PR creation or update.
	RunAuto RunKind = "auto"
	// RunMention is a manual review triggered by a comment mention.
	RunMention RunKind = "mention"
)

// Run is one completed review run.
type Run struct {
	PullRequestID int64
	Kind          RunKind
	Incremental   bool
	FromCommit    *string
	ToCommit      string
	ModelLabel    string
	PromptTokens  int64
	CachedTokens  int64
	// CompletionTokens includes reasoning tokens (llmwire's usage rule).
	CompletionTokens int64
	// CostNanoUSD is nil when the gateway reported no cost: the run counts,
	// its price does not (fail open).
	CostNanoUSD    *int64
	ElapsedMS      int64
	FindingsPosted int
	LinesAdded     int
	LinesRemoved   int
	FilesChanged   int
	// ReviewedAdded and ReviewedRemoved count only the files the prompt
	// carried: Lines* above include size-gated and over-budget ones. Run
	// row only, never the PR's final_lines_*.
	ReviewedAdded   int
	ReviewedRemoved int
	// Summary is the run's PR summary prose, carried onto the PR for the
	// next review. nil keeps the stored one.
	Summary *PRSummary
}

// PRSummary is the PR summary prose the next review builds on.
type PRSummary struct {
	Overview string `json:"overview"`
}

// InsertRun records the run and carries the run's commit and diff size onto
// the PR as the latest known final figures.
// Returns the run id.
func (s *Store) InsertRun(ctx context.Context, r Run) (int64, error) {
	var summary []byte
	if r.Summary != nil {
		var err error
		if summary, err = json.Marshal(r.Summary); err != nil {
			return 0, err
		}
	}
	var id int64
	err := s.inTx(ctx, func(tx pgx.Tx) error {
		if err := tx.QueryRow(ctx, `
		INSERT INTO review_runs (pull_request_id, kind, incremental, from_commit, to_commit, model_label,
			prompt_tokens, cached_tokens, completion_tokens, cost_nano_usd, elapsed_ms, findings_posted,
			lines_added, lines_removed, files_changed, reviewed_lines_added, reviewed_lines_removed)
		VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17)
		RETURNING id`,
			r.PullRequestID, string(r.Kind), r.Incremental, r.FromCommit, r.ToCommit, r.ModelLabel,
			r.PromptTokens, r.CachedTokens, r.CompletionTokens, r.CostNanoUSD, r.ElapsedMS, r.FindingsPosted,
			r.LinesAdded, r.LinesRemoved, r.FilesChanged, r.ReviewedAdded, r.ReviewedRemoved,
		).Scan(&id); err != nil {
			return err
		}
		_, err := tx.Exec(ctx, `
		UPDATE pull_requests SET final_source_commit = $2, final_lines_added = $3, final_lines_removed = $4,
			final_files_changed = $5, last_summary = COALESCE($6, last_summary), updated_at = now()
		WHERE id = $1`, r.PullRequestID, r.ToCommit, r.LinesAdded, r.LinesRemoved, r.FilesChanged, summary)
		return err
	})
	if err != nil {
		return 0, err
	}
	return id, nil
}

// LastSummary is the PR summary prose of the PR's last completed review, or
// nil when none is stored. Active PRs only, like the findings: a merged or
// deleted row never hands its prose to a PR that reuses the id. A declined
// and reopened PR is the same PR and keeps it.
func (s *Store) LastSummary(ctx context.Context, k PRKey) (*PRSummary, error) {
	var raw []byte
	err := s.pool.QueryRow(ctx, `
		SELECT last_summary FROM pull_requests
		WHERE project_key = $1 AND repo_slug = $2 AND pr_id = $3 AND `+activeFilter,
		k.Project, k.Repo, k.PRID).Scan(&raw)
	if errors.Is(err, pgx.ErrNoRows) || (err == nil && raw == nil) {
		return nil, nil
	}
	if err != nil {
		return nil, err
	}
	var out PRSummary
	if err := json.Unmarshal(raw, &out); err != nil {
		return nil, err
	}
	return &out, nil
}

// InsertFinding stores a posted finding.
func (s *Store) InsertFinding(ctx context.Context, f Finding) error {
	_, err := s.pool.Exec(ctx, `
		INSERT INTO findings (pull_request_id, review_run_id, file_path, line_number, severity, confidence, headline,
			comment_text, suggestion, bitbucket_comment_id)
		VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)`,
		f.PullRequestID, f.RunID, f.FilePath, f.LineNumber, f.Severity, f.Confidence, f.Headline,
		f.CommentText, f.Suggestion, f.BitbucketCommentID)
	return err
}

// ExistingFindings returns the findings posted on an open PR, oldest first,
// for the previously-posted block of the prompt and for dedup. A fixed one is
// left out of both, so the same bug coming back is posted again.
func (s *Store) ExistingFindings(ctx context.Context, k PRKey) ([]Finding, error) {
	return queryAll(ctx, s.pool, `
		SELECT `+findingCols+`, f.closed_at
		FROM findings f JOIN pull_requests p ON f.pull_request_id = p.id
		WHERE p.project_key = $1 AND p.repo_slug = $2 AND p.pr_id = $3 AND p.`+activeFilter+`
			AND f.fixed_at IS NULL
		ORDER BY f.id ASC`, []any{k.Project, k.Repo, k.PRID},
		func(row pgx.CollectableRow, f *Finding) error {
			return row.Scan(append(f.scanTargets(), &f.ClosedAt)...)
		})
}

// findingCols is the finding row as ExistingFindings and FindingByComment
// read it, from `findings f`; scanTargets receives it in the same order.
const findingCols = `f.id, f.pull_request_id, f.review_run_id, f.file_path, f.line_number, f.severity, f.confidence,
			f.headline, f.comment_text, f.suggestion, f.bitbucket_comment_id, f.replied_at, f.dispute_text,
			(SELECT x.commit_sha FROM finding_fixes x WHERE x.finding_id = f.id AND x.outcome = 'committed')`

func (f *Finding) scanTargets() []any {
	return []any{&f.ID, &f.PullRequestID, &f.RunID, &f.FilePath, &f.LineNumber, &f.Severity, &f.Confidence,
		&f.Headline, &f.CommentText, &f.Suggestion, &f.BitbucketCommentID, &f.RepliedAt, &f.DisputeText,
		&f.BotFixCommit}
}

// MarkFindingClosed settles a finding whose thread was resolved or deleted
// on the PR. The first stamp wins, so its thread is never read again.
func (s *Store) MarkFindingClosed(ctx context.Context, findingID int64) error {
	_, err := s.pool.Exec(ctx, `UPDATE findings SET closed_at = now() WHERE id = $1 AND closed_at IS NULL`, findingID)
	return err
}

// MarkFindingOpen unsettles a finding whose thread was reopened on the PR.
func (s *Store) MarkFindingOpen(ctx context.Context, findingID int64) error {
	_, err := s.pool.Exec(ctx, `UPDATE findings SET closed_at = NULL WHERE id = $1`, findingID)
	return err
}

// FindingByComment is the open, unfixed finding whose inline comment is
// commentID on that PR, or nil. Scoped to the PR key the webhook
// authenticated, so a comment id from another team's PR finds nothing.
func (s *Store) FindingByComment(ctx context.Context, k PRKey, commentID int) (*Finding, error) {
	return s.openFinding(ctx, k, "f.bitbucket_comment_id", int64(commentID))
}

// FindingByID is FindingByComment for a finding named by its row id, the
// one a stored bot reply points at.
func (s *Store) FindingByID(ctx context.Context, k PRKey, id int64) (*Finding, error) {
	return s.openFinding(ctx, k, "f.id", id)
}

// openFinding is the one query behind FindingByComment and FindingByID:
// scoped to the PR key, the PR active, the finding unfixed. col is a
// literal column name from those two callers, never input.
func (s *Store) openFinding(ctx context.Context, k PRKey, col string, v int64) (*Finding, error) {
	var f Finding
	err := s.pool.QueryRow(ctx, `
		SELECT `+findingCols+`, r.to_commit
		FROM findings f JOIN pull_requests p ON f.pull_request_id = p.id
			JOIN review_runs r ON f.review_run_id = r.id
		WHERE p.project_key = $1 AND p.repo_slug = $2 AND p.pr_id = $3 AND p.`+activeFilter+`
			AND f.fixed_at IS NULL AND `+col+` = $4`,
		k.Project, k.Repo, k.PRID, v).Scan(append(f.scanTargets(), &f.RunCommit)...)
	if errors.Is(err, pgx.ErrNoRows) {
		return nil, nil
	}
	if err != nil {
		return nil, err
	}
	return &f, nil
}

// Bot reply kinds: what noergler posted, for the log line of a reply under it.
const (
	BotReplyQuestion = "question" // a Q&A answer, a notice or a language confirmation
	BotReplyReply    = "reply"    // the answer on a finding's thread
	BotReplyFixed    = "fixed"    // the "Fixed in" reply on a retired finding
)

// InsertBotReply records a reply noergler posted on a PR. A PR without a
// row records nothing (a language confirmation on a PR never reviewed or
// asked about), and a replayed id is kept once.
func (s *Store) InsertBotReply(ctx context.Context, k PRKey, commentID int, kind string) error {
	return s.InsertFindingBotReply(ctx, k, commentID, kind, 0)
}

// InsertFindingBotReply is InsertBotReply for an answer on a finding's
// thread: findingID names the finding (0 for none), so the author's reply
// under that answer finds its finding again.
func (s *Store) InsertFindingBotReply(ctx context.Context, k PRKey, commentID int, kind string, findingID int64) error {
	_, err := s.pool.Exec(ctx, `
		INSERT INTO bot_replies (pull_request_id, comment_id, kind, finding_id)
		SELECT id, $4, $5, NULLIF($6::bigint, 0) FROM pull_requests
		WHERE project_key = $1 AND repo_slug = $2 AND pr_id = $3
		ON CONFLICT DO NOTHING`,
		k.Project, k.Repo, k.PRID, commentID, kind, findingID)
	return err
}

// BotReplyByComment is the reply noergler posted as commentID on that PR,
// or nil. Scoped like FindingByComment: a comment id from another team's PR
// finds nothing, and a merged or deleted PR is not answered on.
func (s *Store) BotReplyByComment(ctx context.Context, k PRKey, commentID int) (*BotReply, error) {
	var b BotReply
	err := s.pool.QueryRow(ctx, `
		SELECT b.comment_id, b.kind, COALESCE(b.finding_id, 0)
		FROM bot_replies b JOIN pull_requests p ON b.pull_request_id = p.id
		WHERE p.project_key = $1 AND p.repo_slug = $2 AND p.pr_id = $3 AND p.`+activeFilter+`
			AND b.comment_id = $4`,
		k.Project, k.Repo, k.PRID, commentID).Scan(&b.CommentID, &b.Kind, &b.FindingID)
	if errors.Is(err, pgx.ErrNoRows) {
		return nil, nil
	}
	if err != nil {
		return nil, err
	}
	return &b, nil
}

// MarkFindingReplied records the one answer on a finding's thread. A
// non-nil dispute is the author's case against the finding. The first
// answer wins: a finding is answered once.
func (s *Store) MarkFindingReplied(ctx context.Context, findingID int64, dispute *string) error {
	_, err := s.pool.Exec(ctx, `
		UPDATE findings SET replied_at = now(), dispute_text = $2::text
		WHERE id = $1 AND replied_at IS NULL`, findingID, dispute)
	return err
}

// MarkFindingFixed retires a finding a later run found fixed. The first
// stamp wins: a finding is fixed once. A zero runID (the run row failed)
// stamps NULL: the thread is resolved either way, and an unstamped row
// would be listed, claimed and resolved again.
func (s *Store) MarkFindingFixed(ctx context.Context, findingID, runID int64) error {
	_, err := s.pool.Exec(ctx, `
		UPDATE findings SET fixed_at = now(), fixed_run_id = NULLIF($2::bigint, 0)
		WHERE id = $1 AND fixed_at IS NULL`, findingID, runID)
	return err
}

// PRCost is the sum of priced runs, billed run-less attempts (an
// unparseable response) and reply and question calls in nano-USD, or nil
// when nothing was priced. Callers treat nil as "no known cost" and never
// block on it. Unpriced calls contribute nothing, so the total is a floor,
// not the bill.
func (s *Store) PRCost(ctx context.Context, k PRKey) (*int64, error) {
	_, cost, err := s.PRCapState(ctx, k)
	return cost, err
}

// PRCapState is PRCost plus the PR row's id, which the cap notice needs on a
// path that does not upsert the PR (a reply, a question). id 0 = no row.
func (s *Store) PRCapState(ctx context.Context, k PRKey) (id int64, cost *int64, err error) {
	err = s.pool.QueryRow(ctx, `SELECT p.id, `+prCostSQL+` FROM pull_requests p
		WHERE p.project_key = $1 AND p.repo_slug = $2 AND p.pr_id = $3`,
		k.Project, k.Repo, k.PRID).Scan(&id, &cost)
	if errors.Is(err, pgx.ErrNoRows) {
		return 0, nil, nil
	}
	return id, cost, err
}

// prCostSQL is PR row p's cost: its priced runs plus the billed run-less
// attempts made during this row's life, NULL when nothing was priced. PRCost,
// the frozen final cost and the riptide rollup all read it, so the figure
// that tripped the cap is the one recorded. created_at bounds the attempts:
// a purged repo claimed again starts from zero, as its runs do.
const prCostSQL = `(SELECT CASE WHEN runs IS NULL AND attempts IS NULL AND calls IS NULL THEN NULL
		ELSE COALESCE(runs, 0) + COALESCE(attempts, 0) + COALESCE(calls, 0) END
	FROM (SELECT
		(SELECT SUM(r.cost_nano_usd) FROM review_runs r WHERE r.pull_request_id = p.id) AS runs,
		(SELECT SUM(a.cost_nano_usd) FROM review_attempts a
		  WHERE a.project_key = p.project_key AND a.repo_slug = p.repo_slug AND a.pr_id = p.pr_id
		    AND a.review_run_id IS NULL AND a.created_at >= p.created_at) AS attempts,
		-- Replies and questions: no attempt, so their call row is the
		-- ledger. Review calls (attempt_id set) are already in the two above.
		(SELECT SUM(c.cost_nano_usd) FROM llm_calls c
		  WHERE c.project_key = p.project_key AND c.repo_slug = p.repo_slug AND c.pr_id = p.pr_id
		    AND c.attempt_id IS NULL AND c.created_at >= p.created_at) AS calls) t)`

// FreezeFinalCost copies the PR's cost total into final_cost_nano_usd at the
// terminal outcome and returns it (nil when unpriced).
func (s *Store) FreezeFinalCost(ctx context.Context, k PRKey) (*int64, error) {
	var cost *int64
	err := s.pool.QueryRow(ctx, `
		UPDATE pull_requests p SET final_cost_nano_usd = `+prCostSQL+`, updated_at = now()
		WHERE project_key = $1 AND repo_slug = $2 AND pr_id = $3
		RETURNING final_cost_nano_usd`, k.Project, k.Repo, k.PRID).Scan(&cost)
	if errors.Is(err, pgx.ErrNoRows) {
		return nil, nil
	}
	return cost, err
}

// RollupFinal is the final diff refresh taken at the terminal outcome; nil
// fields keep what the last run recorded. SourceCommit only fills a PR no
// run recorded one for: one billed for unparseable responses alone.
type RollupFinal struct {
	SourceCommit *string
	MergeCommit  *string
	LinesAdded   *int
	LinesRemoved *int
	FilesChanged *int
}

// ClaimRollup atomically stamps riptide_emitted_at and returns the snapshot.
// nil when already emitted (a redelivered pr:merged must not produce a second
// event) or when the PR has neither a run nor a billed run-less attempt
// (nothing to forward). Attempt-only spend is in the cap and the frozen cost,
// so it must reach FinOps too. The claim happens
// BEFORE the POST, so a failed emission is never retried.
func (s *Store) ClaimRollup(ctx context.Context, k PRKey, final RollupFinal) (*RollupSnapshot, error) {
	var snap RollupSnapshot
	err := s.pool.QueryRow(ctx, `
		WITH claimed AS (
			UPDATE pull_requests p SET
				riptide_emitted_at = now(),
				final_source_commit = COALESCE(p.final_source_commit, $8),
				final_merge_commit = COALESCE($4, p.final_merge_commit),
				final_lines_added = COALESCE($5, p.final_lines_added),
				final_lines_removed = COALESCE($6, p.final_lines_removed),
				final_files_changed = COALESCE($7, p.final_files_changed),
				updated_at = now()
			WHERE p.project_key = $1 AND p.repo_slug = $2 AND p.pr_id = $3
			  AND p.riptide_emitted_at IS NULL
			  AND (EXISTS (SELECT 1 FROM review_runs r WHERE r.pull_request_id = p.id)
			    OR EXISTS (SELECT 1 FROM review_attempts a
			      WHERE a.project_key = p.project_key AND a.repo_slug = p.repo_slug AND a.pr_id = p.pr_id
			        AND a.review_run_id IS NULL AND a.cost_nano_usd IS NOT NULL AND a.created_at >= p.created_at)
			    OR EXISTS (SELECT 1 FROM llm_calls c
			      WHERE c.project_key = p.project_key AND c.repo_slug = p.repo_slug AND c.pr_id = p.pr_id
			        AND c.attempt_id IS NULL AND c.cost_nano_usd IS NOT NULL AND c.created_at >= p.created_at))
			RETURNING p.id, p.created_at, p.final_source_commit, p.final_merge_commit, p.final_lines_added, p.final_lines_removed, p.final_files_changed
		)
		SELECT COUNT(r.id), COALESCE(SUM(r.prompt_tokens), 0), COALESCE(SUM(r.completion_tokens), 0),
			COALESCE(SUM(r.elapsed_ms), 0), COALESCE(SUM(r.findings_posted), 0),
			(SELECT `+prCostSQL+` FROM pull_requests p WHERE p.id = c.id),
			ARRAY(SELECT DISTINCT m FROM unnest(array_agg(r.model_label)) AS m WHERE m IS NOT NULL ORDER BY m),
			COALESCE(MIN(r.created_at), c.created_at),
			c.final_source_commit, c.final_merge_commit, c.final_lines_added, c.final_lines_removed, c.final_files_changed
		FROM claimed c LEFT JOIN review_runs r ON r.pull_request_id = c.id
		GROUP BY c.id, c.created_at, c.final_source_commit, c.final_merge_commit, c.final_lines_added, c.final_lines_removed, c.final_files_changed`,
		k.Project, k.Repo, k.PRID, final.MergeCommit, final.LinesAdded, final.LinesRemoved, final.FilesChanged, final.SourceCommit,
	).Scan(&snap.Runs, &snap.PromptTokens, &snap.CompletionTokens, &snap.ElapsedMS, &snap.Findings, &snap.CostNanoUSD,
		&snap.Models, &snap.FirstReviewAt, &snap.SourceCommit, &snap.MergeCommit, &snap.LinesAdded, &snap.LinesRemoved, &snap.FilesChanged)
	if errors.Is(err, pgx.ErrNoRows) {
		return nil, nil
	}
	if err != nil {
		return nil, err
	}
	return &snap, nil
}
