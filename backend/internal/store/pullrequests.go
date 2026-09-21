package store

import (
	"context"
	"errors"
	"fmt"
	"time"

	"github.com/jackc/pgx/v5"
)

// PRKey identifies a pull request across Bitbucket and the store.
type PRKey struct {
	Project string
	Repo    string
	PRID    int
}

// Tag is `PROJECT/repo#id`, the log and riptide spelling.
func (k PRKey) Tag() string { return fmt.Sprintf("%s/%s#%d", k.Project, k.Repo, k.PRID) }

// PRUpsert is what a review writes about the PR itself.
type PRUpsert struct {
	Key PRKey
	// TeamSlug is the team the webhook route authenticated, never a payload
	// value. A repo belongs to exactly one team, so an update from a
	// different slug means the file moved the repo: the row follows.
	TeamSlug           string
	LastReviewedCommit *string
	Author             *string
	Title              *string
	OpenedAt           *time.Time
}

// UpsertPullRequest inserts or updates the PR row and returns its id.
// opened_at is sticky: the first non-NULL value stays. The pointer, author
// and title are written as given, nil included: the skip paths pass the
// prior pointer back on purpose. A declined PR that sees a
// review again was reopened: declined_at is cleared.
func (s *Store) UpsertPullRequest(ctx context.Context, u PRUpsert) (int64, error) {
	var id int64
	err := s.pool.QueryRow(ctx, `
		INSERT INTO pull_requests (project_key, repo_slug, pr_id, team_slug, last_reviewed_commit, author, title, opened_at)
		VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
		ON CONFLICT (project_key, repo_slug, pr_id) DO UPDATE SET
			team_slug = EXCLUDED.team_slug,
			last_reviewed_commit = EXCLUDED.last_reviewed_commit,
			author = EXCLUDED.author,
			title = EXCLUDED.title,
			opened_at = COALESCE(pull_requests.opened_at, EXCLUDED.opened_at),
			declined_at = NULL,
			updated_at = now()
		RETURNING id`,
		u.Key.Project, u.Key.Repo, u.Key.PRID, u.TeamSlug, u.LastReviewedCommit, u.Author, u.Title, u.OpenedAt,
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

// SummaryComment is the stored comment reference.
type SummaryComment struct {
	ID      int
	Version int
}

// GetSummaryComment returns the reference, or nil when none is stored.
func (s *Store) GetSummaryComment(ctx context.Context, prID int64) (*SummaryComment, error) {
	var id, version *int
	err := s.pool.QueryRow(ctx, `SELECT summary_comment_id, summary_comment_version FROM pull_requests WHERE id = $1`, prID).Scan(&id, &version)
	if errors.Is(err, pgx.ErrNoRows) || (err == nil && id == nil) {
		return nil, nil
	}
	if err != nil {
		return nil, err
	}
	v := 0
	if version != nil {
		v = *version
	}
	return &SummaryComment{ID: *id, Version: v}, nil
}

// SkipState is what the review guard needs to decide whether to skip a PR.
type SkipState struct {
	ID        int64
	IgnoredAt *time.Time
	// Summary is nil when no summary comment is stored.
	Summary *SummaryComment
}

// GetSkipState returns nil when the PR has no row yet (first review).
func (s *Store) GetSkipState(ctx context.Context, k PRKey) (*SkipState, error) {
	var st SkipState
	var id, version *int
	err := s.pool.QueryRow(ctx, `
		SELECT id, ignored_at, summary_comment_id, summary_comment_version FROM pull_requests
		WHERE project_key = $1 AND repo_slug = $2 AND pr_id = $3`,
		k.Project, k.Repo, k.PRID).Scan(&st.ID, &st.IgnoredAt, &id, &version)
	if errors.Is(err, pgx.ErrNoRows) {
		return nil, nil
	}
	if err != nil {
		return nil, err
	}
	if id != nil {
		v := 0
		if version != nil {
			v = *version
		}
		st.Summary = &SummaryComment{ID: *id, Version: v}
	}
	return &st, nil
}

// MarkIgnored flags a PR whose summary comment a human deleted. Idempotent.
func (s *Store) MarkIgnored(ctx context.Context, k PRKey) error {
	return s.exec(ctx, `UPDATE pull_requests SET ignored_at = now(), updated_at = now()
		WHERE project_key = $1 AND repo_slug = $2 AND pr_id = $3 AND ignored_at IS NULL`, k)
}

// Reactivate clears the ignored flag and forgets the old summary comment, so
// the next review posts a fresh one instead of updating the deleted one.
func (s *Store) Reactivate(ctx context.Context, k PRKey) error {
	return s.exec(ctx, `UPDATE pull_requests
		SET ignored_at = NULL, summary_comment_id = NULL, summary_comment_version = NULL, updated_at = now()
		WHERE project_key = $1 AND repo_slug = $2 AND pr_id = $3`, k)
}

// MarkMerged stamps merged_at once.
func (s *Store) MarkMerged(ctx context.Context, k PRKey) error {
	return s.exec(ctx, `UPDATE pull_requests SET merged_at = now(), updated_at = now()
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
}

// InsertRun records the run and carries the run's commit and diff size onto
// the PR as the latest known final figures.
// Returns the run id.
func (s *Store) InsertRun(ctx context.Context, r Run) (int64, error) {
	tx, err := s.pool.Begin(ctx)
	if err != nil {
		return 0, err
	}
	defer func() { _ = tx.Rollback(ctx) }()
	var id int64
	if err := tx.QueryRow(ctx, `
		INSERT INTO review_runs (pull_request_id, kind, incremental, from_commit, to_commit, model_label,
			prompt_tokens, cached_tokens, completion_tokens, cost_nano_usd, elapsed_ms, findings_posted,
			lines_added, lines_removed, files_changed)
		VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15)
		RETURNING id`,
		r.PullRequestID, string(r.Kind), r.Incremental, r.FromCommit, r.ToCommit, r.ModelLabel,
		r.PromptTokens, r.CachedTokens, r.CompletionTokens, r.CostNanoUSD, r.ElapsedMS, r.FindingsPosted,
		r.LinesAdded, r.LinesRemoved, r.FilesChanged,
	).Scan(&id); err != nil {
		return 0, err
	}
	if _, err := tx.Exec(ctx, `
		UPDATE pull_requests SET final_source_commit = $2, final_lines_added = $3, final_lines_removed = $4,
			final_files_changed = $5, updated_at = now()
		WHERE id = $1`, r.PullRequestID, r.ToCommit, r.LinesAdded, r.LinesRemoved, r.FilesChanged); err != nil {
		return 0, err
	}
	return id, tx.Commit(ctx)
}

// Finding is one posted inline comment.
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
// for the previously-posted block of the prompt and for dedup.
func (s *Store) ExistingFindings(ctx context.Context, k PRKey) ([]Finding, error) {
	rows, err := s.pool.Query(ctx, `
		SELECT f.id, f.pull_request_id, f.review_run_id, f.file_path, f.line_number, f.severity, f.confidence,
			f.headline, f.comment_text, f.suggestion, f.bitbucket_comment_id
		FROM findings f JOIN pull_requests p ON f.pull_request_id = p.id
		WHERE p.project_key = $1 AND p.repo_slug = $2 AND p.pr_id = $3 AND p.`+activeFilter+`
		ORDER BY f.id ASC`, k.Project, k.Repo, k.PRID)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	var out []Finding
	for rows.Next() {
		var f Finding
		if err := rows.Scan(&f.ID, &f.PullRequestID, &f.RunID, &f.FilePath, &f.LineNumber, &f.Severity, &f.Confidence,
			&f.Headline, &f.CommentText, &f.Suggestion, &f.BitbucketCommentID); err != nil {
			return nil, err
		}
		out = append(out, f)
	}
	return out, rows.Err()
}

// PRCost is the sum of priced runs in nano-USD, or nil when no run was
// priced (no row, no runs, or every run unpriced). Callers treat nil as "no
// known cost" and never block on it. Unpriced runs contribute nothing, so
// the total is a floor, not the bill.
func (s *Store) PRCost(ctx context.Context, k PRKey) (*int64, error) {
	var cost *int64
	err := s.pool.QueryRow(ctx, `
		SELECT SUM(r.cost_nano_usd) FROM review_runs r JOIN pull_requests p ON r.pull_request_id = p.id
		WHERE p.project_key = $1 AND p.repo_slug = $2 AND p.pr_id = $3`,
		k.Project, k.Repo, k.PRID).Scan(&cost)
	return cost, err
}

// FreezeFinalCost copies the PR's cost total into final_cost_nano_usd at the
// terminal outcome and returns it (nil when unpriced).
func (s *Store) FreezeFinalCost(ctx context.Context, k PRKey) (*int64, error) {
	var cost *int64
	err := s.pool.QueryRow(ctx, `
		UPDATE pull_requests p SET final_cost_nano_usd = (
			SELECT SUM(cost_nano_usd) FROM review_runs WHERE pull_request_id = p.id
		), updated_at = now()
		WHERE project_key = $1 AND repo_slug = $2 AND pr_id = $3
		RETURNING final_cost_nano_usd`, k.Project, k.Repo, k.PRID).Scan(&cost)
	if errors.Is(err, pgx.ErrNoRows) {
		return nil, nil
	}
	return cost, err
}

// RollupFinal is the final diff refresh taken at the terminal outcome; nil
// fields keep what the last run recorded.
type RollupFinal struct {
	MergeCommit  *string
	LinesAdded   *int
	LinesRemoved *int
	FilesChanged *int
}

// RollupSnapshot is what one riptide rollup carries.
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

// ClaimRollup atomically stamps riptide_emitted_at and returns the snapshot.
// nil when already emitted (a redelivered pr:merged must not produce a second
// event) or when the PR has no run (nothing to forward). The claim happens
// BEFORE the POST, so a failed emission is never retried.
func (s *Store) ClaimRollup(ctx context.Context, k PRKey, final RollupFinal) (*RollupSnapshot, error) {
	var snap RollupSnapshot
	err := s.pool.QueryRow(ctx, `
		WITH claimed AS (
			UPDATE pull_requests p SET
				riptide_emitted_at = now(),
				final_merge_commit = COALESCE($4, p.final_merge_commit),
				final_lines_added = COALESCE($5, p.final_lines_added),
				final_lines_removed = COALESCE($6, p.final_lines_removed),
				final_files_changed = COALESCE($7, p.final_files_changed),
				updated_at = now()
			WHERE p.project_key = $1 AND p.repo_slug = $2 AND p.pr_id = $3
			  AND p.riptide_emitted_at IS NULL
			  AND EXISTS (SELECT 1 FROM review_runs r WHERE r.pull_request_id = p.id)
			RETURNING p.id, p.final_source_commit, p.final_merge_commit, p.final_lines_added, p.final_lines_removed, p.final_files_changed
		)
		SELECT COUNT(r.id), COALESCE(SUM(r.prompt_tokens), 0), COALESCE(SUM(r.completion_tokens), 0),
			COALESCE(SUM(r.elapsed_ms), 0), COALESCE(SUM(r.findings_posted), 0), SUM(r.cost_nano_usd),
			ARRAY(SELECT DISTINCT m FROM unnest(array_agg(r.model_label)) AS m ORDER BY m), MIN(r.created_at),
			c.final_source_commit, c.final_merge_commit, c.final_lines_added, c.final_lines_removed, c.final_files_changed
		FROM claimed c JOIN review_runs r ON r.pull_request_id = c.id
		GROUP BY c.id, c.final_source_commit, c.final_merge_commit, c.final_lines_added, c.final_lines_removed, c.final_files_changed`,
		k.Project, k.Repo, k.PRID, final.MergeCommit, final.LinesAdded, final.LinesRemoved, final.FilesChanged,
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
