package store

import (
	"context"
	"encoding/json"
	"errors"
	"time"

	"github.com/jackc/pgx/v5"
)

// RunDetail is everything the run page shows about one attempt.
//
// Everything, comment text and dispute included (decided): the page is
// reached only through its PR's link token, handed out in the PR's summary
// comment, so its audience is the PR's.
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

// PRPage is everything the PR page shows: the PR, every attempt on it and
// every finding posted on it. Comment calls and the cost total are read
// apart (PRCommentCalls, PRCost), as the cap reads them.
type PRPage struct {
	ID         int64
	Key        PRKey
	TeamSlug   string
	Title      string
	MergedAt   *time.Time
	DeclinedAt *time.Time
	// Runs are the PR's attempts, oldest first, numbered from 1 over the
	// whole history. A history over prRunsLimit keeps the newest and says
	// so in More: the numbers stay those of the whole history.
	Runs     []PRRun
	More     bool
	Findings []RunFinding
}

// PRRun is one attempt as the PR page lists it.
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
	// Detail is the run_details record, nil when none decodes: the page
	// reads its stages, files and PR scope.
	Detail *Pipeline
}

// prRunsLimit bounds the PR page's history.
const prRunsLimit = 300

// prRow is the PR row behind a page token.
type prRow struct {
	ID        int64
	Key       PRKey
	TeamSlug  string
	Title     string
	CreatedAt time.Time
	Merged    *time.Time
	Declined  *time.Time
}

func (s *Store) prByToken(ctx context.Context, token string) (p prRow, ok bool, err error) {
	err = s.pool.QueryRow(ctx, `
		SELECT id, project_key, repo_slug, pr_id, team_slug, COALESCE(title, ''), created_at, merged_at, declined_at
		  FROM pull_requests WHERE link_token = $1`, token).Scan(
		&p.ID, &p.Key.Project, &p.Key.Repo, &p.Key.PRID, &p.TeamSlug, &p.Title, &p.CreatedAt, &p.Merged, &p.Declined)
	if errors.Is(err, pgx.ErrNoRows) {
		return prRow{}, false, nil
	}
	return p, err == nil, err
}

// prAttemptsWhere is the PR's history: its team's attempts during the PR
// row's life, as prCostSQL and PRCommentCalls bound it. A repo that changed
// hands keeps its old rows under the old team.
const prAttemptsWhere = `a.team_slug = $1 AND a.project_key = $2 AND a.repo_slug = $3 AND a.pr_id = $4
	AND a.created_at >= $5`

// PRPage reads the PR behind a link token for its page. ok is false when no
// PR has that token.
func (s *Store) PRPage(ctx context.Context, token string) (pg PRPage, ok bool, err error) {
	p, ok, err := s.prByToken(ctx, token)
	if !ok || err != nil {
		return PRPage{}, false, err
	}
	pg = PRPage{ID: p.ID, Key: p.Key, TeamSlug: p.TeamSlug, Title: p.Title, MergedAt: p.Merged, DeclinedAt: p.Declined}

	rows, err := s.pool.Query(ctx, `
		SELECT n, total, kind, outcome, reason, elapsed_ms, created_at, cost,
		       run_id, incremental, from_commit, to_commit, model_label, prompt_tokens, cached_tokens,
		       completion_tokens, run_cost, findings_posted, lines_added, lines_removed, files_changed,
		       fixed, detail
		  FROM (
		SELECT row_number() OVER (ORDER BY a.created_at, a.id) AS n, count(*) OVER () AS total,
		       a.kind, a.outcome, COALESCE(a.reason, '') AS reason, a.elapsed_ms, a.created_at,
		       COALESCE(r.cost_nano_usd, a.cost_nano_usd) AS cost,
		       r.id AS run_id, r.incremental, r.from_commit, r.to_commit, r.model_label, r.prompt_tokens,
		       r.cached_tokens, r.completion_tokens, r.cost_nano_usd AS run_cost, r.findings_posted,
		       r.lines_added, r.lines_removed, r.files_changed,
		       (SELECT count(*) FROM findings f WHERE f.fixed_run_id = r.id) AS fixed,
		       d.detail
		  FROM review_attempts a
		  LEFT JOIN review_runs r ON r.id = a.review_run_id
		  LEFT JOIN run_details d ON d.attempt_id = a.id
		 WHERE `+prAttemptsWhere+`) h
		 WHERE n > total - $6
		 ORDER BY n`,
		p.TeamSlug, p.Key.Project, p.Key.Repo, p.Key.PRID, p.CreatedAt, prRunsLimit)
	if err != nil {
		return PRPage{}, false, err
	}
	defer rows.Close()
	for rows.Next() {
		var r PRRun
		var total int
		var runID *int64
		var inc *bool
		var from, to, model *string
		var prompt, cached, completion, runCost *int64
		var posted, added, removed, files *int
		var raw []byte
		if err := rows.Scan(&r.N, &total, &r.Kind, &r.Outcome, &r.Reason, &r.ElapsedMS, &r.CreatedAt, &r.CostNanoUSD,
			&runID, &inc, &from, &to, &model, &prompt, &cached, &completion, &runCost, &posted,
			&added, &removed, &files, &r.Fixed, &raw); err != nil {
			return PRPage{}, false, err
		}
		pg.More = total > prRunsLimit
		if runID != nil {
			r.Run = &RunFacts{ID: *runID, Incremental: *inc, FromCommit: deref(from), ToCommit: deref(to),
				ModelLabel: deref(model), PromptTokens: *prompt, CachedTokens: *cached, CompletionTokens: *completion,
				CostNanoUSD: runCost, FindingsPosted: *posted, LinesAdded: *added, LinesRemoved: *removed,
				FilesChanged: *files}
		}
		// An undecodable record leaves this row's bar and files out, as the
		// run page leaves its tabs empty.
		r.Detail, _ = decodeDetail(raw)
		pg.Runs = append(pg.Runs, r)
	}
	if err := rows.Err(); err != nil {
		return PRPage{}, false, err
	}
	if pg.Findings, err = s.runFindings(ctx, `f.pull_request_id = $1`, p.ID); err != nil {
		return PRPage{}, false, err
	}
	return pg, true, nil
}

// PRRun reads attempt n of the PR behind a link token for the run page. ok
// is false when no PR has that token or it has no attempt n.
func (s *Store) PRRun(ctx context.Context, token string, n int) (d RunDetail, ok bool, err error) {
	if n < 1 {
		return RunDetail{}, false, nil
	}
	p, ok, err := s.prByToken(ctx, token)
	if !ok || err != nil {
		return RunDetail{}, false, err
	}
	var runID *int64
	err = s.pool.QueryRow(ctx, `
		SELECT a.id, a.kind, a.outcome, COALESCE(a.reason, ''), a.elapsed_ms, a.created_at,
		       a.cost_nano_usd, a.review_run_id
		  FROM review_attempts a
		 WHERE `+prAttemptsWhere+`
		 ORDER BY a.created_at, a.id
		 OFFSET $6 LIMIT 1`,
		p.TeamSlug, p.Key.Project, p.Key.Repo, p.Key.PRID, p.CreatedAt, n-1).Scan(
		&d.ID, &d.Kind, &d.Outcome, &d.Reason, &d.ElapsedMS, &d.CreatedAt, &d.AttemptCostNanoUSD, &runID)
	if errors.Is(err, pgx.ErrNoRows) {
		return RunDetail{}, false, nil
	}
	if err != nil {
		return RunDetail{}, false, err
	}
	d.N, d.Key, d.TeamSlug, d.PRTitle = n, p.Key, p.TeamSlug, p.Title

	if runID != nil {
		var r RunFacts
		var from, to *string
		err = s.pool.QueryRow(ctx, `
			SELECT id, incremental, from_commit, to_commit, model_label, prompt_tokens, cached_tokens,
			       completion_tokens, cost_nano_usd, findings_posted, lines_added, lines_removed, files_changed
			  FROM review_runs WHERE id = $1`, *runID).Scan(
			&r.ID, &r.Incremental, &from, &to, &r.ModelLabel, &r.PromptTokens, &r.CachedTokens,
			&r.CompletionTokens, &r.CostNanoUSD, &r.FindingsPosted, &r.LinesAdded, &r.LinesRemoved, &r.FilesChanged)
		if err != nil && !errors.Is(err, pgx.ErrNoRows) {
			return RunDetail{}, false, err
		}
		if err == nil {
			r.FromCommit, r.ToCommit = deref(from), deref(to)
			d.Run = &r
			if d.Findings, err = s.runFindings(ctx, `f.review_run_id = $1`, r.ID); err != nil {
				return RunDetail{}, false, err
			}
			if d.Fixed, err = s.runFindings(ctx, `f.fixed_run_id = $1`, r.ID); err != nil {
				return RunDetail{}, false, err
			}
		}
	}

	var raw []byte
	if err := s.pool.QueryRow(ctx, `SELECT detail FROM run_details WHERE attempt_id = $1`, d.ID).
		Scan(&raw); err != nil && !errors.Is(err, pgx.ErrNoRows) {
		return RunDetail{}, false, err
	}
	d.Detail, d.DetailErr = decodeDetail(raw)

	call, has, err := s.AttemptCall(ctx, d.ID)
	if err != nil {
		return RunDetail{}, false, err
	}
	if has {
		d.Call = &call
	}
	return d, true, nil
}

// decodeDetail reads a run_details record; none is nil. Display-only: a
// record an older binary wrote in another shape is nil with its error, it
// does not fail the page.
func decodeDetail(raw []byte) (*Pipeline, error) {
	if len(raw) == 0 {
		return nil, nil
	}
	var p Pipeline
	if err := json.Unmarshal(raw, &p); err != nil {
		return nil, err
	}
	return &p, nil
}

func deref(s *string) string {
	if s == nil {
		return ""
	}
	return *s
}

// runFindings reads the page's columns of the findings matching where (one
// placeholder, $1).
func (s *Store) runFindings(ctx context.Context, where string, arg int64) ([]RunFinding, error) {
	rows, err := s.pool.Query(ctx, `
		SELECT f.severity, f.confidence, f.file_path, f.line_number, COALESCE(f.headline, ''),
		       f.comment_text, f.suggestion, f.dispute_text,
		       f.bitbucket_comment_id, f.replied_at IS NOT NULL, f.fixed_at IS NOT NULL,
		       f.review_run_id, f.fixed_run_id
		  FROM findings f
		 WHERE `+where+`
		 ORDER BY f.id`, arg)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	out := []RunFinding{}
	for rows.Next() {
		var f RunFinding
		var comment *int32
		if err := rows.Scan(&f.Severity, &f.Confidence, &f.FilePath, &f.Line, &f.Headline,
			&f.CommentText, &f.Suggestion, &f.DisputeText,
			&comment, &f.Replied, &f.Fixed, &f.RunID, &f.FixedRunID); err != nil {
			return nil, err
		}
		if comment != nil {
			c := int64(*comment)
			f.CommentID = &c
		}
		out = append(out, f)
	}
	return out, rows.Err()
}

// InsertRunDetail writes one attempt's detail record. The caller swallows
// the error: the record is display-only.
func (s *Store) InsertRunDetail(ctx context.Context, attemptID int64, p Pipeline) error {
	detail, err := json.Marshal(p)
	if err != nil {
		return err
	}
	_, err = s.pool.Exec(ctx, `
		INSERT INTO run_details (attempt_id, detail) VALUES ($1, $2)
		ON CONFLICT (attempt_id) DO UPDATE SET detail = EXCLUDED.detail`, attemptID, detail)
	return err
}
