package store

import (
	"context"
	"errors"
	"time"

	"github.com/jackc/pgx/v5"
)

// RunDetail is everything the run detail page shows about one attempt.
//
// Headlines only (decided): the dashboard is unauthenticated, so no
// comment text, suggestion, quoted code or dispute text is read here. The
// full text stays behind Bitbucket's own access control, one link away.
type RunDetail struct {
	ID        int64
	Key       PRKey
	TeamSlug  string
	Kind      RunKind
	Outcome   string
	Reason    string
	ElapsedMS *int64
	CreatedAt time.Time
	// AttemptCostNanoUSD is set only for a billed call with no run.
	AttemptCostNanoUSD *int64

	// PRTitle is empty when no PR row exists (a skip before the upsert).
	PRTitle string
	// Run is nil unless the attempt produced one.
	Run *RunFacts
	// Call is nil when no gateway call was recorded for the attempt.
	Call *LLMCall
	// Findings are the ones this run posted; Fixed the earlier ones this run
	// marked fixed.
	Findings []FindingHeadline
	Fixed    []FindingHeadline
	// PRAttempts is every attempt on the same PR, oldest first, this one
	// included: the page's previous/next and its pull request tab.
	PRAttempts []PRAttempt
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

// FindingHeadline is a posted finding as the page may show it: where, how
// bad, the one-line headline and the Bitbucket comment to open for the rest.
type FindingHeadline struct {
	Severity   string
	Confidence *int
	FilePath   string
	Line       int
	Headline   string
	CommentID  *int64
	Replied    bool
	Fixed      bool
	// FromRunID is the run that posted it; set on Fixed, where it is an
	// earlier run.
	FromRunID int64
}

// PRAttempt is one row of the PR's history.
type PRAttempt struct {
	ID        int64
	Kind      RunKind
	Outcome   string
	Reason    string
	ElapsedMS *int64
	CreatedAt time.Time
	Findings  *int
	// CostNanoUSD is the run's cost, or the attempt's for a billed call
	// with no run: the same figure prCostSQL counts, never both.
	CostNanoUSD *int64
}

// RunDetail reads one attempt for the page. ok is false when there is no
// such attempt.
func (s *Store) RunDetail(ctx context.Context, attemptID int64) (d RunDetail, ok bool, err error) {
	var runID *int64
	err = s.pool.QueryRow(ctx, `
		SELECT a.id, a.team_slug, a.project_key, a.repo_slug, a.pr_id, a.kind, a.outcome,
		       COALESCE(a.reason, ''), a.elapsed_ms, a.created_at, a.cost_nano_usd, a.review_run_id,
		       COALESCE(p.title, '')
		  FROM review_attempts a
		  LEFT JOIN pull_requests p
		    ON p.project_key = a.project_key AND p.repo_slug = a.repo_slug AND p.pr_id = a.pr_id
		 WHERE a.id = $1`, attemptID).Scan(
		&d.ID, &d.TeamSlug, &d.Key.Project, &d.Key.Repo, &d.Key.PRID, &d.Kind, &d.Outcome,
		&d.Reason, &d.ElapsedMS, &d.CreatedAt, &d.AttemptCostNanoUSD, &runID, &d.PRTitle)
	if errors.Is(err, pgx.ErrNoRows) {
		return RunDetail{}, false, nil
	}
	if err != nil {
		return RunDetail{}, false, err
	}

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
			if from != nil {
				r.FromCommit = *from
			}
			if to != nil {
				r.ToCommit = *to
			}
			d.Run = &r
			if d.Findings, err = s.findingHeadlines(ctx, `f.review_run_id = $1`, r.ID); err != nil {
				return RunDetail{}, false, err
			}
			if d.Fixed, err = s.findingHeadlines(ctx, `f.fixed_run_id = $1`, r.ID); err != nil {
				return RunDetail{}, false, err
			}
		}
	}

	call, has, err := s.AttemptCall(ctx, attemptID)
	if err != nil {
		return RunDetail{}, false, err
	}
	if has {
		d.Call = &call
	}

	if d.PRAttempts, err = s.prAttempts(ctx, d.TeamSlug, d.Key); err != nil {
		return RunDetail{}, false, err
	}
	return d, true, nil
}

// findingHeadlines reads the page's columns of the findings matching where
// (one placeholder, $1). Never comment_text, suggestion or dispute_text.
func (s *Store) findingHeadlines(ctx context.Context, where string, arg int64) ([]FindingHeadline, error) {
	rows, err := s.pool.Query(ctx, `
		SELECT f.severity, f.confidence, f.file_path, f.line_number, COALESCE(f.headline, ''),
		       f.bitbucket_comment_id, f.replied_at IS NOT NULL, f.fixed_at IS NOT NULL, f.review_run_id
		  FROM findings f
		 WHERE `+where+`
		 ORDER BY f.id`, arg)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	out := []FindingHeadline{}
	for rows.Next() {
		var f FindingHeadline
		var comment *int32
		if err := rows.Scan(&f.Severity, &f.Confidence, &f.FilePath, &f.Line, &f.Headline,
			&comment, &f.Replied, &f.Fixed, &f.FromRunID); err != nil {
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

// maxPRAttempts bounds a PR's history on the page; a PR with more has been
// pushed to far past what one page should list.
const maxPRAttempts = 200

// prAttempts is the PR's history, oldest first. Scoped to the team as well
// as the PR: a repo that changed hands keeps its old rows under the old team.
func (s *Store) prAttempts(ctx context.Context, team string, k PRKey) ([]PRAttempt, error) {
	rows, err := s.pool.Query(ctx, `
		SELECT a.id, a.kind, a.outcome, COALESCE(a.reason, ''), a.elapsed_ms, a.created_at,
		       r.findings_posted, COALESCE(r.cost_nano_usd, a.cost_nano_usd)
		  FROM review_attempts a
		  LEFT JOIN review_runs r ON r.id = a.review_run_id
		 WHERE a.team_slug = $1 AND a.project_key = $2 AND a.repo_slug = $3 AND a.pr_id = $4
		 ORDER BY a.created_at, a.id
		 LIMIT $5`, team, k.Project, k.Repo, k.PRID, maxPRAttempts)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	out := []PRAttempt{}
	for rows.Next() {
		var a PRAttempt
		if err := rows.Scan(&a.ID, &a.Kind, &a.Outcome, &a.Reason, &a.ElapsedMS, &a.CreatedAt,
			&a.Findings, &a.CostNanoUSD); err != nil {
			return nil, err
		}
		out = append(out, a)
	}
	return out, rows.Err()
}
