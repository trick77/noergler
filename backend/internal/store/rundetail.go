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
	// PRAttempts is the PR's history around this attempt, oldest first and
	// this one always included: up to prAttemptsEachSide before and after
	// it. The page's previous/next and its pull request tab.
	PRAttempts []PRAttempt
	// MoreBefore and MoreAfter say the window cut older or newer attempts,
	// so the page can say rows are missing instead of looking complete.
	MoreBefore bool
	MoreAfter  bool
	// Detail is the run_details record, raw JSON; nil when none was
	// written (an attempt from before it existed, or a skip).
	Detail []byte
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
	// FromAttemptID is the attempt whose run posted it, the key of that
	// run's page. Set on Fixed only (an earlier run); nil there when that
	// attempt is gone or was recorded under another team.
	FromAttemptID *int64
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
			if d.Findings, err = s.findingHeadlines(ctx, `f.review_run_id = $1`, r.ID, d.TeamSlug, false); err != nil {
				return RunDetail{}, false, err
			}
			if d.Fixed, err = s.findingHeadlines(ctx, `f.fixed_run_id = $1`, r.ID, d.TeamSlug, true); err != nil {
				return RunDetail{}, false, err
			}
		}
	}

	if err := s.pool.QueryRow(ctx, `SELECT detail FROM run_details WHERE attempt_id = $1`, attemptID).
		Scan(&d.Detail); err != nil && !errors.Is(err, pgx.ErrNoRows) {
		return RunDetail{}, false, err
	}

	call, has, err := s.AttemptCall(ctx, attemptID)
	if err != nil {
		return RunDetail{}, false, err
	}
	if has {
		d.Call = &call
	}

	if d.PRAttempts, d.MoreBefore, d.MoreAfter, err = s.prAttempts(ctx, d); err != nil {
		return RunDetail{}, false, err
	}
	return d, true, nil
}

// findingHeadlines reads the page's columns of the findings matching where
// (one placeholder, $1). Never comment_text, suggestion or dispute_text.
//
// withFrom also looks up the attempt of the run that posted each finding,
// within team (the page's history is scoped to it), for the fixed table's
// link. The run's own findings never show it, so they skip the lookup.
func (s *Store) findingHeadlines(ctx context.Context, where string, arg int64, team string, withFrom bool) ([]FindingHeadline, error) {
	rows, err := s.pool.Query(ctx, `
		SELECT f.severity, f.confidence, f.file_path, f.line_number, COALESCE(f.headline, ''),
		       f.bitbucket_comment_id, f.replied_at IS NOT NULL, f.fixed_at IS NOT NULL,
		       CASE WHEN $3 THEN
		         (SELECT MIN(a.id) FROM review_attempts a
		           WHERE a.review_run_id = f.review_run_id AND a.team_slug = $2) END
		  FROM findings f
		 WHERE `+where+`
		 ORDER BY f.id`, arg, team, withFrom)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	out := []FindingHeadline{}
	for rows.Next() {
		var f FindingHeadline
		var comment *int32
		if err := rows.Scan(&f.Severity, &f.Confidence, &f.FilePath, &f.Line, &f.Headline,
			&comment, &f.Replied, &f.Fixed, &f.FromAttemptID); err != nil {
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

// prAttemptsEachSide bounds the PR's history on the page: that many runs
// before the one shown and that many after. A window, not the oldest N, so
// the attempt being viewed and its neighbours are always in it.
const prAttemptsEachSide = 100

// prAttempts is the PR's history around d, oldest first, and whether the
// window cut older or newer attempts. Scoped to the team as well as the PR:
// a repo that changed hands keeps its old rows under the old team. Each side
// fetches one row more than it keeps (the older side also holds d itself),
// to tell "cut" from "all there" in the same query and snapshot.
func (s *Store) prAttempts(ctx context.Context, d RunDetail) (out []PRAttempt, moreBefore, moreAfter bool, err error) {
	const cols = `a.id, a.kind, a.outcome, COALESCE(a.reason, '') AS reason, a.elapsed_ms, a.created_at,
		       r.findings_posted, COALESCE(r.cost_nano_usd, a.cost_nano_usd) AS cost,
		       (a.created_at, a.id) <= ($5, $6) AS before`
	const from = `FROM review_attempts a
		  LEFT JOIN review_runs r ON r.id = a.review_run_id
		 WHERE a.team_slug = $1 AND a.project_key = $2 AND a.repo_slug = $3 AND a.pr_id = $4`
	rows, err := s.pool.Query(ctx, `
		SELECT id, kind, outcome, reason, elapsed_ms, created_at, findings_posted, cost, before FROM (
		  (SELECT `+cols+` `+from+` AND (a.created_at, a.id) <= ($5, $6)
		    ORDER BY a.created_at DESC, a.id DESC LIMIT $7 + 2)
		  UNION ALL
		  (SELECT `+cols+` `+from+` AND (a.created_at, a.id) > ($5, $6)
		    ORDER BY a.created_at, a.id LIMIT $7 + 1)
		) h
		ORDER BY created_at, id`,
		d.TeamSlug, d.Key.Project, d.Key.Repo, d.Key.PRID, d.CreatedAt, d.ID, prAttemptsEachSide)
	if err != nil {
		return nil, false, false, err
	}
	defer rows.Close()
	var before, after []PRAttempt
	for rows.Next() {
		var a PRAttempt
		var isBefore bool
		if err := rows.Scan(&a.ID, &a.Kind, &a.Outcome, &a.Reason, &a.ElapsedMS, &a.CreatedAt,
			&a.Findings, &a.CostNanoUSD, &isBefore); err != nil {
			return nil, false, false, err
		}
		if isBefore {
			before = append(before, a)
		} else {
			after = append(after, a)
		}
	}
	if err := rows.Err(); err != nil {
		return nil, false, false, err
	}
	// The before side holds the attempt itself plus up to prAttemptsEachSide
	// older ones; one more than that means older rows exist.
	if len(before) > prAttemptsEachSide+1 {
		before, moreBefore = before[1:], true
	}
	if len(after) > prAttemptsEachSide {
		after, moreAfter = after[:prAttemptsEachSide], true
	}
	return append(before, after...), moreBefore, moreAfter, nil
}

// InsertRunDetail writes one attempt's detail record. The caller swallows
// the error: the record is display-only.
func (s *Store) InsertRunDetail(ctx context.Context, attemptID int64, detail []byte) error {
	_, err := s.pool.Exec(ctx, `
		INSERT INTO run_details (attempt_id, detail) VALUES ($1, $2)
		ON CONFLICT (attempt_id) DO UPDATE SET detail = EXCLUDED.detail`, attemptID, detail)
	return err
}
