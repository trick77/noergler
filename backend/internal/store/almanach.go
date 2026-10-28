package store

import (
	"context"
	"encoding/json"
	"errors"
	"time"

	"github.com/jackc/pgx/v5"
)

// AlmanachNote is one PR's scratch note as a review round leaves it. See
// migrations/0018_almanach.sql: every round replaces the row, ordered by
// RunID; the merge stamps it final; the forwarder marks it emitted.
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

// AlmanachFinal is what pr:merged stamps onto the note.
type AlmanachFinal struct {
	MergedAt time.Time
	// MergeCommit may be empty (Bitbucket does not always send it);
	// SourceCommit is the PR's head at merge.
	MergeCommit  string
	SourceCommit string
	// PRTitle fills the stub when no round has written the row yet.
	PRTitle string
}

// AlmanachReviewTotals is the PR's review aggregate, what riptide's rollup
// carries too. Zero when the PR row is gone.
type AlmanachReviewTotals struct {
	Runs             int
	PromptTokens     int64
	CachedTokens     int64
	CompletionTokens int64
	ElapsedMS        int64
	// CostNanoUSD is nil when no run was priced.
	CostNanoUSD *int64
	Models      []string
}

// AlmanachPending is one final note the forwarder sends, with everything
// Almanach gets beside it.
type AlmanachPending struct {
	AlmanachNote
	MergedAt     time.Time
	MergeCommit  string
	SourceCommit string
	// IssuesPosted and SuggestionsPosted count the findings noergler posted
	// on the PR, by severity, as of finalize.
	IssuesPosted      int
	SuggestionsPosted int
	// NoteCall is the round's own gateway call as recorded; Answered false
	// when none was recorded. NoteCallOutcome and NoteStatus are that row's.
	NoteCall        LLMCall
	NoteCallFound   bool
	NoteCallOutcome string
	NoteStatus      string
	// Rounds counts every note call the PR made, with their summed tokens
	// and cost (nil when none was priced).
	Rounds             int
	RoundsInputTokens  int64
	RoundsOutputTokens int64
	RoundsCostNanoUSD  *int64
	Review             AlmanachReviewTotals
}

// AlmanachCall is one note call's row: the gateway's figures plus what
// became of the note. Call is nil when the gateway never answered.
type AlmanachCall struct {
	Key        PRKey
	TeamSlug   string
	RunID      *int64
	Outcome    string
	NoteStatus string
	Call       *LLMCall
}

// UpsertAlmanachNote writes a round's note. written is false when a later
// round already landed (run id order; the call ran unheld) or the note was
// already emitted. The merge stamps are never touched.
func (s *Store) UpsertAlmanachNote(ctx context.Context, n AlmanachNote) (bool, error) {
	tag, err := s.pool.Exec(ctx, `
		INSERT INTO almanach_notes
		    (project_key, repo_slug, pr_id, team_slug, run_id, reviewed_commit, pr_title,
		     ticket_key, jira_issue_type, change_kind, note, chars, model_label)
		VALUES ($1, $2, $3, $4, $5, NULLIF($6, ''), $7, NULLIF($8, ''), NULLIF($9, ''), NULLIF($10, ''), $11, $12, NULLIF($13, ''))
		ON CONFLICT (project_key, repo_slug, pr_id) DO UPDATE SET
		    run_id = EXCLUDED.run_id, reviewed_commit = EXCLUDED.reviewed_commit, pr_title = EXCLUDED.pr_title,
		    ticket_key = EXCLUDED.ticket_key, jira_issue_type = EXCLUDED.jira_issue_type,
		    change_kind = EXCLUDED.change_kind, note = EXCLUDED.note, chars = EXCLUDED.chars,
		    model_label = EXCLUDED.model_label, updated_at = now()
		WHERE (almanach_notes.run_id IS NULL OR almanach_notes.run_id < EXCLUDED.run_id)
		  AND almanach_notes.almanach_emitted_at IS NULL`,
		n.Key.Project, n.Key.Repo, n.Key.PRID, n.TeamSlug, n.RunID, storable(n.ReviewedCommit), storable(n.PRTitle),
		storable(n.TicketKey), storable(n.JiraIssueType), storable(n.ChangeKind), []byte(n.Note), n.Chars, storable(n.ModelLabel))
	if err != nil {
		return false, err
	}
	return tag.RowsAffected() == 1, nil
}

// FinalizeAlmanachNote stamps the PR's note with the merge facts and the
// posted-findings counts, once: a redelivered pr:merged changes nothing.
// With no note row yet it writes a stub for the late round to fill.
// stamped is false when the row was already final.
func (s *Store) FinalizeAlmanachNote(ctx context.Context, k PRKey, team string, f AlmanachFinal) (bool, error) {
	tag, err := s.pool.Exec(ctx, `
		INSERT INTO almanach_notes
		    (project_key, repo_slug, pr_id, team_slug, pr_title, finalized_at, merged_at, merge_commit, source_commit,
		     issues_posted, suggestions_posted)
		VALUES ($1, $2, $3, $4, $5, now(), $6, NULLIF($7, ''), NULLIF($8, ''),
		    (SELECT COUNT(*) FROM findings f JOIN pull_requests p ON p.id = f.pull_request_id
		      WHERE p.project_key = $1 AND p.repo_slug = $2 AND p.pr_id = $3 AND f.severity = 'issue'),
		    (SELECT COUNT(*) FROM findings f JOIN pull_requests p ON p.id = f.pull_request_id
		      WHERE p.project_key = $1 AND p.repo_slug = $2 AND p.pr_id = $3 AND f.severity = 'suggestion'))
		ON CONFLICT (project_key, repo_slug, pr_id) DO UPDATE SET
		    finalized_at = now(), merged_at = EXCLUDED.merged_at, merge_commit = EXCLUDED.merge_commit,
		    source_commit = EXCLUDED.source_commit, issues_posted = EXCLUDED.issues_posted,
		    suggestions_posted = EXCLUDED.suggestions_posted, updated_at = now()
		WHERE almanach_notes.finalized_at IS NULL`,
		k.Project, k.Repo, k.PRID, team, storable(f.PRTitle), f.MergedAt.UTC(), storable(f.MergeCommit), storable(f.SourceCommit))
	if err != nil {
		return false, err
	}
	return tag.RowsAffected() == 1, nil
}

// DeleteAlmanachNote drops the PR's note: a declined or deleted PR teaches
// Almanach nothing.
func (s *Store) DeleteAlmanachNote(ctx context.Context, k PRKey) error {
	_, err := s.pool.Exec(ctx, `DELETE FROM almanach_notes WHERE project_key = $1 AND repo_slug = $2 AND pr_id = $3`,
		k.Project, k.Repo, k.PRID)
	return err
}

// MarkAlmanachEmitted records that Almanach took the note (2xx). Called
// after the POST, never before.
func (s *Store) MarkAlmanachEmitted(ctx context.Context, k PRKey) error {
	_, err := s.pool.Exec(ctx, `UPDATE almanach_notes SET almanach_emitted_at = now(), updated_at = now()
		WHERE project_key = $1 AND repo_slug = $2 AND pr_id = $3 AND almanach_emitted_at IS NULL`,
		k.Project, k.Repo, k.PRID)
	return err
}

// PendingAlmanachNotes lists the team's final notes Almanach has not taken
// yet, oldest first, with the round's call, the PR's note-call totals and
// its review totals. A stub (no note yet) is not pending.
func (s *Store) PendingAlmanachNotes(ctx context.Context, team string, limit int) ([]AlmanachPending, error) {
	rows, err := s.pool.Query(ctx, `
		SELECT n.project_key, n.repo_slug, n.pr_id, n.team_slug, COALESCE(n.run_id, 0), COALESCE(n.reviewed_commit, ''),
		       n.pr_title, COALESCE(n.ticket_key, ''), COALESCE(n.jira_issue_type, ''), COALESCE(n.change_kind, ''),
		       n.note, COALESCE(n.chars, 0), COALESCE(n.model_label, ''),
		       n.merged_at, COALESCE(n.merge_commit, ''), COALESCE(n.source_commit, ''),
		       COALESCE(n.issues_posted, 0), COALESCE(n.suggestions_posted, 0),
		       c.id IS NOT NULL, COALESCE(c.outcome, ''), COALESCE(c.note_status, ''),
		       COALESCE(c.profile, ''), COALESCE(c.model, ''), COALESCE(c.deployment, ''), COALESCE(c.reasoning_sent, ''),
		       COALESCE(c.finish_reason, ''), COALESCE(c.call_id, ''), COALESCE(c.litellm_version, ''),
		       c.input_tokens, c.input_uncached_tokens, c.cache_read_tokens, c.cache_write_tokens,
		       c.output_tokens, c.reasoning_tokens,
		       c.cost_nano_usd, c.cost_input_nano_usd, c.cost_cache_read_nano_usd, c.cost_cache_write_nano_usd,
		       c.cost_output_nano_usd, c.cost_reasoning_nano_usd, c.cost_tool_usage_nano_usd,
		       c.key_spend_nano_usd, c.headers_ms, c.total_ms, COALESCE(c.warnings, '{}'),
		       rd.rounds, rd.input_tokens, rd.output_tokens, rd.cost,
		       rv.runs, rv.prompt_tokens, rv.cached_tokens, rv.completion_tokens, rv.elapsed_ms, rv.cost, rv.models
		  FROM almanach_notes n
		  LEFT JOIN LATERAL (
		       SELECT * FROM almanach_calls c
		        WHERE c.project_key = n.project_key AND c.repo_slug = n.repo_slug AND c.pr_id = n.pr_id
		          AND c.run_id = n.run_id
		        ORDER BY c.id DESC LIMIT 1) c ON true
		  LEFT JOIN LATERAL (
		       SELECT COUNT(*)::int AS rounds, COALESCE(SUM(input_tokens), 0)::bigint AS input_tokens,
		              COALESCE(SUM(output_tokens), 0)::bigint AS output_tokens, SUM(cost_nano_usd) AS cost
		         FROM almanach_calls a
		        WHERE a.project_key = n.project_key AND a.repo_slug = n.repo_slug AND a.pr_id = n.pr_id) rd ON true
		  LEFT JOIN LATERAL (
		       SELECT COUNT(r.id)::int AS runs, COALESCE(SUM(r.prompt_tokens), 0)::bigint AS prompt_tokens,
		              COALESCE(SUM(r.cached_tokens), 0)::bigint AS cached_tokens,
		              COALESCE(SUM(r.completion_tokens), 0)::bigint AS completion_tokens,
		              COALESCE(SUM(r.elapsed_ms), 0)::bigint AS elapsed_ms, SUM(r.cost_nano_usd) AS cost,
		              ARRAY(SELECT DISTINCT m FROM unnest(array_agg(r.model_label)) AS m WHERE m IS NOT NULL ORDER BY m) AS models
		         FROM pull_requests p LEFT JOIN review_runs r ON r.pull_request_id = p.id
		        WHERE p.project_key = n.project_key AND p.repo_slug = n.repo_slug AND p.pr_id = n.pr_id) rv ON true
		 WHERE n.team_slug = $1 AND n.finalized_at IS NOT NULL AND n.note IS NOT NULL AND n.almanach_emitted_at IS NULL
		 ORDER BY n.finalized_at, n.pr_id
		 LIMIT $2`, team, limit)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	var out []AlmanachPending
	for rows.Next() {
		var p AlmanachPending
		var note []byte
		var mergedAt *time.Time
		var runs, prompt, cached, completion, elapsed *int64
		var models []string
		var rounds, rin, rout *int64
		if err := rows.Scan(&p.Key.Project, &p.Key.Repo, &p.Key.PRID, &p.TeamSlug, &p.RunID, &p.ReviewedCommit,
			&p.PRTitle, &p.TicketKey, &p.JiraIssueType, &p.ChangeKind, &note, &p.Chars, &p.ModelLabel,
			&mergedAt, &p.MergeCommit, &p.SourceCommit, &p.IssuesPosted, &p.SuggestionsPosted,
			&p.NoteCallFound, &p.NoteCallOutcome, &p.NoteStatus,
			&p.NoteCall.Profile, &p.NoteCall.Model, &p.NoteCall.Deployment, &p.NoteCall.ReasoningSent,
			&p.NoteCall.FinishReason, &p.NoteCall.CallID, &p.NoteCall.GatewayVer,
			&p.NoteCall.InputTokens, &p.NoteCall.InputUncachedTokens, &p.NoteCall.CacheReadTokens, &p.NoteCall.CacheWriteTokens,
			&p.NoteCall.OutputTokens, &p.NoteCall.ReasoningTokens,
			&p.NoteCall.CostNanoUSD, &p.NoteCall.CostInputNanoUSD, &p.NoteCall.CostCacheReadNanoUSD, &p.NoteCall.CostCacheWriteNanoUSD,
			&p.NoteCall.CostOutputNanoUSD, &p.NoteCall.CostReasoningNanoUSD, &p.NoteCall.CostToolUsageNanoUSD,
			&p.NoteCall.KeySpendNanoUSD, &p.NoteCall.HeadersMS, &p.NoteCall.TotalMS, &p.NoteCall.Warnings,
			&rounds, &rin, &rout, &p.RoundsCostNanoUSD,
			&runs, &prompt, &cached, &completion, &elapsed, &p.Review.CostNanoUSD, &models); err != nil {
			return nil, err
		}
		p.Note = json.RawMessage(note)
		if mergedAt != nil {
			p.MergedAt = *mergedAt
		}
		p.Rounds, p.RoundsInputTokens, p.RoundsOutputTokens = int(orZero(rounds)), orZero(rin), orZero(rout)
		p.Review = AlmanachReviewTotals{Runs: int(orZero(runs)), PromptTokens: orZero(prompt), CachedTokens: orZero(cached),
			CompletionTokens: orZero(completion), ElapsedMS: orZero(elapsed), CostNanoUSD: p.Review.CostNanoUSD, Models: models}
		if p.Review.Models == nil {
			p.Review.Models = []string{}
		}
		out = append(out, p)
	}
	return out, rows.Err()
}

func orZero(p *int64) int64 {
	if p == nil {
		return 0
	}
	return *p
}

// InsertAlmanachCall records one note call, the only ledger of it. The
// gateway's strings are made storable like llm_calls' (writeCall).
func (s *Store) InsertAlmanachCall(ctx context.Context, a AlmanachCall) error {
	c := a.Call
	if c == nil {
		c = &LLMCall{}
	}
	warnings := make([]string, 0, len(c.Warnings))
	for _, w := range c.Warnings {
		warnings = append(warnings, storable(w))
	}
	_, err := s.pool.Exec(ctx, `
		INSERT INTO almanach_calls
		    (team_slug, project_key, repo_slug, pr_id, run_id, outcome, note_status,
		     profile, model, deployment, reasoning_sent, finish_reason, call_id, litellm_version,
		     input_tokens, input_uncached_tokens, cache_read_tokens, cache_write_tokens,
		     output_tokens, reasoning_tokens,
		     cost_nano_usd, cost_input_nano_usd, cost_cache_read_nano_usd, cost_cache_write_nano_usd,
		     cost_output_nano_usd, cost_reasoning_nano_usd, cost_tool_usage_nano_usd,
		     key_spend_nano_usd, headers_ms, total_ms, warnings)
		VALUES ($1, $2, $3, $4, $5, $6, $7,
		        NULLIF($8, ''), NULLIF($9, ''), NULLIF($10, ''), NULLIF($11, ''), NULLIF($12, ''), NULLIF($13, ''), NULLIF($14, ''),
		        $15, $16, $17, $18, $19, $20,
		        $21, $22, $23, $24, $25, $26, $27,
		        $28, $29, $30, $31)`,
		a.TeamSlug, a.Key.Project, a.Key.Repo, a.Key.PRID, a.RunID, storable(a.Outcome), a.NoteStatus,
		storable(c.Profile), storable(c.Model), storable(c.Deployment), storable(c.ReasoningSent),
		storable(c.FinishReason), storable(c.CallID), storable(c.GatewayVer),
		c.InputTokens, c.InputUncachedTokens, c.CacheReadTokens, c.CacheWriteTokens,
		c.OutputTokens, c.ReasoningTokens,
		c.CostNanoUSD, c.CostInputNanoUSD, c.CostCacheReadNanoUSD, c.CostCacheWriteNanoUSD,
		c.CostOutputNanoUSD, c.CostReasoningNanoUSD, c.CostToolUsageNanoUSD,
		c.KeySpendNanoUSD, c.HeadersMS, c.TotalMS, warnings)
	return err
}

// AlmanachNoteState is what a caller reads back before deciding to emit.
type AlmanachNoteState struct {
	Finalized bool
	HasNote   bool
	Emitted   bool
}

// AlmanachNoteStatus reads the PR's note row; ok false when there is none.
func (s *Store) AlmanachNoteStatus(ctx context.Context, k PRKey) (st AlmanachNoteState, ok bool, err error) {
	err = s.pool.QueryRow(ctx, `SELECT finalized_at IS NOT NULL, note IS NOT NULL, almanach_emitted_at IS NOT NULL
		FROM almanach_notes WHERE project_key = $1 AND repo_slug = $2 AND pr_id = $3`,
		k.Project, k.Repo, k.PRID).Scan(&st.Finalized, &st.HasNote, &st.Emitted)
	if errors.Is(err, pgx.ErrNoRows) {
		return AlmanachNoteState{}, false, nil
	}
	if err != nil {
		return AlmanachNoteState{}, false, err
	}
	return st, true, nil
}
