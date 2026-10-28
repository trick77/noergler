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

// What UpsertAlmanachNote did with a round's note, each an
// almanach_calls.note_status value: written; superseded (a later round
// already landed); already_emitted (the merge sent the note while this round
// was in flight); no_row (the PR was declined or deleted meanwhile and its
// row is gone).
const (
	NoteWritten        = "written"
	NoteSuperseded     = "superseded"
	NoteAlreadyEmitted = "already_emitted"
	NoteNoRow          = "no_row"
	// NoteStoreFailed is the caller's word for an upsert the database
	// refused; the store never returns it.
	NoteStoreFailed = "store_failed"
)

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

// AlmanachReviewTotals is the PR's review aggregate: the runs, their
// tokens, and the PR's cost as prCostSQL counts it (the one figure the cap,
// the frozen cost and the rollup use). Zero and nil when the PR row is gone.
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

// UpsertAlmanachNote writes a round's note and says what happened:
// NoteWritten, NoteSuperseded (a later round already landed; the call ran
// unheld), NoteAlreadyEmitted (the merge sent the note while this round was
// in flight) or NoteNoRow (the PR has no open row: declined, deleted or
// purged while the round ran, so nothing is written back). The merge stamps
// are never touched; the team slug follows the round, as the PR row follows
// a repo that changed hands.
func (s *Store) UpsertAlmanachNote(ctx context.Context, n AlmanachNote) (string, error) {
	tag, err := s.pool.Exec(ctx, `
		INSERT INTO almanach_notes
		    (project_key, repo_slug, pr_id, team_slug, run_id, reviewed_commit, pr_title,
		     ticket_key, jira_issue_type, change_kind, note, chars, model_label)
		SELECT $1, $2, $3, $4, $5, NULLIF($6, ''), $7, NULLIF($8, ''), NULLIF($9, ''), NULLIF($10, ''), $11, $12, NULLIF($13, '')
		 WHERE EXISTS (SELECT 1 FROM pull_requests p
		                WHERE p.project_key = $1 AND p.repo_slug = $2 AND p.pr_id = $3
		                  AND p.declined_at IS NULL AND p.deleted_at IS NULL)
		ON CONFLICT (project_key, repo_slug, pr_id) DO UPDATE SET
		    team_slug = EXCLUDED.team_slug, run_id = EXCLUDED.run_id, reviewed_commit = EXCLUDED.reviewed_commit,
		    pr_title = EXCLUDED.pr_title, ticket_key = EXCLUDED.ticket_key, jira_issue_type = EXCLUDED.jira_issue_type,
		    change_kind = EXCLUDED.change_kind, note = EXCLUDED.note, chars = EXCLUDED.chars,
		    model_label = EXCLUDED.model_label, updated_at = now()
		WHERE (almanach_notes.run_id IS NULL OR almanach_notes.run_id < EXCLUDED.run_id)
		  AND almanach_notes.almanach_emitted_at IS NULL`,
		n.Key.Project, n.Key.Repo, n.Key.PRID, n.TeamSlug, n.RunID, storable(n.ReviewedCommit), storable(n.PRTitle),
		storable(n.TicketKey), storable(n.JiraIssueType), storable(n.ChangeKind), []byte(n.Note), n.Chars, storable(n.ModelLabel))
	if err != nil {
		return "", err
	}
	if tag.RowsAffected() == 1 {
		return NoteWritten, nil
	}
	st, ok, err := s.AlmanachNoteStatus(ctx, n.Key)
	switch {
	case err != nil:
		return "", err
	case !ok || !st.PROpen:
		// A decline, a delete or a purge landed while the round ran
		// unheld: nothing for the note to land on.
		return NoteNoRow, nil
	case st.Emitted:
		return NoteAlreadyEmitted, nil
	}
	return NoteSuperseded, nil
}

// FinalizeAlmanachNote stamps the PR's note with the merge facts and the
// posted-findings counts, once: a redelivered pr:merged changes nothing.
// With no note row yet it writes a stub for the late round to fill, but
// only for a PR noergler has a row for: a merge it never reviewed leaves
// nothing behind. stamped is false when the row was already final or
// nothing was written.
func (s *Store) FinalizeAlmanachNote(ctx context.Context, k PRKey, team string, f AlmanachFinal) (bool, error) {
	tag, err := s.pool.Exec(ctx, `
		INSERT INTO almanach_notes
		    (project_key, repo_slug, pr_id, team_slug, pr_title, finalized_at, merged_at, merge_commit, source_commit,
		     issues_posted, suggestions_posted)
		SELECT $1, $2, $3, $4, $5, now(), $6, NULLIF($7, ''), NULLIF($8, ''),
		       COUNT(f.id) FILTER (WHERE f.severity = 'issue'), COUNT(f.id) FILTER (WHERE f.severity = 'suggestion')
		  FROM pull_requests p LEFT JOIN findings f ON f.pull_request_id = p.id
		 WHERE p.project_key = $1 AND p.repo_slug = $2 AND p.pr_id = $3
		 GROUP BY p.id
		ON CONFLICT (project_key, repo_slug, pr_id) DO UPDATE SET
		    team_slug = EXCLUDED.team_slug, finalized_at = now(), merged_at = EXCLUDED.merged_at,
		    merge_commit = EXCLUDED.merge_commit, source_commit = EXCLUDED.source_commit,
		    issues_posted = EXCLUDED.issues_posted, suggestions_posted = EXCLUDED.suggestions_posted, updated_at = now()
		WHERE almanach_notes.finalized_at IS NULL`,
		k.Project, k.Repo, k.PRID, team, storable(f.PRTitle), f.MergedAt.UTC(), storable(f.MergeCommit), storable(f.SourceCommit))
	if err != nil {
		return false, err
	}
	return tag.RowsAffected() == 1, nil
}

// DeleteAlmanachNote drops the PR's note: a declined or deleted PR teaches
// Almanach nothing. A final note is kept whatever the caller says: a merged
// PR deleted afterwards still happened, and its note may be unsent.
func (s *Store) DeleteAlmanachNote(ctx context.Context, k PRKey) error {
	_, err := s.pool.Exec(ctx, `DELETE FROM almanach_notes
		WHERE project_key = $1 AND repo_slug = $2 AND pr_id = $3 AND finalized_at IS NULL`,
		k.Project, k.Repo, k.PRID)
	return err
}

// MarkAlmanachEmitted records that Almanach took the note of round runID.
// Called after the POST, never before, and only for the round that was
// sent: a later round that landed in between stays pending.
func (s *Store) MarkAlmanachEmitted(ctx context.Context, k PRKey, runID int64) error {
	_, err := s.pool.Exec(ctx, `UPDATE almanach_notes SET almanach_emitted_at = now(), updated_at = now()
		WHERE project_key = $1 AND repo_slug = $2 AND pr_id = $3 AND run_id = $4 AND almanach_emitted_at IS NULL`,
		k.Project, k.Repo, k.PRID, runID)
	return err
}

// PendingAlmanachNotes lists the team's final notes Almanach has not taken
// yet, oldest first, with the round's call, the PR incarnation's note-call
// totals and its review totals. A stub (no note yet) is not pending. A
// note whose PR row is gone (purged after the merge) still lists, with no
// review totals: the note is the payload.
func (s *Store) PendingAlmanachNotes(ctx context.Context, team string, limit int) ([]AlmanachPending, error) {
	rows, err := s.pool.Query(ctx, `
		SELECT n.project_key, n.repo_slug, n.pr_id, n.team_slug, n.run_id, COALESCE(n.reviewed_commit, ''),
		       n.pr_title, COALESCE(n.ticket_key, ''), COALESCE(n.jira_issue_type, ''), COALESCE(n.change_kind, ''),
		       n.note, n.chars, COALESCE(n.model_label, ''),
		       n.merged_at, COALESCE(n.merge_commit, ''), COALESCE(n.source_commit, ''),
		       n.issues_posted, n.suggestions_posted,
		       c.id IS NOT NULL, COALESCE(c.outcome, ''), COALESCE(c.note_status, ''), COALESCE(c.profile, ''), `+callSelectTail("c")+`,
		       rd.rounds, rd.input_tokens, rd.output_tokens, rd.cost,
		       rv.runs, rv.prompt_tokens, rv.cached_tokens, rv.completion_tokens, rv.elapsed_ms, rv.cost, rv.models
		  FROM almanach_notes n
		  LEFT JOIN pull_requests pp
		    ON pp.project_key = n.project_key AND pp.repo_slug = n.repo_slug AND pp.pr_id = n.pr_id
		  LEFT JOIN LATERAL (
		       SELECT * FROM almanach_calls c
		        WHERE c.project_key = n.project_key AND c.repo_slug = n.repo_slug AND c.pr_id = n.pr_id
		          AND c.run_id = n.run_id
		        ORDER BY c.id DESC LIMIT 1) c ON true
		  -- This incarnation's calls only: a purged repo claimed again starts
		  -- from zero, as prCostSQL counts. No PR row left: every call.
		  LEFT JOIN LATERAL (
		       SELECT COUNT(*)::int AS rounds, COALESCE(SUM(a.input_tokens), 0)::bigint AS input_tokens,
		              COALESCE(SUM(a.output_tokens), 0)::bigint AS output_tokens, SUM(a.cost_nano_usd) AS cost
		         FROM almanach_calls a
		        WHERE a.project_key = n.project_key AND a.repo_slug = n.repo_slug AND a.pr_id = n.pr_id
		          AND a.created_at >= COALESCE(pp.created_at, a.created_at)) rd ON true
		  LEFT JOIN LATERAL (
		       SELECT COUNT(r.id)::int AS runs, COALESCE(SUM(r.prompt_tokens), 0)::bigint AS prompt_tokens,
		              COALESCE(SUM(r.cached_tokens), 0)::bigint AS cached_tokens,
		              COALESCE(SUM(r.completion_tokens), 0)::bigint AS completion_tokens,
		              COALESCE(SUM(r.elapsed_ms), 0)::bigint AS elapsed_ms,
		              (SELECT `+prCostSQL+` FROM pull_requests p WHERE p.id = pp.id) AS cost,
		              ARRAY(SELECT DISTINCT m FROM unnest(array_agg(r.model_label)) AS m WHERE m IS NOT NULL ORDER BY m) AS models
		         FROM review_runs r WHERE r.pull_request_id = pp.id) rv ON true
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
		targets := []any{&p.Key.Project, &p.Key.Repo, &p.Key.PRID, &p.TeamSlug, &p.RunID, &p.ReviewedCommit,
			&p.PRTitle, &p.TicketKey, &p.JiraIssueType, &p.ChangeKind, &note, &p.Chars, &p.ModelLabel,
			&p.MergedAt, &p.MergeCommit, &p.SourceCommit, &p.IssuesPosted, &p.SuggestionsPosted,
			&p.NoteCallFound, &p.NoteCallOutcome, &p.NoteStatus}
		targets = append(targets, callScan(&p.NoteCall)...)
		targets = append(targets, &p.Rounds, &p.RoundsInputTokens, &p.RoundsOutputTokens, &p.RoundsCostNanoUSD,
			&p.Review.Runs, &p.Review.PromptTokens, &p.Review.CachedTokens, &p.Review.CompletionTokens,
			&p.Review.ElapsedMS, &p.Review.CostNanoUSD, &p.Review.Models)
		if err := rows.Scan(targets...); err != nil {
			return nil, err
		}
		p.Note = json.RawMessage(note)
		if p.Review.Models == nil {
			p.Review.Models = []string{}
		}
		out = append(out, p)
	}
	return out, rows.Err()
}

// InsertAlmanachCall records one note call, the only ledger of it, with
// the same column handling as llm_calls (callArgs).
func (s *Store) InsertAlmanachCall(ctx context.Context, a AlmanachCall) error {
	c := a.Call
	if c == nil {
		c = &LLMCall{}
	}
	args := append([]any{a.TeamSlug, a.Key.Project, a.Key.Repo, a.Key.PRID, a.RunID, storable(a.Outcome), a.NoteStatus}, callArgs(*c)...)
	_, err := s.pool.Exec(ctx, `
		INSERT INTO almanach_calls (team_slug, project_key, repo_slug, pr_id, run_id, outcome, note_status, `+callColumns+`)
		VALUES (`+placeholders(len(args))+`)`, args...)
	return err
}

// AlmanachNoteState is what a caller reads back before deciding to emit.
type AlmanachNoteState struct {
	Finalized bool
	HasNote   bool
	Emitted   bool
	// PROpen is whether the PR row a round could write to is still there:
	// not declined, not deleted, not purged.
	PROpen bool
}

// AlmanachNoteStatus reads the PR's note row; ok false when there is none.
func (s *Store) AlmanachNoteStatus(ctx context.Context, k PRKey) (st AlmanachNoteState, ok bool, err error) {
	err = s.pool.QueryRow(ctx, `SELECT finalized_at IS NOT NULL, note IS NOT NULL, almanach_emitted_at IS NOT NULL,
		       EXISTS (SELECT 1 FROM pull_requests p
		                WHERE p.project_key = $1 AND p.repo_slug = $2 AND p.pr_id = $3
		                  AND p.declined_at IS NULL AND p.deleted_at IS NULL)
		FROM almanach_notes WHERE project_key = $1 AND repo_slug = $2 AND pr_id = $3`,
		k.Project, k.Repo, k.PRID).Scan(&st.Finalized, &st.HasNote, &st.Emitted, &st.PROpen)
	if errors.Is(err, pgx.ErrNoRows) {
		return AlmanachNoteState{}, false, nil
	}
	if err != nil {
		return AlmanachNoteState{}, false, err
	}
	return st, true, nil
}
