package store

import (
	"context"
	"encoding/json"
	"testing"
	"time"
)

func noteFor(k PRKey, runID int64, text string) AlmanachNote {
	return AlmanachNote{Key: k, TeamSlug: "platform", RunID: runID, ReviewedCommit: "c" + text, PRTitle: "t",
		TicketKey: "PLAT-1", JiraIssueType: "Bug", ChangeKind: "bugfix",
		Note: json.RawMessage(`{"summary":"` + text + `"}`), Chars: len(text), ModelLabel: "gpt-5.5-high"}
}

func pending(t *testing.T, s *Store, team string) []AlmanachPending {
	t.Helper()
	out, err := s.PendingAlmanachNotes(context.Background(), team, 10)
	if err != nil {
		t.Fatal(err)
	}
	return out
}

func upsertNote(t *testing.T, s *Store, n AlmanachNote) string {
	t.Helper()
	got, err := s.UpsertAlmanachNote(context.Background(), n)
	if err != nil {
		t.Fatal(err)
	}
	return got
}

func finalize(t *testing.T, s *Store, k PRKey, team string, f AlmanachFinal) bool {
	t.Helper()
	if f.MergedAt.IsZero() {
		f.MergedAt = time.Now()
	}
	st, err := s.FinalizeAlmanachNote(context.Background(), k, team, f)
	if err != nil {
		t.Fatal(err)
	}
	return st
}

// Rounds replace the note in run-id order, whatever order they land in: the
// note call runs unheld, so a slow round 1 may finish after round 2. The
// team slug follows the latest round, as the PR row follows a repo that
// changed hands.
func TestAlmanachNoteFollowsTheLatestRound(t *testing.T) {
	s := testStore(t)
	ctx := context.Background()
	upsert(t, s, key, "a")

	if got := upsertNote(t, s, noteFor(key, 2, "two")); got != NoteWritten {
		t.Fatalf("round 2 = %q, want written", got)
	}
	if got := upsertNote(t, s, noteFor(key, 1, "one")); got != NoteSuperseded {
		t.Fatalf("late round 1 = %q, want superseded", got)
	}
	moved := noteFor(key, 3, "three")
	moved.TeamSlug = "mobile"
	if got := upsertNote(t, s, moved); got != NoteWritten {
		t.Fatalf("round 3 = %q, want written", got)
	}
	var text, team string
	if err := s.pool.QueryRow(ctx, `SELECT note->>'summary', team_slug FROM almanach_notes WHERE pr_id = $1`, key.PRID).Scan(&text, &team); err != nil {
		t.Fatal(err)
	}
	if text != "three" || team != "mobile" {
		t.Errorf("note = %q under %q, want round 3's under mobile", text, team)
	}
	finalize(t, s, key, "mobile", AlmanachFinal{})
	if got := pending(t, s, "platform"); len(got) != 0 {
		t.Errorf("the old team still drains the note: %+v", got)
	}
	if got := pending(t, s, "mobile"); len(got) != 1 {
		t.Errorf("the new team does not: %+v", got)
	}
}

// pr:merged stamps once, with the posted-findings counts by severity; a
// redelivery changes nothing, and the stamps survive a later round. A merge
// before any round leaves a stub that is not pending until the note lands;
// a merge of a PR noergler has no row for leaves nothing.
func TestAlmanachFinalizeStampsOnceAndFillsAStub(t *testing.T) {
	s := testStore(t)
	ctx := context.Background()
	prID := upsert(t, s, key, "a")
	runID := run(t, s, prID, "a", nano(600), "m")
	for _, sev := range []string{"issue", "issue", "suggestion"} {
		if err := s.InsertFinding(ctx, Finding{PullRequestID: prID, RunID: runID, FilePath: "a.go", LineNumber: 1,
			Severity: sev, CommentText: "c"}); err != nil {
			t.Fatal(err)
		}
	}

	if !finalize(t, s, key, "platform", AlmanachFinal{MergeCommit: "m1", SourceCommit: "s1", PRTitle: "t"}) {
		t.Fatal("first finalize must stamp")
	}
	if got := pending(t, s, "platform"); len(got) != 0 {
		t.Fatalf("a stub without a note is pending: %+v", got)
	}
	if finalize(t, s, key, "platform", AlmanachFinal{MergeCommit: "other"}) {
		t.Fatal("second finalize must not stamp")
	}

	// The late round fills the stub and keeps the stamps.
	if got := upsertNote(t, s, noteFor(key, runID, "late")); got != NoteWritten {
		t.Fatalf("late note = %q, want written", got)
	}
	got := pending(t, s, "platform")
	if len(got) != 1 {
		t.Fatalf("pending = %d, want 1", len(got))
	}
	p := got[0]
	if p.MergeCommit != "m1" || p.SourceCommit != "s1" || p.IssuesPosted != 2 || p.SuggestionsPosted != 1 || p.MergedAt.IsZero() {
		t.Errorf("final = merge %q source %q issues %d suggestions %d at %v", p.MergeCommit, p.SourceCommit, p.IssuesPosted, p.SuggestionsPosted, p.MergedAt)
	}
	if p.Review.Runs != 1 || p.Review.CostNanoUSD == nil || *p.Review.CostNanoUSD != 600 || len(p.Review.Models) != 1 {
		t.Errorf("review totals = %+v", p.Review)
	}
	if p.TicketKey != "PLAT-1" || p.JiraIssueType != "Bug" || p.ChangeKind != "bugfix" || string(p.Note) != `{"summary": "late"}` {
		t.Errorf("note facts = %+v %s", p, p.Note)
	}

	// Never reviewed, never a row: a merge leaves nothing behind.
	unknown := PRKey{Project: "PLAT", Repo: "api", PRID: 99}
	if finalize(t, s, unknown, "platform", AlmanachFinal{}) {
		t.Error("a merge of a PR noergler never saw wrote a stub")
	}
	if _, ok, _ := s.AlmanachNoteStatus(ctx, unknown); ok {
		t.Error("stub row for an unknown PR")
	}
}

// The review cost a note reports is the PR's cost as the cap and the rollup
// count it: runs, billed run-less attempts and comment calls, one figure.
func TestAlmanachReviewCostIsThePRCost(t *testing.T) {
	s := testStore(t)
	ctx := context.Background()
	prID := upsert(t, s, key, "a")
	run(t, s, prID, "a", nano(500), "m")
	if _, err := s.InsertAttempt(ctx, Attempt{Key: key, TeamSlug: "platform", Kind: RunAuto, Outcome: "unparseable",
		CostNanoUSD: nano(400)}); err != nil {
		t.Fatal(err)
	}
	if err := s.InsertCommentCall(ctx, "platform", key, CallReply, 1, LLMCall{Profile: "p", CostNanoUSD: nano(100)}); err != nil {
		t.Fatal(err)
	}
	upsertNote(t, s, noteFor(key, 1, "n"))
	finalize(t, s, key, "platform", AlmanachFinal{})
	got := pending(t, s, "platform")
	if len(got) != 1 || got[0].Review.CostNanoUSD == nil || *got[0].Review.CostNanoUSD != 1000 {
		t.Fatalf("review cost = %+v, want 1000", got)
	}
}

// Emitted is stamped for the round that was sent, after Almanach took it:
// a round that landed in between stays pending, an emitted note leaves the
// list, and a straggler after the emit is told so.
func TestAlmanachEmitIsPerRound(t *testing.T) {
	s := testStore(t)
	ctx := context.Background()
	upsert(t, s, key, "a")
	upsertNote(t, s, noteFor(key, 5, "five"))
	finalize(t, s, key, "platform", AlmanachFinal{})
	if got := pending(t, s, "platform"); len(got) != 1 || got[0].RunID != 5 {
		t.Fatalf("pending = %+v, want round 5", got)
	}

	// The forwarder read round 5; round 6 lands before the stamp.
	if got := upsertNote(t, s, noteFor(key, 6, "six")); got != NoteWritten {
		t.Fatalf("round 6 = %q", got)
	}
	if err := s.MarkAlmanachEmitted(ctx, key, 5); err != nil {
		t.Fatal(err)
	}
	if got := pending(t, s, "platform"); len(got) != 1 || got[0].RunID != 6 {
		t.Fatalf("round 6 must stay pending after round 5's stamp: %+v", got)
	}
	if err := s.MarkAlmanachEmitted(ctx, key, 6); err != nil {
		t.Fatal(err)
	}
	if got := pending(t, s, "platform"); len(got) != 0 {
		t.Fatalf("emitted note still pending: %+v", got)
	}
	if got := upsertNote(t, s, noteFor(key, 7, "late")); got != NoteAlreadyEmitted {
		t.Fatalf("round after emit = %q, want already_emitted", got)
	}
	st, ok, err := s.AlmanachNoteStatus(ctx, key)
	if err != nil || !ok || !st.Finalized || !st.HasNote || !st.Emitted {
		t.Errorf("status = %+v, %v, %v", st, ok, err)
	}
}

// Declined and deleted drop a note that is not final; a final note stays,
// whoever asks: a merged PR deleted afterwards still happened.
// A round whose PR was declined, deleted or purged while its note call ran
// writes nothing: the upsert needs an open PR row, and says so.
func TestAlmanachNoteNeedsAnOpenPR(t *testing.T) {
	s := testStore(t)
	ctx := context.Background()
	never := PRKey{Project: "PLAT", Repo: "api", PRID: 70}
	if got := upsertNote(t, s, noteFor(never, 1, "n")); got != NoteNoRow {
		t.Errorf("never-seen PR = %q, want no_row", got)
	}

	declined := PRKey{Project: "PLAT", Repo: "api", PRID: 71}
	upsert(t, s, declined, "a")
	upsertNote(t, s, noteFor(declined, 1, "one"))
	if err := s.MarkDeclined(ctx, declined); err != nil {
		t.Fatal(err)
	}
	if got := upsertNote(t, s, noteFor(declined, 2, "two")); got != NoteNoRow {
		t.Errorf("declined PR = %q, want no_row", got)
	}
	var text string
	if err := s.pool.QueryRow(ctx, `SELECT note->>'summary' FROM almanach_notes WHERE pr_id = $1`, declined.PRID).Scan(&text); err != nil {
		t.Fatal(err)
	}
	if text != "one" {
		t.Errorf("note = %q, the late round wrote over a declined PR", text)
	}

	deleted := PRKey{Project: "PLAT", Repo: "api", PRID: 72}
	upsert(t, s, deleted, "a")
	if err := s.MarkDeleted(ctx, deleted); err != nil {
		t.Fatal(err)
	}
	if got := upsertNote(t, s, noteFor(deleted, 1, "n")); got != NoteNoRow {
		t.Errorf("deleted PR = %q, want no_row", got)
	}
	if _, ok, _ := s.AlmanachNoteStatus(ctx, deleted); ok {
		t.Error("a note row for a deleted PR")
	}
}

func TestAlmanachDeleteKeepsAFinalNote(t *testing.T) {
	s := testStore(t)
	ctx := context.Background()
	open := PRKey{Project: "PLAT", Repo: "api", PRID: 8}
	upsert(t, s, open, "a")
	upsertNote(t, s, noteFor(open, 1, "x"))
	if err := s.DeleteAlmanachNote(ctx, open); err != nil {
		t.Fatal(err)
	}
	if _, ok, _ := s.AlmanachNoteStatus(ctx, open); ok {
		t.Error("open note still there after delete")
	}

	upsert(t, s, key, "a")
	upsertNote(t, s, noteFor(key, 1, "n"))
	finalize(t, s, key, "platform", AlmanachFinal{})
	if err := s.DeleteAlmanachNote(ctx, key); err != nil {
		t.Fatal(err)
	}
	if got := pending(t, s, "platform"); len(got) != 1 {
		t.Errorf("a final note was deleted: %+v", got)
	}
}

// The note call is Almanach's ledger alone: a priced call row changes
// neither the PR's cost, nor the team's spend, nor whether a rollup emits.
// The pending note still carries the round's call and the PR's round totals.
func TestAlmanachCallCountsTowardNothing(t *testing.T) {
	s := testStore(t)
	ctx := context.Background()
	upsert(t, s, key, "a")
	runID := int64(41)
	for i, cost := range []*int64{nano(900), nano(100)} {
		if err := s.InsertAlmanachCall(ctx, AlmanachCall{Key: key, TeamSlug: "platform", RunID: &runID, Outcome: "ok",
			NoteStatus: "written", Call: &LLMCall{Profile: "p", Model: "m", CostNanoUSD: cost, InputTokens: nano(int64(1000 * (i + 1))),
				CacheReadTokens: nano(900), OutputTokens: nano(50), CallID: "call-" + string(rune('a'+i)), Warnings: []string{"w \xff"}}}); err != nil {
			t.Fatal(err)
		}
	}
	// A call the gateway never answered still leaves its row.
	if err := s.InsertAlmanachCall(ctx, AlmanachCall{Key: key, TeamSlug: "platform", RunID: &runID, Outcome: "timed_out", NoteStatus: "no_answer"}); err != nil {
		t.Fatal(err)
	}

	if c, _ := s.PRCost(ctx, key); c != nil {
		t.Errorf("PRCost = %d, want NULL: the note call is not the PR's cost", *c)
	}
	if tot, err := s.TotalsSince(ctx, time.Now().Add(-time.Hour)); err != nil || tot.CostNanoUSD != nil {
		t.Errorf("team spend = %+v, %v; want unpriced: the note call is not dashboard spend", tot, err)
	}
	if snap, err := s.ClaimRollup(ctx, key, RollupFinal{}); err != nil || snap != nil {
		t.Errorf("ClaimRollup = %+v, %v; want nil: a note call alone emits no rollup", snap, err)
	}

	upsertNote(t, s, noteFor(key, runID, "n"))
	finalize(t, s, key, "platform", AlmanachFinal{})
	got := pending(t, s, "platform")
	if len(got) != 1 {
		t.Fatalf("pending = %d, want 1", len(got))
	}
	p := got[0]
	// The round's call is the latest row for its run id: the unanswered one.
	if !p.NoteCallFound || p.NoteCallOutcome != "timed_out" || p.NoteStatus != "no_answer" || p.NoteCall.CostNanoUSD != nil {
		t.Errorf("note call = found %v outcome %q status %q cost %v", p.NoteCallFound, p.NoteCallOutcome, p.NoteStatus, p.NoteCall.CostNanoUSD)
	}
	if p.Rounds != 3 || p.RoundsInputTokens != 3000 || p.RoundsOutputTokens != 100 || p.RoundsCostNanoUSD == nil || *p.RoundsCostNanoUSD != 1000 {
		t.Errorf("rounds = %d in %d out %d cost %v", p.Rounds, p.RoundsInputTokens, p.RoundsOutputTokens, p.RoundsCostNanoUSD)
	}
	var warning string
	if err := s.pool.QueryRow(ctx, `SELECT warnings[1] FROM almanach_calls WHERE call_id = 'call-a'`).Scan(&warning); err != nil || warning != "w �" {
		t.Errorf("warning = %q, %v; want the storable form", warning, err)
	}
}

// Releasing the project purges the team's PR rows and the notes that are
// not final; a final note stays, with zero review totals, because the
// forwarder still has to send it. Round totals count this incarnation only.
func TestAlmanachNotesAcrossPurge(t *testing.T) {
	s := testStore(t)
	ctx := context.Background()
	open := PRKey{Project: "PLAT", Repo: "api", PRID: 8}
	upsert(t, s, open, "a")
	upsertNote(t, s, noteFor(open, 1, "open"))
	upsert(t, s, key, "a")
	runID := int64(1)
	if err := s.InsertAlmanachCall(ctx, AlmanachCall{Key: key, TeamSlug: "platform", RunID: &runID, Outcome: "ok",
		NoteStatus: "written", Call: &LLMCall{Profile: "p", CostNanoUSD: nano(5)}}); err != nil {
		t.Fatal(err)
	}
	upsertNote(t, s, noteFor(key, 1, "final"))
	finalize(t, s, key, "platform", AlmanachFinal{})

	if _, err := s.PurgeProject(ctx, "platform", key.Project, nil); err != nil {
		t.Fatal(err)
	}
	if _, ok, _ := s.AlmanachNoteStatus(ctx, open); ok {
		t.Error("an open note survived the purge")
	}
	got := pending(t, s, "platform")
	if len(got) != 1 || got[0].Review.Runs != 0 || got[0].Review.CostNanoUSD != nil || got[0].Rounds != 1 {
		t.Fatalf("pending after purge = %+v, want the final note with zero review totals and its one round", got)
	}

	// Claimed again: a new incarnation starts its round count from zero.
	time.Sleep(10 * time.Millisecond)
	upsert(t, s, key, "b")
	if err := s.MarkAlmanachEmitted(ctx, key, 1); err != nil {
		t.Fatal(err)
	}
	runID = 2
	if err := s.InsertAlmanachCall(ctx, AlmanachCall{Key: key, TeamSlug: "platform", RunID: &runID, Outcome: "ok",
		NoteStatus: "written", Call: &LLMCall{Profile: "p", CostNanoUSD: nano(7)}}); err != nil {
		t.Fatal(err)
	}
	// The old row is emitted and final, so the new incarnation's note cannot
	// land on it; that is the PR key's life, and a reopened PR is rare
	// enough to log. What this pins is the round bound.
	var rounds int
	if err := s.pool.QueryRow(ctx, `SELECT COUNT(*) FROM almanach_calls a JOIN pull_requests p
		ON p.project_key = a.project_key AND p.repo_slug = a.repo_slug AND p.pr_id = a.pr_id
		WHERE a.pr_id = $1 AND a.created_at >= p.created_at`, key.PRID).Scan(&rounds); err != nil {
		t.Fatal(err)
	}
	if rounds != 1 {
		t.Errorf("rounds in the new incarnation = %d, want 1", rounds)
	}
}
