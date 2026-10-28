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

func pending(t *testing.T, s *Store) []AlmanachPending {
	t.Helper()
	out, err := s.PendingAlmanachNotes(context.Background(), "platform", 10)
	if err != nil {
		t.Fatal(err)
	}
	return out
}

// Rounds replace the note in run-id order, whatever order they land in: the
// note call runs unheld, so a slow round 1 may finish after round 2.
func TestAlmanachNoteFollowsTheLatestRound(t *testing.T) {
	s := testStore(t)
	ctx := context.Background()

	if w, err := s.UpsertAlmanachNote(ctx, noteFor(key, 2, "two")); err != nil || !w {
		t.Fatalf("round 2 = %v, %v; want written", w, err)
	}
	if w, err := s.UpsertAlmanachNote(ctx, noteFor(key, 1, "one")); err != nil || w {
		t.Fatalf("late round 1 = %v, %v; want superseded", w, err)
	}
	if w, err := s.UpsertAlmanachNote(ctx, noteFor(key, 3, "three")); err != nil || !w {
		t.Fatalf("round 3 = %v, %v; want written", w, err)
	}
	var text string
	if err := s.pool.QueryRow(ctx, `SELECT note->>'summary' FROM almanach_notes WHERE pr_id = $1`, key.PRID).Scan(&text); err != nil {
		t.Fatal(err)
	}
	if text != "three" {
		t.Errorf("note = %q, want round 3's", text)
	}
}

// pr:merged stamps once, with the posted-findings counts by severity; a
// redelivery changes nothing, and the stamps survive a later round. A merge
// before any round leaves a stub that is not pending until the note lands.
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

	final := AlmanachFinal{MergedAt: time.Now(), MergeCommit: "m1", SourceCommit: "s1", PRTitle: "t"}
	if st, err := s.FinalizeAlmanachNote(ctx, key, "platform", final); err != nil || !st {
		t.Fatalf("finalize = %v, %v; want stamped", st, err)
	}
	if got := pending(t, s); len(got) != 0 {
		t.Fatalf("a stub without a note is pending: %+v", got)
	}
	if st, err := s.FinalizeAlmanachNote(ctx, key, "platform", AlmanachFinal{MergedAt: time.Now(), MergeCommit: "other"}); err != nil || st {
		t.Fatalf("second finalize = %v, %v; want not stamped", st, err)
	}

	// The late round fills the stub and keeps the stamps.
	if w, err := s.UpsertAlmanachNote(ctx, noteFor(key, runID, "late")); err != nil || !w {
		t.Fatalf("late note = %v, %v; want written", w, err)
	}
	got := pending(t, s)
	if len(got) != 1 {
		t.Fatalf("pending = %d, want 1", len(got))
	}
	p := got[0]
	if p.MergeCommit != "m1" || p.SourceCommit != "s1" || p.IssuesPosted != 2 || p.SuggestionsPosted != 1 {
		t.Errorf("final = merge %q source %q issues %d suggestions %d", p.MergeCommit, p.SourceCommit, p.IssuesPosted, p.SuggestionsPosted)
	}
	if p.Review.Runs != 1 || p.Review.CostNanoUSD == nil || *p.Review.CostNanoUSD != 600 || len(p.Review.Models) != 1 {
		t.Errorf("review totals = %+v", p.Review)
	}
	if p.TicketKey != "PLAT-1" || p.JiraIssueType != "Bug" || p.ChangeKind != "bugfix" || string(p.Note) != `{"summary": "late"}` {
		t.Errorf("note facts = %+v %s", p, p.Note)
	}
}

// Emitted is set after Almanach took the note; an emitted note leaves the
// pending list and is never rewritten by a straggling round. Declined and
// deleted delete.
func TestAlmanachEmitAndDelete(t *testing.T) {
	s := testStore(t)
	ctx := context.Background()
	if _, err := s.UpsertAlmanachNote(ctx, noteFor(key, 5, "n")); err != nil {
		t.Fatal(err)
	}
	if _, err := s.FinalizeAlmanachNote(ctx, key, "platform", AlmanachFinal{MergedAt: time.Now()}); err != nil {
		t.Fatal(err)
	}
	if got := pending(t, s); len(got) != 1 {
		t.Fatalf("pending = %d, want 1", len(got))
	}
	if err := s.MarkAlmanachEmitted(ctx, key); err != nil {
		t.Fatal(err)
	}
	if got := pending(t, s); len(got) != 0 {
		t.Fatalf("emitted note still pending: %+v", got)
	}
	if w, err := s.UpsertAlmanachNote(ctx, noteFor(key, 6, "late")); err != nil || w {
		t.Fatalf("round after emit = %v, %v; want refused", w, err)
	}
	st, ok, err := s.AlmanachNoteStatus(ctx, key)
	if err != nil || !ok || !st.Finalized || !st.HasNote || !st.Emitted {
		t.Errorf("status = %+v, %v, %v", st, ok, err)
	}

	other := PRKey{Project: "PLAT", Repo: "api", PRID: 8}
	if _, err := s.UpsertAlmanachNote(ctx, noteFor(other, 1, "x")); err != nil {
		t.Fatal(err)
	}
	if err := s.DeleteAlmanachNote(ctx, other); err != nil {
		t.Fatal(err)
	}
	if _, ok, _ := s.AlmanachNoteStatus(ctx, other); ok {
		t.Error("deleted note still there")
	}
}

// The note call is Almanach's ledger alone: a priced call row changes
// neither the PR's cost, nor the team's spend, nor whether a rollup emits.
// The pending note still carries the call and the PR's round totals.
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

	if _, err := s.UpsertAlmanachNote(ctx, noteFor(key, runID, "n")); err != nil {
		t.Fatal(err)
	}
	if _, err := s.FinalizeAlmanachNote(ctx, key, "platform", AlmanachFinal{MergedAt: time.Now()}); err != nil {
		t.Fatal(err)
	}
	got := pending(t, s)
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

// Releasing the project purges the team's PR rows and leaves the notes:
// they are the export payload, and the forwarder still has to send them.
func TestAlmanachNotesSurvivePurge(t *testing.T) {
	s := testStore(t)
	ctx := context.Background()
	upsert(t, s, key, "a")
	if _, err := s.UpsertAlmanachNote(ctx, noteFor(key, 1, "n")); err != nil {
		t.Fatal(err)
	}
	if _, err := s.FinalizeAlmanachNote(ctx, key, "platform", AlmanachFinal{MergedAt: time.Now()}); err != nil {
		t.Fatal(err)
	}
	if _, err := s.PurgeProject(ctx, "platform", key.Project, nil); err != nil {
		t.Fatal(err)
	}
	got := pending(t, s)
	if len(got) != 1 || got[0].Review.Runs != 0 {
		t.Fatalf("pending after purge = %+v, want the note with zero review totals", got)
	}
}
