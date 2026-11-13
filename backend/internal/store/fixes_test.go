package store

import (
	"context"
	"testing"
	"time"
)

// A fix request is one row whatever came of it; a finding is committed
// once, and its commit rides every read of the finding.
func TestFindingFixes_RecordedCommittedOnceAndCounted(t *testing.T) {
	s := testStore(t)
	ctx := context.Background()
	id := upsert(t, s, key, "a")
	runID := run(t, s, id, "a", nil, "m")
	if err := s.InsertFinding(ctx, Finding{PullRequestID: id, RunID: runID, FilePath: "a.go", LineNumber: 2,
		Severity: "issue", CommentText: "unchecked", BitbucketCommentID: num(501)}); err != nil {
		t.Fatal(err)
	}
	f, err := s.FindingByComment(ctx, key, 501)
	if err != nil || f == nil || f.BotFixCommit != nil {
		t.Fatalf("finding = %+v, err = %v", f, err)
	}

	for _, x := range []FindingFix{
		{FindingID: f.ID, TeamSlug: "payments", Outcome: FixNoWrite},
		{FindingID: f.ID, TeamSlug: "payments", Outcome: FixCommitted, CommitSHA: "abc"},
	} {
		if ok, err := s.RecordFindingFix(ctx, key, x); err != nil || !ok {
			t.Fatalf("record %+v: %v, %v", x, ok, err)
		}
	}
	if ok, err := s.RecordFindingFix(ctx, key, FindingFix{FindingID: f.ID, TeamSlug: "payments", Outcome: FixCommitted, CommitSHA: "def"}); err != nil || ok {
		t.Errorf("a second commit of one finding = %v, %v: want dropped", ok, err)
	}
	if _, err := s.RecordFindingFix(ctx, key, FindingFix{FindingID: f.ID, TeamSlug: "payments", Outcome: FixCommitted}); err == nil {
		t.Error("a committed fix without its sha must be refused by the schema")
	}
	if _, err := s.RecordFindingFix(ctx, key, FindingFix{FindingID: f.ID, TeamSlug: "payments", Outcome: "maybe"}); err == nil {
		t.Error("an unknown outcome must be refused by the schema")
	}

	got, _ := s.FindingByComment(ctx, key, 501)
	if got.BotFixCommit == nil || *got.BotFixCommit != "abc" {
		t.Errorf("BotFixCommit = %v, want abc", got.BotFixCommit)
	}
	byID, _ := s.FindingByID(ctx, key, f.ID)
	if byID == nil || byID.BotFixCommit == nil || byID.RunCommit != "a" {
		t.Errorf("FindingByID = %+v", byID)
	}
	other := key
	other.PRID++
	if f, err := s.FindingByID(ctx, other, f.ID); err != nil || f != nil {
		t.Errorf("another PR's lookup = %+v, %v: the id is scoped to its PR", f, err)
	}
	existing, _ := s.ExistingFindings(ctx, key)
	if len(existing) != 1 || existing[0].BotFixCommit == nil {
		t.Errorf("existing = %+v, want the commit for the review", existing)
	}

	counts, err := s.FixCountsSince(ctx, time.Now().Add(-time.Hour))
	if err != nil {
		t.Fatal(err)
	}
	want := []FixCount{{"payments", FixCommitted, 1}, {"payments", FixNoWrite, 1}}
	if len(counts) != 2 || counts[0] != want[0] || counts[1] != want[1] {
		t.Errorf("counts = %+v, want %+v", counts, want)
	}
	if counts, _ := s.FixCountsSince(ctx, time.Now().Add(time.Hour)); len(counts) != 0 {
		t.Errorf("a later window counts %+v", counts)
	}
	days, err := s.FixesDaily(ctx, time.Now().Add(-time.Hour))
	if err != nil || len(days) != 1 || days[0].Requested != 2 || days[0].Committed != 1 {
		t.Errorf("days = %+v, %v: want one day, 2 requests, 1 committed", days, err)
	}
}

// noergler's answer on a finding's thread keeps its finding.
func TestBotReplies_KeepTheirFinding(t *testing.T) {
	s := testStore(t)
	ctx := context.Background()
	id := upsert(t, s, key, "a")
	runID := run(t, s, id, "a", nil, "m")
	if err := s.InsertFinding(ctx, Finding{PullRequestID: id, RunID: runID, FilePath: "a.go", LineNumber: 2,
		Severity: "issue", CommentText: "unchecked", BitbucketCommentID: num(501)}); err != nil {
		t.Fatal(err)
	}
	f, _ := s.FindingByComment(ctx, key, 501)
	if err := s.InsertFindingBotReply(ctx, key, 700, BotReplyReply, f.ID); err != nil {
		t.Fatal(err)
	}
	if err := s.InsertBotReply(ctx, key, 701, BotReplyQuestion); err != nil {
		t.Fatal(err)
	}
	if b, _ := s.BotReplyByComment(ctx, key, 700); b == nil || b.FindingID != f.ID {
		t.Errorf("answer = %+v, want finding %d", b, f.ID)
	}
	if b, _ := s.BotReplyByComment(ctx, key, 701); b == nil || b.FindingID != 0 {
		t.Errorf("Q&A answer = %+v, want no finding", b)
	}
}

// A Q&A answer keeps its question's code line; its fix is recorded against
// the answer, committed once, counted with the findings', gone with its PR.
func TestAnswerFixes_AnchorKeptCommittedOnceGoneWithThePR(t *testing.T) {
	s := testStore(t)
	ctx := context.Background()
	id := upsert(t, s, key, "a")
	a := &BotReplyAnchor{Path: "a.go", Line: 7, Commit: "src1"}
	if err := s.InsertAnswerBotReply(ctx, key, 950, a, 0); err != nil {
		t.Fatal(err)
	}
	if err := s.InsertAnswerBotReply(ctx, key, 951, nil, 0); err != nil {
		t.Fatal(err)
	}
	// A reply of the fix flow about answer 950 keeps its line and names it.
	if err := s.InsertAnswerBotReply(ctx, key, 952, a, 950); err != nil {
		t.Fatal(err)
	}
	if b, _ := s.BotReplyByComment(ctx, key, 950); b == nil || b.Kind != BotReplyAnswer || b.Anchor == nil || *b.Anchor != *a || b.Source != 0 {
		t.Errorf("answer = %+v, want its anchor %+v", b, a)
	}
	if b, _ := s.BotReplyByComment(ctx, key, 952); b == nil || b.Source != 950 || b.Anchor == nil {
		t.Errorf("reply about the answer = %+v", b)
	}
	if b, _ := s.BotReplyByComment(ctx, key, 951); b == nil || b.Anchor != nil {
		t.Errorf("answer without a line = %+v", b)
	}

	if sha, err := s.AnswerFixCommit(ctx, 950); err != nil || sha != "" {
		t.Errorf("before = %q, %v", sha, err)
	}
	for _, x := range []FindingFix{
		{AnswerCommentID: 950, TeamSlug: "payments", Outcome: FixConflict},
		{AnswerCommentID: 950, TeamSlug: "payments", Outcome: FixCommitted, CommitSHA: "abc"},
	} {
		if ok, err := s.RecordFindingFix(ctx, key, x); err != nil || !ok {
			t.Fatalf("record %+v: %v, %v", x, ok, err)
		}
	}
	if ok, err := s.RecordFindingFix(ctx, key, FindingFix{AnswerCommentID: 950, TeamSlug: "payments", Outcome: FixCommitted, CommitSHA: "def"}); err != nil || ok {
		t.Errorf("a second commit of one answer = %v, %v: want dropped", ok, err)
	}
	if sha, _ := s.AnswerFixCommit(ctx, 950); sha != "abc" {
		t.Errorf("AnswerFixCommit = %q, want abc", sha)
	}
	other := key
	other.PRID++
	if ok, err := s.RecordFindingFix(ctx, other, FindingFix{AnswerCommentID: 960, TeamSlug: "payments", Outcome: FixError}); err != nil || ok {
		t.Errorf("an answer on a PR without a row = %v, %v: want nothing recorded", ok, err)
	}
	if counts, _ := s.FixCountsSince(ctx, time.Now().Add(-time.Hour)); len(counts) != 2 {
		t.Errorf("counts = %+v, want committed and conflict", counts)
	}

	if _, err := s.pool.Exec(ctx, `DELETE FROM pull_requests WHERE id = $1`, id); err != nil {
		t.Fatal(err)
	}
	if counts, _ := s.FixCountsSince(ctx, time.Now().Add(-time.Hour)); len(counts) != 0 {
		t.Errorf("after the PR's purge counts = %+v", counts)
	}
}
