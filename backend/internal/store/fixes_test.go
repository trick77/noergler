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
		if ok, err := s.RecordFindingFix(ctx, x); err != nil || !ok {
			t.Fatalf("record %+v: %v, %v", x, ok, err)
		}
	}
	if ok, err := s.RecordFindingFix(ctx, FindingFix{FindingID: f.ID, TeamSlug: "payments", Outcome: FixCommitted, CommitSHA: "def"}); err != nil || ok {
		t.Errorf("a second commit of one finding = %v, %v: want dropped", ok, err)
	}
	if _, err := s.RecordFindingFix(ctx, FindingFix{FindingID: f.ID, TeamSlug: "payments", Outcome: FixCommitted}); err == nil {
		t.Error("a committed fix without its sha must be refused by the schema")
	}
	if _, err := s.RecordFindingFix(ctx, FindingFix{FindingID: f.ID, TeamSlug: "payments", Outcome: "maybe"}); err == nil {
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
