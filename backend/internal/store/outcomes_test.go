package store

import (
	"context"
	"testing"
	"time"
)

// Each finding of a merged PR lands in one outcome, first match wins; a fix
// noergler committed is fixed without a later claim. An open PR, a task, a
// rule-removed task and a legacy row without a thread count nothing.
func TestFindingOutcomesSince_ClassifiesMergedFindings(t *testing.T) {
	s := testStore(t)
	ctx := context.Background()
	merged := upsert(t, s, key, "a")
	runID := run(t, s, merged, "a", nil, "m")
	insert := func(prID, runID int64, comment int, conf *int, blocking *string) int64 {
		t.Helper()
		if err := s.InsertFinding(ctx, Finding{PullRequestID: prID, RunID: runID, FilePath: "a.go", LineNumber: comment,
			Severity: "issue", Confidence: conf, CommentText: "c", BitbucketCommentID: num(comment),
			BlockingOutcome: blocking}); err != nil {
			t.Fatal(err)
		}
		k := key
		if prID != merged {
			k = PRKey{Project: key.Project, Repo: key.Repo, PRID: 8}
		}
		f, err := s.FindingByComment(ctx, k, comment)
		if err != nil {
			t.Fatal(err)
		}
		return f.ID
	}
	fixedAndDisputed := insert(merged, runID, 1, num(96), nil)
	disputed := insert(merged, runID, 2, num(92), nil)
	closed := insert(merged, runID, 3, num(85), nil)
	insert(merged, runID, 4, nil, nil) // open, no confidence
	removed := insert(merged, runID, 5, num(95), str("rule_removed"))
	insert(merged, runID, 7, num(97), str("task")) // a task is no finding
	committed := insert(merged, runID, 8, num(91), nil)
	if _, err := s.RecordFindingFix(ctx, key, FindingFix{FindingID: committed, TeamSlug: "platform",
		Outcome: FixCommitted, CommitSHA: "abc"}); err != nil {
		t.Fatal(err)
	}
	// A legacy row with no comment id: no thread anyone could resolve.
	if err := s.InsertFinding(ctx, Finding{PullRequestID: merged, RunID: runID, FilePath: "a.go", LineNumber: 9,
		Severity: "issue", Confidence: num(99), CommentText: "legacy"}); err != nil {
		t.Fatal(err)
	}
	for _, err := range []error{
		s.MarkFindingFixed(ctx, fixedAndDisputed, runID),
		s.MarkFindingReplied(ctx, fixedAndDisputed, str("not a bug")),
		s.MarkFindingReplied(ctx, disputed, str("the caller checks it")),
		s.MarkFindingClosed(ctx, closed),
		s.MarkFindingClosed(ctx, removed),
	} {
		if err != nil {
			t.Fatal(err)
		}
	}
	open := upsert(t, s, PRKey{Project: key.Project, Repo: key.Repo, PRID: 8}, "b")
	insert(open, run(t, s, open, "b", nil, "m"), 6, num(99), nil)
	if err := s.MarkMerged(ctx, key); err != nil {
		t.Fatal(err)
	}

	counts, err := s.FindingOutcomesSince(ctx, time.Now().Add(-time.Hour))
	if err != nil {
		t.Fatal(err)
	}
	got := map[[2]string]int{}
	for _, c := range counts {
		if c.TeamSlug != "platform" || c.Severity != "issue" {
			t.Errorf("row %+v: want team platform, severity issue", c)
		}
		got[[2]string{c.Band, c.Outcome}] += c.Count
	}
	want := map[[2]string]int{
		{BandHigh, OutcomeFixed}:   1,
		{BandMid, OutcomeDisputed}: 1,
		{BandMid, OutcomeFixed}:    1, // committed by noergler, no claim yet
		{BandLow, OutcomeClosed}:   1,
		{BandUnknown, OutcomeOpen}: 1,
	}
	if len(got) != len(want) {
		t.Errorf("outcomes = %v, want %v", got, want)
	}
	for k, n := range want {
		if got[k] != n {
			t.Errorf("%v = %d, want %d (all: %v)", k, got[k], n, got)
		}
	}
	if later, _ := s.FindingOutcomesSince(ctx, time.Now().Add(time.Hour)); len(later) != 0 {
		t.Errorf("a window after the merge counts %v, want nothing", later)
	}
}
