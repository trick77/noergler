package store

import (
	"context"
	"testing"
)

// The runs feed lists every act its own row (user, 2026-11-17): a review,
// an engagement with its own calls' time and cost (its fix call aside), a
// fix with its commit and source, a SNAPSHOT task. failed reaches a failed
// engagement and an errored fix too; skipped stays a review's.
func TestFeedListsEveryAct(t *testing.T) {
	s := testStore(t)
	ctx := context.Background()
	prID := upsert(t, s, key, "a")
	runID := run(t, s, prID, "a", nano(500), "m")
	addAttempt(t, s, Attempt{Key: key, TeamSlug: "platform", Kind: RunAuto, Outcome: "ok", RunID: &runID})
	addAttempt(t, s, Attempt{Key: key, TeamSlug: "platform", Kind: RunAuto, Outcome: "skipped", Reason: "head_unchanged"})

	if _, err := s.RecordEngagement(ctx, key, 77, EngagementMention, EngagementAnswered); err != nil {
		t.Fatal(err)
	}
	if _, err := s.RecordEngagement(ctx, key, 78, EngagementReply, EngagementFailed); err != nil {
		t.Fatal(err)
	}
	ms := int64(1200)
	for _, c := range []struct {
		kind string
		cost int64
	}{{CallQuestion, 300}, {CallQuestion, 200}, {CallFix, 9000}} {
		cost := c.cost
		if err := s.InsertCommentCall(ctx, "platform", key, c.kind, 77, LLMCall{Profile: "m", CostNanoUSD: &cost, TotalMS: &ms}); err != nil {
			t.Fatal(err)
		}
	}
	// One call priced, one not: the engagement's cost is unpriced, never the
	// priced part alone.
	if _, err := s.RecordEngagement(ctx, key, 79, EngagementReply, EngagementAnswered); err != nil {
		t.Fatal(err)
	}
	priced := int64(400)
	for _, c := range []*int64{&priced, nil} {
		if err := s.InsertCommentCall(ctx, "platform", key, CallReply, 79, LLMCall{Profile: "m", CostNanoUSD: c}); err != nil {
			t.Fatal(err)
		}
	}
	if err := s.RecordSnapshotTasks(ctx, prID, []int{91}); err != nil {
		t.Fatal(err)
	}
	if _, err := s.RecordFindingFix(ctx, key, FindingFix{ConflictCommentID: 95, TeamSlug: "platform",
		Outcome: FixCommitted, CommitSHA: "abc123"}); err != nil {
		t.Fatal(err)
	}
	if _, err := s.RecordFindingFix(ctx, key, FindingFix{AnswerCommentID: 96, TeamSlug: "platform", Outcome: FixError}); err != nil {
		t.Fatal(err)
	}

	rows, err := s.RecentAttempts(ctx, "", AttemptsAll, 50)
	if err != nil {
		t.Fatal(err)
	}
	byAct := map[string][]AttemptRow{}
	for _, r := range rows {
		if r.TeamSlug != "platform" || r.Key != key {
			t.Errorf("row %+v: want the PR's key and team", r)
		}
		byAct[r.Act] = append(byAct[r.Act], r)
	}
	if len(byAct[ActReview]) != 2 || len(byAct[ActEngagement]) != 3 || len(byAct[ActFix]) != 2 || len(byAct[ActTask]) != 1 {
		t.Fatalf("acts = %+v", byAct)
	}
	for _, e := range byAct[ActEngagement] {
		if e.ID == 77 && (e.Outcome != EngagementAnswered || e.Detail != EngagementMention || e.Unpriced ||
			e.CostNanoUSD == nil || *e.CostNanoUSD != 500 || e.ElapsedMS == nil || *e.ElapsedMS != 2400) {
			t.Errorf("engagement 77 = %+v: want answered by mention, its two question calls' cost and time", e)
		}
		if e.ID == 79 && (!e.Unpriced || e.CostNanoUSD != nil) {
			t.Errorf("engagement 79 = %+v: one unpriced call makes it unpriced", e)
		}
		if e.ID == 78 && (e.Unpriced || e.CostNanoUSD != nil) {
			t.Errorf("engagement 78 = %+v: no call, no cost, not unpriced", e)
		}
	}
	for _, f := range byAct[ActFix] {
		if f.Outcome == FixCommitted && (f.Detail != "conflict" || f.Commit == nil || *f.Commit != "abc123") {
			t.Errorf("committed fix = %+v", f)
		}
		if f.Outcome == FixError && f.Detail != "answer" {
			t.Errorf("failed fix = %+v", f)
		}
	}
	if task := byAct[ActTask][0]; task.ID != 91 || task.Outcome != "posted" || task.CostNanoUSD != nil {
		t.Errorf("task = %+v", task)
	}

	failed, err := s.RecentAttempts(ctx, "", AttemptsFailed, 50)
	if err != nil {
		t.Fatal(err)
	}
	if len(failed) != 2 {
		t.Errorf("failed = %+v, want the failed engagement and the errored fix", failed)
	}
	skipped, _ := s.RecentAttempts(ctx, "", AttemptsSkipped, 50)
	if len(skipped) != 1 || skipped[0].Act != ActReview {
		t.Errorf("skipped = %+v, want the one review", skipped)
	}
}
