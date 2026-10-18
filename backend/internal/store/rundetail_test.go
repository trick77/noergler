package store

import (
	"context"
	"testing"
)

func TestRunDetail(t *testing.T) {
	s := testStore(t)
	ctx := context.Background()
	prID := upsert(t, s, key, "a")
	earlier := run(t, s, prID, "a", nano(600), "m")
	if err := s.InsertFinding(ctx, Finding{PullRequestID: prID, RunID: earlier, FilePath: "a.go", LineNumber: 3,
		Severity: "medium", Headline: strp("old one"), CommentText: "SECRET BODY"}); err != nil {
		t.Fatal(err)
	}
	runID := run(t, s, prID, "b", nano(900), "m")
	if err := s.InsertFinding(ctx, Finding{PullRequestID: prID, RunID: runID, FilePath: "b.go", LineNumber: 7,
		Severity: "high", Headline: strp("new one"), CommentText: "SECRET BODY"}); err != nil {
		t.Fatal(err)
	}
	if _, err := s.pool.Exec(ctx, `UPDATE findings SET fixed_at = now(), fixed_run_id = $1 WHERE review_run_id = $2`, runID, earlier); err != nil {
		t.Fatal(err)
	}
	earlierAttempt, err := s.InsertAttempt(ctx, Attempt{Key: key, TeamSlug: "platform", Kind: RunAuto, Outcome: "ok", RunID: &earlier})
	if err != nil {
		t.Fatal(err)
	}
	skip, err := s.InsertAttempt(ctx, Attempt{Key: key, TeamSlug: "platform", Kind: RunAuto, Outcome: "skipped", Reason: "head_unchanged"})
	if err != nil {
		t.Fatal(err)
	}
	id, err := s.InsertAttempt(ctx, Attempt{Key: key, TeamSlug: "platform", Kind: RunAuto, Outcome: "ok",
		RunID: &runID, Call: &LLMCall{Profile: "m", CallID: "c1"}})
	if err != nil {
		t.Fatal(err)
	}
	// Another team's attempt on the same PR key is not this PR's history.
	if _, err := s.InsertAttempt(ctx, Attempt{Key: key, TeamSlug: "other", Kind: RunAuto, Outcome: "error"}); err != nil {
		t.Fatal(err)
	}

	d, ok, err := s.RunDetail(ctx, id)
	if err != nil || !ok {
		t.Fatalf("RunDetail = %v, %v", ok, err)
	}
	if d.Run == nil || d.Run.ID != runID || d.Run.CostNanoUSD == nil || *d.Run.CostNanoUSD != 900 {
		t.Errorf("run = %+v", d.Run)
	}
	if d.Call == nil || d.Call.CallID != "c1" {
		t.Errorf("call = %+v", d.Call)
	}
	if len(d.Findings) != 1 || d.Findings[0].Headline != "new one" || d.Findings[0].Line != 7 {
		t.Errorf("findings = %+v", d.Findings)
	}
	// The earlier run's own page is keyed by its ATTEMPT id, not its run
	// id: the link must open the attempt that produced the finding.
	if len(d.Fixed) != 1 || d.Fixed[0].Headline != "old one" || d.Fixed[0].FromAttemptID == nil ||
		*d.Fixed[0].FromAttemptID != earlierAttempt {
		t.Errorf("fixed = %+v, want from attempt %d", d.Fixed, earlierAttempt)
	}
	if len(d.PRAttempts) != 3 || d.PRAttempts[0].ID != earlierAttempt || d.PRAttempts[1].ID != skip || d.PRAttempts[2].ID != id {
		t.Errorf("pr attempts = %+v", d.PRAttempts)
	}
	if c := d.PRAttempts[2].CostNanoUSD; c == nil || *c != 900 {
		t.Errorf("the ok attempt's cost = %v, want its run's 900", c)
	}

	// A skip has no run, no call, no findings, and is still a page.
	d, ok, err = s.RunDetail(ctx, skip)
	if err != nil || !ok || d.Run != nil || d.Call != nil || len(d.Findings) != 0 || d.Reason != "head_unchanged" {
		t.Errorf("skip detail = %+v, %v, %v", d, ok, err)
	}

	if _, ok, err := s.RunDetail(ctx, id+1000); ok || err != nil {
		t.Errorf("unknown id = %v, %v; want not found", ok, err)
	}
}

func strp(s string) *string { return &s }

// A long PR's history is a window around the attempt shown, so the attempt
// and its neighbours are always in it, whichever end of the PR it is at.
func TestRunDetailHistoryWindow(t *testing.T) {
	s := testStore(t)
	ctx := context.Background()
	var ids []int64
	for i := 0; i < 2*prAttemptsEachSide+10; i++ {
		id, err := s.InsertAttempt(ctx, Attempt{Key: key, TeamSlug: "platform", Kind: RunAuto, Outcome: "skipped", Reason: "head_unchanged"})
		if err != nil {
			t.Fatal(err)
		}
		ids = append(ids, id)
	}
	for _, at := range []int{2, len(ids) - 1, len(ids) / 2} {
		d, ok, err := s.RunDetail(ctx, ids[at])
		if err != nil || !ok {
			t.Fatalf("RunDetail = %v, %v", ok, err)
		}
		found := -1
		for i, a := range d.PRAttempts {
			if a.ID == ids[at] {
				found = i
			}
			if i > 0 && d.PRAttempts[i-1].ID >= a.ID {
				t.Fatalf("history not oldest first at %d", i)
			}
		}
		if found < 0 || len(d.PRAttempts) > 2*prAttemptsEachSide+1 {
			t.Errorf("attempt %d: found at %d of %d", at, found, len(d.PRAttempts))
		}
		if at > 0 && at < len(ids)-1 && (found == 0 || found == len(d.PRAttempts)-1) {
			t.Errorf("attempt %d lost a neighbour", at)
		}
		// The page is told when rows are cut, on the side they are cut.
		if wantBefore := at > prAttemptsEachSide; d.MoreBefore != wantBefore {
			t.Errorf("attempt %d: MoreBefore = %v, want %v", at, d.MoreBefore, wantBefore)
		}
		if wantAfter := len(ids)-1-at > prAttemptsEachSide; d.MoreAfter != wantAfter {
			t.Errorf("attempt %d: MoreAfter = %v, want %v", at, d.MoreAfter, wantAfter)
		}
	}
}
