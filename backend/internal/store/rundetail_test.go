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
	if len(d.Fixed) != 1 || d.Fixed[0].Headline != "old one" || d.Fixed[0].FromRunID != earlier {
		t.Errorf("fixed = %+v", d.Fixed)
	}
	if len(d.PRAttempts) != 2 || d.PRAttempts[0].ID != skip || d.PRAttempts[1].ID != id {
		t.Errorf("pr attempts = %+v", d.PRAttempts)
	}
	if c := d.PRAttempts[1].CostNanoUSD; c == nil || *c != 900 {
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
