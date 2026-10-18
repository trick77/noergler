package store

import (
	"context"
	"errors"
	"reflect"
	"testing"
	"time"
)

func TestAttemptCallRoundTrip(t *testing.T) {
	s := testStore(t)
	ctx := context.Background()
	call := LLMCall{
		Profile: "gpt-5.5-medium", Model: "gpt-5.5-2026-08-06", Deployment: "azure/gpt-5.5-swc",
		ReasoningSent: "medium", FinishReason: "stop", CallID: "b7e2", GatewayVer: "1.80.0",
		InputTokens: nano(41210), InputUncachedTokens: nano(12794), CacheReadTokens: nano(28416),
		OutputTokens: nano(2830), ReasoningTokens: nano(1920),
		CostNanoUSD: nano(47_845_000), CostInputNanoUSD: nano(15_993_000),
		CostCacheReadNanoUSD: nano(3_552_000), CostOutputNanoUSD: nano(28_300_000),
		CostReasoningNanoUSD: nano(19_200_000), KeySpendNanoUSD: nano(812_400_000_000),
		HeadersMS: nano(58310), TotalMS: nano(58420),
		Warnings: []string{"compatibility reasoning: mapped to effort"},
	}
	id, err := s.InsertAttempt(ctx, Attempt{Key: key, TeamSlug: "platform", Kind: RunAuto, Outcome: "ok", Call: &call})
	if err != nil {
		t.Fatal(err)
	}
	got, ok, err := s.AttemptCall(ctx, id)
	if err != nil || !ok {
		t.Fatalf("AttemptCall = %v, %v", ok, err)
	}
	if !reflect.DeepEqual(got, call) {
		t.Errorf("round trip:\n got %+v\nwant %+v", got, call)
	}
	// A lane the gateway did not send stays absent, never zero.
	if got.CacheWriteTokens != nil || got.CostCacheWriteNanoUSD != nil || got.CostToolUsageNanoUSD != nil {
		t.Error("an absent lane came back as a value")
	}
}

// A pre-flight skip, a timeout and a pre-flight too-large got no answer:
// there is nothing to show, and the page says so instead of rendering zeros.
func TestAttemptWithoutCall(t *testing.T) {
	s := testStore(t)
	ctx := context.Background()
	id, err := s.InsertAttempt(ctx, Attempt{Key: key, TeamSlug: "platform", Kind: RunAuto, Outcome: "timed_out"})
	if err != nil {
		t.Fatal(err)
	}
	if _, ok, err := s.AttemptCall(ctx, id); ok || err != nil {
		t.Errorf("AttemptCall = %v, %v; want none", ok, err)
	}
	// An empty warning list is stored as one, not as NULL.
	id, err = s.InsertAttempt(ctx, Attempt{Key: key, TeamSlug: "platform", Kind: RunAuto, Outcome: "ok",
		Call: &LLMCall{Profile: "p"}})
	if err != nil {
		t.Fatal(err)
	}
	got, _, err := s.AttemptCall(ctx, id)
	if err != nil || got.Warnings == nil || len(got.Warnings) != 0 {
		t.Errorf("warnings = %#v, %v; want an empty list", got.Warnings, err)
	}
}

// The call's cost is detail. The review's cost is on the run (or the
// attempt, for a billed call with no run), and the PR total must not count
// the same call a second time.
func TestCallCostNeverReachesThePRTotal(t *testing.T) {
	s := testStore(t)
	ctx := context.Background()
	id := upsert(t, s, key, "a")
	run(t, s, id, "a", nano(600), "m")
	if _, err := s.InsertAttempt(ctx, Attempt{Key: key, TeamSlug: "platform", Kind: RunAuto, Outcome: "ok",
		Call: &LLMCall{Profile: "m", CostNanoUSD: nano(600)}}); err != nil {
		t.Fatal(err)
	}
	if c, _ := s.PRCost(ctx, key); c == nil || *c != 600 {
		t.Errorf("PRCost = %v, want 600 (the run only)", c)
	}
}

// A call row the database would refuse (gateway header bytes that are not
// UTF-8, a NUL) is cleaned up and stored; and a refused call write never
// takes the attempt, or the billed cost on it, down with it.
func TestCallWriteNeverCostsTheAttempt(t *testing.T) {
	s := testStore(t)
	ctx := context.Background()
	upsert(t, s, key, "a")
	id, err := s.InsertAttempt(ctx, Attempt{Key: key, TeamSlug: "platform", Kind: RunAuto, Outcome: "unparseable",
		CostNanoUSD: nano(400), Call: &LLMCall{Profile: "p", Deployment: "azure/\xff\xfe-model", CallID: "a\x00b",
			Warnings: []string{"bad \xff"}}})
	if err != nil {
		t.Fatalf("a dirty header must be stored, not refused: %v", err)
	}
	got, ok, err := s.AttemptCall(ctx, id)
	if err != nil || !ok || got.CallID != "ab" || got.Deployment != "azure/�-model" {
		t.Errorf("stored call = %+v, %v, %v", got, ok, err)
	}

	// The unique index refuses a second call for one attempt. Here the
	// attempt is written first, so its cost survives the refusal.
	if _, err := s.pool.Exec(ctx, `INSERT INTO llm_calls (attempt_id, team_slug, project_key, repo_slug, pr_id, kind, profile)
		VALUES ($1, 'platform', 'PLAT', 'api', 7, 'review', 'p')`, id); err == nil {
		t.Error("two calls for one attempt were accepted")
	}
	if c, _ := s.PRCost(ctx, key); c == nil || *c != 400 {
		t.Errorf("PRCost = %v, want the attempt's 400", c)
	}

	// The call write itself failing (here: a constraint no new row meets;
	// the table stays, the PR total reads it) still returns the attempt's
	// id with the error, and the attempt's cost is in the PR total.
	if _, err := s.pool.Exec(ctx, `ALTER TABLE llm_calls ADD CONSTRAINT refuse_all CHECK (false) NOT VALID`); err != nil {
		t.Fatal(err)
	}
	id, err = s.InsertAttempt(ctx, Attempt{Key: key, TeamSlug: "platform", Kind: RunAuto, Outcome: "unparseable",
		CostNanoUSD: nano(100), Call: &LLMCall{Profile: "p"}})
	if !errors.Is(err, ErrCallNotRecorded) || id == 0 {
		t.Fatalf("InsertAttempt = %d, %v; want the attempt id and ErrCallNotRecorded", id, err)
	}
	if c, _ := s.PRCost(ctx, key); c == nil || *c != 500 {
		t.Errorf("PRCost = %v, want 500: the second attempt's cost survived its call", c)
	}
}

// A reply's and a question's call is the only record of it: the PR cost, the
// cap check's id and the monthly spend count it, the review call (detail
// beside its run) is never counted twice, and the page lists them newest
// first.
func TestCommentCallsCountTowardThePR(t *testing.T) {
	s := testStore(t)
	ctx := context.Background()
	prID := upsert(t, s, key, "c1")
	runID := run(t, s, prID, "c1", nano(600), "gpt-5.5")
	// The review's call carries the same 600 as its run: detail, not money.
	if _, err := s.InsertAttempt(ctx, Attempt{Key: key, TeamSlug: "platform", Kind: RunAuto, Outcome: "ok",
		RunID: &runID, Call: &LLMCall{Profile: "p", CostNanoUSD: nano(600)}}); err != nil {
		t.Fatal(err)
	}
	if err := s.InsertCommentCall(ctx, "platform", key, CallReply, 501, LLMCall{Profile: "p", Model: "m", CostNanoUSD: nano(30)}); err != nil {
		t.Fatal(err)
	}
	if err := s.InsertCommentCall(ctx, "platform", key, CallQuestion, 502, LLMCall{Profile: "p", CostNanoUSD: nano(70)}); err != nil {
		t.Fatal(err)
	}
	// Unpriced: counted as nothing, never as zero cost or a failure.
	if err := s.InsertCommentCall(ctx, "platform", key, CallQuestion, 502, LLMCall{Profile: "p"}); err != nil {
		t.Fatal(err)
	}

	id, cost, err := s.PRCapState(ctx, key)
	if err != nil || id != prID || cost == nil || *cost != 700 {
		t.Errorf("PRCapState = %d, %v, %v; want row %d, 700", id, cost, err, prID)
	}
	tot, err := s.TotalsSince(ctx, time.Now().Add(-time.Hour))
	if err != nil || tot.CostNanoUSD == nil || *tot.CostNanoUSD != 700 || tot.Runs != 1 {
		t.Errorf("totals = %+v, %v; want 700 over 1 run", tot, err)
	}
	calls, more, err := s.PRCommentCalls(ctx, key)
	if err != nil || more || len(calls) != 3 {
		t.Fatalf("PRCommentCalls = %+v, %v, %v", calls, more, err)
	}
	if calls[2].Kind != CallReply || *calls[2].CommentID != 501 || calls[2].Model != "m" || calls[0].CostNanoUSD != nil {
		t.Errorf("calls = %+v, want newest first, the reply last", calls)
	}
}

// Only a reply or a question may be a comment call, and only a review call
// hangs off an attempt: the ledger split rests on it.
func TestCommentCallKinds(t *testing.T) {
	s := testStore(t)
	ctx := context.Background()
	if err := s.InsertCommentCall(ctx, "platform", key, CallReview, 1, LLMCall{Profile: "p"}); err == nil {
		t.Error("a review call written without its attempt")
	}
	if _, err := s.pool.Exec(ctx, `INSERT INTO llm_calls (team_slug, project_key, repo_slug, pr_id, kind, profile)
		VALUES ('platform', 'P', 'r', 1, 'review', 'p')`); err == nil {
		t.Error("the database took a review call with no attempt")
	}
}
