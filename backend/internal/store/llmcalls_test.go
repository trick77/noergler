package store

import (
	"context"
	"reflect"
	"testing"
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

// A skip, a timeout and a pre-flight too-large made no call: there is
// nothing to show, and the page says so instead of rendering zeros.
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
