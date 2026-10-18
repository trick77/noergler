package inference

import (
	"context"
	"encoding/json"
	"strconv"
	"strings"
	"testing"
)

// toolCallBody is a completion asking for the named tools, one call each.
func toolCallBody(names ...string) string {
	var calls []string
	for i, n := range names {
		calls = append(calls, `{"id":"call_`+strconv.Itoa(i)+`","type":"function","function":{"name":"`+n+`","arguments":"{}"}}`)
	}
	return `{"choices":[{"message":{"role":"assistant","content":"","tool_calls":[` + strings.Join(calls, ",") + `]},` +
		`"finish_reason":"tool_calls"}],"usage":{"prompt_tokens":10,"completion_tokens":2,"total_tokens":12}}`
}

func countingTool(name string, runs *int) MentionTool {
	return MentionTool{Name: name, Description: "d", Run: func(context.Context, string) string {
		*runs++
		return `{"ok":true}`
	}}
}

func toolClient(t *testing.T, f *fakeGateway) *Client {
	t.Helper()
	c := reviewClient(t, f)
	c.tools = true
	return c
}

func TestMentionToolLoop(t *testing.T) {
	f := &fakeGateway{costHeader: "0.001", chatBodies: []string{toolCallBody("team_config"), chatBody(`{"answer": "5 dollars"}`)}}
	c := toolClient(t, f)
	runs := 0

	got := c.Mention(context.Background(), MentionRequest{Prompt: "cap?", Tools: []MentionTool{countingTool("team_config", &runs)}})

	if got.Outcome != OutcomeOK || got.Answer != "5 dollars" {
		t.Fatalf("got %+v (err %v)", got, got.Err)
	}
	if runs != 1 || f.gotChat != 2 || len(got.Rounds) != 2 {
		t.Errorf("runs = %d, calls = %d, rounds = %d, want 1, 2, 2", runs, f.gotChat, len(got.Rounds))
	}
	// The second request replays the call and carries its result.
	var body struct {
		Messages []struct {
			Role       string `json:"role"`
			ToolCallID string `json:"tool_call_id"`
		} `json:"messages"`
		Tools []any `json:"tools"`
	}
	if err := json.Unmarshal(f.chatRequests[1], &body); err != nil {
		t.Fatal(err)
	}
	if n := len(body.Messages); n != 4 || body.Messages[2].Role != "assistant" || body.Messages[3].Role != "tool" ||
		body.Messages[3].ToolCallID != "call_0" {
		t.Errorf("second request messages = %+v", body.Messages)
	}
	if got.Cost.NanoUSD == nil || *got.Cost.NanoUSD != 2_000_000 {
		t.Errorf("summed cost = %v, want both rounds", got.Cost.NanoUSD)
	}
	if got.Cost.PromptTokens != 20 {
		t.Errorf("summed prompt tokens = %d", got.Cost.PromptTokens)
	}
}

// A model that keeps calling gets MaxToolRounds rounds, then one asking for
// no tool, which forces the answer. The tools stay defined: a history with
// tool calls and no tools is refused by some routes.
func TestMentionToolRoundsAreCapped(t *testing.T) {
	f := &fakeGateway{chatBodies: []string{toolCallBody("t"), toolCallBody("t"), toolCallBody("t"), toolCallBody("t")}}
	c := toolClient(t, f)
	runs := 0

	got := c.Mention(context.Background(), MentionRequest{Prompt: "x", Tools: []MentionTool{countingTool("t", &runs)}})

	if f.gotChat != MaxToolRounds+1 || runs != MaxToolRounds {
		t.Errorf("calls = %d, runs = %d, want %d, %d", f.gotChat, runs, MaxToolRounds+1, MaxToolRounds)
	}
	var last struct {
		Tools      []any `json:"tools"`
		ToolChoice any   `json:"tool_choice"`
	}
	_ = json.Unmarshal(f.chatRequests[len(f.chatRequests)-1], &last)
	if len(last.Tools) == 0 || last.ToolChoice != "none" {
		t.Errorf("last round tools = %d, tool_choice = %v, want the tools with none", len(last.Tools), last.ToolChoice)
	}
	if got.Outcome != OutcomeOK {
		t.Errorf("outcome = %v", got.Outcome)
	}
}

// Calls past the per-turn cap still get a result (every call id needs one),
// but the tool does not run; a repeated call runs once.
func TestMentionToolCallsPerTurnAreCapped(t *testing.T) {
	f := &fakeGateway{chatBodies: []string{toolCallBody("a", "a", "b", "c", "d", "e"), chatBody(`{"answer": "done"}`)}}
	c := toolClient(t, f)
	counts := map[string]*int{}
	var tools []MentionTool
	for _, n := range []string{"a", "b", "c", "d", "e"} {
		counts[n] = new(int)
		tools = append(tools, countingTool(n, counts[n]))
	}

	got := c.Mention(context.Background(), MentionRequest{Prompt: "x", Tools: tools})

	if *counts["a"] != 1 || *counts["b"] != 1 || *counts["c"] != 1 || *counts["d"] != 0 || *counts["e"] != 0 {
		t.Errorf("runs a=%d b=%d c=%d d=%d e=%d", *counts["a"], *counts["b"], *counts["c"], *counts["d"], *counts["e"])
	}
	if got.CappedCalls != 2 {
		t.Errorf("capped = %d, want 2", got.CappedCalls)
	}
	var body struct {
		Messages []struct {
			Role string `json:"role"`
		} `json:"messages"`
	}
	_ = json.Unmarshal(f.chatRequests[1], &body)
	tool := 0
	for _, m := range body.Messages {
		if m.Role == "tool" {
			tool++
		}
	}
	if tool != 6 {
		t.Errorf("tool results = %d, want one per call (6)", tool)
	}
}

// A model without tool support answers as before: one call, no tools.
func TestMentionWithoutToolSupportIsOneCall(t *testing.T) {
	f := &fakeGateway{chatBody: chatBody(`{"answer": "hi"}`)}
	c := reviewClient(t, f)
	runs := 0

	got := c.Mention(context.Background(), MentionRequest{Prompt: "x", Tools: []MentionTool{countingTool("t", &runs)}})

	if got.Answer != "hi" || f.gotChat != 1 || runs != 0 || strings.Contains(string(f.lastChatBody), `"tools"`) {
		t.Errorf("answer %q, calls %d, runs %d, body %s", got.Answer, f.gotChat, runs, f.lastChatBody)
	}
}

func TestSumCostsUnpricedIfAnyIs(t *testing.T) {
	one, two := int64(1), int64(2)
	if c := SumCosts([]CallCost{{NanoUSD: &one}, {NanoUSD: &two}}); c.NanoUSD == nil || *c.NanoUSD != 3 {
		t.Errorf("priced sum = %v", c.NanoUSD)
	}
	if c := SumCosts([]CallCost{{NanoUSD: &one}, {}}); c.NanoUSD != nil {
		t.Errorf("a partial sum must be unpriced, got %d", *c.NanoUSD)
	}
	spend1, spend2 := int64(100), int64(103)
	if c := SumCosts([]CallCost{{KeySpendNanoUSD: &spend1}, {KeySpendNanoUSD: &spend2}}); *c.KeySpendNanoUSD != 103 {
		t.Errorf("key spend is a gauge: the last reading, got %d", *c.KeySpendNanoUSD)
	}
}
