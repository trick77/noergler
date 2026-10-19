package inference

import (
	"context"
	"encoding/json"
	"strconv"
	"strings"
	"testing"

	"github.com/trick77/llmwire/llmwiretest"
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
	if runs != 1 || f.gotChat != 2 || len(got.Rounds) != 2 || got.ToolsOff != "" {
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

// On the Responses protocol the second round replays the assistant turn with
// its reasoning item: a function_call sent back without it is refused.
func TestMentionToolLoopReplaysReasoning(t *testing.T) {
	const alias = "gateway-alias"
	f := &fakeGateway{chatBodies: []string{
		responsesBody(`{"type":"function_call","id":"fc_0","call_id":"call_0","name":"team_config","arguments":"{}"}`),
		responsesBody(`{"type":"message","role":"assistant","content":[{"type":"output_text","text":"{\"answer\": \"5 dollars\"}"}]}`),
	}}
	srv := f.start(t, alias)
	c := newTestClient(t, srv, alias, func(o *Options) {
		o.Model = llmwiretest.ResponsesModel
		o.ContextWindow = 1_000_000
		o.Env = env(srv.URL, alias, map[string]string{"LLMWIRE_LITELLM_MODELS": llmwiretest.ResponsesModel + "=" + alias})
	})
	c.tools = true
	runs := 0

	got := c.Mention(context.Background(), MentionRequest{Prompt: "cap?", Tools: []MentionTool{countingTool("team_config", &runs)}})

	if got.Outcome != OutcomeOK || got.Answer != "5 dollars" || runs != 1 || f.gotChat != 2 {
		t.Fatalf("got %+v (err %v), runs = %d, calls = %d", got, got.Err, runs, f.gotChat)
	}
	var body map[string]any
	if err := json.Unmarshal(f.chatRequests[1], &body); err != nil {
		t.Fatal(err)
	}
	if !(llmwiretest.Request{Body: body}).ReplayedReasoning() {
		t.Errorf("second request input = %v, want the reasoning item replayed", body["input"])
	}
	// The call goes back with its result, paired by call_id, never the item id.
	var call, output bool
	input, _ := body["input"].([]any)
	for _, raw := range input {
		item, _ := raw.(map[string]any)
		call = call || item["type"] == "function_call" && item["call_id"] == "call_0"
		output = output || item["type"] == "function_call_output" && item["call_id"] == "call_0"
	}
	if !call || !output {
		t.Errorf("second request input = %v, want the call and its output under call_0", input)
	}
	// Without it Azure sends the reasoning back unencrypted and it cannot be replayed.
	if inc, _ := body["include"].([]any); len(inc) == 0 || inc[0] != "reasoning.encrypted_content" {
		t.Errorf("include = %v, want reasoning.encrypted_content", body["include"])
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
	if got.ToolsOff != "the model's profile has no tool support" {
		t.Errorf("ToolsOff = %q", got.ToolsOff)
	}
}

// A prompt near the ceiling has no room for the loop's growth: one plain
// call, not a refusal in round two.
func TestMentionNearTheCeilingSkipsTheTools(t *testing.T) {
	f := &fakeGateway{chatBody: chatBody(`{"answer": "hi"}`)}
	c := toolClient(t, f)
	runs := 0

	got := c.Mention(context.Background(), MentionRequest{Prompt: "x", PromptTokens: c.FitCeiling() - ToolHeadroomTokens/2,
		Tools: []MentionTool{countingTool("t", &runs)}})

	if got.Outcome != OutcomeOK || strings.Contains(string(f.lastChatBody), `"tools"`) {
		t.Errorf("outcome %v, body %s", got.Outcome, f.lastChatBody)
	}
	if !strings.HasPrefix(got.ToolsOff, "no room: ") {
		t.Errorf("ToolsOff = %q", got.ToolsOff)
	}
}

// A failed later round keeps the billed rounds' total.
func TestMentionToolLoopErrorKeepsTheCost(t *testing.T) {
	f := &fakeGateway{costHeader: "0.001", chatBodies: []string{toolCallBody("t"), `not json`}}
	c := toolClient(t, f)
	runs := 0

	got := c.Mention(context.Background(), MentionRequest{Prompt: "x", Tools: []MentionTool{countingTool("t", &runs)}})

	if got.Outcome == OutcomeOK || len(got.Rounds) != 1 || got.Cost.NanoUSD == nil || *got.Cost.NanoUSD != 1_000_000 {
		t.Errorf("outcome %v, rounds %d, cost %v", got.Outcome, len(got.Rounds), got.Cost.NanoUSD)
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
