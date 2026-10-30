package inference

import (
	"context"
	"strings"

	"github.com/trick77/llmwire"
)

// MentionTool is a function the model may call while answering a mention.
// The tools are the caller's; inference only runs the loop.
//
// Run answers one call from the model's raw JSON arguments. It must never
// call Bitbucket (the loop runs off the worker) and must return only what a
// commenter may read: an explicit allowlist, never a secret or a disable
// reason (what is allowed is the caller's decision, see INVARIANTS).
type MentionTool struct {
	Name        string
	Description string
	// Parameters is the JSON schema of the arguments; nil takes none.
	Parameters map[string]any
	Run        func(ctx context.Context, arguments string) string
}

// Loop limits. A tool round is not a retry: each sends what the last one
// asked for. The caps bound the cost a question can run up: MiMo answered
// one round with 38 parallel calls and was cut off mid-JSON.
const (
	MaxToolRounds       = 3
	MaxToolCallsPerTurn = 4
)

// ToolHeadroomTokens is what a tool loop may add to the prompt: the tool
// schemas, the assistant's calls and the JSON results, a few hundred tokens
// a round. A prompt without this much room under the fit ceiling is
// answered in one plain call, rather than refused mid-loop as too large.
const ToolHeadroomTokens = 4_000

// toolCapResult is what a call past MaxToolCallsPerTurn gets: every call id
// needs a tool message, or the next request is refused.
const toolCapResult = `{"error": "too many tool calls in one turn; answer with what you have"}`

// mentionWithTools is Mention's loop: up to MaxToolRounds rounds with the
// tools on offer, then one asking for none, which forces a text answer. Each round
// is a full Chat with the whole history; there is no streaming.
func (c *Client) mentionWithTools(ctx context.Context, req MentionRequest) MentionResult {
	if err := c.tooLarge(req.PromptTokens); err != nil {
		return MentionResult{Outcome: OutcomeTooLarge, Err: err}
	}
	tools := make([]llmwire.Tool, len(req.Tools))
	byName := make(map[string]MentionTool, len(req.Tools))
	for i, t := range req.Tools {
		params := t.Parameters
		if params == nil {
			params = map[string]any{"type": "object", "properties": map[string]any{}, "additionalProperties": false}
		}
		tools[i] = llmwire.Tool{Name: t.Name, Description: t.Description, Parameters: params}
		byName[t.Name] = t
	}
	messages := []llmwire.Message{llmwire.System(MentionSystemMessage), llmwire.User(req.Prompt)}

	var res MentionResult
	for round := 0; ; round++ {
		// The tools stay defined in every round: a history holding tool
		// calls without them is refused by some routes (Anthropic, Bedrock).
		// The last round asks for none instead; a profile that only takes
		// "auto" relaxes that, and a call it makes anyway is not run.
		final := round == MaxToolRounds
		chat := llmwire.ChatRequest{Model: c.model, Reasoning: c.reasoning(), Messages: messages, Tools: tools,
			ToolChoice: llmwire.ToolChoice{Mode: llmwire.ToolChoiceAuto}}
		if final {
			chat.ToolChoice = llmwire.ToolChoice{Mode: llmwire.ToolChoiceNone}
		}
		resp, _, err := c.wire.Chat(ctx, chat)
		if err != nil {
			// The rounds before were billed: their total stands.
			res.Outcome, res.Err, res.Cost = classifyCallError(ctx, err), err, SumCosts(res.Rounds)
			// Past the first round the PR fitted: what overflowed is the
			// tool history, and "the PR is too large" would be untrue.
			res.ToolOverflow = res.Outcome == OutcomeTooLarge && round > 0
			return res
		}
		res.Rounds = append(res.Rounds, CostFrom(resp))
		if len(resp.ToolCalls) == 0 || final {
			res.Outcome = OutcomeOK
			// A stray </think> was seen after a multi-turn tool history
			// (llmwire FINDINGS §2): it is markup, never the answer.
			res.Answer = ParseMention(strings.ReplaceAll(resp.Content, "</think>", ""))
			res.Cost = SumCosts(res.Rounds)
			return res
		}
		// The turn as the model sent it: on the Responses protocol it carries
		// the reasoning item, and a rebuilt turn without it is refused.
		messages = append(messages, resp.Assistant())
		answered := map[string]string{}
		for i, call := range resp.ToolCalls {
			result := toolCapResult
			if i < MaxToolCallsPerTurn {
				key := call.Name + "\x00" + call.Arguments
				if prior, ok := answered[key]; ok {
					result = prior
				} else if t, ok := byName[call.Name]; ok {
					result = t.Run(ctx, call.Arguments)
					answered[key] = result
				} else {
					result = `{"error": "unknown tool"}`
				}
				res.ToolCalls = append(res.ToolCalls, call.Name)
			} else {
				res.CappedCalls++
			}
			messages = append(messages, llmwire.ToolResult(call.ID, result))
		}
	}
}

// SumCosts is the cost of a sequence of calls as one figure. Unpriced if any
// call was: a partial sum would undercount and read as priced. The key spend
// is a gauge, so the last reading stands; the split is never summed.
func SumCosts(rounds []CallCost) CallCost {
	switch len(rounds) {
	case 0:
		return CallCost{}
	case 1:
		return rounds[0]
	}
	last := rounds[len(rounds)-1]
	out := CallCost{
		KeySpendNanoUSD: last.KeySpendNanoUSD,
		CallID:          last.CallID,
		GatewayVersion:  last.GatewayVersion,
	}
	var total int64
	priced := true
	for _, r := range rounds {
		out.PromptTokens += r.PromptTokens
		out.CachedTokens += r.CachedTokens
		out.CompletionTokens += r.CompletionTokens
		out.ZeroWithTokens = out.ZeroWithTokens || r.ZeroWithTokens
		if r.NanoUSD == nil {
			priced = false
			continue
		}
		total += *r.NanoUSD
	}
	if priced {
		out.NanoUSD = &total
	}
	return out
}
