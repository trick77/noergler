package inference

import (
	"context"
	"encoding/json"
	"slices"
	"strings"

	"github.com/trick77/llmwire"
)

// answer is the one call a mention and a finding reply share: the fit
// pre-flight, the guarded system message, an optional strict schema, no
// retry. A non-OK outcome carries no response.
func (c *Client) answer(ctx context.Context, prompt string, promptTokens int, format *llmwire.ResponseFormat) (*llmwire.ChatResponse, Outcome, error) {
	if promptTokens > 0 && c.Ready() {
		if ceiling := c.FitCeiling(); promptTokens > ceiling {
			return nil, OutcomeTooLarge, &TooLargeError{PromptTokens: promptTokens, Ceiling: ceiling}
		}
	}
	chat := llmwire.ChatRequest{
		Model:     c.model,
		Reasoning: c.reasoning(),
		Messages: []llmwire.Message{
			llmwire.System(MentionSystemMessage),
			llmwire.User(prompt),
		},
	}
	if format != nil {
		chat.ResponseFormat = *format
	}
	resp, _, err := c.wire.Chat(ctx, chat)
	if err != nil {
		return nil, classifyCallError(ctx, err), err
	}
	return resp, OutcomeOK, nil
}

// Placeholders of the finding-reply template, beside the shared
// {repo_instructions}.
const (
	PlaceholderFinding     = "{finding}"
	PlaceholderFileExcerpt = "{file_excerpt}"
	PlaceholderReplyBy     = "{reply_by}"
	PlaceholderReply       = "{reply}"
)

// ReplyIntents are what a reply on a finding's thread can be. "disputes"
// covers the implicit case too: "we do this in the UI" disputes the finding
// without saying it is wrong.
var ReplyIntents = []string{"disputes", "acknowledges", "question", "other"}

// FindingReplyPromptRequest is one finding-reply prompt's inputs.
type FindingReplyPromptRequest struct {
	Template         string
	RepoInstructions string
	// Finding is the finding as posted: location, severity and its comment.
	Finding string
	// FileExcerpt is the file around the finding's line at the PR head,
	// numbered, or empty when it could not be read.
	FileExcerpt string
	// ReplyBy says who wrote the reply: the PR author or someone else.
	ReplyBy string
	// Reply is the replier's text: intent only, never evidence.
	Reply string
}

// RenderFindingReplyPrompt assembles the prompt in one substitution pass,
// so a reply containing "{file_excerpt}" stays literal.
func RenderFindingReplyPrompt(req FindingReplyPromptRequest) string {
	excerpt := req.FileExcerpt
	if strings.TrimSpace(excerpt) == "" {
		excerpt = "_The file could not be read at the PR head._"
	}
	return strings.NewReplacer(
		PlaceholderRepoInstructions, req.RepoInstructions,
		PlaceholderFinding, req.Finding,
		PlaceholderFileExcerpt, excerpt,
		PlaceholderReplyBy, req.ReplyBy,
		PlaceholderReply, req.Reply,
	).Replace(req.Template)
}

// FindingReplySchemaName is the strict schema's name on the wire.
const FindingReplySchemaName = "finding_reply"

func findingReplyFormat() llmwire.ResponseFormat {
	intents := make([]any, len(ReplyIntents))
	for i, s := range ReplyIntents {
		intents[i] = s
	}
	return llmwire.ResponseFormat{
		Kind: llmwire.FormatJSONSchema,
		Name: FindingReplySchemaName,
		Schema: map[string]any{
			"type":                 "object",
			"additionalProperties": false,
			"required":             []any{"intent", "reply", "agents_md_line"},
			"properties": map[string]any{
				"intent":         map[string]any{"type": "string", "enum": intents},
				"reply":          map[string]any{"type": "string", "minLength": 1},
				"agents_md_line": map[string]any{"type": []any{"string", "null"}},
			},
		},
		Strict: true,
	}
}

// FindingReply is the model's answer to a reply on a finding's thread.
type FindingReply struct {
	Intent string
	Reply  string
	// AgentsMDLine is the suggested AGENTS.md bullet, unchecked; empty when
	// the model proposed none. CheckAgentsMDLine decides whether it is shown.
	AgentsMDLine string
}

// ParseFindingReply reads the model's envelope. An unknown intent reads as
// "other", so a model off the enum still gets its reply posted, but never
// records a dispute nobody made.
func ParseFindingReply(content string) (FindingReply, bool) {
	var raw struct {
		Intent       string  `json:"intent"`
		Reply        string  `json:"reply"`
		AgentsMDLine *string `json:"agents_md_line"`
	}
	if json.Unmarshal([]byte(stripFence(strings.TrimSpace(content))), &raw) != nil {
		return FindingReply{}, false
	}
	out := FindingReply{Intent: raw.Intent, Reply: strings.TrimSpace(raw.Reply)}
	if !slices.Contains(ReplyIntents, out.Intent) {
		out.Intent = "other"
	}
	if raw.AgentsMDLine != nil {
		out.AgentsMDLine = strings.TrimSpace(*raw.AgentsMDLine)
	}
	if out.Reply == "" {
		return FindingReply{}, false
	}
	return out, true
}

// AgentsMDLineMax bounds a suggested bullet: AGENTS.md lines are terse.
const AgentsMDLineMax = 200

// CheckAgentsMDLine is the blunt check on a suggested AGENTS.md bullet: one
// line, a "- " bullet, at most AgentsMDLineMax runes, and not already in the
// AGENTS.md the review read (case-insensitive). The reason is log
// vocabulary, empty when the line stands.
func CheckAgentsMDLine(line, agentsMD string) string {
	switch {
	case strings.ContainsAny(line, "\r\n"):
		return "multi_line"
	case !strings.HasPrefix(line, "- "):
		return "no_bullet"
	case len([]rune(line)) > AgentsMDLineMax:
		return "too_long"
	case strings.Contains(strings.ToLower(agentsMD), strings.ToLower(strings.TrimSpace(line[2:]))):
		return "already_present"
	}
	return ""
}

// FindingReplyRequest is one finding-reply call's inputs.
type FindingReplyRequest struct {
	Prompt       string
	PromptTokens int
}

// FindingReplyResult is one finding-reply call.
type FindingReplyResult struct {
	Outcome Outcome
	Reply   FindingReply
	Cost    CallCost
	Err     error
}

// FindingReply answers a reply on a finding's thread: one call, a strict
// schema, no retry, like a mention.
func (c *Client) FindingReply(ctx context.Context, req FindingReplyRequest) FindingReplyResult {
	format := findingReplyFormat()
	resp, outcome, err := c.answer(ctx, req.Prompt, req.PromptTokens, &format)
	if outcome != OutcomeOK {
		return FindingReplyResult{Outcome: outcome, Err: err}
	}
	out := FindingReplyResult{Cost: CostFrom(resp)}
	reply, ok := ParseFindingReply(resp.Content)
	if !ok {
		out.Outcome = OutcomeUnparseable
		return out
	}
	out.Reply = reply
	return out
}
