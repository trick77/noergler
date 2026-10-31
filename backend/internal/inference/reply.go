package inference

import (
	"context"
	"encoding/json"
	"fmt"
	"slices"
	"strings"

	"github.com/trick77/llmwire"
	"github.com/trick77/noergler/internal/lang"
)

// answer is the one call a mention and a finding reply share: the fit
// pre-flight, the guarded system message, an optional strict schema, no
// retry. A non-OK outcome carries no response.
func (c *Client) answer(ctx context.Context, prompt string, promptTokens int, format *llmwire.ResponseFormat) (*llmwire.ChatResponse, Outcome, error) {
	if err := c.tooLarge(promptTokens); err != nil {
		return nil, OutcomeTooLarge, err
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
	PlaceholderReply       = "{reply}"
	PlaceholderAuthorName  = "{author_first_name}"
	// PlaceholderDisputeSignals is the repo's open dispute signals. Only the
	// finding-reply template carries it: a review never sees them.
	PlaceholderDisputeSignals = "{dispute_signals}"
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
	// FileExcerpt is the file around the finding's line at the commit it was
	// posted at, numbered, or empty when it could not be read.
	FileExcerpt string
	// Reply is the PR author's text: intent only, never evidence.
	Reply string
	// AuthorFirstName is the author's first name, or empty when unknown.
	AuthorFirstName string
	// DisputeSignals is the rule of each open dispute signal of the repo,
	// listed as [1]..[n]; the model's matches_signal names one by number.
	DisputeSignals []string
	// Language is the author's; the zero value writes German.
	Language lang.Lang
}

// RenderDisputeSignals is the numbered list the prompt shows, one rule per
// line, "_None._" without one. A rule is one line already (CheckAgentsMDLine).
func RenderDisputeSignals(rules []string) string {
	if len(rules) == 0 {
		return "_None._"
	}
	var b strings.Builder
	for i, r := range rules {
		if i > 0 {
			b.WriteByte('\n')
		}
		fmt.Fprintf(&b, "[%d] %s", i+1, r)
	}
	return b.String()
}

// RenderFindingReplyPrompt assembles the prompt in one substitution pass,
// so a reply containing "{file_excerpt}" stays literal.
func RenderFindingReplyPrompt(req FindingReplyPromptRequest) string {
	excerpt := req.FileExcerpt
	if strings.TrimSpace(excerpt) == "" {
		excerpt = "_The file could not be read._"
	}
	authorName := req.AuthorFirstName
	if authorName == "" {
		authorName = "_Unknown._"
	}
	return strings.NewReplacer(
		PlaceholderRepoInstructions, req.RepoInstructions,
		PlaceholderFinding, req.Finding,
		PlaceholderFileExcerpt, excerpt,
		PlaceholderReply, req.Reply,
		PlaceholderAuthorName, authorName,
		PlaceholderDisputeSignals, RenderDisputeSignals(req.DisputeSignals),
		PlaceholderOutputLanguage, OutputLanguageBlock(req.Language),
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
			"required":             []any{"intent", "reply", "agents_md_line", "matches_signal"},
			"properties": map[string]any{
				"intent":         map[string]any{"type": "string", "enum": intents},
				"reply":          map[string]any{"type": "string", "minLength": 1},
				"agents_md_line": map[string]any{"type": []any{"string", "null"}},
				"matches_signal": map[string]any{"type": []any{"integer", "null"}},
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
	// MatchesSignal is the listed dispute signal's number (1-based) the
	// reply's reason is the same rule as, 0 for none. Unchecked: the caller
	// maps it to the list it sent and treats anything else as none.
	MatchesSignal int
}

// ParseFindingReply reads the model's envelope. An unknown intent reads as
// "other", so a model off the enum still gets its reply posted, but never
// records a dispute nobody made. A matches_signal that is no integer reads
// as none: it never costs the reply.
func ParseFindingReply(content string) (FindingReply, bool) {
	var raw struct {
		Intent        string          `json:"intent"`
		Reply         string          `json:"reply"`
		AgentsMDLine  *string         `json:"agents_md_line"`
		MatchesSignal json.RawMessage `json:"matches_signal"`
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
	var n int
	if len(raw.MatchesSignal) > 0 && json.Unmarshal(raw.MatchesSignal, &n) == nil && n > 0 {
		out.MatchesSignal = n
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
