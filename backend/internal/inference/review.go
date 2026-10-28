package inference

import (
	"context"
	"errors"
	"strings"

	"github.com/trick77/llmwire"
)

// Outcome says how a review call ended. Every value but OutcomeOK is a skip
// the caller turns into a posted notice rather than a silent no-review.
type Outcome int

const (
	// OutcomeOK is a response that parsed.
	OutcomeOK Outcome = iota
	// OutcomeTimedOut is the hard wall-clock cap firing.
	OutcomeTimedOut
	// OutcomeTooLarge is the prompt exceeding the model's real limit, whether
	// caught by the pre-flight check, a 413, or a 400 naming an overflow.
	OutcomeTooLarge
	// OutcomeUnparseable is an empty or non-JSON response, including a refusal.
	OutcomeUnparseable
	// OutcomeError is anything else: a dial failure, a 5xx, a 401.
	OutcomeError
)

func (o Outcome) String() string {
	switch o {
	case OutcomeOK:
		return "ok"
	case OutcomeTimedOut:
		return "timed_out"
	case OutcomeTooLarge:
		return "too_large"
	case OutcomeUnparseable:
		return "unparseable"
	default:
		return "error"
	}
}

// ReviewResult is one review call: what the model said, what it cost, and how
// it ended.
type ReviewResult struct {
	Outcome Outcome
	Review  ParsedReview
	Cost    CallCost
	// Raw is the model's content verbatim, the assistant turn the Almanach
	// note call replays so the gateway's prefix cache can serve the
	// conversation. Empty when the gateway never answered.
	Raw string
	// Err carries the underlying failure for OutcomeError, OutcomeTimedOut and
	// OutcomeTooLarge. Nil on success.
	Err error
}

// contextOverflowMarkers identify a 400 that is really an overflow. The
// pre-flight fit check catches almost every case; this is the backstop for a
// model whose real limit is below its advertised window.
var contextOverflowMarkers = []string{
	"context length",
	"context window",
	"context_length_exceeded",
	"context_window_exceeded",
}

// ReviewRequest is one review call's inputs. The prompt is already assembled.
type ReviewRequest struct {
	Prompt string
	// PromptTokens is the caller's count of Prompt, used for the pre-flight
	// fit check. Zero skips the check.
	PromptTokens int
	// ResponseSchema, when set, is sent as a strict JSON schema.
	ResponseSchema *llmwire.ResponseFormat
	// Images are the screenshot pairs to describe, sent in a second user
	// message after the prompt so the prompt's prefix stays cacheable. Sent
	// only to a model whose profile has vision; PromptTokens includes them.
	Images []VisualImage
}

// VisualImage is one changed screenshot, cut to what changed.
type VisualImage struct {
	// ID is what the model answers under in visual_changes: "V1", "V2".
	ID   string
	Path string
	// Before and After are data: URIs; Before is "" for an added screenshot.
	Before, After string
	// Cropped: both show the changed region only, not the whole screen.
	Cropped bool
}

// visualMessage is the user turn carrying the screenshots: per pair a label,
// then the before and after image.
func visualMessage(images []VisualImage) llmwire.Message {
	parts := []llmwire.Part{{Kind: llmwire.PartText, Text: VisualIntro}}
	for _, img := range images {
		parts = append(parts, llmwire.Part{Kind: llmwire.PartText, Text: VisualLabel(img)})
		if img.Before != "" {
			parts = append(parts, llmwire.Part{Kind: llmwire.PartImage, URL: img.Before})
		}
		parts = append(parts, llmwire.Part{Kind: llmwire.PartImage, URL: img.After})
	}
	return llmwire.Message{Role: llmwire.RoleUser, Parts: parts}
}

// VisualIntro heads the screenshot turn. The rules are in review.txt; this
// only says what follows. Exported so the caller counts it into the fit
// check with the labels.
const VisualIntro = "Changed test screenshots of this PR, for `visual_changes`. Answer each id once."

// VisualLabel is the text before one pair's images.
func VisualLabel(img VisualImage) string {
	label := img.ID + ": `" + img.Path + "`"
	switch {
	case img.Before == "":
		return label + " (added; one image)"
	case img.Cropped:
		return label + " (before, then after; cropped to the changed region)"
	}
	return label + " (before, then after)"
}

// Vision reports whether the model's profile takes images, as read at
// startup. Asked of the profile, never probed.
func (c *Client) Vision() bool { return c.vision }

// Review runs one review call.
//
// The pre-flight check refuses a prompt that cannot fit before spending
// anything: the whole PR is reviewed in one call, so an overflow is a skip,
// not a partial review.
func (c *Client) Review(ctx context.Context, req ReviewRequest) ReviewResult {
	// Against the fit ceiling, not the compression budget: the budget is what
	// compression aims at, the ceiling is what will not fit. Needs a resolved
	// window, or the unresolved 0 would reject everything.
	if req.PromptTokens > 0 && c.Ready() {
		if ceiling := c.FitCeiling(); req.PromptTokens > ceiling {
			return ReviewResult{
				Outcome: OutcomeTooLarge,
				Review:  ParsedReview{Summary: NewReviewSummary()},
				Err: &TooLargeError{
					PromptTokens: req.PromptTokens,
					Ceiling:      ceiling,
				},
			}
		}
	}

	chat := llmwire.ChatRequest{
		Model:     c.model,
		Reasoning: c.reasoning(),
		Messages: []llmwire.Message{
			llmwire.System(ReviewSystemMessage),
			llmwire.User(req.Prompt),
		},
	}
	if req.ResponseSchema != nil {
		chat.ResponseFormat = *req.ResponseSchema
	}
	// Never to a model without vision: llmwire refuses the whole request,
	// and the caller has already said "not compared" for these.
	if len(req.Images) > 0 && c.vision {
		chat.Messages = append(chat.Messages, visualMessage(req.Images))
	}

	resp, warnings, err := c.wire.Chat(ctx, chat)
	if err != nil {
		return ReviewResult{
			Outcome: classifyCallError(ctx, err),
			Review:  ParsedReview{Summary: NewReviewSummary()},
			Err:     err,
		}
	}

	out := ReviewResult{Review: ParseReview(resp.Content), Cost: CostFrom(resp), Raw: resp.Content}
	out.Cost.Warnings = WarningLines(warnings)
	// The parser is pure, so its operator-facing lines surface here, where the
	// caller's context carries team and pr_tag. Emitting them inside the
	// parser would leave them unbound.
	for _, d := range out.Review.Diagnostics {
		c.log.Log(ctx, d.Level, d.Message)
	}
	if out.Review.ParseFailed {
		out.Outcome = OutcomeUnparseable
	}
	return out
}

// MentionRequest is one mention call's inputs.
type MentionRequest struct {
	Prompt       string
	PromptTokens int
	// Tools the model may call. Ignored on a model whose profile has no
	// tool support: the answer is then one plain call, as before.
	Tools []MentionTool
}

// MentionResult is one mention, one call or a tool loop of several.
type MentionResult struct {
	Outcome Outcome
	// Answer is the rendered markdown, empty unless the outcome is OK.
	Answer string
	// Cost is the whole answer's, SumCosts of Rounds.
	Cost CallCost
	// Rounds is each call's own cost, one per round, for the per-call log.
	Rounds []CallCost
	// ToolCalls names each tool run, in order; CappedCalls counts the calls
	// past MaxToolCallsPerTurn that were refused.
	ToolCalls   []string
	CappedCalls int
	// ToolOverflow is OutcomeTooLarge caused by the tool history, not by
	// the PR: the first round fitted.
	ToolOverflow bool
	// ToolsOff is why the tools were not offered, "" when they were or the
	// request had none: a model without them answers settings questions
	// with "cannot see", so the caller logs it.
	ToolsOff string
	Err      error
}

// Mention answers a developer's question about a PR.
//
// Unlike a review there is no schema and no unparseable outcome: the envelope
// parser falls back to the raw text, so a model that ignores the envelope
// still produces a usable reply.
func (c *Client) Mention(ctx context.Context, req MentionRequest) MentionResult {
	toolsOff := ""
	switch {
	case len(req.Tools) == 0:
	case !c.tools:
		toolsOff = "the model's profile has no tool support"
	case c.Ready() && req.PromptTokens+ToolHeadroomTokens > c.FitCeiling():
		toolsOff = "no room: " + itoa(req.PromptTokens) + "+" + itoa(ToolHeadroomTokens) +
			" tokens exceed the fit ceiling of " + itoa(c.FitCeiling())
	default:
		return c.mentionWithTools(ctx, req)
	}
	resp, outcome, err := c.answer(ctx, req.Prompt, req.PromptTokens, nil)
	if outcome != OutcomeOK {
		return MentionResult{Outcome: outcome, ToolsOff: toolsOff, Err: err}
	}
	cost := CostFrom(resp)
	return MentionResult{Answer: ParseMention(resp.Content), Cost: cost, Rounds: []CallCost{cost}, ToolsOff: toolsOff}
}

// Tools reports whether the model's profile takes tool calls, as read at
// startup. Asked of the profile, never probed.
func (c *Client) Tools() bool { return c.tools }

// TooLargeError is the pre-flight refusal: the assembled prompt exceeds the
// context window less the reply reserve.
type TooLargeError struct {
	PromptTokens int
	Ceiling      int
}

func (e *TooLargeError) Error() string {
	return "prompt of " + itoa(e.PromptTokens) + " tokens exceeds the fit ceiling of " + itoa(e.Ceiling)
}

// classifyCallError maps a failed call to an outcome.
//
// llmwire applies CallTimeout to a context it derives itself, so the caller's
// ctx.Err() stays nil and its sentinels are plain errors that never unwrap to
// context.DeadlineExceeded. A non-streaming Chat arms the stall guard with the
// call cap under stallHeaders, so the 300s cap surfaces as ErrNoResponseHeaders
// rather than ErrCallCap. Both are the model being too slow and both must read
// as a timeout, or a hung model is reported to the operator as a generic error.
func classifyCallError(ctx context.Context, err error) Outcome {
	if isOverflow(err) {
		return OutcomeTooLarge
	}
	if errors.Is(err, llmwire.ErrCallCap) || errors.Is(err, llmwire.ErrNoResponseHeaders) {
		return OutcomeTimedOut
	}
	// A caller's own deadline counts too: the review queue bounds a job.
	if errors.Is(err, context.DeadlineExceeded) || errors.Is(ctx.Err(), context.DeadlineExceeded) {
		return OutcomeTimedOut
	}
	return OutcomeError
}

// isOverflow reports whether an error is a context-window overflow: a 413, or
// a 400 whose body names one of the markers.
func isOverflow(err error) bool {
	var api *llmwire.APIError
	if !errors.As(err, &api) {
		return false
	}
	if api.StatusCode == 413 {
		return true
	}
	if api.StatusCode != 400 {
		return false
	}
	body := strings.ToLower(api.Message + " " + api.Code + " " + api.Type + " " + api.Param)
	for _, marker := range contextOverflowMarkers {
		if strings.Contains(body, marker) {
			return true
		}
	}
	return false
}
