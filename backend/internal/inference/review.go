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
}

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

	resp, _, err := c.wire.Chat(ctx, chat)
	if err != nil {
		return ReviewResult{
			Outcome: classifyCallError(ctx, err),
			Review:  ParsedReview{Summary: NewReviewSummary()},
			Err:     err,
		}
	}

	out := ReviewResult{Review: ParseReview(resp.Content), Cost: CostFrom(resp)}
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
}

// MentionResult is one mention call.
type MentionResult struct {
	Outcome Outcome
	// Answer is the rendered markdown, empty unless the outcome is OK.
	Answer string
	Cost   CallCost
	Err    error
}

// Mention answers a developer's question about a PR.
//
// Unlike a review there is no schema and no unparseable outcome: the envelope
// parser falls back to the raw text, so a model that ignores the envelope
// still produces a usable reply.
func (c *Client) Mention(ctx context.Context, req MentionRequest) MentionResult {
	if req.PromptTokens > 0 && c.Ready() {
		if ceiling := c.FitCeiling(); req.PromptTokens > ceiling {
			return MentionResult{
				Outcome: OutcomeTooLarge,
				Err:     &TooLargeError{PromptTokens: req.PromptTokens, Ceiling: ceiling},
			}
		}
	}

	resp, _, err := c.wire.Chat(ctx, llmwire.ChatRequest{
		Model:     c.model,
		Reasoning: c.reasoning(),
		Messages: []llmwire.Message{
			llmwire.System(MentionSystemMessage),
			llmwire.User(req.Prompt),
		},
	})
	if err != nil {
		return MentionResult{Outcome: classifyCallError(ctx, err), Err: err}
	}
	return MentionResult{Answer: ParseMention(resp.Content), Cost: CostFrom(resp)}
}

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
