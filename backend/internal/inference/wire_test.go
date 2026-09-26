package inference

import (
	"context"
	"encoding/json"
	"testing"

	"github.com/trick77/llmwire"
	"github.com/trick77/llmwire/llmwiretest"
)

// wireBody is the shape noergler cares about in an outgoing request.
type wireBody struct {
	Model    string `json:"model"`
	Messages []struct {
		Role    string `json:"role"`
		Content any    `json:"content"`
	} `json:"messages"`
	ResponseFormat json.RawMessage `json:"response_format"`
	Stream         *bool           `json:"stream"`
}

// The prompt is the only thing the model sees, so what reaches the wire is
// asserted rather than assumed: the guardrails must be in the system role,
// where untrusted PR content cannot override them.
func TestReviewRequestOnTheWire(t *testing.T) {
	f := &fakeGateway{chatBody: chatBody(`{"overview": "ok"}`)}
	c := reviewClient(t, f)

	got := c.Review(context.Background(), ReviewRequest{Prompt: "PLEASE REVIEW"})
	if got.Outcome != OutcomeOK {
		t.Fatalf("Outcome = %v (err: %v)", got.Outcome, got.Err)
	}

	var body wireBody
	if err := json.Unmarshal(f.lastChatBody, &body); err != nil {
		t.Fatalf("decoding the captured request: %v", err)
	}

	// The gateway alias goes on the wire, not the profile id.
	if body.Model != "gateway-alias" {
		t.Errorf("model = %q, want the gateway alias", body.Model)
	}
	if len(body.Messages) != 2 {
		t.Fatalf("got %d messages, want system + user", len(body.Messages))
	}
	if body.Messages[0].Role != "system" {
		t.Errorf("first message role = %q, want system", body.Messages[0].Role)
	}
	if s, _ := body.Messages[0].Content.(string); s != ReviewSystemMessage {
		t.Error("the system message must be the guardrail constant verbatim")
	}
	if body.Messages[1].Role != "user" {
		t.Errorf("second message role = %q, want user", body.Messages[1].Role)
	}
	if s, _ := body.Messages[1].Content.(string); s != "PLEASE REVIEW" {
		t.Errorf("user content = %q, want the prompt", s)
	}
	// Code review is analysis a reader keeps: never the minimal setting,
	// which on some models is thinking switched off.
	if got := sentReasoning(t, f.lastChatBody); got != llmwiretest.BalancedSent {
		t.Errorf("reasoning = %q, want the balanced level %q", got, llmwiretest.BalancedSent)
	}
	// Chat only, never streaming: a LiteLLM stream carries no cost header.
	if body.Stream != nil && *body.Stream {
		t.Error("the request must not ask for a stream")
	}
}

// A response schema, when the caller supplies one, must reach the wire: it is
// what makes the model answer parseable JSON.
func TestReviewSchemaOnTheWire(t *testing.T) {
	f := &fakeGateway{chatBody: chatBody(`{"overview": "ok"}`)}
	c := reviewClient(t, f)

	schema := llmwire.ResponseFormat{
		Kind: llmwire.FormatJSONSchema,
		Name: "review_response",
		Schema: map[string]any{
			"type":                 "object",
			"properties":           map[string]any{"overview": map[string]any{"type": "string"}},
			"required":             []any{"overview"},
			"additionalProperties": false,
		},
		Strict: true,
	}

	got := c.Review(context.Background(), ReviewRequest{Prompt: "x", ResponseSchema: &schema})
	if got.Outcome != OutcomeOK {
		t.Fatalf("Outcome = %v (err: %v)", got.Outcome, got.Err)
	}

	var body wireBody
	if err := json.Unmarshal(f.lastChatBody, &body); err != nil {
		t.Fatalf("decoding the captured request: %v", err)
	}
	if len(body.ResponseFormat) == 0 {
		t.Fatal("response_format is absent from the request")
	}
	var rf struct {
		Type       string `json:"type"`
		JSONSchema struct {
			Name   string `json:"name"`
			Strict *bool  `json:"strict"`
		} `json:"json_schema"`
	}
	if err := json.Unmarshal(body.ResponseFormat, &rf); err != nil {
		t.Fatalf("decoding response_format: %v", err)
	}
	if rf.Type != "json_schema" {
		t.Errorf("response_format type = %q, want json_schema", rf.Type)
	}
	if rf.JSONSchema.Name != "review_response" {
		t.Errorf("schema name = %q, want review_response", rf.JSONSchema.Name)
	}
	if rf.JSONSchema.Strict == nil || !*rf.JSONSchema.Strict {
		t.Error("the schema must be strict")
	}
}

// A mention carries its own system message, not the review one.
func TestMentionRequestOnTheWire(t *testing.T) {
	f := &fakeGateway{chatBody: chatBody(`{"answer": "hi"}`)}
	c := reviewClient(t, f)

	if got := c.Mention(context.Background(), MentionRequest{Prompt: "why?"}); got.Outcome != OutcomeOK {
		t.Fatalf("Outcome = %v (err: %v)", got.Outcome, got.Err)
	}

	var body wireBody
	if err := json.Unmarshal(f.lastChatBody, &body); err != nil {
		t.Fatalf("decoding the captured request: %v", err)
	}
	if len(body.Messages) != 2 {
		t.Fatalf("got %d messages, want system + user", len(body.Messages))
	}
	if got := sentReasoning(t, f.lastChatBody); got != llmwiretest.BalancedSent {
		t.Errorf("reasoning = %q, want the balanced level %q", got, llmwiretest.BalancedSent)
	}
	if s, _ := body.Messages[0].Content.(string); s != MentionSystemMessage {
		t.Error("a mention must carry the mention system message, not the review one")
	}
	// No schema: the envelope parser falls back to raw text instead.
	if len(body.ResponseFormat) != 0 {
		t.Error("a mention must not send a response schema")
	}
}
