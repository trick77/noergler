package inference

import (
	"context"
	"encoding/json"
	"testing"

	"github.com/trick77/llmwire"
)

// sentMessages decodes the messages of the last completion request.
func sentMessages(t *testing.T, body []byte) []map[string]any {
	t.Helper()
	var sent struct {
		Messages []map[string]any `json:"messages"`
	}
	if err := json.Unmarshal(body, &sent); err != nil {
		t.Fatal(err)
	}
	return sent.Messages
}

// The screenshots ride in their own user turn after the prompt, which stays
// the plain text it was: the gateway's prefix cache keys on it.
func TestReviewSendsScreenshotsAfterThePrompt(t *testing.T) {
	f := &fakeGateway{chatBody: chatBody(`{"overview": "x", "verdict": {"decision": "approve"}}`)}
	c := reviewClient(t, f)
	c.vision = true

	got := c.Review(context.Background(), ReviewRequest{Prompt: "review this", Images: []VisualImage{
		{ID: "V1", Path: "a-snapshots/x.png", Before: "data:image/png;base64,AA", After: "data:image/png;base64,BB", Cropped: true},
		{ID: "V2", Path: "a-snapshots/new.png", After: "data:image/png;base64,CC"},
	}})
	if got.Outcome != OutcomeOK {
		t.Fatalf("Outcome = %v (%v)", got.Outcome, got.Err)
	}
	msgs := sentMessages(t, f.lastChatBody)
	if len(msgs) != 3 {
		t.Fatalf("messages = %d, want system, prompt, screenshots", len(msgs))
	}
	if msgs[1]["content"] != "review this" {
		t.Errorf("the prompt turn changed: %v", msgs[1]["content"])
	}
	parts, ok := msgs[2]["content"].([]any)
	if !ok {
		t.Fatalf("screenshot turn is not parts: %v", msgs[2]["content"])
	}
	images := 0
	for _, p := range parts {
		if p.(map[string]any)["type"] == string(llmwire.PartImage) {
			images++
		}
	}
	// Before and after for V1, the after alone for the added V2.
	if images != 3 {
		t.Errorf("image parts = %d, want 3", images)
	}
}

// A model without vision gets no images: llmwire would refuse the request.
func TestReviewSendsNoScreenshotsWithoutVision(t *testing.T) {
	f := &fakeGateway{chatBody: chatBody(`{"overview": "x", "verdict": {"decision": "approve"}}`)}
	c := reviewClient(t, f)

	got := c.Review(context.Background(), ReviewRequest{Prompt: "review this", Images: []VisualImage{
		{ID: "V1", Path: "a-snapshots/x.png", After: "data:image/png;base64,BB"},
	}})
	if got.Outcome != OutcomeOK {
		t.Fatalf("Outcome = %v (%v)", got.Outcome, got.Err)
	}
	if n := len(sentMessages(t, f.lastChatBody)); n != 2 {
		t.Errorf("messages = %d, want system and prompt only", n)
	}
}

func TestParseVisualChanges(t *testing.T) {
	got := ParseReview(`{"overview": "x", "visual_changes": [` +
		`{"id": " V1 ", "change": " Button moved below the form; explained by login.css. "},` +
		`{"id": "V2", "change": "  "}, {"change": "no id"}, {"id": 3, "change": null}]}`)
	if len(got.Visual) != 1 || got.Visual[0] != (VisualChange{ID: "V1", Change: "Button moved below the form; explained by login.css."}) {
		t.Errorf("visual = %+v", got.Visual)
	}
	if none := ParseReview(`{"overview": "x", "visual_changes": null}`); none.Visual != nil {
		t.Errorf("null visual_changes = %+v, want none", none.Visual)
	}
}
