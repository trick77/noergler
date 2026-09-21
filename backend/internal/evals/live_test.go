package evals_test

import (
	"context"
	"encoding/json"
	"log/slog"
	"net/http"
	"net/http/httptest"
	"os"
	"strings"
	"testing"

	"github.com/trick77/noergler/internal/evals"
	"github.com/trick77/noergler/internal/inference"
)

// One run over the real client against a fake OpenAI-compatible endpoint.
//
// The scoring tests stub the Reviewer, so nothing there proves an eval can
// actually reach a gateway, assemble a real prompt and parse a real response.
// This pins the whole path the way cmd/evals drives it, including that the
// prompt template's placeholders are gone by the time the request is sent:
// a prompt shipped with a literal {files} in it would score whatever the
// model made of the placeholder.
func TestEvalRunReachesAGatewayAndScoresTheResponse(t *testing.T) {
	// A real llmwire profile: the registry is fixed, so an invented id is
	// rejected before any request. This is the profile cmd/evals defaults to.
	const profile, alias = "mimo-v2.5-pro", "eval-target"

	var sawPrompt string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if strings.HasSuffix(r.URL.Path, "/models") {
			w.Header().Set("Content-Type", "application/json")
			_, _ = w.Write([]byte(`{"data":[{"id":"` + alias +
				`","max_input_tokens":2000000}]}`))
			return
		}
		var body struct {
			Messages []struct {
				Content string `json:"content"`
			} `json:"messages"`
		}
		_ = json.NewDecoder(r.Body).Decode(&body)
		for _, m := range body.Messages {
			sawPrompt += m.Content
		}
		answer, _ := json.Marshal(map[string]any{
			"summary": map[string]any{"overview": "one real bug"},
			"findings": []map[string]any{{
				"file": "internal/review/summary.go", "line": 20,
				"severity": "issue",
				"comment":  "r.Ticket is nil here, so this dereference will panic",
			}},
		})
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"id":"1","object":"chat.completion","model":"` + alias +
			`","choices":[{"index":0,"finish_reason":"stop","message":{"role":"assistant","content":` +
			string(mustJSON(string(answer))) + `}}],"usage":{"prompt_tokens":10,"completion_tokens":5}}`))
	}))
	defer srv.Close()

	env := map[string]string{
		"LLMWIRE_LITELLM_MODELS":   profile + "=" + alias,
		"LLMWIRE_LITELLM_BASE_URL": srv.URL,
		"LLMWIRE_LITELLM_API_KEY":  "k",
	}
	client, err := inference.New(inference.Options{
		Model: profile, ReasoningEffort: "medium", APIKey: "k",
		Logger: slog.New(slog.NewTextHandler(discard{}, nil)),
		Env: func(n string) (string, bool) {
			v, ok := env[n]
			return v, ok
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	ctx := context.Background()
	if err := client.Startup(ctx); err != nil {
		t.Fatal(err)
	}

	cases, err := evals.LoadCorpus()
	if err != nil {
		t.Fatal(err)
	}
	var only []evals.Case
	for _, c := range cases {
		if c.Name == "nil-deref" {
			only = append(only, c)
		}
	}
	if len(only) != 1 {
		t.Fatalf("corpus has %d nil-deref case(s), want 1", len(only))
	}

	score := evals.Run(ctx, client, readTemplate(t), only, func(s string) int { return len(s) / 4 })
	if score.Seeded != 1 || score.Caught != 1 {
		t.Errorf("caught %d of %d, want 1 of 1; findings=%+v",
			score.Caught, score.Seeded, score.Results[0].Findings)
	}
	if score.Results[0].Outcome != inference.OutcomeOK.String() {
		t.Errorf("outcome = %s, want %s", score.Results[0].Outcome, inference.OutcomeOK)
	}
	// The assembled prompt must carry the case's diff and no placeholder.
	if !strings.Contains(sawPrompt, "internal/review/summary.go") {
		t.Error("the prompt the gateway saw does not contain the case's file")
	}
	for _, ph := range []string{
		inference.PlaceholderFiles, inference.PlaceholderCumulativePRDiff,
		inference.PlaceholderPreviouslyPosted, inference.PlaceholderRepoInstructions,
		inference.PlaceholderTicketContext, inference.PlaceholderComplianceInstructions,
	} {
		if strings.Contains(sawPrompt, ph) {
			t.Errorf("prompt still contains %s: the eval would score the placeholder", ph)
		}
	}
}

func readTemplate(t *testing.T) string {
	t.Helper()
	// The real template, not a stand-in: an eval that scores a toy prompt
	// says nothing about the one that ships.
	b, err := os.ReadFile("../../../prompts/review.txt")
	if err != nil {
		t.Fatal(err)
	}
	return string(b)
}

type discard struct{}

func (discard) Write(p []byte) (int, error) { return len(p), nil }

func mustJSON(s string) []byte {
	b, err := json.Marshal(s)
	if err != nil {
		panic(err)
	}
	return b
}
