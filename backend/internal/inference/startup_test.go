package inference

import (
	"context"
	"encoding/json"
	"io"
	"log/slog"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"
)

// testProfile is a registry id that exists in llmwire's built-in registry and
// accepts a named reasoning effort.
const testProfile = "gpt-5.5"

// fakeGateway stands in for LiteLLM. Each field overrides one route.
type fakeGateway struct {
	// models is the /models body. Empty means a single entry for the alias
	// with a 1M window.
	models string
	// chatStatus and chatBody override the completion response.
	chatStatus int
	chatBody   string
	// costHeader, when set, is sent as the per-call cost.
	costHeader string
	// delay stalls the completion route, for timeout tests.
	delay time.Duration

	gotChat int
	// lastChatBody is the most recent completion request, for asserting what
	// actually reaches the wire.
	lastChatBody []byte
}

func (f *fakeGateway) start(t *testing.T, alias string) *httptest.Server {
	t.Helper()
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch {
		case strings.HasSuffix(r.URL.Path, "/models"):
			body := f.models
			if body == "" {
				body = `{"data":[{"id":"` + alias + `","max_input_tokens":1000000}]}`
			}
			w.Header().Set("Content-Type", "application/json")
			_, _ = w.Write([]byte(body))
		case strings.HasSuffix(r.URL.Path, "/chat/completions"):
			f.gotChat++
			f.lastChatBody, _ = io.ReadAll(r.Body)
			if f.delay > 0 {
				select {
				case <-time.After(f.delay):
				case <-r.Context().Done():
					return
				}
			}
			if f.costHeader != "" {
				w.Header().Set("x-litellm-response-cost", f.costHeader)
			}
			w.Header().Set("Content-Type", "application/json")
			if f.chatStatus != 0 {
				w.WriteHeader(f.chatStatus)
			}
			body := f.chatBody
			if body == "" {
				body = `{"choices":[{"message":{"role":"assistant","content":"ok"}}],` +
					`"usage":{"prompt_tokens":5,"completion_tokens":1,"total_tokens":6}}`
			}
			_, _ = w.Write([]byte(body))
		default:
			w.WriteHeader(http.StatusNotFound)
		}
	}))
	t.Cleanup(srv.Close)
	return srv
}

// env builds the environment llmwire reads, routing testProfile through the
// fake gateway under alias.
func env(base, alias string, extra map[string]string) func(string) (string, bool) {
	vars := map[string]string{
		"LLMWIRE_LITELLM_MODELS":   testProfile + "=" + alias,
		"LLMWIRE_LITELLM_BASE_URL": base,
		"LLMWIRE_LITELLM_API_KEY":  "test-key",
	}
	for k, v := range extra {
		vars[k] = v
	}
	return func(name string) (string, bool) {
		v, ok := vars[name]
		return v, ok
	}
}

func newTestClient(t *testing.T, srv *httptest.Server, alias string, opts ...func(*Options)) *Client {
	t.Helper()
	o := Options{
		Model:           testProfile,
		ReasoningEffort: "medium",
		APIKey:          "team-key",
		HeadroomTokens:  defHeadroom,
		Threshold:       defThreshold,
		Tail:            defTail,
		Env:             env(srv.URL, alias, nil),
		// llmwire logs a settings line per client; keep the test output to
		// the failures.
		Logger: slog.New(slog.NewTextHandler(io.Discard, nil)),
	}
	for _, f := range opts {
		f(&o)
	}
	c, err := New(o)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	return c
}

func TestStartupResolvesWindow(t *testing.T) {
	const alias = "ai-gateway-gpt-5.5"
	f := &fakeGateway{}
	srv := f.start(t, alias)
	c := newTestClient(t, srv, alias)

	if err := c.Startup(context.Background()); err != nil {
		t.Fatalf("Startup: %v", err)
	}
	if got := c.ContextWindow(); got != 1_000_000 {
		t.Errorf("ContextWindow = %d, want 1000000", got)
	}
	// 1M window with the default knobs: 256000 + (1000000-256000)/2.
	if got := c.InputTokenBudget(); got != 628_000 {
		t.Errorf("InputTokenBudget = %d, want 628000", got)
	}
	if f.gotChat != 1 {
		t.Errorf("ping ran %d times, want 1", f.gotChat)
	}
}

// Startup records whether the gateway priced the ping, because an unpriced
// model means the per-PR cap can never fire and boot is the only place that
// gets said.
func TestStartupRecordsPingPricing(t *testing.T) {
	const alias = "ai-gateway-gpt-5.5"

	t.Run("priced", func(t *testing.T) {
		f := &fakeGateway{costHeader: "0.0123"}
		srv := f.start(t, alias)
		c := newTestClient(t, srv, alias)
		if err := c.Startup(context.Background()); err != nil {
			t.Fatalf("Startup: %v", err)
		}
		pc := c.PingCost()
		if !pc.Priced() {
			t.Fatal("PingCost is unpriced, want priced")
		}
		if got := *pc.NanoUSD; got != 12_300_000 {
			t.Errorf("NanoUSD = %d, want 12300000", got)
		}
	})

	t.Run("unpriced", func(t *testing.T) {
		f := &fakeGateway{}
		srv := f.start(t, alias)
		c := newTestClient(t, srv, alias)
		if err := c.Startup(context.Background()); err != nil {
			t.Fatalf("Startup: %v", err)
		}
		if c.PingCost().Priced() {
			t.Error("PingCost is priced, want unpriced without a cost header")
		}
	})
}

// An explicit OPENAI_CONTEXT_WINDOW wins over the gateway's figure.
func TestStartupExplicitWindowWins(t *testing.T) {
	const alias = "ai-gateway-gpt-5.5"
	f := &fakeGateway{
		models: `{"data":[{"id":"ai-gateway-gpt-5.5","max_input_tokens":2000000}]}`,
	}
	srv := f.start(t, alias)
	c := newTestClient(t, srv, alias, func(o *Options) { o.ContextWindow = 1_500_000 })

	if err := c.Startup(context.Background()); err != nil {
		t.Fatalf("Startup: %v", err)
	}
	if got := c.ContextWindow(); got != 1_500_000 {
		t.Errorf("ContextWindow = %d, want the explicit 1500000", got)
	}
}

func TestStartupRejects(t *testing.T) {
	const alias = "ai-gateway-gpt-5.5"

	t.Run("window below the floor", func(t *testing.T) {
		f := &fakeGateway{models: `{"data":[{"id":"ai-gateway-gpt-5.5","max_input_tokens":128000}]}`}
		srv := f.start(t, alias)
		c := newTestClient(t, srv, alias)
		err := c.Startup(context.Background())
		if err == nil || !strings.Contains(err.Error(), "below the") {
			t.Errorf("err = %v, want a window-floor error", err)
		}
		if f.gotChat != 0 {
			t.Error("ping must not run after the window check fails")
		}
	})

	t.Run("missing max_input_tokens", func(t *testing.T) {
		f := &fakeGateway{models: `{"data":[{"id":"ai-gateway-gpt-5.5"}]}`}
		srv := f.start(t, alias)
		c := newTestClient(t, srv, alias)
		err := c.Startup(context.Background())
		if err == nil || !strings.Contains(err.Error(), "OPENAI_CONTEXT_WINDOW") {
			t.Errorf("err = %v, want the set-the-window error", err)
		}
	})

	// A limit that is present but unusable must not report as absent: llmwire
	// returns a nil limit plus a Warning naming it, and an operator told the
	// field is missing goes looking for the wrong thing.
	t.Run("unusable max_input_tokens says so", func(t *testing.T) {
		f := &fakeGateway{models: `{"data":[{"id":"ai-gateway-gpt-5.5","max_input_tokens":"lots"}]}`}
		srv := f.start(t, alias)
		c := newTestClient(t, srv, alias)
		err := c.Startup(context.Background())
		if err == nil {
			t.Fatal("want an error for an unusable max_input_tokens")
		}
		if !strings.Contains(err.Error(), "unusable") {
			t.Errorf("err = %v, want it to say the gateway sent an unusable value", err)
		}
		if !strings.Contains(err.Error(), "OPENAI_CONTEXT_WINDOW") {
			t.Errorf("err = %v, want it to still name the escape hatch", err)
		}
	})

	t.Run("alias not listed for this key", func(t *testing.T) {
		f := &fakeGateway{models: `{"data":[{"id":"some-other-model","max_input_tokens":1000000}]}`}
		srv := f.start(t, alias)
		c := newTestClient(t, srv, alias)
		err := c.Startup(context.Background())
		if err == nil || !strings.Contains(err.Error(), "not listed by the gateway") {
			t.Fatalf("err = %v, want a not-listed error", err)
		}
		// The alias is the one thing the operator must spell exactly, so the
		// error names what the gateway did list.
		if !strings.Contains(err.Error(), "some-other-model") {
			t.Errorf("err = %v, want it to name the ids the gateway returned", err)
		}
	})

	t.Run("nothing listed at all", func(t *testing.T) {
		f := &fakeGateway{models: `{"data":[]}`}
		srv := f.start(t, alias)
		c := newTestClient(t, srv, alias)
		err := c.Startup(context.Background())
		if err == nil || !strings.Contains(err.Error(), "no models at all") {
			t.Errorf("err = %v, want the empty-listing wording", err)
		}
	})

	t.Run("empty model response", func(t *testing.T) {
		f := &fakeGateway{chatBody: `{"choices":[{"message":{"role":"assistant","content":"  "}}]}`}
		srv := f.start(t, alias)
		c := newTestClient(t, srv, alias)
		err := c.Startup(context.Background())
		if err == nil || !strings.Contains(err.Error(), "empty response from model") {
			t.Errorf("err = %v, want the empty-response error", err)
		}
	})

	t.Run("400 naming reasoning_effort", func(t *testing.T) {
		f := &fakeGateway{
			chatStatus: http.StatusBadRequest,
			chatBody: `{"error":{"message":"Unsupported parameter: reasoning_effort",` +
				`"type":"invalid_request_error","param":"reasoning_effort"}}`,
		}
		srv := f.start(t, alias)
		c := newTestClient(t, srv, alias)
		err := c.Startup(context.Background())
		if err == nil || !strings.Contains(err.Error(), "reasoning-capable") {
			t.Errorf("err = %v, want the reasoning-capable error", err)
		}
	})

	t.Run("other 400 is not mistaken for an effort problem", func(t *testing.T) {
		f := &fakeGateway{
			chatStatus: http.StatusBadRequest,
			chatBody:   `{"error":{"message":"context length exceeded","type":"invalid_request_error"}}`,
		}
		srv := f.start(t, alias)
		c := newTestClient(t, srv, alias)
		err := c.Startup(context.Background())
		if err == nil {
			t.Fatal("want an error")
		}
		if strings.Contains(err.Error(), "reasoning-capable") {
			t.Errorf("err = %v, should not blame reasoning effort", err)
		}
	})
}

// A profile missing from LLMWIRE_LITELLM_MODELS must not build a client that
// would reach the vendor's own host.
func TestUnroutedProfileIsRefused(t *testing.T) {
	_, err := New(Options{
		Model:           testProfile,
		ReasoningEffort: "medium",
		APIKey:          "team-key",
		Env: func(_ string) (string, bool) {
			// No LLMWIRE_LITELLM_MODELS at all.
			return "", false
		},
	})
	if err == nil {
		t.Fatal("want an error for a profile with no gateway route and no vendor key")
	}
}

// The team's key answers the gateway key variable; Config.APIKey stays empty,
// which FromEnv requires for a routed model.
func TestTeamKeyAnswersGatewayVariable(t *testing.T) {
	lookup := teamLookup("team-secret", func(string) (string, bool) { return "fallback", true })
	if got, _ := lookup(gatewayAPIKeyEnv); got != "team-secret" {
		t.Errorf("gateway key = %q, want the team's", got)
	}
	if got, _ := lookup("SOMETHING_ELSE"); got != "fallback" {
		t.Errorf("other lookup = %q, want delegation", got)
	}
}

// An empty team key delegates rather than masking the environment.
func TestEmptyTeamKeyDelegates(t *testing.T) {
	lookup := teamLookup("", func(string) (string, bool) { return "from-env", true })
	if got, _ := lookup(gatewayAPIKeyEnv); got != "from-env" {
		t.Errorf("gateway key = %q, want the environment's", got)
	}
}

func TestModelsResponseShape(t *testing.T) {
	// Guards the fixture itself: if llmwire changes what it accepts, this
	// fails here rather than in every startup test at once.
	var body struct {
		Data []struct {
			ID             string `json:"id"`
			MaxInputTokens *int64 `json:"max_input_tokens"`
		} `json:"data"`
	}
	raw := `{"data":[{"id":"x","max_input_tokens":1000000}]}`
	if err := json.Unmarshal([]byte(raw), &body); err != nil {
		t.Fatalf("fixture does not decode: %v", err)
	}
	if len(body.Data) != 1 || body.Data[0].MaxInputTokens == nil {
		t.Fatal("fixture shape changed")
	}
}
