package inference

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/trick77/llmwire"
	"github.com/trick77/llmwire/llmwiretest"
)

// testProfile is llmwiretest's synthetic chat model: tests assert what
// noergler asks for, never a real model's id, levels or rates.
const testProfile = llmwiretest.ChatModel

// fakeGateway stands in for LiteLLM. Each field overrides one route.
type fakeGateway struct {
	// models is the /models body. Empty means a single entry for the alias
	// with a 1M window.
	models string
	// chatStatus and chatBody override the completion response.
	chatStatus int
	chatBody   string
	// chatBodies, when set, answer the completions in turn; the last one
	// repeats. chatRequests holds every completion request.
	chatBodies   []string
	chatRequests [][]byte
	// costHeader, when set, is sent as the per-call cost.
	costHeader string
	// delay stalls the completion route, for timeout tests.
	delay time.Duration

	gotChat   int
	gotModels int
	// lastChatBody is the most recent completion request, for asserting what
	// actually reaches the wire.
	lastChatBody []byte
}

func (f *fakeGateway) start(t *testing.T, alias string) *httptest.Server {
	t.Helper()
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch {
		case strings.HasSuffix(r.URL.Path, "/models"):
			f.gotModels++
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
			if n := len(f.chatBodies); n > 0 {
				body = f.chatBodies[min(f.gotChat, n)-1]
			}
			f.chatRequests = append(f.chatRequests, f.lastChatBody)
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
		Model:          testProfile,
		Registry:       testRegistry,
		APIKey:         "team-key",
		HeadroomTokens: defHeadroom,
		Threshold:      defThreshold,
		Tail:           defTail,
		Env:            env(srv.URL, alias, nil),
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
	const alias = "gateway-alias"
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
	// Read off the profile, never probed: the test model takes tools.
	if !c.Tools() {
		t.Error("Tools() = false for a profile with tool support")
	}
}

// Startup records whether the gateway priced the ping, because an unpriced
// model means the per-PR cap can never fire and boot is the only place that
// gets said.
func TestStartupRecordsPingPricing(t *testing.T) {
	const alias = "gateway-alias"

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

// The floor is what a 1.05M model accepts as input, not 1M: the floor itself
// passes, one token under it does not.
func TestStartupWindowFloorBoundary(t *testing.T) {
	const alias = "gateway-alias"
	for _, tc := range []struct {
		window int
		ok     bool
	}{{MinContextWindow, true}, {MinContextWindow - 1, false}} {
		f := &fakeGateway{models: fmt.Sprintf(`{"data":[{"id":"gateway-alias","max_input_tokens":%d}]}`, tc.window)}
		srv := f.start(t, alias)
		c := newTestClient(t, srv, alias)
		err := c.Startup(context.Background())
		if tc.ok && err != nil {
			t.Errorf("window %d: Startup: %v", tc.window, err)
		}
		if !tc.ok && (err == nil || !strings.Contains(err.Error(), "below the")) {
			t.Errorf("window %d: err = %v, want a window-floor error", tc.window, err)
		}
	}
}

// An explicit OPENAI_CONTEXT_WINDOW wins over the gateway's figure.
func TestStartupExplicitWindowWins(t *testing.T) {
	const alias = "gateway-alias"
	f := &fakeGateway{
		models: `{"data":[{"id":"gateway-alias","max_input_tokens":2000000}]}`,
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

// A gateway that lists the alias without a usable max_input_tokens leaves the
// window to the profile's context, and the log says where it came from. The
// gateway's model_info is optional; the profile is a vendor fact llmwire holds.
func TestStartupFallsBackToProfileContext(t *testing.T) {
	const alias = "gateway-alias"
	for _, tc := range []struct{ name, models, wantLog string }{
		{"absent", `{"data":[{"id":"gateway-alias"}]}`, "llmwire profile"},
		{"unusable", `{"data":[{"id":"gateway-alias","max_input_tokens":"lots"}]}`, "unusable"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			f := &fakeGateway{models: tc.models}
			srv := f.start(t, alias)
			var buf bytes.Buffer
			c := newTestClient(t, srv, alias, onModel(srv, wideModel, alias), func(o *Options) {
				o.Logger = slog.New(slog.NewTextHandler(&buf, nil))
			})
			if err := c.Startup(context.Background()); err != nil {
				t.Fatalf("Startup: %v", err)
			}
			if got := c.ContextWindow(); got != wideInput {
				t.Errorf("ContextWindow = %d, want the profile's input %d", got, wideInput)
			}
			if !strings.Contains(buf.String(), tc.wantLog) {
				t.Errorf("log = %q, want it to contain %q", buf.String(), tc.wantLog)
			}
		})
	}
}

// onModel runs the client on model, routed through srv under alias.
func onModel(srv *httptest.Server, model, alias string) func(*Options) {
	return func(o *Options) {
		o.Model = model
		o.Env = routed(srv.URL, model, alias)
	}
}

func TestStartupRejects(t *testing.T) {
	const alias = "gateway-alias"

	t.Run("window below the floor", func(t *testing.T) {
		f := &fakeGateway{models: `{"data":[{"id":"gateway-alias","max_input_tokens":128000}]}`}
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

	t.Run("missing max_input_tokens, profile names no context", func(t *testing.T) {
		f := &fakeGateway{models: `{"data":[{"id":"gateway-alias"}]}`}
		srv := f.start(t, alias)
		c := newTestClient(t, srv, alias, onModel(srv, noLimitModel, alias))
		err := c.Startup(context.Background())
		if err == nil || !strings.Contains(err.Error(), "OPENAI_CONTEXT_WINDOW") {
			t.Errorf("err = %v, want the set-the-window error", err)
		}
	})

	// The profile's input, not its context, is held to the floor: the
	// context alone would pass. The error says the number is the profile's
	// and keeps what the gateway sent, so the operator does not go looking
	// for it on the gateway.
	t.Run("unusable max_input_tokens, profile input below the floor", func(t *testing.T) {
		f := &fakeGateway{models: `{"data":[{"id":"gateway-alias","max_input_tokens":"lots"}]}`}
		srv := f.start(t, alias)
		c := newTestClient(t, srv, alias, onModel(srv, tightModel, alias))
		err := c.Startup(context.Background())
		if err == nil || !strings.Contains(err.Error(), "below the") {
			t.Fatalf("err = %v, want a window-floor error", err)
		}
		for _, want := range []string{"llmwire profile", "unusable"} {
			if !strings.Contains(err.Error(), want) {
				t.Errorf("err = %v, want it to contain %q", err, want)
			}
		}
		if f.gotChat != 0 {
			t.Error("ping must not run after the window check fails")
		}
		if c.ContextWindow() != 0 {
			t.Errorf("ContextWindow = %d after a refused window, want 0", c.ContextWindow())
		}
	})

	// A limit the gateway does report is never overridden by the profile:
	// the gateway is the one that can cap below the vendor.
	t.Run("gateway limit below the floor beats a wide profile", func(t *testing.T) {
		f := &fakeGateway{models: `{"data":[{"id":"gateway-alias","max_input_tokens":128000}]}`}
		srv := f.start(t, alias)
		c := newTestClient(t, srv, alias, onModel(srv, wideModel, alias))
		err := c.Startup(context.Background())
		if err == nil || !strings.Contains(err.Error(), "below the") {
			t.Errorf("err = %v, want a window-floor error", err)
		}
	})

	// A limit that is present but unusable must not report as absent: llmwire
	// returns a nil limit plus a Warning naming it, and an operator told the
	// field is missing goes looking for the wrong thing.
	t.Run("unusable max_input_tokens says so", func(t *testing.T) {
		f := &fakeGateway{models: `{"data":[{"id":"gateway-alias","max_input_tokens":"lots"}]}`}
		srv := f.start(t, alias)
		c := newTestClient(t, srv, alias, onModel(srv, noLimitModel, alias))
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

}

// A profile missing from LLMWIRE_LITELLM_MODELS must not build a client that
// would reach the vendor's own host.
func TestUnroutedProfileIsRefused(t *testing.T) {
	_, err := New(Options{
		Model:    testProfile,
		Registry: testRegistry,
		APIKey:   "team-key",
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

// sentReasoning is the reasoning setting a captured request carried, in
// llmwire's ReasoningSent vocabulary rather than any vendor's wire spelling.
func sentReasoning(t *testing.T, body []byte) string {
	t.Helper()
	var m map[string]any
	if err := json.Unmarshal(body, &m); err != nil {
		t.Fatalf("decoding the captured request: %v", err)
	}
	return llmwiretest.Request{Body: m}.Reasoning()
}

// deepestLevel is a level the synthetic profile takes that is not its
// balanced one, so a test can tell a configured level from the default.
func deepestLevel(t *testing.T) string {
	t.Helper()
	p, err := llmwiretest.Registry().Lookup(testProfile)
	if err != nil {
		t.Fatal(err)
	}
	levels := p.Reasoning.EffortValues
	return levels[len(levels)-1]
}

// Unset, the review runs at the model's balanced level, and the label a run
// row stores names what was actually sent.
func TestStartupDefaultsToTheBalancedLevel(t *testing.T) {
	const alias = "gateway-alias"
	f := &fakeGateway{}
	srv := f.start(t, alias)
	c := newTestClient(t, srv, alias)

	if err := c.Startup(context.Background()); err != nil {
		t.Fatalf("Startup: %v", err)
	}
	if got := sentReasoning(t, f.lastChatBody); got != llmwiretest.BalancedSent {
		t.Errorf("ping reasoning = %q, want the balanced level %q", got, llmwiretest.BalancedSent)
	}
	if got, want := c.Label(), testProfile+"-"+llmwiretest.BalancedSent; got != want {
		t.Errorf("Label = %q, want %q", got, want)
	}
}

// A configured level is the operator's choice and is sent as given.
func TestStartupKeepsAConfiguredLevel(t *testing.T) {
	const alias = "gateway-alias"
	level := deepestLevel(t)
	f := &fakeGateway{}
	srv := f.start(t, alias)
	c := newTestClient(t, srv, alias, func(o *Options) { o.ReasoningEffort = level })

	if err := c.Startup(context.Background()); err != nil {
		t.Fatalf("Startup: %v", err)
	}
	if got := sentReasoning(t, f.lastChatBody); got != level {
		t.Errorf("ping reasoning = %q, want %q", got, level)
	}
	if got, want := c.Label(), testProfile+"-"+level; got != want {
		t.Errorf("Label = %q, want %q", got, want)
	}
}

// What the profile knows is checked against the profile, at boot and before
// any request: no network probe for a level or a capability.
func TestStartupChecksTheProfileBeforeTheNetwork(t *testing.T) {
	const alias = "gateway-alias"

	t.Run("a level the model does not take", func(t *testing.T) {
		f := &fakeGateway{}
		srv := f.start(t, alias)
		c := newTestClient(t, srv, alias, func(o *Options) { o.ReasoningEffort = "not-a-level" })
		err := c.Startup(context.Background())
		if err == nil {
			t.Fatal("want an error for a level the profile does not list")
		}
		p, _ := llmwiretest.Registry().Lookup(testProfile)
		for _, want := range append([]string{"not-a-level"}, p.Reasoning.EffortValues...) {
			if !strings.Contains(err.Error(), want) {
				t.Errorf("err = %v, want it to name %q", err, want)
			}
		}
		if f.gotModels+f.gotChat != 0 {
			t.Errorf("%d request(s) sent, want none", f.gotModels+f.gotChat)
		}
	})

	t.Run("a model without strict JSON output", func(t *testing.T) {
		f := &fakeGateway{}
		srv := f.start(t, alias)
		c := newTestClient(t, srv, alias, onModel(srv, llmwiretest.BudgetModel, alias))
		err := c.Startup(context.Background())
		if err == nil || !strings.Contains(err.Error(), "valid choices are") ||
			!strings.Contains(err.Error(), llmwiretest.ChatModel) {
			t.Fatalf("err = %v, want a capability error listing the valid choices", err)
		}
		if f.gotModels+f.gotChat != 0 {
			t.Errorf("%d request(s) sent, want none", f.gotModels+f.gotChat)
		}
	})

	t.Run("a level on a model that takes none", func(t *testing.T) {
		f := &fakeGateway{}
		srv := f.start(t, alias)
		c := newTestClient(t, srv, alias, onModel(srv, "noergler-budget", alias), func(o *Options) { o.ReasoningEffort = "some-level" })
		err := c.Startup(context.Background())
		if err == nil || !strings.Contains(err.Error(), "takes no named level") {
			t.Fatalf("err = %v, want it to say the model takes no named level", err)
		}
		if f.gotModels+f.gotChat != 0 {
			t.Errorf("%d request(s) sent, want none", f.gotModels+f.gotChat)
		}
	})

	// A level the model lists but that switches reasoning off would bypass
	// the reasoning-capable check.
	t.Run("a level that switches reasoning off", func(t *testing.T) {
		f := &fakeGateway{}
		srv := f.start(t, alias)
		c := newTestClient(t, srv, alias, func(o *Options) { o.ReasoningEffort = "none" })
		err := c.Startup(context.Background())
		if err == nil || !strings.Contains(err.Error(), "switches reasoning off") {
			t.Fatalf("err = %v, want the reasoning-off refusal", err)
		}
		if f.gotModels+f.gotChat != 0 {
			t.Errorf("%d request(s) sent, want none", f.gotModels+f.gotChat)
		}
	})

	// With no level set, a model that reasons only when asked and names no
	// balanced level would get nothing sent and run without reasoning.
	t.Run("no level on a model that reasons only when asked", func(t *testing.T) {
		f := &fakeGateway{}
		srv := f.start(t, alias)
		c := newTestClient(t, srv, alias, onModel(srv, "noergler-optin", alias))
		err := c.Startup(context.Background())
		if err == nil {
			t.Fatal("want an error")
		}
		msg := err.Error()
		// Only levels that would pass: "none" is refused as reasoning off.
		if !strings.Contains(msg, "set reasoning_effort (accepted: low, high)") {
			t.Errorf("err = %v, want it to offer only the levels that pass", err)
		}
		if !strings.Contains(msg, pickAnotherModel) {
			t.Errorf("err = %v, want it to name the other remedy", err)
		}
		if f.gotModels+f.gotChat != 0 {
			t.Errorf("%d request(s) sent, want none", f.gotModels+f.gotChat)
		}
	})

	// No named level to offer: picking another model is the only remedy.
	t.Run("no level on an opt-in model that takes no named level", func(t *testing.T) {
		f := &fakeGateway{}
		srv := f.start(t, alias)
		c := newTestClient(t, srv, alias, onModel(srv, "noergler-optin-budget", alias))
		err := c.Startup(context.Background())
		if err == nil {
			t.Fatal("want an error")
		}
		msg := err.Error()
		if strings.Contains(msg, "accepted") || strings.Contains(msg, "set reasoning_effort") {
			t.Errorf("err = %v, want no level offered", err)
		}
		if !strings.Contains(msg, pickAnotherModel) {
			t.Errorf("err = %v, want it to name the other remedy", err)
		}
	})

	t.Run("a model that does not reason", func(t *testing.T) {
		f := &fakeGateway{}
		srv := f.start(t, alias)
		c := newTestClient(t, srv, alias, onModel(srv, "noergler-plain", alias))
		err := c.Startup(context.Background())
		if err == nil || !strings.Contains(err.Error(), "reasoning") ||
			!strings.Contains(err.Error(), llmwiretest.ChatModel) {
			t.Fatalf("err = %v, want a reasoning-capable error naming a valid choice", err)
		}
		if f.gotModels+f.gotChat != 0 {
			t.Errorf("%d request(s) sent, want none", f.gotModels+f.gotChat)
		}
	})
}

// testRegistry is llmwiretest's synthetic profiles, routable through the
// gateway the way production routes every model, plus a chat model that does
// not reason.
var testRegistry = func() *llmwire.Registry {
	doc := strings.Replace(string(llmwiretest.Profiles()),
		"providers:\n", "providers:\n  litellm: {}\n", 1) + plainProfile
	reg, err := llmwire.NewRegistry([]byte(doc))
	if err != nil {
		panic("test registry: " + err.Error())
	}
	return reg
}()

// plainProfile adds chat models with strict JSON output: one that does not
// reason, one that reasons by token budget and so has no named level, and one
// that reasons only when asked and names no balanced level; and three that
// reason like testProfile and differ only in limits (see wideModel).
const plainProfile = `
  - id: noergler-plain
    display_name: noergler plain
    provider: llmwiretest
    verified: source-derived
    max_tokens_param: max_tokens
    output: {json_object: true, json_schema: true, strict_schema: true}
    limits: {context: 128000, max_output: 16384}
  - id: noergler-budget
    display_name: noergler budget
    provider: llmwiretest
    verified: source-derived
    max_tokens_param: max_tokens
    reasoning:
      supported: true
      enabled_by_default: true
      can_be_disabled: true
      control: budget_tokens
      budget_param: thinking_budget
      min_budget: 1024
    output: {json_object: true, json_schema: true, strict_schema: true}
    limits: {context: 128000, max_output: 16384}
  - id: noergler-optin-budget
    display_name: noergler opt-in budget
    provider: llmwiretest
    verified: source-derived
    max_tokens_param: max_tokens
    reasoning:
      supported: true
      enabled_by_default: false
      can_be_disabled: true
      control: budget_tokens
      budget_param: thinking_budget
      min_budget: 1024
    output: {json_object: true, json_schema: true, strict_schema: true}
    limits: {context: 128000, max_output: 16384}
  - id: noergler-optin
    display_name: noergler opt-in
    provider: llmwiretest
    verified: source-derived
    max_tokens_param: max_tokens
    reasoning:
      supported: true
      enabled_by_default: false
      can_be_disabled: true
      control: effort
      effort_values: [none, low, high]
    output: {json_object: true, json_schema: true, strict_schema: true}
    limits: {context: 128000, max_output: 16384}
  - id: noergler-wide
    display_name: noergler wide
    provider: llmwiretest
    verified: source-derived
    max_tokens_param: max_tokens
    reasoning:
      supported: true
      enabled_by_default: true
      can_be_disabled: true
      control: effort
      effort_values: [none, low, medium, high]
      balanced: medium
    output: {json_object: true, json_schema: true, strict_schema: true}
    limits: {context: 1050000, max_output: 128000}
  - id: noergler-tight
    display_name: noergler tight
    provider: llmwiretest
    verified: source-derived
    max_tokens_param: max_tokens
    reasoning:
      supported: true
      enabled_by_default: true
      can_be_disabled: true
      control: effort
      effort_values: [none, low, medium, high]
      balanced: medium
    output: {json_object: true, json_schema: true, strict_schema: true}
    limits: {context: 1000000, max_output: 128000}
  - id: noergler-nolimit
    display_name: noergler no limit
    provider: llmwiretest
    verified: source-derived
    max_tokens_param: max_tokens
    reasoning:
      supported: true
      enabled_by_default: true
      can_be_disabled: true
      control: effort
      effort_values: [none, low, medium, high]
      balanced: medium
    output: {json_object: true, json_schema: true, strict_schema: true}
`

// wideModel's input (context less max output) clears the window floor;
// tightModel's context would, its input does not; noLimitModel names none.
const (
	wideModel    = "noergler-wide"
	tightModel   = "noergler-tight"
	noLimitModel = "noergler-nolimit"
	wideInput    = 1_050_000 - 128_000
)

// pickAnotherModel is the remedy every opt-in refusal names.
const pickAnotherModel = "pick a model that reasons by default or names a balanced level"

// routed is env for a model other than testProfile.
func routed(base, model, alias string) func(string) (string, bool) {
	vars := map[string]string{
		"LLMWIRE_LITELLM_MODELS":   model + "=" + alias,
		"LLMWIRE_LITELLM_BASE_URL": base,
		"LLMWIRE_LITELLM_API_KEY":  "test-key",
	}
	return func(name string) (string, bool) {
		v, ok := vars[name]
		return v, ok
	}
}

// Whatever the configuration, a ping that went out with reasoning off fails
// startup: reviews would run the same way.
func TestPingRefusesReasoningOff(t *testing.T) {
	const alias = "gateway-alias"
	f := &fakeGateway{}
	srv := f.start(t, alias)
	// Straight to ping, past the profile check that would refuse the level.
	c := newTestClient(t, srv, alias, func(o *Options) { o.ReasoningEffort = "none" })
	p, _ := testRegistry.Lookup(testProfile)
	err := c.ping(context.Background(), p)
	if err == nil || !strings.Contains(err.Error(), "reasoning off") {
		t.Fatalf("err = %v, want the reasoning-off refusal", err)
	}
}

// Nothing sent to a model that reasons only when asked is reasoning off too.
func TestPingRefusesNothingSentOnAnOptInModel(t *testing.T) {
	const alias = "gateway-alias"
	f := &fakeGateway{}
	srv := f.start(t, alias)
	c := newTestClient(t, srv, alias, onModel(srv, "noergler-optin", alias))
	p, _ := testRegistry.Lookup("noergler-optin")
	err := c.ping(context.Background(), p)
	if err == nil || !strings.Contains(err.Error(), "reasoning off") {
		t.Fatalf("err = %v, want the reasoning-off refusal", err)
	}
}

// A configured level on such a model is fine: it asks for reasoning.
func TestStartupAcceptsALevelOnAnOptInModel(t *testing.T) {
	const alias = "gateway-alias"
	f := &fakeGateway{}
	srv := f.start(t, alias)
	c := newTestClient(t, srv, alias, onModel(srv, "noergler-optin", alias), func(o *Options) { o.ReasoningEffort = "high" })
	if err := c.Startup(context.Background()); err != nil {
		t.Fatalf("Startup: %v", err)
	}
	if got := c.Label(); got != "noergler-optin-high" {
		t.Errorf("Label = %q, want noergler-optin-high", got)
	}
}

// The profile's input limit needs both numbers: a context alone would read
// the output share as input, and an output cap at or over the context is no
// input at all.
func TestProfileInputLimit(t *testing.T) {
	for _, tc := range []struct {
		l    llmwire.Limits
		want int
	}{
		{llmwire.Limits{Context: 1_050_000, MaxOutput: 128_000}, 922_000},
		{llmwire.Limits{Context: 1_000_000}, 0},
		{llmwire.Limits{Context: 128_000, MaxOutput: 128_000}, 0},
		{llmwire.Limits{}, 0},
	} {
		if got := profileInputLimit(tc.l); got != tc.want {
			t.Errorf("profileInputLimit(%+v) = %d, want %d", tc.l, got, tc.want)
		}
	}
}
