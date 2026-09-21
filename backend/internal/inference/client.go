package inference

import (
	"fmt"
	"log/slog"
	"net/http"
	"os"
	"strings"
	"time"

	"github.com/trick77/llmwire"

	"github.com/trick77/noergler/internal/config"
	"github.com/trick77/noergler/internal/httpstats"
)

// CallTimeout is the hard wall-clock cap on a single LLM call. noergler
// imposes it; the model is unaware of any deadline.
const CallTimeout = 300 * time.Second

// Client is one team's inference client.
//
// Safe for concurrent use: llmwire's Client guards its own state and wraps
// *http.Client. noergler does not serialize calls here. Python held a
// process-wide asyncio lock, which existed because asyncio makes accidental
// concurrency easy, not because the transport needed it; serialization is the
// review queue's policy, not this package's property.
type Client struct {
	wire   *llmwire.Client
	model  string
	effort string

	// window is the context window resolved at startup, 0 until Startup runs.
	// An explicit OPENAI_CONTEXT_WINDOW wins and is set here directly.
	window int
	// pingCost is what the startup ping cost, for the boot-time pricing line.
	pingCost CallCost
	// budget knobs, read from config once.
	headroom  int
	threshold int
	tail      float64

	// log emits the parser's diagnostics, which ParseReview returns as data
	// rather than logging itself. Never nil; New substitutes a discarding
	// logger, as review.New does.
	log *slog.Logger
}

// Options is what a team's client needs beyond its llmwire profile.
type Options struct {
	// Model is an llmwire profile id (gpt-5.5), never a gateway alias.
	Model string
	// ReasoningEffort is sent on every call. Not validated locally: an
	// unusable value is the gateway's 400, which Startup maps to a readable
	// error. Python validated against a hardcoded set that is wrong for the
	// configured model in both directions, so that set is not ported.
	ReasoningEffort string
	// APIKey is the team's key, answered to llmwire through Lookup rather than
	// Config.APIKey: FromEnv refuses a key set directly on a gateway-routed
	// model.
	APIKey string
	// ContextWindow overrides the gateway's figure. 0 means resolve it.
	ContextWindow int

	HeadroomTokens int
	Threshold      int
	Tail           float64

	Logger *slog.Logger
	// Env is where llmwire reads its settings. Nil means the process
	// environment; tests pass a map.
	Env func(string) (string, bool)
}

// New builds a team's client.
//
// Config.BaseURL is deliberately left empty. FromEnv short-circuits when it is
// set, skipping gateway routing entirely, which would send a profile missing
// from LLMWIRE_LITELLM_MODELS to api.openai.com with whatever key is around.
// Leaving it empty makes that case an error instead.
func New(opt Options) (*Client, error) {
	if strings.TrimSpace(opt.Model) == "" {
		return nil, fmt.Errorf("inference: model is required")
	}

	env := opt.Env
	if env == nil {
		env = os.LookupEnv
	}

	// The parser's diagnostics are logged from Review, which may run before a
	// caller has set a logger in tests. Discard rather than panic: a missing
	// log line must not take a review down.
	log := opt.Logger
	if log == nil {
		log = slog.New(slog.DiscardHandler)
	}

	c := &Client{
		model:     opt.Model,
		effort:    opt.ReasoningEffort,
		window:    opt.ContextWindow,
		headroom:  opt.HeadroomTokens,
		threshold: opt.Threshold,
		tail:      opt.Tail,
		log:       log,
	}

	wire, err := llmwire.FromEnv(opt.Model, llmwire.Config{
		CallTimeout: CallTimeout,
		Logger:      opt.Logger,
		Lookup:      teamLookup(opt.APIKey, env),
		HTTPClient:  countingClient(),
		// APIKey stays empty: the team's key is answered through Lookup.
	})
	if err != nil {
		return nil, fmt.Errorf("inference: build client for %s: %w", opt.Model, err)
	}
	c.wire = wire
	return c, nil
}

// countingClient is llmwire's own default client with the transport wrapped so
// inference requests reach the per-review HTTP totals. Without it the totals
// line reported inference=0 on every review while the call plainly happened,
// because bitbucket and jira are the only wrapped transports.
//
// Supplying an HTTPClient means llmwire skips building its own, so the tuned
// pieces are reproduced here: the stdlib default is cloned (keeping proxy
// support, dial timeouts and pooling) and ResponseHeaderTimeout is set LONGER
// than llmwire's header bound so llmwire's guard still reports the timeout
// under its own named bound rather than the transport's generic one.
func countingClient() *http.Client {
	return &http.Client{Transport: httpstats.Transport("inference", tunedTransport())}
}

// tunedTransport is the transport countingClient wraps, split out so a test
// can assert the tuning without unwrapping the counting layer.
func tunedTransport() *http.Transport {
	tr := http.DefaultTransport.(*http.Transport).Clone()
	tr.ResponseHeaderTimeout = llmwire.DefaultHeaderTimeout + headerBackstopHeadroom
	return tr
}

// headerBackstopHeadroom mirrors llmwire's unexported constant of the same
// name. It is the margin that keeps llmwire's header guard ahead of the
// transport's backstop; see countingClient.
//
// llmwire computes the backstop from its RESOLVED HeaderTimeout, while
// countingClient uses DefaultHeaderTimeout. The two agree only because this
// Config never sets HeaderTimeout; set it there and set it here too.
const headerBackstopHeadroom = 30 * time.Second

// teamLookup answers the gateway's key variable with this team's key and
// delegates everything else, so per-team keys need no llmwire change.
func teamLookup(apiKey string, env func(string) (string, bool)) func(string) (string, bool) {
	return func(name string) (string, bool) {
		if name == gatewayAPIKeyEnv && apiKey != "" {
			return apiKey, true
		}
		return env(name)
	}
}

// gatewayAPIKeyEnv is the variable llmwire reads a gateway-routed model's key
// from. Each team answers it with its own TEAM_<SLUG>_OPENAI_API_KEY.
//
//nolint:gosec // G101: the name of an env var, not a key
const gatewayAPIKeyEnv = "LLMWIRE_LITELLM_API_KEY"

// Model is the llmwire profile id this client was built for.
func (c *Client) Model() string { return c.model }

// Label is the model string a reader sees: the profile id with the reasoning
// effort appended, as Python's model_label renders it. It is what the summary
// footnote shows and what a run row stores.
//
// Not Model(): that one names the profile llmwire routes on, and the effort is
// part of what produced a review, so a run recorded without it cannot be told
// apart from the same model at another effort. The profile id rather than the
// gateway alias is deliberate: the alias is the operator's private naming and
// has no business in a PR comment.
func (c *Client) Label() string { return config.ModelLabel(c.model, c.effort) }

// ContextWindow is the resolved window in tokens, 0 before Startup.
func (c *Client) ContextWindow() int { return c.window }

// Ready reports whether Startup has resolved a context window.
//
// Worth checking before InputTokenBudget in any path that could run before
// Startup: an unresolved window is 0, and the curve floors at 2000 rather than
// failing, which would silently compress every PR to nothing.
func (c *Client) Ready() bool { return c.window > 0 }

// InputTokenBudget is the usable input budget: the diminishing-trust curve
// applied to the context window. Governs diff COMPRESSION and prompt hygiene,
// deciding how much goes into the prompt.
//
// It is not the fit ceiling: see FitCeiling, which is larger. Compressing to
// this budget and then refusing anything above it would skip PRs the model can
// hold.
//
// Meaningless until Startup has run, since an unresolved window yields the
// floor. Every caller runs after Startup, which resolves the window or
// disables the team; Ready reports the difference.
func (c *Client) InputTokenBudget() int {
	return UsableContextBudget(c.window, c.headroom, c.threshold, c.tail)
}

// FitCeiling is the most the assembled prompt may occupy: the whole context
// window less the reply reserve.
//
// Distinct from InputTokenBudget on purpose. The budget is what compression
// aims at; the ceiling is what actually will not fit. With a 1M window and the
// default knobs the budget is 628k and the ceiling 936k, so a prompt that
// compression filled plus the cumulative diff and the posted findings can
// exceed the budget and still be reviewable.
func (c *Client) FitCeiling() int {
	return c.window - OutputTokenReserve
}
