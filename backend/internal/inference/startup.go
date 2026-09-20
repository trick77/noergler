package inference

import (
	"context"
	"errors"
	"fmt"
	"strings"

	"github.com/trick77/llmwire"
)

// pingPrompt is the startup probe. Short on purpose: it proves the route, the
// key and that the model answers, and costs almost nothing.
const pingPrompt = "Reply with: ok"

// Startup runs the per-team checks. Any error disables that team only; it
// never aborts the process (AGENTS.md).
//
// Order matters: the registry lookup and route check are local and cheap, the
// window resolve is one HTTP call, and the ping is an inference call. A team
// misconfigured in an obvious way fails before it costs anything.
//
// There is no local reasoning-effort set. Python validated against a hardcoded
// {minimal, low, medium, high}, which is wrong for the configured model in
// both directions: it rejected a valid xhigh and accepted an invalid minimal.
// llmwire validates the level against the profile before sending, and the
// gateway's 400 covers the rest; both are mapped in mapPingError.
func (c *Client) Startup(ctx context.Context) error {
	profile, err := c.profile()
	if err != nil {
		return err
	}
	// A profile not listed in LLMWIRE_LITELLM_MODELS is not gateway-routed, so
	// FromEnv would send it to the vendor's own host. Disable the team rather
	// than talk to api.openai.com with a gateway key.
	if profile.Gateway == "" {
		return fmt.Errorf("model %q is not routed through the gateway: add it to %s",
			c.model, llmwire.GatewayModelsEnv)
	}

	if err := c.resolveWindow(ctx, profile); err != nil {
		return err
	}
	return c.ping(ctx)
}

// profile looks the model up in the client's registry, which FromEnv has
// already rewritten with the gateway routes.
func (c *Client) profile() (*llmwire.Profile, error) {
	p, err := c.wire.Registry().Lookup(c.model)
	if err != nil {
		return nil, fmt.Errorf("model %q: %w", c.model, err)
	}
	return p, nil
}

// resolveWindow sets the context window from the gateway, unless an explicit
// OPENAI_CONTEXT_WINDOW already did.
//
// GET /models with the team's key is both the access check (a model the key
// may not use is simply not listed) and the source of the window, so there is
// no separate catalog to keep in sync with the gateway's names.
func (c *Client) resolveWindow(ctx context.Context, profile *llmwire.Profile) error {
	entries, _, err := c.wire.ListModels(ctx)
	if err != nil {
		return fmt.Errorf("list models: %w", err)
	}

	// The gateway lists the operator's alias, not the profile id. That is
	// WireModelID (what goes on the wire); Gateway names the proxy itself
	// ("litellm") and is only the routed/not-routed marker.
	alias := profile.WireModelID
	var found *llmwire.ModelEntry
	for i := range entries {
		if entries[i].ID == alias {
			found = &entries[i]
			break
		}
	}
	if found == nil {
		return fmt.Errorf("model %q (gateway alias %q) is not listed by the gateway for this key",
			c.model, alias)
	}

	// An explicit window wins: it is the escape hatch for an endpoint whose
	// real cap differs from what it advertises.
	if c.window > 0 {
		return c.checkWindowFloor()
	}
	if found.MaxInputTokens == nil || *found.MaxInputTokens <= 0 {
		return fmt.Errorf("model %q is listed by the gateway without a usable `max_input_tokens`. "+
			"Set OPENAI_CONTEXT_WINDOW (or the team's inference.context_window) to the real limit",
			c.model)
	}
	c.window = int(*found.MaxInputTokens)
	return c.checkWindowFloor()
}

// checkWindowFloor refuses a window too small to hold a real PR. A whole PR is
// reviewed in one call, so a small-context model cannot hold it coherently.
func (c *Client) checkWindowFloor() error {
	if c.window < MinContextWindow {
		return fmt.Errorf("context window %d is below the %d minimum: a whole PR is reviewed in one call",
			c.window, MinContextWindow)
	}
	return nil
}

// ping proves the route, the key and that the model answers.
func (c *Client) ping(ctx context.Context) error {
	resp, _, err := c.wire.Chat(ctx, llmwire.ChatRequest{
		Model:     c.model,
		Reasoning: llmwire.ReasoningEffort(c.effort),
		Messages:  []llmwire.Message{llmwire.User(pingPrompt)},
	})
	if err != nil {
		return mapPingError(err, c.effort)
	}
	if strings.TrimSpace(resp.Content) == "" {
		return errors.New("empty response from model")
	}
	return nil
}

// mapPingError turns a rejection into something an operator can act on.
//
// Two shapes reach here for a bad effort level. llmwire refuses a level the
// profile does not list before sending, carrying the accepted set; a gateway
// that rejects it anyway answers 400 naming the parameter. Both mean the same
// thing to an operator.
func mapPingError(err error, effort string) error {
	var unsupported *llmwire.UnsupportedError
	if errors.As(err, &unsupported) {
		if len(unsupported.Accepted) > 0 {
			return fmt.Errorf("reasoning_effort=%q is not accepted by %s (accepted: %s)",
				effort, unsupported.Model, strings.Join(unsupported.Accepted, ", "))
		}
		return fmt.Errorf("requires a reasoning-capable model (reasoning_effort=%q): %w", effort, err)
	}

	var api *llmwire.APIError
	if errors.As(err, &api) && api.StatusCode == 400 && mentionsReasoningEffort(api) {
		return fmt.Errorf("requires a reasoning-capable model (reasoning_effort=%q rejected): %w", effort, err)
	}
	return fmt.Errorf("ping: %w", err)
}

// mentionsReasoningEffort reports whether a 400 blamed the reasoning parameter,
// by the field the endpoint names or by its message.
func mentionsReasoningEffort(api *llmwire.APIError) bool {
	if strings.Contains(strings.ToLower(api.Param), "reasoning") {
		return true
	}
	return strings.Contains(strings.ToLower(api.Message), "reasoning_effort")
}
