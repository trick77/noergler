package inference

import (
	"context"
	"errors"
	"fmt"
	"sort"
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

// listedIDs names what the gateway did list, for the error that says an alias
// is not among them. Capped: a gateway may serve hundreds of models, and an
// error line nobody can read helps nobody.
func listedIDs(entries []llmwire.ModelEntry) string {
	if len(entries) == 0 {
		return " (the gateway listed no models at all for this key)"
	}
	ids := make([]string, 0, len(entries))
	for _, e := range entries {
		ids = append(ids, e.ID)
	}
	sort.Strings(ids)
	const max = 12
	if len(ids) > max {
		return fmt.Sprintf(" (it listed %d models, including %s)",
			len(ids), strings.Join(ids[:max], ", "))
	}
	return " (it listed " + strings.Join(ids, ", ") + ")"
}

// aliasWarnings reports what llmwire said about this alias, so an unusable
// max_input_tokens does not read as a missing one.
func aliasWarnings(warnings []llmwire.Warning, alias string) string {
	var hits []string
	for _, w := range warnings {
		if strings.Contains(w.Details, alias) || strings.Contains(w.Feature, alias) {
			hits = append(hits, w.String())
		}
	}
	if len(hits) == 0 {
		return ""
	}
	return " (the gateway sent one, and it was unusable: " + strings.Join(hits, "; ") + ")"
}

// resolveWindow sets the context window from the gateway, unless an explicit
// OPENAI_CONTEXT_WINDOW already did.
//
// GET /models with the team's key is both the access check (a model the key
// may not use is simply not listed) and the source of the window, so there is
// no separate catalog to keep in sync with the gateway's names.
func (c *Client) resolveWindow(ctx context.Context, profile *llmwire.Profile) error {
	// The warnings are kept: llmwire reports a limit that is present but
	// unusable (a bool, a string, a fraction, zero) as a nil limit plus a
	// Warning naming it. Dropping them turns "the gateway sent nonsense" into
	// "the gateway sent nothing", and the operator is told to supply a window
	// the gateway is in fact advertising.
	entries, warnings, err := c.wire.ListModels(ctx)
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
		// Name what the gateway did list: the alias is the one thing the
		// operator has to get exactly right, and the listing is in hand.
		return fmt.Errorf("model %q (gateway alias %q) is not listed by the gateway for this key%s",
			c.model, alias, listedIDs(entries))
	}

	// An explicit window wins: it is the escape hatch for an endpoint whose
	// real cap differs from what it advertises.
	if c.window > 0 {
		if err := c.checkWindowFloor(); err != nil {
			return err
		}
		c.logWindow(ctx, "OPENAI_CONTEXT_WINDOW override")
		return nil
	}
	if found.MaxInputTokens == nil || *found.MaxInputTokens <= 0 {
		// A warning about this alias means the field was there and unusable,
		// which is a different fix from the field being absent.
		return fmt.Errorf("model %q is listed by the gateway without a usable `max_input_tokens`%s. "+
			"Set OPENAI_CONTEXT_WINDOW (or the team's inference.context_window) to the real limit",
			c.model, aliasWarnings(warnings, alias))
	}
	c.window = int(*found.MaxInputTokens)
	if err := c.checkWindowFloor(); err != nil {
		return err
	}
	c.logWindow(ctx, "gateway max_input_tokens")
	return nil
}

// logWindow records the window that was actually resolved. The config dump
// runs before any team starts, so without this line the resolved value never
// reaches the log and the operator only ever sees the configured one.
func (c *Client) logWindow(ctx context.Context, source string) {
	c.log.InfoContext(ctx, fmt.Sprintf("context window %d for %s (%s)", c.window, c.Label(), source))
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
	// Whether the gateway priced the ping is the one cheap answer to "will the
	// per-PR cap ever apply for this model". Recorded for the caller to report
	// at boot; cost still fails open either way.
	c.pingCost = CostFrom(resp)
	return nil
}

// PingCost is what the startup ping cost, as the gateway reported it. Valid
// only after Startup. An unpriced ping means summaries carry no cost and the
// per-PR cap never fires for this model.
func (c *Client) PingCost() CallCost { return c.pingCost }

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
