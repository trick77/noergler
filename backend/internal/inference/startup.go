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
// Order matters: the profile and route checks are local and cheap, the
// window resolve is one HTTP call, and the ping is an inference call. A team
// misconfigured in an obvious way fails before it costs anything. What the
// profile knows (capabilities, reasoning levels) is asked of the profile,
// never probed over the network.
func (c *Client) Startup(ctx context.Context) error {
	profile, err := c.profile()
	if err != nil {
		return err
	}
	// A profile not listed in LLMWIRE_LITELLM_MODELS is not gateway-routed, so
	// FromEnv would send it to the vendor's own host. Disable the team rather
	// than send a gateway key there.
	if profile.Gateway == "" {
		return fmt.Errorf("model %q is not routed through the gateway: add it to %s",
			c.model, llmwire.GatewayModelsEnv)
	}

	if err := c.resolveWindow(ctx, profile); err != nil {
		return err
	}
	return c.ping(ctx, profile)
}

// reviewNeeds is what a review asks of a model: strict JSON-schema output,
// which is what makes the answer parseable.
var reviewNeeds = llmwire.Needs{JSONSchema: true}

// profile looks the model up in the client's registry, which FromEnv has
// already rewritten with the gateway routes, and checks it can do the job:
// review output, reasoning, and the configured level if there is one.
func (c *Client) profile() (*llmwire.Profile, error) {
	reg := c.wire.Registry()
	p, err := reg.Require(c.model, reviewNeeds)
	if err != nil {
		return nil, fmt.Errorf("model %q: %w", c.model, err)
	}
	// Policy, not a model fact: a review is analysis a reader keeps, and a
	// model that cannot reason is not good enough at it.
	if !p.Reasoning.Supported {
		return nil, fmt.Errorf("model %q does not reason, and noergler needs a reasoning-capable model; valid choices are %s",
			c.model, strings.Join(reasoningModels(reg), ", "))
	}
	if c.effort != "" && !p.Reasoning.Accepts(c.effort) {
		if len(p.Reasoning.EffortValues) == 0 {
			return nil, fmt.Errorf("reasoning_effort=%q: model %q takes no named level; unset it to use the model's default",
				c.effort, c.model)
		}
		return nil, fmt.Errorf("reasoning_effort=%q is not accepted by %s (accepted: %s; unset it for the model's balanced level)",
			c.effort, c.model, strings.Join(p.Reasoning.EffortValues, ", "))
	}
	// A listed level can still be the off switch, which would run reviews
	// without reasoning past the check above. "none" is how llmwire spells
	// that level.
	if c.effort == "none" {
		return nil, fmt.Errorf("reasoning_effort=%q switches reasoning off on %s, and noergler needs reasoning; pick another level or unset it",
			c.effort, c.model)
	}
	// Unset sends the balanced intent. A model that names no balanced level
	// and reasons only when asked then gets nothing, and runs without it.
	if c.effort == "" && p.Reasoning.Balanced == "" && !p.Reasoning.EnabledByDefault {
		remedy := "pick a model that reasons by default or names a balanced level"
		// Offer only levels the checks above would pass: "none" is off.
		var levels []string
		for _, l := range p.Reasoning.EffortValues {
			if l != "none" {
				levels = append(levels, l)
			}
		}
		if len(levels) > 0 {
			remedy = fmt.Sprintf("set reasoning_effort (accepted: %s), or %s", strings.Join(levels, ", "), remedy)
		}
		return nil, fmt.Errorf("model %q reasons only when asked and names no balanced level; %s", c.model, remedy)
	}
	return p, nil
}

// reasoningModels lists the registry's models that could run a review.
func reasoningModels(reg *llmwire.Registry) []string {
	var out []string
	for _, id := range reg.ChatModels(reviewNeeds) {
		if p, err := reg.Lookup(id); err == nil && p.Reasoning.Supported {
			out = append(out, id)
		}
	}
	return out
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
		// Not "OPENAI_CONTEXT_WINDOW": a team's inference.context_window lands
		// in the same field, and naming the env var would blame a setting the
		// operator never touched. Same two sources the error below names.
		c.logWindow(ctx, "configured override")
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

// ping proves the route, the key and that the model answers. It carries the
// reviews' reasoning setting, so what llmwire sent is the label's.
func (c *Client) ping(ctx context.Context, profile *llmwire.Profile) error {
	resp, _, err := c.wire.Chat(ctx, llmwire.ChatRequest{
		Model:     c.model,
		Reasoning: c.reasoning(),
		Messages:  []llmwire.Message{llmwire.User(pingPrompt)},
	})
	if err != nil {
		return fmt.Errorf("ping: %w", err)
	}
	c.sent = resp.ReasoningSent
	// The backstop for any other route to "off": reviews send the same
	// setting, so they would run without reasoning. Nothing sent to a model
	// that reasons only when asked is off too.
	if c.sent == "off" || (c.sent == "" && !profile.Reasoning.EnabledByDefault) {
		return fmt.Errorf("ping: the model ran with reasoning off, and noergler needs reasoning")
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
