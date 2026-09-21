package teams

import (
	"context"
	"errors"
	"fmt"
	"log/slog"
	"os"

	"github.com/trick77/noergler/internal/config"
	"github.com/trick77/noergler/internal/inference"
	"github.com/trick77/noergler/internal/jira"
	"github.com/trick77/noergler/internal/logging"
	"github.com/trick77/noergler/internal/review"
	"github.com/trick77/noergler/internal/riptide"
)

// Deps are the shared objects every team is built on. They outlive any one
// team and are the caller's to close.
//
// Claims and ReviewStore are the same *store.Store in production and are
// split here so a startup test can supply the reconciliation half without
// standing up the whole review surface.
type Deps struct {
	Claims      ClaimStore
	ReviewStore review.Store
	Bitbucket   review.BitbucketClient
	Tokens      review.TokenCounter
	Log         *slog.Logger
	// ReadFile loads a prompt template. Nil means os.ReadFile; tests pass a
	// map-backed one.
	ReadFile func(string) ([]byte, error)
	// Env is where llmwire reads its settings. Nil means the process
	// environment.
	Env func(string) (string, bool)
}

// Boot reconciles the config against the DB and starts every team.
//
// Reconcile runs first and to completion: review.New copies config.Review by
// value, so a Reviewer built before the DB has spoken would hold the
// teams.yaml author lists forever.
//
// A shared-layer failure (either claims table unreadable) comes back as err
// and aborts boot. Everything after that is per-team: one team's fault
// disables that team and nothing else.
func Boot(ctx context.Context, app *config.App, d Deps) (*Registry, error) {
	seedErrors, err := Reconcile(ctx, d.Claims, app.Teams, app.Order, d.Log)
	if err != nil {
		return nil, err
	}

	g := &Registry{
		enabled:  map[string]*Runtime{},
		disabled: map[string]string{},
		log:      d.Log,
	}
	for slug, reason := range app.Disabled {
		g.disabled[slug] = reason
	}

	for _, slug := range app.Order {
		team, ok := app.Teams[slug]
		if !ok {
			continue
		}
		// Per-team startup is a boundary: every line it emits carries the slug.
		teamCtx := logging.WithTeam(ctx, slug)
		if reason, seeded := seedErrors[slug]; seeded {
			g.disable(teamCtx, slug, reason)
			continue
		}
		rt, reason := safeStart(teamCtx, app, team, d)
		if reason != "" {
			g.disable(teamCtx, slug, reason)
			continue
		}
		g.enabled[slug] = rt
		d.Log.InfoContext(teamCtx, fmt.Sprintf("team_ready team=%s model=%s riptide=%s",
			slug,
			config.ModelLabel(team.LLM.Model, team.LLM.ReasoningEffort),
			onOff(rt.Riptide != nil && rt.Riptide.Enabled())))
	}

	enabled, disabled := g.Status()
	// The three startup lines are alerted on in Splunk, which matches the
	// rendered message, so the lists stay inside the string in the exact
	// ['a', 'b'] form the alert matches. Do not move them to attributes.
	summary := fmt.Sprintf("teams_ready enabled=%s disabled=%s", quotedList(enabled), quotedList(disabled))
	if len(disabled) > 0 {
		d.Log.WarnContext(ctx, summary)
	} else {
		d.Log.InfoContext(ctx, summary)
	}
	if len(enabled) == 0 {
		d.Log.ErrorContext(ctx, "no team is enabled; /ready reports 503 until the config is fixed")
	}
	return g, nil
}

// safeStart contains a panic in one team's startup.
//
// A panic becomes that team's disable reason. Without this a nil map or a
// slice index deep in a client constructor takes the whole instance down.
// Nothing a single team does may do that.
func safeStart(ctx context.Context, app *config.App, team *config.Team, d Deps) (rt *Runtime, reason string) {
	defer func() {
		if r := recover(); r != nil {
			rt, reason = nil, fmt.Sprintf("startup failed: panic: %v", r)
		}
	}()
	return Start(ctx, app, team, d)
}

func (g *Registry) disable(ctx context.Context, slug, reason string) {
	g.disabled[slug] = reason
	g.log.ErrorContext(ctx, fmt.Sprintf("team_disabled team=%s reason=%s", slug, reason))
}

// quotedList renders a slug list as ['a', 'b'] - single quotes, ", " between
// entries, [] when empty. The Splunk alert on teams_ready matches that exact
// form, so the rendering is pinned.
func quotedList(items []string) string {
	out := "["
	for i, s := range items {
		if i > 0 {
			out += ", "
		}
		out += "'" + s + "'"
	}
	return out + "]"
}

func onOff(b bool) string {
	if b {
		return "on"
	}
	return "off"
}

// Start builds one team. A non-empty reason means the team is disabled and
// the Runtime is nil.
//
// Every failure here is that team's alone. Nothing a single team does may
// take the instance down, which is why this returns a reason rather than an
// error.
func Start(ctx context.Context, app *config.App, team *config.Team, d Deps) (*Runtime, string) {
	readFile := d.ReadFile
	if readFile == nil {
		readFile = os.ReadFile
	}

	// Both templates are loaded before anything else, so a missing file
	// disables this team rather than aborting boot for all of them.
	reviewTmpl, err := readFile(team.Review.ReviewPromptTemplate)
	if err != nil {
		return nil, fmt.Sprintf("prompt template not found: %s", team.Review.ReviewPromptTemplate)
	}
	mentionTmpl, err := readFile(team.Review.MentionPromptTemplate)
	if err != nil {
		return nil, fmt.Sprintf("prompt template not found: %s", team.Review.MentionPromptTemplate)
	}

	llm, err := inference.New(inference.Options{
		Model:           team.LLM.Model,
		ReasoningEffort: team.LLM.ReasoningEffort,
		APIKey:          team.LLM.APIKey,
		ContextWindow:   team.LLM.ContextWindow,
		HeadroomTokens:  app.Trust.HeadroomTokens,
		Threshold:       app.Trust.Threshold,
		Tail:            app.Trust.Tail,
		Env:             d.Env,
		// Without this llmwire falls back to slog.Default(), so its per-call
		// line (tokens, cost, provenance, gateway call id) never reaches the
		// Splunk handler at all.
		//
		// The slug is bound onto the LOGGER, not taken from the context the
		// way the rest of the service does it: llmwire logs through plain
		// Debug/Info calls, never the *Context variants, so a context binding
		// would not reach it and its lines would land without team=.
		Logger: d.Log.With("team", team.Slug),
	})
	if err != nil {
		return nil, fmt.Sprintf("LLM check failed: %v", err)
	}
	if err := llm.Startup(ctx); err != nil {
		return nil, fmt.Sprintf("LLM check failed: %v", err)
	}
	// Said once at boot: an unpriced model means summaries carry no cost and
	// the per-PR cap never fires. Cost fails open, so this
	// log line is the only signal an operator gets.
	if pc := llm.PingCost(); pc.Priced() {
		d.Log.InfoContext(ctx, fmt.Sprintf("Model %s ping priced by the gateway: $%.3f",
			llm.Label(), float64(*pc.NanoUSD)/1e9))
	} else {
		d.Log.WarnContext(ctx, fmt.Sprintf("Model %s is not priced by the gateway "+
			"(no usable cost on the ping): summaries will carry no cost and the "+
			"per-PR cost cap never applies", llm.Label()))
	}

	var jr *jira.Client
	if team.Jira.URL != "" {
		jr, err = jira.New(team.Jira, d.Log)
		if err != nil {
			return nil, fmt.Sprintf("Jira client: %v", err)
		}
	}

	var rt *riptide.Emitter
	if team.Riptide != nil {
		rt = riptide.New(team.Riptide.URL, team.Riptide.Token, d.Log)
		if rt.Enabled() {
			if _, err := rt.VerifyAtStartup(ctx); err != nil {
				// Only a rejected token disables the team: emitting for weeks
				// against a bad token loses the rollups silently. Anything
				// else is riptide being down, which is best-effort anyway.
				if errors.Is(err, riptide.ErrAuth) {
					return nil, fmt.Sprintf("riptide check failed: %v", err)
				}
				d.Log.WarnContext(ctx, "riptide ping failed, team stays enabled", "team", team.Slug, "error", err)
			}
		} else {
			d.Log.InfoContext(ctx, "riptide disabled for this team; event forwarding off", "team", team.Slug)
		}
	}

	// A typed nil in an interface passes a != nil check, so the optional
	// dependencies go in as untyped nil when absent.
	var jiraDep review.JiraClient
	if jr != nil {
		jiraDep = jr
	}
	var riptideDep review.RiptideEmitter
	if rt != nil {
		riptideDep = rt
	}

	reviewer := review.New(review.Options{
		TeamSlug:        team.Slug,
		Bitbucket:       d.Bitbucket,
		LLM:             llm,
		Store:           d.ReviewStore,
		Jira:            jiraDep,
		Riptide:         riptideDep,
		Tokens:          d.Tokens,
		Config:          team.Review,
		Template:        string(reviewTmpl),
		MentionTemplate: string(mentionTmpl),
		Log:             d.Log,
	})
	return NewRuntime(team, reviewer, llm, jr, rt), ""
}
