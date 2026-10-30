package teams

import (
	"context"
	"errors"
	"fmt"
	"log/slog"

	"github.com/trick77/noergler/internal/almanach"
	"github.com/trick77/noergler/internal/bitbucket"
	"github.com/trick77/noergler/internal/config"
	"github.com/trick77/noergler/internal/inference"
	"github.com/trick77/noergler/internal/review"
	"github.com/trick77/noergler/internal/riptide"
	"github.com/trick77/noergler/internal/secretbox"
	"github.com/trick77/ship/kit/logging"
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
	// Bitbucket is the shared bot; a team's own bot is a clone of it.
	Bitbucket *bitbucket.Client
	Tokens    review.TokenCounter
	Log       *slog.Logger
	// Credentials and Box hold and open the accounts teams push. Nil
	// Credentials (startup tests) starts every team on the shared accounts.
	Credentials CredentialStore
	Box         *secretbox.Box
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
// A shared-layer failure (either claims table unreadable, the credentials
// unreadable, a secrets key that does not match the stored one) comes back
// as err and aborts boot. Everything after that is per-team: one team's
// fault disables that team and nothing else, and a fault in a team's own
// Bitbucket or Jira account does not even do that (accounts.go).
func Boot(ctx context.Context, app *config.App, d Deps) (*Registry, error) {
	seedErrors, err := Reconcile(ctx, d.Claims, app.Teams, app.Order, d.Log)
	if err != nil {
		return nil, err
	}
	var opened map[string]map[Kind]Opened
	if d.Credentials != nil {
		// Every configured team counts, disabled ones too: their rows are
		// not orphans, a fix to teams.yaml brings them back.
		known := map[string]bool{}
		for slug := range app.Teams {
			known[slug] = true
		}
		for slug := range app.Disabled {
			known[slug] = true
		}
		if opened, err = OpenCredentials(ctx, d.Credentials, d.Box, known, d.Log); err != nil {
			return nil, err
		}
	}

	g := NewRegistry(nil, nil, d.Log).WithNames(app.Names)
	for slug, reason := range app.Disabled {
		g.disabled[slug] = reason
	}

	for _, slug := range app.Order {
		team, ok := app.Teams[slug]
		if !ok {
			continue
		}
		// Per-team startup is a boundary: every line it emits carries the slug.
		teamCtx := logging.With(ctx, "team", slug)
		if reason, seeded := seedErrors[slug]; seeded {
			g.disable(teamCtx, slug, reason)
			continue
		}
		rt, reason := safeStart(teamCtx, app, team, d, opened[slug])
		if reason != "" {
			g.disable(teamCtx, slug, reason)
			continue
		}
		g.enabled[slug] = rt
		d.Log.InfoContext(teamCtx, fmt.Sprintf("team_ready team=%s model=%s riptide=%s",
			slug,
			modelLabel(rt, team),
			onOff(rt.Riptide != nil && rt.Riptide.Enabled())))
	}

	enabled, disabled := g.Status()
	// The three startup lines are alerted on in Splunk, which matches the
	// rendered message, so the lists stay inside the string in the exact
	// ['a', 'b'] form the alert matches. Do not move them to attributes.
	summary := fmt.Sprintf("teams_ready enabled=%s disabled=%s", config.QuotedList(enabled), config.QuotedList(disabled))
	if len(disabled) > 0 {
		d.Log.WarnContext(ctx, summary)
	} else {
		d.Log.InfoContext(ctx, summary)
	}
	if len(enabled) == 0 {
		d.Log.ErrorContext(ctx, "no team is enabled; /readyz reports 503 until the config is fixed")
	}
	return g, nil
}

// safeStart contains a panic in one team's startup.
//
// A panic becomes that team's disable reason. Without this a nil map or a
// slice index deep in a client constructor takes the whole instance down.
// Nothing a single team does may do that.
func safeStart(ctx context.Context, app *config.App, team *config.Team, d Deps, opened map[Kind]Opened) (rt *Runtime, reason string) {
	defer func() {
		if r := recover(); r != nil {
			rt, reason = nil, fmt.Sprintf("startup failed: panic: %v", r)
		}
	}()
	return Start(ctx, app, team, d, opened)
}

func (g *Registry) disable(ctx context.Context, slug, reason string) {
	g.disabled[slug] = reason
	g.log.ErrorContext(ctx, fmt.Sprintf("team_disabled team=%s reason=%s", slug, reason))
}

// modelLabel is the started client's label, which names the reasoning level
// llmwire resolved: an unset level would otherwise print as the bare model.
func modelLabel(rt *Runtime, team *config.Team) string {
	if rt != nil && rt.LLM != nil {
		return rt.LLM.Label()
	}
	return config.ModelLabel(team.LLM.Model, team.LLM.ReasoningEffort)
}

func onOff(b bool) string {
	if b {
		return "on"
	}
	return "off"
}

// verifyForwarder checks one of the team's forwarding endpoints (riptide,
// Almanach) at startup and returns the disable reason, if any. Only a
// rejected token (errAuth) disables the team: emitting for weeks against a
// bad token loses the data silently. Anything else is the service being
// down, which is best-effort anyway, and the team stays enabled. Not enabled
// (no URL or token) logs off, what the team goes without.
func (d Deps) verifyForwarder(ctx context.Context, slug, name string, enabled bool,
	verify func(context.Context) (string, error), errAuth error, off string) string {
	if !enabled {
		d.Log.InfoContext(ctx, name+" disabled for this team; "+off, "team", slug)
		return ""
	}
	if _, err := verify(ctx); err != nil {
		if errors.Is(err, errAuth) {
			return fmt.Sprintf("%s check failed: %v", name, err)
		}
		d.Log.WarnContext(ctx, name+" ping failed, team stays enabled", "team", slug, logging.Err(err))
	}
	return ""
}

// Start builds one team. A non-empty reason means the team is disabled and
// the Runtime is nil.
//
// Every failure here is that team's alone. Nothing a single team does may
// take the instance down, which is why this returns a reason rather than an
// error. opened is the team's own accounts, nil for none.
func Start(ctx context.Context, app *config.App, team *config.Team, d Deps, opened map[Kind]Opened) (*Runtime, string) {
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
		d.Log.InfoContext(ctx, fmt.Sprintf("Model %s ping priced by the gateway: $%s",
			llm.Label(), inference.FormatUSD3(*pc.NanoUSD)))
	} else {
		d.Log.WarnContext(ctx, fmt.Sprintf("Model %s is not priced by the gateway "+
			"(no usable cost on the ping): summaries will carry no cost and the "+
			"per-PR cost cap never applies", llm.Label()))
	}

	// The team's own accounts never disable it: a bad one is reported and
	// the team can push a working one (accounts.go). Bound to the team, so
	// its clients' lines carry team= like the LLM client's.
	acc, err := newAccounts(team.Slug, AccountDeps{
		Store: d.Credentials, Box: d.Box, Shared: d.Bitbucket, Jira: team.Jira,
		Log: d.Log.With("team", team.Slug),
	})
	if err != nil {
		return nil, fmt.Sprintf("Jira client: %v", err)
	}
	acc.start(ctx, opened)

	var rt *riptide.Emitter
	if team.Riptide != nil {
		rt = riptide.New(team.Riptide.URL, team.Riptide.Token, d.Log)
		if reason := d.verifyForwarder(ctx, team.Slug, "riptide", rt.Enabled(), rt.VerifyAtStartup,
			riptide.ErrAuth, "event forwarding off"); reason != "" {
			return nil, reason
		}
	}
	var al *almanach.Emitter
	if team.Almanach != nil {
		al = almanach.New(team.Almanach.URL, team.Almanach.Token, d.Log)
		if reason := d.verifyForwarder(ctx, team.Slug, "almanach", al.Enabled(), al.VerifyAtStartup,
			almanach.ErrAuth, "no notes"); reason != "" {
			return nil, reason
		}
	}

	// A typed nil in an interface passes a != nil check, so the optional
	// dependencies go in as untyped nil when absent.
	var jiraDep review.JiraClient
	if team.Jira.URL != "" {
		jiraDep = acc.jira
	}
	var riptideDep review.RiptideEmitter
	if rt != nil {
		riptideDep = rt
	}
	var almanachDep review.AlmanachEmitter
	if al != nil {
		almanachDep = al
	}

	// The Reviewer asks the live snapshot, which exists only once the
	// Runtime does; nothing reviews before Start returns.
	var runtime *Runtime
	reviewer := review.New(review.Options{
		Owns:      func(project, repo string) bool { return runtime.Team().Owns(project, repo) },
		Team:      func() *config.Team { return runtime.Team() },
		TeamSlug:  team.Slug,
		Bitbucket: acc.bot,
		LLM:       llm,
		Store:     d.ReviewStore,
		Jira:      jiraDep,
		Riptide:   riptideDep,
		Almanach:  almanachDep,
		Tokens:    d.Tokens,

		AlmanachNoteMaxChars: app.AlmanachLimits.NoteMaxChars,
		Config:               team.Review,
		Log:                  d.Log,

		DisplayNamePattern: app.Bitbucket.DisplayNamePattern,
		PublicURL:          app.Server.PublicURL,
	})
	runtime = NewRuntime(team, reviewer, llm, rt)
	runtime.accounts = acc
	runtime.Almanach = al
	return runtime, ""
}
