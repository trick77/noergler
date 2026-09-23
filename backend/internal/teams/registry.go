package teams

import (
	"context"
	"log/slog"
	"sort"

	"github.com/trick77/noergler/internal/review"
	"github.com/trick77/noergler/internal/webhook"
)

// Registry is the fixed set of teams after startup.
//
// Both maps are written once, before the server accepts traffic, and only
// read afterwards, so a lookup needs no lock. A team that failed to start is
// disabled for the life of the process: fixing it is a config change and a
// redeploy, which is what the 503 tells the caller.
type Registry struct {
	enabled  map[string]*Runtime
	disabled map[string]string
	// names is the teams.yaml display name by slug, disabled teams included:
	// a disabled team has no Runtime to ask.
	names map[string]string
	log   *slog.Logger
}

// NewRegistry builds a registry directly, for tests and for callers that
// start teams themselves. Boot is the normal way in.
func NewRegistry(enabled map[string]*Runtime, disabled map[string]string, log *slog.Logger) *Registry {
	if enabled == nil {
		enabled = map[string]*Runtime{}
	}
	if disabled == nil {
		disabled = map[string]string{}
	}
	return &Registry{enabled: enabled, disabled: disabled, names: map[string]string{}, log: log}
}

// WithNames sets the display names of teams without a Runtime. Call it before
// the registry is shared, like everything else that writes it.
func (g *Registry) WithNames(names map[string]string) *Registry {
	for slug, name := range names {
		g.names[slug] = name
	}
	return g
}

// Name is the team's display name, the slug when it has none. An enabled
// team answers from its Runtime, a disabled one from teams.yaml.
func (g *Registry) Name(slug string) string {
	if rt, ok := g.enabled[slug]; ok && rt != nil {
		if team := rt.Team(); team != nil && team.Name != "" {
			return team.Name
		}
	}
	if name := g.names[slug]; name != "" {
		return name
	}
	return slug
}

// Lookup resolves a slug.
//
// ok reports whether the team takes traffic. When it does not, reason is
// non-empty if the slug is known but disabled (503) and empty if the slug is
// unknown (404). The two are deliberately distinguishable: they leak nothing
// the webhook route does not already leak, and telling an operator "disabled"
// rather than "unknown" is the difference between reading the startup log and
// hunting a typo.
func (g *Registry) Lookup(slug string) (rt *Runtime, reason string, ok bool) {
	if rt, found := g.enabled[slug]; found {
		return rt, "", true
	}
	if why, found := g.disabled[slug]; found {
		return nil, why, false
	}
	return nil, "", false
}

// Status feeds httpapi.TeamStatus. Slugs only: a disable reason carries
// gateway URLs, error bodies and env var names and belongs in the log, not
// on an unauthenticated probe.
func (g *Registry) Status() (enabled, disabled []string) {
	enabled = make([]string, 0, len(g.enabled))
	for slug := range g.enabled {
		enabled = append(enabled, slug)
	}
	disabled = make([]string, 0, len(g.disabled))
	for slug := range g.disabled {
		disabled = append(disabled, slug)
	}
	sort.Strings(enabled)
	sort.Strings(disabled)
	return enabled, disabled
}

// Scheduler stages a review's inference off the worker. An alias, not a
// second declaration: it has to be the very type the review package names
// in its own signature, or a Reviewer cannot satisfy this package's
// interface.
type Scheduler = review.Scheduler

// Review matches queue.ReviewFunc. The team was authenticated by the webhook
// route, so the worker only has to find it again.
//
// queue.run already binds team= into the context, so this does not.
func (g *Registry) Review(ctx context.Context, team string, p *webhook.Payload, sched Scheduler) bool {
	rt, _, ok := g.Lookup(team)
	if !ok {
		// A team can only be disabled at startup, so this means the queue
		// outlived a config the process no longer has. Drop it loudly.
		g.log.ErrorContext(ctx, "queued review for unknown team dropped", "team", team)
		return false
	}
	// Staged: the gateway call runs on the inference pool and the posting
	// comes back to the worker, so a 200s+ inference no longer holds the
	// queue. True means the work outlives this call and the PR's hold must
	// survive with it.
	return rt.Reviewer.ReviewPullRequestStaged(ctx, p, team, false, sched)
}

// There is deliberately no Close. Each team's clients hold an http.Client
// whose transport needs no teardown, and the pool is the caller's.
