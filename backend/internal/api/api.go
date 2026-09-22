// Package api is the inbound HTTP surface: the Bitbucket webhook, team
// self-service and onboarding.
//
// It sits on top of internal/httpapi, which owns the mux, the middleware and
// the two JSON writers. Keeping the handlers here rather than there is what
// lets httpapi stay a transport layer with a test that links nothing: these
// routes pull in the store, the queue and the review pipeline.
package api

import (
	"log/slog"

	"github.com/trick77/noergler/internal/bitbucket"
	"github.com/trick77/noergler/internal/httpapi"
	"github.com/trick77/noergler/internal/onboarding"
	"github.com/trick77/noergler/internal/queue"
	"github.com/trick77/noergler/internal/store"
	"github.com/trick77/noergler/internal/teams"
	"github.com/trick77/noergler/internal/webhook"
	"github.com/trick77/noergler/web"
)

// Submitter is the review queue as the routes use it.
//
// Consumer-side, so a route test can assert which handler the dispatch table
// picked without starting a worker.
type Submitter interface {
	Submit(key store.PRKey, payload *webhook.Payload, team string) string
	SubmitJob(key store.PRKey, team string, fn queue.JobFunc) string
}

// BotClient is the instance's Bitbucket client as the routes use it: the bot
// read proof, plus the per-request clone that carries the caller's own token.
//
// WithToken returns an interface rather than *bitbucket.Client so a route
// test can supply a fake; bitbucketClient below adapts the real one.
type BotClient interface {
	onboarding.BotClient
	WithToken(token string) onboarding.AdminClient
}

// bitbucketClient adapts *bitbucket.Client, whose WithToken returns its own
// concrete type, to BotClient.
type bitbucketClient struct{ *bitbucket.Client }

func (c bitbucketClient) WithToken(token string) onboarding.AdminClient {
	return c.Client.WithToken(token)
}

// Bot wraps the instance client for Deps.Bitbucket.
func Bot(c *bitbucket.Client) BotClient { return bitbucketClient{c} }

// Deps is what the routes need.
type Deps struct {
	Teams *teams.Registry
	Queue Submitter
	Log   *slog.Logger
	// Store persists team settings.
	Store SettingsStore
	// Claims is the claim half of the store, for onboarding.
	Claims onboarding.ClaimStore
	// Bitbucket is the instance bot client. Onboarding writes go through a
	// per-request clone carrying the caller.s own token.
	Bitbucket BotClient
	// PublicURL is this instance.s base URL. Empty disables /onboard.
	PublicURL string
	// BitbucketURL is the Bitbucket base the dashboard links PR tags to.
	// Serving it is the only way the browser can learn it: it is read from
	// BITBUCKET_URL in this process and the SPA ships as static files.
	BitbucketURL string
	// BotUsername is the instance's Bitbucket account, read from
	// BITBUCKET_USERNAME. The @mention trigger is instance-wide, not
	// per-team.
	BotUsername string
	// Dashboard is the review queue's read side, for the live panel.
	Dashboard DashboardQueue
	// DashboardStore is the read-only half of the store the dashboard uses.
	DashboardStore DashboardStore
}

// Register adds every route this package owns to the server.
func Register(srv *httpapi.Server, d Deps) {
	srv.HandleFunc("POST /webhook/{team}", d.webhook)
	srv.HandleFunc("GET /teams/{team}", d.getTeam)
	srv.HandleFunc("PUT /teams/{team}/settings", d.putTeamSettings)
	srv.HandleFunc("POST /onboard/{team}", d.onboard)

	// The contract for the four routes above, and the page that renders it.
	// NOT inside the dashboard block: the self-service API exists without the
	// dashboard, so its documentation has to as well.
	srv.HandleFunc("GET /api/openapi.yaml", d.openapiSpec)
	srv.HandleFunc("GET /api/docs", d.docs)

	// Read-only, unauthenticated, cross-team. Registered only when the
	// dashboard's dependencies are wired, so an instance that does not want
	// it simply does not pass them and the routes do not exist.
	if d.Dashboard != nil && d.DashboardStore != nil {
		srv.HandleFunc("GET /api/dashboard/live", d.live)
		srv.HandleFunc("GET /api/dashboard/runs", d.runs)
		srv.HandleFunc("GET /api/dashboard/metrics", d.metrics)
		srv.HandleFunc("GET /api/dashboard/teams", d.teamsView)

		// The SPA on the catch-all, registered LAST and only alongside the
		// API it reads. Go's ServeMux picks the most specific pattern, so
		// every route above and every probe still wins over "/".
		srv.Handle("/", web.Handler())
	}
}
