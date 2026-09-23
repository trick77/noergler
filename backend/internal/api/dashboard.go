package api

import (
	"context"
	"net/http"
	"strconv"
	"time"

	"github.com/trick77/noergler/internal/config"
	"github.com/trick77/noergler/internal/httpapi"
	"github.com/trick77/noergler/internal/queue"
	"github.com/trick77/noergler/internal/review"
	"github.com/trick77/noergler/internal/store"
)

// The dashboard is read-only and cross-team by nature: it answers "what is
// this instance doing", which no per-team route can. It carries no auth of
// its own, so the ingress is the boundary.
//
// What it therefore must NOT serve: a team's disable reason. /health withholds
// those deliberately, because a reason can name an env var
// (TEAM_X_OPENAI_API_KEY), and an unauthenticated route that leaks the name of
// a secret is worse than one that says nothing. Slugs and states only, same as
// /health.
//
// Every body here is a struct, never a map: encoding/json sorts map keys, and
// this repo pins struct bodies.

// DashboardQueue is the review queue as the dashboard reads it. Consumer-side,
// so a route test needs no worker.
type DashboardQueue interface {
	Snapshot() queue.Snapshot
}

// DashboardStore is the read half of the store the dashboard uses. Every
// method is read-only; nothing here writes.
type DashboardStore interface {
	RecentAttempts(ctx context.Context, team string, outcome store.AttemptFilter, limit int) ([]store.AttemptRow, error)
	OutcomeBreakdown(ctx context.Context, since time.Time) ([]store.OutcomeCount, error)
	TotalsSince(ctx context.Context, since time.Time) (store.Totals, error)
	TotalsByTeam(ctx context.Context, since time.Time) ([]store.TeamTotals, error)
	DailyByTeam(ctx context.Context, since time.Time) ([]store.Bucket, error)
	DailyAttempts(ctx context.Context, since time.Time) ([]store.AttemptBucket, error)
	ActivityByTeam(ctx context.Context) ([]store.TeamActivity, error)
}

// nanoPerUSD converts the store's BIGINT nano-USD to the USD the edge shows.
const nanoPerUSD = 1_000_000_000

// usd renders nano-USD as a fixed-point decimal string with 3 decimals, the
// same precision the cost log lines carry.
//
// A string, never a float: a JSON number would hand the browser a binary
// float of a decimal amount. Never an exponent either - 1E-9 breaks strict
// parsers, which is why this formats rather than letting the encoder choose.
// nil stays nil: an unpriced run is not a free one.
func usd(nano *int64) *string {
	if nano == nil {
		return nil
	}
	s := strconv.FormatFloat(float64(*nano)/nanoPerUSD, 'f', 3, 64)
	return &s
}

// teamNamer answers a slug with the team's display name, for the pages that
// build their rows from the store and so know only the slug.
//
// The map is built ONCE per request and read per row, so a name cannot
// change halfway down a response. It is built with one Lookup per enabled
// team, not from a single Runtime snapshot: Name comes from teams.yaml and
// ApplySettings never touches it, so two teams read a moment apart cannot
// disagree about it.
//
// That is the invariant to keep. Make a display name settable through
// PUT /teams/{slug}/settings and this loop starts reading N independent
// copy-on-write snapshots, and a concurrent write lands between two of them.
//
// A disabled team has no Runtime, so its name comes from teams.yaml via the
// registry; without that the roster mixes names and slugs. A team dropped
// from teams.yaml still owns review_runs rows forever, so the rows outlive
// the config that names them: that miss answers with the slug, which is what
// the logs, the webhook path and team= already print, rather than a blank
// cell.
func (d Deps) teamNamer() func(slug string) string {
	names := map[string]string{}
	enabled, disabled := d.Teams.Status()
	for _, slug := range append(enabled, disabled...) {
		names[slug] = d.Teams.Name(slug)
	}
	return func(slug string) string {
		if name, ok := names[slug]; ok {
			return name
		}
		return slug
	}
}

// Team states. "ready" is not the same claim as "configured": a team can
// pass every startup check and still review nothing, because it owns no
// repositories. Nothing else reports that, and an operator reading a green
// pill beside an idle team has no way to tell the two apart.
const (
	teamReady    = "ready"
	teamNoScope  = "no_repos"
	teamDisabled = "disabled"
)

type liveTeam struct {
	Slug string `json:"slug"`
	// Name is the display name from teams.yaml, defaulted to the slug when
	// the team has none or is no longer configured. The slug stays on the
	// wire beside it: it is the identity the logs and the webhook path use,
	// and the page keys and links off it.
	Name string `json:"name"`
	PRs  int    `json:"prs"`
	// LastReviewed is the last success. LastRun is the last run of any
	// outcome, which is what "run" means on every page; the two differ
	// exactly when the latest run was skipped or failed, and showing only
	// the success under "last run" hid that.
	LastReviewed  *time.Time `json:"last_reviewed"`
	LastRun       time.Time  `json:"last_run"`
	LastOutcome   string     `json:"last_outcome"`
	LastReason    string     `json:"last_reason,omitempty"`
	LastReasonMsg string     `json:"last_reason_label,omitempty"`
}

// scopeSize counts the repositories a team owns.
//
// 0 means no projects AND no repos: the team reviews nothing, whatever else
// is configured. -1 means at least one whole-project claim, which covers
// every repo in that project, now and every one added later; that is
// unbounded coverage, not a count, and returning a number for it would be a
// guess that goes stale the moment a repo is added.
func scopeSize(scopes []config.ProjectScope) int {
	n := 0
	for _, sc := range scopes {
		if len(sc.Repos) == 0 {
			return -1
		}
		n += len(sc.Repos)
	}
	return n
}

// teamState folds the startup verdict and the team's scope into one pill.
func teamState(enabled bool, repos int) string {
	if !enabled {
		return teamDisabled
	}
	if repos == 0 {
		return teamNoScope
	}
	return teamReady
}

type liveItem struct {
	Tag string `json:"tag"`
	// Team is the slug; TeamName is what the page prints.
	Team     string    `json:"team"`
	TeamName string    `json:"team_name"`
	Kind     string    `json:"kind,omitempty"`
	Since    time.Time `json:"since"`
	WaitedMS int64     `json:"waited_ms,omitempty"`
}

type liveBody struct {
	PoolCapacity int        `json:"pool_capacity"`
	PoolPerTeam  int        `json:"pool_per_team"`
	Staged       int        `json:"staged"`
	Depth        int        `json:"depth"`
	Running      []liveItem `json:"running"`
	Waiting      []liveItem `json:"waiting"`
	Teams        []liveTeam `json:"teams"`
	// BitbucketURL is the instance's Bitbucket base, so the page can turn a
	// PROJECT/repo#id tag into a link. The browser has no other way to know
	// it: BITBUCKET_URL is read in this process and the SPA is static files.
	// Empty when unset, and the page then renders the tag as plain text.
	BitbucketURL string `json:"bitbucket_url"`
}

// live is the "what is going on right now" panel.
func (d Deps) live(w http.ResponseWriter, r *http.Request) {
	snap := d.Dashboard.Snapshot()
	nameOf := d.teamNamer()

	body := liveBody{
		PoolCapacity: snap.Capacity,
		PoolPerTeam:  snap.PerTeam,
		Staged:       snap.Staged,
		Depth:        snap.Depth,
		Running:      make([]liveItem, 0, len(snap.Running)),
		Waiting:      make([]liveItem, 0, len(snap.Waiting)),
		BitbucketURL: d.BitbucketURL,
	}
	for _, it := range snap.Running {
		body.Running = append(body.Running, liveItem{
			Tag: it.Key.Tag(), Team: it.Team, TeamName: nameOf(it.Team), Kind: it.Kind,
			Since: it.Since, WaitedMS: it.Waited.Milliseconds(),
		})
	}
	for _, it := range snap.Waiting {
		body.Waiting = append(body.Waiting, liveItem{
			Tag: it.Key.Tag(), Team: it.Team, TeamName: nameOf(it.Team), Since: it.Since,
		})
	}

	// Enabled teams with at least one run, nothing else. A disabled team
	// and one that has never run have nothing live to show; the Teams page
	// lists every team with its state.
	enabled, _ := d.Teams.Status()
	activity := d.activityByTeam(r.Context())

	body.Teams = make([]liveTeam, 0, len(enabled))
	for _, slug := range enabled {
		a, ok := activity[slug]
		if !ok || a.LastRun == nil {
			continue
		}
		body.Teams = append(body.Teams, liveTeam{
			Slug: slug, Name: nameOf(slug), PRs: a.PRs,
			LastReviewed: a.LastReviewed, LastRun: *a.LastRun,
			LastOutcome: a.LastOutcome, LastReason: a.LastReason,
			LastReasonMsg: review.SkipReason(a.LastReason).Label(),
		})
	}

	httpapi.WriteJSON(w, http.StatusOK, body)
}

type runRow struct {
	Tag string `json:"tag"`
	// Team is the slug the row was stored under; TeamName is what the page
	// prints. These rows outlive teams.yaml, so a name is not always there.
	Team      string    `json:"team"`
	TeamName  string    `json:"team_name"`
	Kind      string    `json:"kind"`
	Outcome   string    `json:"outcome"`
	Reason    string    `json:"reason,omitempty"`
	ReasonMsg string    `json:"reason_label,omitempty"`
	ElapsedMS *int64    `json:"elapsed_ms"`
	Findings  *int      `json:"findings"`
	Cost      *string   `json:"cost_usd"`
	CreatedAt time.Time `json:"created_at"`
}

type runsBody struct {
	Runs []runRow `json:"runs"`
	// BitbucketURL as on liveBody: this page polls its own endpoint, so it
	// cannot read the base off the live response.
	BitbucketURL string `json:"bitbucket_url"`
}

// attemptFilter reads ?outcome=. An unknown value is every row rather than an
// error: the route is unauthenticated and read-only, and a mistyped query
// string should show the feed, not a 4xx the page has no way to explain.
func attemptFilter(v string) store.AttemptFilter {
	switch store.AttemptFilter(v) {
	case store.AttemptsFailed:
		return store.AttemptsFailed
	case store.AttemptsSkipped:
		return store.AttemptsSkipped
	default:
		return store.AttemptsAll
	}
}

// runs is the feed: every attempt, including the ones that produced no run.
func (d Deps) runs(w http.ResponseWriter, r *http.Request) {
	limit, _ := strconv.Atoi(r.URL.Query().Get("limit"))
	rows, err := d.DashboardStore.RecentAttempts(
		r.Context(), r.URL.Query().Get("team"),
		attemptFilter(r.URL.Query().Get("outcome")), limit,
	)
	if err != nil {
		d.Log.ErrorContext(r.Context(), "dashboard: RecentAttempts failed: "+err.Error())
		httpapi.WriteDetail(w, http.StatusInternalServerError, "could not read runs")
		return
	}

	nameOf := d.teamNamer()
	body := runsBody{Runs: make([]runRow, 0, len(rows)), BitbucketURL: d.BitbucketURL}
	for _, a := range rows {
		body.Runs = append(body.Runs, runRow{
			Tag: a.Key.Tag(), Team: a.TeamSlug, TeamName: nameOf(a.TeamSlug), Kind: string(a.Kind),
			Outcome: a.Outcome, Reason: a.Reason,
			ReasonMsg: review.SkipReason(a.Reason).Label(),
			ElapsedMS: a.ElapsedMS, Findings: a.Findings,
			Cost: usd(a.CostNanoUSD), CreatedAt: a.CreatedAt,
		})
	}
	httpapi.WriteJSON(w, http.StatusOK, body)
}

type totalsBody struct {
	Runs             int     `json:"runs"`
	PromptTokens     int64   `json:"prompt_tokens"`
	CachedTokens     int64   `json:"cached_tokens"`
	CompletionTokens int64   `json:"completion_tokens"`
	FindingsPosted   int64   `json:"findings_posted"`
	Cost             *string `json:"cost_usd"`
	// Unpriced is runs the gateway did not price. Reported beside the cost,
	// never inside it.
	Unpriced int `json:"unpriced_runs"`
}

type teamTotalsBody struct {
	Team     string `json:"team"`
	TeamName string `json:"team_name"`
	totalsBody
}

type dayBody struct {
	Day  string  `json:"day"`
	Team string  `json:"team"`
	Runs int     `json:"runs"`
	Cost *string `json:"cost_usd"`
}

type outcomeDayBody struct {
	Day     string `json:"day"`
	Outcome string `json:"outcome"`
	Count   int    `json:"count"`
}

type breakdownBody struct {
	Outcome string `json:"outcome"`
	Reason  string `json:"reason,omitempty"`
	Label   string `json:"label,omitempty"`
	Count   int    `json:"count"`
}

type metricsBody struct {
	Since time.Time `json:"since"`
	// Until is the exclusive end of the window, so the page can lay out a
	// month's days without deciding where the month ends itself.
	Until time.Time `json:"until"`
	// Window is "month" or "rolling", so the page can title itself honestly
	// rather than assuming which one it asked for.
	Window    string           `json:"window"`
	Totals    totalsBody       `json:"totals"`
	ByTeam    []teamTotalsBody `json:"by_team"`
	Daily     []dayBody        `json:"daily"`
	Attempts  []outcomeDayBody `json:"daily_attempts"`
	Breakdown []breakdownBody  `json:"breakdown"`
}

func toTotals(t store.Totals) totalsBody {
	return totalsBody{
		Runs: t.Runs, PromptTokens: t.PromptTokens, CachedTokens: t.CachedTokens,
		CompletionTokens: t.CompletionTokens, FindingsPosted: t.FindingsPosted,
		Cost: usd(t.CostNanoUSD), Unpriced: t.Unpriced,
	}
}

// monthStart is midnight on the first of the month that t falls in, in the
// instance's own location. Local, not UTC: an operator reading a spend figure
// means their month, and a UTC boundary would move the figure for anyone west
// of it on the first of the month.
func monthStart(t time.Time) time.Time {
	return time.Date(t.Year(), t.Month(), 1, 0, 0, 0, 0, t.Location())
}

// dayStart is local midnight on t's own day.
//
// NOT time.Truncate(24h): that works on absolute time since the epoch, so it
// lands on UTC midnight whatever the location. On a TZ=Europe/Zurich pod that
// is 02:00 local, putting two hours of the day inside a window that claims to
// start at midnight, and it disagrees with both monthStart above and the
// date_trunc('day') the daily queries bucket by.
func dayStart(t time.Time) time.Time {
	return time.Date(t.Year(), t.Month(), t.Day(), 0, 0, 0, 0, t.Location())
}

// activityByTeam is the per-team history the live panel and the roster show
// beside a team's state.
//
// It degrades rather than failing the request: the queue and the registry are
// in memory and always answer, so a DB hiccup should cost the history and not
// the panel an operator opened to see what is running. But it LOGS: the same
// read failing silently meant every team showed "0 PRs, never" with nothing
// anywhere to say why.
func (d Deps) activityByTeam(ctx context.Context) map[string]store.TeamActivity {
	activity := map[string]store.TeamActivity{}
	rows, err := d.DashboardStore.ActivityByTeam(ctx)
	if err != nil {
		d.Log.ErrorContext(ctx, "dashboard: ActivityByTeam failed, reporting teams without history: "+err.Error())
		return activity
	}
	for _, a := range rows {
		activity[a.TeamSlug] = a
	}
	return activity
}

// metrics is spend and throughput for the CURRENT CALENDAR MONTH, which is
// the window a key's spend is actually budgeted and invoiced in. A rolling
// 14 days answers a different question and cannot be reconciled with a bill.
//
// ?days=N overrides it with a rolling window, clamped: an unbounded one would
// seq-scan however much history exists. Absent means the month.
func (d Deps) metrics(w http.ResponseWriter, r *http.Request) {
	now := time.Now()
	since := monthStart(now)
	if raw := r.URL.Query().Get("days"); raw != "" {
		days, _ := strconv.Atoi(raw)
		if days <= 0 || days > 90 {
			days = 14
		}
		since = dayStart(now.AddDate(0, 0, -days))
	}
	ctx := r.Context()

	totals, err := d.DashboardStore.TotalsSince(ctx, since)
	if err != nil {
		d.Log.ErrorContext(ctx, "dashboard: TotalsSince failed: "+err.Error())
		httpapi.WriteDetail(w, http.StatusInternalServerError, "could not read metrics")
		return
	}
	byTeam, err := d.DashboardStore.TotalsByTeam(ctx, since)
	if err != nil {
		d.Log.ErrorContext(ctx, "dashboard: TotalsByTeam failed: "+err.Error())
		httpapi.WriteDetail(w, http.StatusInternalServerError, "could not read metrics")
		return
	}
	daily, err := d.DashboardStore.DailyByTeam(ctx, since)
	if err != nil {
		d.Log.ErrorContext(ctx, "dashboard: DailyByTeam failed: "+err.Error())
		httpapi.WriteDetail(w, http.StatusInternalServerError, "could not read metrics")
		return
	}
	attempts, err := d.DashboardStore.DailyAttempts(ctx, since)
	if err != nil {
		d.Log.ErrorContext(ctx, "dashboard: DailyAttempts failed: "+err.Error())
		httpapi.WriteDetail(w, http.StatusInternalServerError, "could not read metrics")
		return
	}
	breakdown, err := d.DashboardStore.OutcomeBreakdown(ctx, since)
	if err != nil {
		d.Log.ErrorContext(ctx, "dashboard: OutcomeBreakdown failed: "+err.Error())
		httpapi.WriteDetail(w, http.StatusInternalServerError, "could not read metrics")
		return
	}

	window := "month"
	until := monthStart(now).AddDate(0, 1, 0)
	if r.URL.Query().Get("days") != "" {
		window = "rolling"
		until = dayStart(now.AddDate(0, 0, 1))
	}

	body := metricsBody{
		Since:     since,
		Until:     until,
		Window:    window,
		Totals:    toTotals(totals),
		ByTeam:    make([]teamTotalsBody, 0, len(byTeam)),
		Daily:     make([]dayBody, 0, len(daily)),
		Attempts:  make([]outcomeDayBody, 0, len(attempts)),
		Breakdown: make([]breakdownBody, 0, len(breakdown)),
	}
	nameOf := d.teamNamer()
	for _, t := range byTeam {
		body.ByTeam = append(body.ByTeam, teamTotalsBody{
			Team: t.TeamSlug, TeamName: nameOf(t.TeamSlug), totalsBody: toTotals(t.Totals),
		})
	}
	for _, b := range daily {
		body.Daily = append(body.Daily, dayBody{
			Day: b.Day.Format(time.DateOnly), Team: b.TeamSlug,
			Runs: b.Runs, Cost: usd(b.CostNanoUSD),
		})
	}
	for _, b := range attempts {
		body.Attempts = append(body.Attempts, outcomeDayBody{
			Day: b.Day.Format(time.DateOnly), Outcome: b.Outcome, Count: b.Count,
		})
	}
	for _, c := range breakdown {
		e := breakdownBody{Outcome: c.Outcome, Reason: c.Reason, Count: c.Count}
		if c.Reason != "" {
			e.Label = review.SkipReason(c.Reason).Label()
		}
		body.Breakdown = append(body.Breakdown, e)
	}
	httpapi.WriteJSON(w, http.StatusOK, body)
}

type claimBody struct {
	Project string `json:"project"`
	Repo    string `json:"repo,omitempty"`
}

type teamBody struct {
	Slug    string `json:"slug"`
	Name    string `json:"name"`
	Enabled bool   `json:"enabled"`
	State   string `json:"state"`
	Repos   int    `json:"repos"`
	PRs     int    `json:"prs"`
	// Same split as liveTeam: last success, and last run of any outcome.
	LastReviewed *time.Time  `json:"last_reviewed"`
	LastRun      *time.Time  `json:"last_run"`
	Claims       []claimBody `json:"claims"`
	// COUNTS, not names. The author lists are Bitbucket usernames, and this
	// route is unauthenticated and cross-team: serving them here would hand
	// out a roster of who works on what, on the same response that
	// deliberately withholds a team's disable reason. A count still answers
	// "is this team configured"; the names stay on the team's own
	// authenticated /teams/{slug} route.
	AutoReviewAuthors int `json:"auto_review_authors"`
	IgnoreAuthors     int `json:"ignore_authors"`
	// exclude_repos stays whole: a repo glob is not a person.
	ExcludeRepos []string `json:"exclude_repos"`
}

type teamsBody struct {
	Teams []teamBody `json:"teams"`
}

// teamsView is the roster: who is configured, what they own, how they are set
// up. Settings are already served per team to their own owner; here they are
// read across teams, which is the whole point of an operator view.
func (d Deps) teamsView(w http.ResponseWriter, r *http.Request) {
	enabled, disabled := d.Teams.Status()
	ctx := r.Context()

	activity := d.activityByTeam(ctx)

	body := teamsBody{Teams: make([]teamBody, 0, len(enabled)+len(disabled))}
	add := func(slug string, isEnabled bool) {
		t := teamBody{
			// Name comes from the registry for a disabled team, which has no
			// snapshot; the snapshot below overwrites it for an enabled one.
			Slug: slug, Name: d.Teams.Name(slug), Enabled: isEnabled,
			Claims:       []claimBody{},
			ExcludeRepos: []string{},
		}
		if a, ok := activity[slug]; ok {
			t.PRs, t.LastReviewed, t.LastRun = a.PRs, a.LastReviewed, a.LastRun
		}
		// ONE Runtime snapshot per team per request: Runtime is
		// copy-on-write behind an atomic.Pointer, so reading Team() twice
		// would let a concurrent settings write land between the two and
		// report a team that never existed in that combination.
		if rt, _, ok := d.Teams.Lookup(slug); ok && rt != nil {
			if team := rt.Team(); team != nil {
				if team.Name != "" {
					t.Name = team.Name
				}
				t.AutoReviewAuthors = len(team.Review.AutoReviewAuthors)
				t.IgnoreAuthors = len(team.Review.IgnoreAuthors)
				t.ExcludeRepos = append(t.ExcludeRepos, team.Review.ExcludeRepos...)
				// A scope with no repo list is the whole project; one repo
				// per claim row otherwise, so the page can show what a team
				// actually owns rather than a project it half-owns.
				t.Repos = scopeSize(team.Projects)
				for _, sc := range team.Projects {
					if len(sc.Repos) == 0 {
						t.Claims = append(t.Claims, claimBody{Project: sc.Key})
						continue
					}
					for _, repo := range sc.Repos {
						t.Claims = append(t.Claims, claimBody{Project: sc.Key, Repo: repo})
					}
				}
			}
		}
		t.State = teamState(isEnabled, t.Repos)
		body.Teams = append(body.Teams, t)
	}
	for _, slug := range enabled {
		add(slug, true)
	}
	for _, slug := range disabled {
		add(slug, false)
	}
	httpapi.WriteJSON(w, http.StatusOK, body)
}
