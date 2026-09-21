package api

import (
	"context"
	"net/http"
	"strconv"
	"time"

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
	RecentAttempts(ctx context.Context, team string, limit int) ([]store.AttemptRow, error)
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

type liveTeam struct {
	Slug    string     `json:"slug"`
	Enabled bool       `json:"enabled"`
	PRs     int        `json:"prs"`
	LastRun *time.Time `json:"last_run"`
}

type liveItem struct {
	Tag      string    `json:"tag"`
	Team     string    `json:"team"`
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
}

// live is the "what is going on right now" panel.
func (d Deps) live(w http.ResponseWriter, r *http.Request) {
	snap := d.Dashboard.Snapshot()

	body := liveBody{
		PoolCapacity: snap.Capacity,
		PoolPerTeam:  snap.PerTeam,
		Staged:       snap.Staged,
		Depth:        snap.Depth,
		Running:      make([]liveItem, 0, len(snap.Running)),
		Waiting:      make([]liveItem, 0, len(snap.Waiting)),
	}
	for _, it := range snap.Running {
		body.Running = append(body.Running, liveItem{
			Tag: it.Key.Tag(), Team: it.Team, Kind: it.Kind,
			Since: it.Since, WaitedMS: it.Waited.Milliseconds(),
		})
	}
	for _, it := range snap.Waiting {
		body.Waiting = append(body.Waiting, liveItem{
			Tag: it.Key.Tag(), Team: it.Team, Since: it.Since,
		})
	}

	enabled, disabled := d.Teams.Status()
	activity := map[string]store.TeamActivity{}
	if rows, err := d.DashboardStore.ActivityByTeam(r.Context()); err == nil {
		for _, a := range rows {
			activity[a.TeamSlug] = a
		}
	}

	body.Teams = make([]liveTeam, 0, len(enabled)+len(disabled))
	for _, slug := range enabled {
		t := liveTeam{Slug: slug, Enabled: true}
		if a, ok := activity[slug]; ok {
			t.PRs, t.LastRun = a.PRs, a.LastRun
		}
		body.Teams = append(body.Teams, t)
	}
	for _, slug := range disabled {
		// State only. The reason stays in the log; see the note above.
		t := liveTeam{Slug: slug}
		if a, ok := activity[slug]; ok {
			t.PRs, t.LastRun = a.PRs, a.LastRun
		}
		body.Teams = append(body.Teams, t)
	}

	httpapi.WriteJSON(w, http.StatusOK, body)
}

type runRow struct {
	Tag       string    `json:"tag"`
	Team      string    `json:"team"`
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
}

// runs is the feed: every attempt, including the ones that produced no run.
func (d Deps) runs(w http.ResponseWriter, r *http.Request) {
	limit, _ := strconv.Atoi(r.URL.Query().Get("limit"))
	rows, err := d.DashboardStore.RecentAttempts(r.Context(), r.URL.Query().Get("team"), limit)
	if err != nil {
		d.Log.ErrorContext(r.Context(), "dashboard: RecentAttempts failed: "+err.Error())
		httpapi.WriteDetail(w, http.StatusInternalServerError, "could not read runs")
		return
	}

	body := runsBody{Runs: make([]runRow, 0, len(rows))}
	for _, a := range rows {
		body.Runs = append(body.Runs, runRow{
			Tag: a.Key.Tag(), Team: a.TeamSlug, Kind: string(a.Kind),
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
	Team string `json:"team"`
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
	Since     time.Time        `json:"since"`
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

// metrics is spend and throughput over a window. days defaults to 14 and is
// clamped: an unbounded window would seq-scan however much history exists.
func (d Deps) metrics(w http.ResponseWriter, r *http.Request) {
	days, _ := strconv.Atoi(r.URL.Query().Get("days"))
	if days <= 0 || days > 90 {
		days = 14
	}
	since := time.Now().AddDate(0, 0, -days).Truncate(24 * time.Hour)
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

	body := metricsBody{
		Since:     since,
		Totals:    toTotals(totals),
		ByTeam:    make([]teamTotalsBody, 0, len(byTeam)),
		Daily:     make([]dayBody, 0, len(daily)),
		Attempts:  make([]outcomeDayBody, 0, len(attempts)),
		Breakdown: make([]breakdownBody, 0, len(breakdown)),
	}
	for _, t := range byTeam {
		body.ByTeam = append(body.ByTeam, teamTotalsBody{Team: t.TeamSlug, totalsBody: toTotals(t.Totals)})
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
	Slug              string      `json:"slug"`
	Enabled           bool        `json:"enabled"`
	PRs               int         `json:"prs"`
	LastRun           *time.Time  `json:"last_run"`
	Claims            []claimBody `json:"claims"`
	AutoReviewAuthors []string    `json:"auto_review_authors"`
	IgnoreAuthors     []string    `json:"ignore_authors"`
	ExcludeRepos      []string    `json:"exclude_repos"`
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

	activity := map[string]store.TeamActivity{}
	if rows, err := d.DashboardStore.ActivityByTeam(ctx); err == nil {
		for _, a := range rows {
			activity[a.TeamSlug] = a
		}
	}

	body := teamsBody{Teams: make([]teamBody, 0, len(enabled)+len(disabled))}
	add := func(slug string, enabled bool) {
		t := teamBody{
			Slug: slug, Enabled: enabled,
			Claims:            []claimBody{},
			AutoReviewAuthors: []string{},
			IgnoreAuthors:     []string{},
			ExcludeRepos:      []string{},
		}
		if a, ok := activity[slug]; ok {
			t.PRs, t.LastRun = a.PRs, a.LastRun
		}
		// ONE Runtime snapshot per team per request: Runtime is
		// copy-on-write behind an atomic.Pointer, so reading Team() twice
		// would let a concurrent settings write land between the two and
		// report a team that never existed in that combination.
		if rt, _, ok := d.Teams.Lookup(slug); ok && rt != nil {
			if team := rt.Team(); team != nil {
				t.AutoReviewAuthors = append(t.AutoReviewAuthors, team.Review.AutoReviewAuthors...)
				t.IgnoreAuthors = append(t.IgnoreAuthors, team.Review.IgnoreAuthors...)
				t.ExcludeRepos = append(t.ExcludeRepos, team.Review.ExcludeRepos...)
				// A scope with no repo list is the whole project; one repo
				// per claim row otherwise, so the page can show what a team
				// actually owns rather than a project it half-owns.
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
