package api

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"log/slog"
	"net/http"
	"net/http/httptest"
	"slices"
	"strings"
	"testing"
	"time"

	"github.com/trick77/noergler/internal/almanach"
	"github.com/trick77/noergler/internal/config"
	"github.com/trick77/noergler/internal/forward"
	"github.com/trick77/noergler/internal/httpapi"
	"github.com/trick77/noergler/internal/inference"
	"github.com/trick77/noergler/internal/queue"
	"github.com/trick77/noergler/internal/riptide"
	"github.com/trick77/noergler/internal/store"
	"github.com/trick77/noergler/internal/teams"
	"github.com/trick77/noergler/web"
	"github.com/trick77/ship/kit/logging"
	"github.com/trick77/toolkit/llmwire"
	"github.com/trick77/toolkit/llmwire/llmwiretest"
)

type fakeDashQueue struct{ snap queue.Snapshot }

func (f *fakeDashQueue) Snapshot() queue.Snapshot { return f.snap }

type fakeDashStore struct {
	attempts  []store.AttemptRow
	breakdown []store.OutcomeCount
	totals    store.Totals
	reviewed  int
	seen      int
	// dailySince and attemptsSince record where the daily reads started.
	dailySince    *time.Time
	attemptsSince *time.Time
	byTeam        []store.TeamTotals
	fixCounts     []store.FixCount
	daily         []store.Bucket
	dailyAtt      []store.AttemptBucket
	activity      []store.TeamActivity
	// ignored is the per-team count of notes Almanach ignored; ignoredSince
	// records the window the handler asked for.
	// ignoredErr fails the count read alone; ignoredReads counts the reads.
	ignored      map[string]int
	ignoredSince time.Time
	ignoredErr   error
	ignoredReads int
	// page and run are served for prToken (run for runN only); any other
	// token or number is "not found".
	page    store.PRPage
	run     store.RunDetail
	prToken string
	runN    int
	prCost  *int64
	// prCostErr fails only the PR total, after the detail read worked.
	prCostErr error
	// calls are the PR's reply and question calls.
	calls []store.CommentCall

	err error
	// lastTeam, lastOutcome and lastLimit record what the handler asked for.
	lastTeam    string
	lastOutcome store.AttemptFilter
	lastLimit   int
}

func (f *fakeDashStore) RecentAttempts(_ context.Context, team string, outcome store.AttemptFilter, limit int) ([]store.AttemptRow, error) {
	f.lastTeam, f.lastOutcome, f.lastLimit = team, outcome, limit
	return f.attempts, f.err
}
func (f *fakeDashStore) PRPage(_ context.Context, token string) (store.PRPage, bool, error) {
	if f.err != nil {
		return store.PRPage{}, false, f.err
	}
	return f.page, token == f.prToken, nil
}
func (f *fakeDashStore) PRRun(_ context.Context, token string, n int) (store.RunDetail, bool, error) {
	if f.err != nil {
		return store.RunDetail{}, false, f.err
	}
	return f.run, token == f.prToken && n == f.runN, nil
}
func (f *fakeDashStore) PRCost(context.Context, store.PRKey) (*int64, error) {
	if f.prCostErr != nil {
		return nil, f.prCostErr
	}
	return f.prCost, f.err
}
func (f *fakeDashStore) PRCommentCalls(context.Context, store.PRKey) ([]store.CommentCall, bool, error) {
	return f.calls, false, f.err
}
func (f *fakeDashStore) OutcomeBreakdown(context.Context, time.Time) ([]store.OutcomeCount, error) {
	return f.breakdown, f.err
}
func (f *fakeDashStore) FixCountsSince(context.Context, time.Time) ([]store.FixCount, error) {
	return f.fixCounts, f.err
}
func (f *fakeDashStore) TotalsSince(context.Context, time.Time) (store.Totals, error) {
	return f.totals, f.err
}
func (f *fakeDashStore) PRCountsSince(context.Context, time.Time) (int, int, error) {
	return f.reviewed, f.seen, f.err
}
func (f *fakeDashStore) TotalsByTeam(context.Context, time.Time) ([]store.TeamTotals, error) {
	return f.byTeam, f.err
}
func (f *fakeDashStore) DailyByTeam(_ context.Context, since time.Time) ([]store.Bucket, error) {
	f.dailySince = &since
	return f.daily, f.err
}
func (f *fakeDashStore) DailyAttempts(_ context.Context, since time.Time) ([]store.AttemptBucket, error) {
	f.attemptsSince = &since
	return f.dailyAtt, f.err
}
func (f *fakeDashStore) ActivityByTeam(context.Context) ([]store.TeamActivity, error) {
	return f.activity, f.err
}
func (f *fakeDashStore) AlmanachIgnoredByTeam(_ context.Context, since time.Time) (map[string]int, error) {
	f.ignoredSince = since
	f.ignoredReads++
	if f.ignoredErr != nil {
		return nil, f.ignoredErr
	}
	return f.ignored, f.err
}

type dashHarness struct {
	srv *httpapi.Server
	q   *fakeDashQueue
	st  *fakeDashStore
	llm *inference.Client
	rt  *teams.Runtime
}

// newDashLLM builds a team client that is never started: the roster reads its
// label, which needs no gateway.
func newDashLLM(t *testing.T) *inference.Client {
	t.Helper()
	reg, err := llmwire.NewRegistry([]byte(strings.Replace(string(llmwiretest.Profiles()),
		"providers:\n", "providers:\n  litellm: {}\n", 1)))
	if err != nil {
		t.Fatal(err)
	}
	env := map[string]string{
		"LLMWIRE_LITELLM_MODELS":   llmwiretest.ChatModel + "=gateway-alias",
		"LLMWIRE_LITELLM_BASE_URL": "https://gateway.invalid",
	}
	c, err := inference.New(inference.Options{
		Model: llmwiretest.ChatModel, Registry: reg, APIKey: "k",
		Logger: slog.New(slog.DiscardHandler),
		Env: func(n string) (string, bool) {
			v, ok := env[n]
			return v, ok
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	return c
}

func newDashHarness(t *testing.T) *dashHarness {
	t.Helper()
	return newDashHarnessWith(t, func(*config.Team) {})
}

// newDashHarnessWith lets a test adjust the team before its Runtime exists.
func newDashHarnessWith(t *testing.T, adjust func(*config.Team)) *dashHarness {
	t.Helper()
	team := &config.Team{
		Slug:     "platform",
		Name:     "Platform Engineering",
		Projects: []config.ProjectScope{{Key: "INF"}, {Key: "SHARED", Repos: []string{"lib"}}},
	}
	team.Review.ExcludeRepos = []string{"*-infra"}
	team.Review.ExcludeSourceBranches = []string{"dependabot/*"}
	team.Review.IgnoreAuthors = []string{"renovate"}
	adjust(team)
	llm := newDashLLM(t)
	rt := teams.NewRuntime(team, &fakeReviewer{}, llm, nil)

	var buf bytes.Buffer
	log := slog.New(logging.NewSplunkHandler(&buf, slog.LevelDebug, "noergler", "test"))
	reg := teams.NewRegistry(
		map[string]*teams.Runtime{"platform": rt},
		map[string]string{"mobile": "TEAM_MOBILE_OPENAI_API_KEY is unset"},
		log,
	).WithNames(map[string]string{"mobile": "Mobile Apps"})
	q := &fakeDashQueue{}
	st := &fakeDashStore{}
	srv := httpapi.New(reg.Status, log)
	Register(srv, Deps{
		Teams: reg, Queue: &fakeQueue{}, Log: log,
		BitbucketURL: "https://bitbucket.example.com",
		Dashboard:    q, DashboardStore: st,
	})
	return &dashHarness{srv: srv, q: q, st: st, llm: llm, rt: rt}
}

func (h *dashHarness) get(t *testing.T, path string) *httptest.ResponseRecorder {
	t.Helper()
	w := httptest.NewRecorder()
	h.srv.Handler().ServeHTTP(w, httptest.NewRequest(http.MethodGet, path, nil))
	return w
}

// probe GETs a probe path as a direct local caller: kit's probes answer no
// one else.
func (h *dashHarness) probe(t *testing.T, path string) *httptest.ResponseRecorder {
	t.Helper()
	w := httptest.NewRecorder()
	req := httptest.NewRequest(http.MethodGet, path, nil)
	req.RemoteAddr = "127.0.0.1:40000"
	h.srv.Handler().ServeHTTP(w, req)
	return w
}

// The dashboard is unauthenticated, so anything it serves is public. A
// disable reason can name an env var, which is why /readyz withholds them
// and why this must too.
func TestDashboardNeverServesADisableReason(t *testing.T) {
	h := newDashHarness(t)

	for _, path := range []string{"/api/dashboard/live", "/api/dashboard/teams"} {
		body := h.get(t, path).Body.String()
		if strings.Contains(body, "TEAM_MOBILE_OPENAI_API_KEY") {
			t.Errorf("%s leaked the disable reason: %s", path, body)
		}
		// Live lists only enabled teams that have run; the roster is where
		// a disabled team must still show up.
		if path == "/api/dashboard/teams" && !strings.Contains(body, "mobile") {
			t.Errorf("%s must still list the disabled team, got %s", path, body)
		}
	}
}

// The pool is six slots wide by default, not one. A panel that showed a
// single in-flight row would misreport what the instance is doing.
func TestLiveReportsThePoolAndBothLists(t *testing.T) {
	h := newDashHarness(t)
	now := time.Now()
	h.q.snap = queue.Snapshot{
		Capacity: 6, PerTeam: 2, Staged: 3, Depth: 1,
		Running: []queue.RunningItem{{
			Key:  store.PRKey{Project: "INF", Repo: "api", PRID: 9},
			Team: "platform", Kind: "review", Since: now, Waited: 2 * time.Second,
		}},
		Waiting: []queue.WaitingItem{{
			Key:  store.PRKey{Project: "INF", Repo: "api", PRID: 10},
			Team: "platform", Since: now,
		}},
	}

	var got liveBody
	if err := json.Unmarshal(h.get(t, "/api/dashboard/live").Body.Bytes(), &got); err != nil {
		t.Fatal(err)
	}
	if got.PoolCapacity != 6 || got.PoolPerTeam != 2 || got.Staged != 3 {
		t.Errorf("pool = %+v, want 6/2 staged 3", got)
	}
	if len(got.Running) != 1 || got.Running[0].Tag != "INF/api#9" {
		t.Errorf("running = %+v", got.Running)
	}
	if got.Running[0].WaitedMS != 2000 {
		t.Errorf("waited = %dms, want 2000", got.Running[0].WaitedMS)
	}
	if len(got.Waiting) != 1 || got.Waiting[0].Tag != "INF/api#10" {
		t.Errorf("waiting = %+v", got.Waiting)
	}
}

// Empty LISTS must serialise as [] rather than null, so a page can map over
// them unguarded. The harness store has no activity, so no team has run and
// teams is empty too.
func TestLiveSerialisesEmptyListsAsArrays(t *testing.T) {
	h := newDashHarness(t)
	body := h.get(t, "/api/dashboard/live").Body.String()
	for _, field := range []string{`"running":[]`, `"waiting":[]`, `"teams":[]`} {
		if !strings.Contains(body, field) {
			t.Errorf("want %s in %s", field, body)
		}
	}
	if strings.Contains(body, `"running":null`) || strings.Contains(body, `"waiting":null`) {
		t.Errorf("a list serialised as null: %s", body)
	}
}

func TestRunsRendersOutcomesAndCost(t *testing.T) {
	h := newDashHarness(t)
	ms := int64(72400)
	n := 6
	cost := int64(218_000_000)
	h.st.attempts = []store.AttemptRow{
		{
			Attempt: store.Attempt{
				Key:      store.PRKey{Project: "INF", Repo: "api", PRID: 9},
				TeamSlug: "platform", Kind: store.RunAuto, Outcome: "ok",
				ElapsedMS: &ms,
			},
			CreatedAt: time.Now(), Findings: &n, CostNanoUSD: &cost,
		},
		{
			Attempt: store.Attempt{
				Key:      store.PRKey{Project: "INF", Repo: "api", PRID: 8},
				TeamSlug: "platform", Kind: store.RunAuto,
				Outcome: "skipped", Reason: "head_unchanged",
			},
			CreatedAt: time.Now(),
		},
	}

	var got runsBody
	if err := json.Unmarshal(h.get(t, "/api/dashboard/runs").Body.Bytes(), &got); err != nil {
		t.Fatal(err)
	}
	if len(got.Runs) != 2 {
		t.Fatalf("runs = %d, want 2", len(got.Runs))
	}
	if got.Runs[0].Cost == nil || *got.Runs[0].Cost != "0.218" {
		t.Errorf("cost = %v, want the 3-decimal string 0.218", got.Runs[0].Cost)
	}
	// The stored reason is machine-readable; the label is what a person
	// reads. Both travel, so the page never has to own the vocabulary.
	if got.Runs[1].Reason != "head_unchanged" {
		t.Errorf("reason = %q", got.Runs[1].Reason)
	}
	if got.Runs[1].ReasonMsg != "HEAD unchanged since last review" {
		t.Errorf("label = %q", got.Runs[1].ReasonMsg)
	}
}

// An unpriced run is not a free run: its cost stays null rather than
// becoming "0.000".
func TestRunsKeepsAnUnpricedRunUnpriced(t *testing.T) {
	h := newDashHarness(t)
	h.st.attempts = []store.AttemptRow{{
		Attempt:   store.Attempt{Key: store.PRKey{Project: "INF", Repo: "api", PRID: 9}, Outcome: "ok"},
		CreatedAt: time.Now(),
	}}

	var got runsBody
	if err := json.Unmarshal(h.get(t, "/api/dashboard/runs").Body.Bytes(), &got); err != nil {
		t.Fatal(err)
	}
	if got.Runs[0].Cost != nil {
		t.Errorf("cost = %q, want null for an unpriced run", *got.Runs[0].Cost)
	}
}

func TestRunsPassesTeamAndLimitThrough(t *testing.T) {
	h := newDashHarness(t)
	h.get(t, "/api/dashboard/runs?team=payments&limit=25")
	if h.st.lastTeam != "payments" || h.st.lastLimit != 25 {
		t.Errorf("store asked for team=%q limit=%d", h.st.lastTeam, h.st.lastLimit)
	}
}

func TestRunsPassesTheOutcomeFilterThrough(t *testing.T) {
	h := newDashHarness(t)
	for _, tc := range []struct {
		query string
		want  store.AttemptFilter
	}{
		{"", store.AttemptsAll},
		{"?outcome=failed", store.AttemptsFailed},
		{"?outcome=skipped", store.AttemptsSkipped},
		// Unknown values are the whole feed, not a 4xx: the route is
		// unauthenticated and read-only, and a mistyped query string should
		// show the runs rather than an error the page cannot explain.
		{"?outcome=nonsense", store.AttemptsAll},
		{"?outcome=OK", store.AttemptsAll},
	} {
		h.get(t, "/api/dashboard/runs"+tc.query)
		if h.st.lastOutcome != tc.want {
			t.Errorf("%q asked the store for %q, want %q", tc.query, h.st.lastOutcome, tc.want)
		}
	}
}

// The name is what the page prints, the slug is what the logs and the webhook
// path use, so both ride the wire. A disabled team keeps its teams.yaml name.
// A team dropped from teams.yaml while its rows live on falls back to the slug
// rather than serving a blank cell.
func TestDashboardNamesTeamsAndFallsBackToTheSlug(t *testing.T) {
	h := newDashHarness(t)
	h.st.attempts = []store.AttemptRow{
		{
			Attempt:   store.Attempt{TeamSlug: "platform", Key: store.PRKey{Project: "INF", Repo: "api", PRID: 1}, Outcome: "ok"},
			CreatedAt: time.Now(),
		},
		{
			// A team with rows but no live config: gone from teams.yaml.
			Attempt:   store.Attempt{TeamSlug: "retired", Key: store.PRKey{Project: "OLD", Repo: "x", PRID: 2}, Outcome: "ok"},
			CreatedAt: time.Now(),
		},
	}

	var runs runsBody
	if err := json.Unmarshal(h.get(t, "/api/dashboard/runs").Body.Bytes(), &runs); err != nil {
		t.Fatal(err)
	}
	if runs.Runs[0].TeamName != "Platform Engineering" || runs.Runs[0].Team != "platform" {
		t.Errorf("configured team = %q/%q", runs.Runs[0].Team, runs.Runs[0].TeamName)
	}
	if runs.Runs[1].TeamName != "retired" {
		t.Errorf("unknown team named %q, want the slug", runs.Runs[1].TeamName)
	}

	var live liveBody
	if err := json.Unmarshal(h.get(t, "/api/dashboard/live").Body.Bytes(), &live); err != nil {
		t.Fatal(err)
	}
	for _, team := range live.Teams {
		// mobile is disabled and has no Runtime: its name comes from
		// teams.yaml via the registry, not the slug.
		want := map[string]string{"platform": "Platform Engineering", "mobile": "Mobile Apps"}[team.Slug]
		if team.Name != want {
			t.Errorf("live team %q named %q, want %q", team.Slug, team.Name, want)
		}
	}

	var roster teamsBody
	if err := json.Unmarshal(h.get(t, "/api/dashboard/teams").Body.Bytes(), &roster); err != nil {
		t.Fatal(err)
	}
	for _, team := range roster.Teams {
		want := map[string]string{"platform": "Platform Engineering", "mobile": "Mobile Apps"}[team.Slug]
		if team.Name != want {
			t.Errorf("roster team %q named %q, want %q", team.Slug, team.Name, want)
		}
	}
}

// The browser has no other way to learn the Bitbucket base: BITBUCKET_URL is
// read in this process and the SPA ships as static files. Both pages poll
// their own endpoint, so both bodies carry it.
func TestDashboardServesTheBitbucketBase(t *testing.T) {
	h := newDashHarness(t)

	var live liveBody
	if err := json.Unmarshal(h.get(t, "/api/dashboard/live").Body.Bytes(), &live); err != nil {
		t.Fatal(err)
	}
	if live.BitbucketURL != "https://bitbucket.example.com" {
		t.Errorf("live bitbucket_url = %q", live.BitbucketURL)
	}

	var runs runsBody
	if err := json.Unmarshal(h.get(t, "/api/dashboard/runs").Body.Bytes(), &runs); err != nil {
		t.Fatal(err)
	}
	if runs.BitbucketURL != "https://bitbucket.example.com" {
		t.Errorf("runs bitbucket_url = %q", runs.BitbucketURL)
	}
}

func TestMetricsReportsUnpricedBesideTheTotal(t *testing.T) {
	h := newDashHarness(t)
	cost := int64(31_470_000_000)
	h.st.totals = store.Totals{Runs: 1284, CostNanoUSD: &cost, Unpriced: 23}

	var got metricsBody
	if err := json.Unmarshal(h.get(t, "/api/dashboard/metrics").Body.Bytes(), &got); err != nil {
		t.Fatal(err)
	}
	if got.Totals.Cost == nil || *got.Totals.Cost != "31.470" {
		t.Errorf("cost = %v, want 31.470", got.Totals.Cost)
	}
	if got.Totals.Unpriced != 23 {
		t.Errorf("unpriced = %d, want 23 reported separately", got.Totals.Unpriced)
	}
}

// Lines reviewed leave as plain integers under the names the page reads, on
// the totals, each team and each day.
func TestMetricsCarriesLinesReviewed(t *testing.T) {
	h := newDashHarness(t)
	h.st.totals = store.Totals{Runs: 3, LinesAdded: 1200, LinesRemoved: 340, LinesUncounted: 1}
	h.st.byTeam = []store.TeamTotals{{TeamSlug: "platform", Totals: store.Totals{Runs: 3, LinesAdded: 1200, LinesRemoved: 340}}}
	h.st.daily = []store.Bucket{{Day: time.Now().UTC(), TeamSlug: "platform", Runs: 3, Lines: 1540}}

	var got struct {
		Totals map[string]any   `json:"totals"`
		ByTeam []map[string]any `json:"by_team"`
		Daily  []map[string]any `json:"daily"`
	}
	if err := json.Unmarshal(h.get(t, "/api/dashboard/metrics?days=7").Body.Bytes(), &got); err != nil {
		t.Fatal(err)
	}
	if got.Totals["lines_added"] != 1200.0 || got.Totals["lines_removed"] != 340.0 ||
		got.Totals["lines_uncounted_runs"] != 1.0 {
		t.Errorf("totals = %v, want lines_added 1200, lines_removed 340", got.Totals)
	}
	if len(got.ByTeam) != 1 || got.ByTeam[0]["lines_added"] != 1200.0 || got.ByTeam[0]["lines_removed"] != 340.0 {
		t.Errorf("by_team = %v", got.ByTeam)
	}
	if len(got.Daily) != 1 || got.Daily[0]["lines"] != 1540.0 {
		t.Errorf("daily = %v, want lines 1540", got.Daily)
	}
}

// Fix requests per team: requested sums every outcome, committed the ones
// that landed; a team without any reads zero, never null.
func TestMetricsCountsFixRequests(t *testing.T) {
	h := newDashHarness(t)
	h.st.byTeam = []store.TeamTotals{{TeamSlug: "platform"}, {TeamSlug: "payments"}}
	h.st.fixCounts = []store.FixCount{
		{TeamSlug: "platform", Outcome: store.FixCommitted, Count: 3},
		{TeamSlug: "platform", Outcome: store.FixNoWrite, Count: 2},
	}

	var got struct {
		ByTeam []struct {
			Team  string    `json:"team"`
			Fixes fixesBody `json:"fixes"`
		} `json:"by_team"`
	}
	if err := json.Unmarshal(h.get(t, "/api/dashboard/metrics?days=7").Body.Bytes(), &got); err != nil {
		t.Fatal(err)
	}
	if len(got.ByTeam) != 2 {
		t.Fatalf("by_team = %+v", got.ByTeam)
	}
	p := got.ByTeam[0].Fixes
	if p.Requested != 5 || p.Committed != 3 || len(p.Outcomes) != 2 || p.Outcomes[1] != (fixOutcomeCount{store.FixNoWrite, 2}) {
		t.Errorf("platform fixes = %+v", p)
	}
	if q := got.ByTeam[1].Fixes; q.Requested != 0 || q.Committed != 0 || q.Outcomes == nil {
		t.Errorf("payments fixes = %+v, want zero with an empty list", q)
	}
}

// One read, two sums: per day for the chart (every team), per team for the
// columns (every day, one entry per outcome).
func TestMetricsCarriesFixesPerDay(t *testing.T) {
	h := newDashHarness(t)
	d1 := time.Date(2026, 11, 12, 0, 0, 0, 0, time.UTC)
	d2 := d1.AddDate(0, 0, 1)
	h.st.byTeam = []store.TeamTotals{{TeamSlug: "platform"}}
	h.st.fixCounts = []store.FixCount{
		{Day: d1, TeamSlug: "payments", Outcome: store.FixNoWrite, Count: 1},
		{Day: d1, TeamSlug: "platform", Outcome: store.FixCommitted, Count: 2},
		{Day: d2, TeamSlug: "platform", Outcome: store.FixCommitted, Count: 1},
		{Day: d2, TeamSlug: "platform", Outcome: store.FixConflict, Count: 1},
	}

	var got struct {
		ByTeam []struct {
			Team  string    `json:"team"`
			Fixes fixesBody `json:"fixes"`
		} `json:"by_team"`
		FixesDaily []fixDayBody `json:"fixes_daily"`
	}
	if err := json.Unmarshal(h.get(t, "/api/dashboard/metrics?days=7").Body.Bytes(), &got); err != nil {
		t.Fatal(err)
	}
	want := []fixDayBody{{"2026-11-12", 3, 2}, {"2026-11-13", 2, 1}}
	if len(got.FixesDaily) != 2 || got.FixesDaily[0] != want[0] || got.FixesDaily[1] != want[1] {
		t.Errorf("fixes_daily = %+v, want %+v", got.FixesDaily, want)
	}
	p := got.ByTeam[0].Fixes
	if p.Requested != 4 || p.Committed != 3 || len(p.Outcomes) != 2 ||
		p.Outcomes[0] != (fixOutcomeCount{store.FixCommitted, 3}) || p.Outcomes[1] != (fixOutcomeCount{store.FixConflict, 1}) {
		t.Errorf("platform fixes = %+v", p)
	}
}

func TestMetricsFixesDailyIsAnEmptyListWithoutRequests(t *testing.T) {
	h := newDashHarness(t)
	var got map[string]any
	if err := json.Unmarshal(h.get(t, "/api/dashboard/metrics?days=7").Body.Bytes(), &got); err != nil {
		t.Fatal(err)
	}
	if v, ok := got["fixes_daily"].([]any); !ok || len(v) != 0 {
		t.Errorf("fixes_daily = %#v, want []", got["fixes_daily"])
	}
}

// A team whose only activity in the window was a fix request still shows.
func TestMetricsShowsATeamWithFixesOnly(t *testing.T) {
	h := newDashHarness(t)
	h.st.byTeam = []store.TeamTotals{{TeamSlug: "platform"}}
	h.st.fixCounts = []store.FixCount{{TeamSlug: "payments", Outcome: store.FixCommitted, Count: 1}}

	var got struct {
		ByTeam []struct {
			Team  string    `json:"team"`
			Runs  int       `json:"runs"`
			Fixes fixesBody `json:"fixes"`
		} `json:"by_team"`
	}
	if err := json.Unmarshal(h.get(t, "/api/dashboard/metrics?days=7").Body.Bytes(), &got); err != nil {
		t.Fatal(err)
	}
	if len(got.ByTeam) != 2 || got.ByTeam[1].Team != "payments" || got.ByTeam[1].Fixes.Committed != 1 || got.ByTeam[1].Runs != 0 {
		t.Errorf("by_team = %+v, want payments after platform with its fix", got.ByTeam)
	}
}

// PR counts sit beside the run totals, never in place of them: the tile
// shows PRs, its note the runs.
func TestMetricsReportsPRCountsBesideTheRuns(t *testing.T) {
	h := newDashHarness(t)
	h.st.totals = store.Totals{Runs: 619}
	h.st.reviewed, h.st.seen = 45, 120

	var got metricsBody
	if err := json.Unmarshal(h.get(t, "/api/dashboard/metrics?days=30").Body.Bytes(), &got); err != nil {
		t.Fatal(err)
	}
	if got.PRs.Reviewed != 45 || got.PRs.Seen != 120 || got.Totals.Runs != 619 {
		t.Errorf("prs = %+v, runs = %d, want 45/120 and 619", got.PRs, got.Totals.Runs)
	}
}

// The daily series reach six days before the window so the 7-day trend has a
// full week behind the first day; the window itself does not move.
func TestMetricsDailySeriesLeadTheWindowByAWeek(t *testing.T) {
	h := newDashHarness(t)
	var got metricsBody
	if err := json.Unmarshal(h.get(t, "/api/dashboard/metrics?days=30").Body.Bytes(), &got); err != nil {
		t.Fatal(err)
	}
	if want := got.Since.AddDate(0, 0, -6); !got.LeadSince.Equal(want) {
		t.Errorf("lead_since = %s, want %s (six days before since)", got.LeadSince, want)
	}
	if h.st.dailySince == nil || !h.st.dailySince.Equal(got.LeadSince) ||
		h.st.attemptsSince == nil || !h.st.attemptsSince.Equal(got.LeadSince) {
		t.Errorf("daily read from %v, attempts from %v, want both from lead_since %s",
			h.st.dailySince, h.st.attemptsSince, got.LeadSince)
	}
}

// The default window is the CURRENT CALENDAR MONTH, not a rolling 14 days:
// a key's spend is budgeted and invoiced by the month, and a rolling figure
// cannot be reconciled with a bill.
func TestMetricsDefaultsToTheCurrentMonth(t *testing.T) {
	h := newDashHarness(t)

	var got metricsBody
	if err := json.Unmarshal(h.get(t, "/api/dashboard/metrics").Body.Bytes(), &got); err != nil {
		t.Fatal(err)
	}
	if got.Window != "month" {
		t.Errorf("window = %q, want month", got.Window)
	}
	// UTC, matching the DB session the daily buckets are truncated in.
	now := time.Now().UTC()
	if got.Since.Day() != 1 || got.Since.Month() != now.Month() || got.Since.Year() != now.Year() {
		t.Errorf("since = %s, want the first of this month", got.Since)
	}
	if _, off := got.Since.Zone(); got.Since.Hour() != 0 || got.Since.Minute() != 0 || off != 0 {
		t.Errorf("since = %s, want UTC midnight", got.Since)
	}
	// Exclusive end, so the page can lay out the month without deciding
	// where it ends itself.
	if !got.Until.After(now) || got.Until.Day() != 1 {
		t.Errorf("until = %s, want the first of next month", got.Until)
	}
}

// monthStart stays in the location it is given (the handler passes UTC).
func TestMonthStartKeepsTheLocation(t *testing.T) {
	at := time.Date(2026, 3, 17, 14, 30, 0, 0, time.Local)
	got := monthStart(at)
	want := time.Date(2026, 3, 1, 0, 0, 0, 0, time.Local)
	if !got.Equal(want) {
		t.Errorf("monthStart = %s, want %s", got, want)
	}
	if got.Location() != time.Local {
		t.Errorf("location = %s, want local", got.Location())
	}
}

// ?days= still gives a rolling window, and an unbounded one would seq-scan
// however much history exists.
func TestMetricsClampsARollingWindow(t *testing.T) {
	h := newDashHarness(t)
	for _, q := range []string{"?days=0", "?days=9999", "?days=abc"} {
		var got metricsBody
		if err := json.Unmarshal(h.get(t, "/api/dashboard/metrics"+q).Body.Bytes(), &got); err != nil {
			t.Fatalf("%s: %v", q, err)
		}
		if got.Window != "rolling" {
			t.Errorf("%s: window = %q, want rolling", q, got.Window)
		}
		if days := time.Since(got.Since).Hours() / 24; days > 91 {
			t.Errorf("%s: window = %.0f days, want it clamped", q, days)
		}
	}
}

// A dashboard that renders zeroes when the database is down is worse than
// one that says it could not read: the first looks like a quiet instance.
func TestMetricsFailsLoudlyWhenTheStoreDoes(t *testing.T) {
	h := newDashHarness(t)
	h.st.err = errors.New("database is down")

	if code := h.get(t, "/api/dashboard/metrics").Code; code != http.StatusInternalServerError {
		t.Errorf("status = %d, want 500", code)
	}
}

func TestRunsFailsLoudlyWhenTheStoreDoes(t *testing.T) {
	h := newDashHarness(t)
	h.st.err = errors.New("database is down")

	if code := h.get(t, "/api/dashboard/runs").Code; code != http.StatusInternalServerError {
		t.Errorf("status = %d, want 500", code)
	}
}

// Live is different on purpose: the queue is in memory and always answers,
// so a broken store costs the per-team history but not the panel. An
// operator watching a failing instance needs to see what is running.
func TestLiveStillAnswersWhenTheStoreIsDown(t *testing.T) {
	h := newDashHarness(t)
	h.st.err = errors.New("database is down")
	h.q.snap = queue.Snapshot{Capacity: 6, PerTeam: 2}

	w := h.get(t, "/api/dashboard/live")
	if w.Code != http.StatusOK {
		t.Fatalf("status = %d, want 200", w.Code)
	}
	var got liveBody
	if err := json.Unmarshal(w.Body.Bytes(), &got); err != nil {
		t.Fatal(err)
	}
	if got.PoolCapacity != 6 {
		t.Errorf("pool = %d, want the queue's answer regardless of the store", got.PoolCapacity)
	}
	// No history means no team has a run to show, so the list is empty
	// rather than the request failing.
	if len(got.Teams) != 0 {
		t.Errorf("teams = %d, want none without history", len(got.Teams))
	}
}

func TestTeamsStillAnswersWhenTheStoreIsDown(t *testing.T) {
	h := newDashHarness(t)
	h.st.err = errors.New("database is down")

	w := h.get(t, "/api/dashboard/teams")
	if w.Code != http.StatusOK {
		t.Fatalf("status = %d, want 200", w.Code)
	}
	var got teamsBody
	if err := json.Unmarshal(w.Body.Bytes(), &got); err != nil {
		t.Fatal(err)
	}
	if len(got.Teams) != 2 {
		t.Errorf("teams = %d, want the roster from the registry", len(got.Teams))
	}
}

// The chart series travel as day/team/count rows, and a day with no cost
// keeps a null rather than a zero.
func TestMetricsRendersTheSeries(t *testing.T) {
	h := newDashHarness(t)
	day := time.Now().Truncate(24 * time.Hour)
	cost := int64(1_250_000_000)
	h.st.daily = []store.Bucket{
		{Day: day, TeamSlug: "platform", Runs: 4, FindingsPosted: 9, CostNanoUSD: &cost},
		{Day: day, TeamSlug: "mobile", Runs: 1, Unpriced: 1},
	}
	h.st.dailyAtt = []store.AttemptBucket{{Day: day, Outcome: "skipped", Count: 7}}
	h.st.byTeam = []store.TeamTotals{{TeamSlug: "platform", Totals: store.Totals{Runs: 4, CostNanoUSD: &cost}}}
	h.st.breakdown = []store.OutcomeCount{
		{Outcome: "skipped", Reason: "empty_diff", Count: 7},
		{Outcome: "error", Count: 1},
	}

	var got metricsBody
	if err := json.Unmarshal(h.get(t, "/api/dashboard/metrics").Body.Bytes(), &got); err != nil {
		t.Fatal(err)
	}
	if len(got.Daily) != 2 || got.Daily[0].Day != day.UTC().Format(time.DateOnly) {
		t.Errorf("daily = %+v", got.Daily)
	}
	if got.Daily[0].Cost == nil || *got.Daily[0].Cost != "1.250" {
		t.Errorf("cost = %v, want 1.250", got.Daily[0].Cost)
	}
	if got.Daily[1].Cost != nil {
		t.Errorf("an unpriced day must stay null, got %q", *got.Daily[1].Cost)
	}
	if got.Daily[0].FindingsPosted != 9 || got.Daily[0].Unpriced != 0 ||
		got.Daily[1].FindingsPosted != 0 || got.Daily[1].Unpriced != 1 {
		t.Errorf("findings/unpriced per day = %+v", got.Daily)
	}
	if len(got.Attempts) != 1 || got.Attempts[0].Count != 7 {
		t.Errorf("attempts = %+v", got.Attempts)
	}
	if len(got.ByTeam) != 1 || got.ByTeam[0].Team != "platform" {
		t.Errorf("by team = %+v", got.ByTeam)
	}
	// A skip carries its human label; an outcome that is not a skip has
	// none, because its own name is already the answer.
	if got.Breakdown[0].Label != "Empty diff" {
		t.Errorf("breakdown label = %q", got.Breakdown[0].Label)
	}
	if got.Breakdown[1].Label != "" {
		t.Errorf("a non-skip needs no label, got %q", got.Breakdown[1].Label)
	}
}

func TestTeamsListsClaimsAndSettings(t *testing.T) {
	h := newDashHarness(t)

	var got teamsBody
	if err := json.Unmarshal(h.get(t, "/api/dashboard/teams").Body.Bytes(), &got); err != nil {
		t.Fatal(err)
	}
	if len(got.Teams) != 2 {
		t.Fatalf("teams = %d, want the enabled and the disabled one", len(got.Teams))
	}
	var platform teamBody
	for _, tm := range got.Teams {
		if tm.Slug == "platform" {
			platform = tm
		}
	}
	if !platform.Enabled {
		t.Error("platform must be enabled")
	}
	// A whole-project claim carries no repo; a narrowed one carries each.
	if len(platform.Claims) != 2 {
		t.Fatalf("claims = %+v, want one per scope entry", platform.Claims)
	}
	if platform.Claims[0].Project != "INF" || platform.Claims[0].Repo != "" {
		t.Errorf("whole-project claim = %+v", platform.Claims[0])
	}
	if platform.Claims[1].Repo != "lib" {
		t.Errorf("narrowed claim = %+v", platform.Claims[1])
	}
	if len(platform.ExcludeRepos) != 1 || platform.ExcludeRepos[0] != "*-infra" {
		t.Errorf("exclude repos = %+v", platform.ExcludeRepos)
	}
	if !slices.Equal(platform.ExcludeSourceBranches, []string{"dependabot/*"}) ||
		platform.ExcludeTargetBranches == nil || len(platform.ExcludeTargetBranches) != 0 {
		t.Errorf("branches = %v / %v", platform.ExcludeSourceBranches, platform.ExcludeTargetBranches)
	}
}

// The roster names the model a team reviews with: the client's own label,
// the one its runs record. Never the gateway alias, and nothing for a
// disabled team, whose configured model may be why it is disabled.
func TestTeamsNamesTheModel(t *testing.T) {
	h := newDashHarness(t)

	raw := h.get(t, "/api/dashboard/teams").Body.String()
	var got teamsBody
	if err := json.Unmarshal([]byte(raw), &got); err != nil {
		t.Fatal(err)
	}
	for _, tm := range got.Teams {
		switch tm.Slug {
		case "platform":
			if tm.Model == "" || tm.Model != h.llm.Label() {
				t.Errorf("model = %q, want the client's label %q", tm.Model, h.llm.Label())
			}
		case "mobile":
			if tm.Model != "" {
				t.Errorf("disabled team model = %q, want none", tm.Model)
			}
		}
	}
	if strings.Contains(raw, "gateway-alias") {
		t.Error("the gateway alias is the operator's naming and must not be served")
	}
	if strings.Count(raw, `"model"`) != 1 {
		t.Errorf("want one model key, for the enabled team only: %s", raw)
	}
}

// The model source says whether the label is the instance default, and is
// absent with the model on a disabled team.
func TestTeamsNamesTheModelSource(t *testing.T) {
	source := func(h *dashHarness) map[string]string {
		var got teamsBody
		if err := json.Unmarshal(h.get(t, "/api/dashboard/teams").Body.Bytes(), &got); err != nil {
			t.Fatal(err)
		}
		out := map[string]string{}
		for _, tm := range got.Teams {
			out[tm.Slug] = tm.ModelSource
		}
		return out
	}
	if got := source(newDashHarness(t)); got["platform"] != "default" || got["mobile"] != "" {
		t.Errorf("model source = %v, want platform default, mobile absent", got)
	}
	own := newDashHarnessWith(t, func(team *config.Team) { team.OwnModel = true })
	if got := source(own)["platform"]; got != "team" {
		t.Errorf("model source = %q, want team", got)
	}
}

// A forwarder (riptide, Almanach) shows as a fixed word from the last call
// (the startup ping, then each emit), never the URL or the token; off
// without an emitter, absent for a disabled team. set points the team's
// emitter at a URL with a token and pings; get reads the roster's word.
func forwarderOnRoster(t *testing.T, name string, set func(h *dashHarness, url, token string), get func(teamBody) string) {
	t.Helper()
	h := newDashHarness(t)
	state := func() (map[string]string, string) {
		raw := h.get(t, "/api/dashboard/teams").Body.String()
		var got teamsBody
		if err := json.Unmarshal([]byte(raw), &got); err != nil {
			t.Fatal(err)
		}
		out := map[string]string{}
		for _, tm := range got.Teams {
			out[tm.Slug] = get(tm)
		}
		return out, raw
	}
	if got, _ := state(); got["platform"] != forward.StatusOff || got["mobile"] != "" {
		t.Errorf("%s = %v, want platform off, mobile absent", name, got)
	}

	up := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		_, _ = w.Write([]byte(`{"status":"ok","team":"platform"}`))
	}))
	t.Cleanup(up.Close)
	set(h, up.URL, name+"-secret")
	got, raw := state()
	if got["platform"] != forward.StatusOK {
		t.Errorf("%s = %q, want ok", name, got["platform"])
	}
	for _, leak := range []string{up.URL, name + "-secret"} {
		if strings.Contains(raw, leak) {
			t.Errorf("%q reached the unauthenticated roster", leak)
		}
	}

	set(h, "http://127.0.0.1:1", "t")
	if got, _ := state(); got["platform"] != forward.StatusFailing {
		t.Errorf("%s = %q, want failing", name, got["platform"])
	}
}

func TestTeamsShowsRiptide(t *testing.T) {
	forwarderOnRoster(t, "riptide", func(h *dashHarness, url, token string) {
		h.rt.Riptide = riptide.New(url, token, slog.New(slog.DiscardHandler))
		_, _ = h.rt.Riptide.VerifyAtStartup(context.Background())
	}, func(tm teamBody) string { return tm.Riptide })
}

func TestTeamsShowsAlmanach(t *testing.T) {
	forwarderOnRoster(t, "almanach", func(h *dashHarness, url, token string) {
		h.rt.Almanach = almanach.New(url, token, slog.New(slog.DiscardHandler))
		_, _ = h.rt.Almanach.VerifyAtStartup(context.Background())
	}, func(tm teamBody) string { return tm.Almanach })
}

// The roster counts a team's notes Almanach ignored in the last 7 days,
// beside the feed's word, which stays ok for them. None with the feed off;
// a failed count read leaves the roster answering without it.
func TestTeamsCountsIgnoredAlmanachNotes(t *testing.T) {
	h := newDashHarness(t)
	h.st.ignored = map[string]int{"platform": 3}
	ignoredOf := func() int {
		var got teamsBody
		if err := json.Unmarshal(h.get(t, "/api/dashboard/teams").Body.Bytes(), &got); err != nil {
			t.Fatal(err)
		}
		for _, tm := range got.Teams {
			if tm.Slug == "platform" {
				return tm.AlmanachIgnored7d
			}
		}
		t.Fatal("platform missing")
		return 0
	}
	if n := ignoredOf(); n != 0 {
		t.Errorf("feed off: ignored = %d, want none shown", n)
	}
	if h.st.ignoredReads != 0 {
		t.Errorf("feed off everywhere: %d count reads, want none", h.st.ignoredReads)
	}

	up := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		_, _ = w.Write([]byte(`{"status":"ok","team":"platform"}`))
	}))
	t.Cleanup(up.Close)
	h.rt.Almanach = almanach.New(up.URL, "t", slog.New(slog.DiscardHandler))
	_, _ = h.rt.Almanach.VerifyAtStartup(context.Background())
	if n := ignoredOf(); n != 3 {
		t.Errorf("ignored = %d, want 3", n)
	}
	if ago := time.Since(h.st.ignoredSince); ago < 7*24*time.Hour-time.Minute || ago > 7*24*time.Hour+time.Minute {
		t.Errorf("window = %v, want 7 days", ago)
	}

	// The count read alone failing: the roster still answers, without it.
	h.st.ignoredErr = errors.New("db down")
	if n := ignoredOf(); n != 0 {
		t.Errorf("count read failed: ignored = %d, want none shown", n)
	}
}

// A team's own accounts show on the roster as source, bot and the fixed
// health status; never a reason, never a token. A disabled team has none.
func TestTeamsShowsTheAccountsHealth(t *testing.T) {
	h := newDashHarness(t)
	hh := &harness{rt: h.rt, log: slog.New(slog.DiscardHandler)}
	hh.withAccounts(t, map[teams.Kind]teams.Opened{
		teams.KindBitbucket: {Username: "plat-bot", Undecryptable: true},
		teams.KindJira:      {Token: "revoked"},
	})

	raw := h.get(t, "/api/dashboard/teams").Body.String()
	var got struct {
		Teams []struct {
			Slug             string         `json:"slug"`
			BitbucketAccount map[string]any `json:"bitbucket_account"`
			JiraAccount      map[string]any `json:"jira_account"`
		} `json:"teams"`
	}
	if err := json.Unmarshal([]byte(raw), &got); err != nil {
		t.Fatal(err)
	}
	for _, tm := range got.Teams {
		switch tm.Slug {
		case "platform":
			if bb := tm.BitbucketAccount; bb["source"] != "team" || bb["username"] != "plat-bot" || bb["status"] != "undecryptable" {
				t.Errorf("bitbucket = %v", bb)
			}
			if jr := tm.JiraAccount; jr["source"] != "team" || jr["status"] != "rejected" {
				t.Errorf("jira = %v", jr)
			}
		case "mobile":
			if tm.BitbucketAccount != nil || tm.JiraAccount != nil {
				t.Errorf("disabled team accounts = %v / %v, want null", tm.BitbucketAccount, tm.JiraAccount)
			}
		}
	}
	for _, leak := range []string{"revoked", "does not decrypt", "HTTP 401", "TEAM_MOBILE"} {
		if strings.Contains(raw, leak) {
			t.Errorf("%q reached the unauthenticated roster: %s", leak, raw)
		}
	}
}

// The routes exist only when their dependencies are wired, so an instance
// that does not want a dashboard does not serve one.
func TestDashboardRoutesAbsentWithoutDeps(t *testing.T) {
	var buf bytes.Buffer
	log := slog.New(logging.NewSplunkHandler(&buf, slog.LevelDebug, "noergler", "test"))
	reg := teams.NewRegistry(nil, nil, log)
	srv := httpapi.New(reg.Status, log)
	Register(srv, Deps{Teams: reg, Queue: &fakeQueue{}, Log: log})

	w := httptest.NewRecorder()
	srv.Handler().ServeHTTP(w, httptest.NewRequest(http.MethodGet, "/api/dashboard/live", nil))
	if w.Code != http.StatusNotFound {
		t.Errorf("status = %d, want 404 when the dashboard is not wired", w.Code)
	}
}

// The SPA sits on "/" as a catch-all. Go's ServeMux prefers the most
// specific pattern, but that is a property worth pinning: a shell served in
// place of the webhook would swallow deliveries and answer 200, and a probe
// answering HTML would read as healthy to a checker that only reads status.
func TestSPACatchAllNeverShadowsTheRealRoutes(t *testing.T) {
	h := newDashHarness(t)

	// The probes' own bodies: liveness a plain "ok", readiness JSON either way.
	if w := h.probe(t, "/healthz"); w.Code != http.StatusOK || strings.TrimSpace(w.Body.String()) != "ok" {
		t.Errorf("/healthz = %d %q, want the probe's 200 ok", w.Code, w.Body.String())
	}
	if ct := h.probe(t, "/readyz").Header().Get("Content-Type"); !strings.Contains(ct, "json") {
		t.Errorf("/readyz served %q, want the probe's JSON", ct)
	}

	// The webhook is POST-only, so a GET must not be answered by the shell
	// with a 200.
	w := httptest.NewRecorder()
	h.srv.Handler().ServeHTTP(w, httptest.NewRequest(http.MethodGet, "/webhook/platform", nil))
	if w.Code == http.StatusOK {
		t.Errorf("GET /webhook/platform = 200, want a method error rather than the shell")
	}

	// An unknown dashboard endpoint stays JSON-shaped 404, never a page.
	if body := h.get(t, "/api/dashboard/typo").Body.String(); strings.Contains(body, "<!doctype") {
		t.Errorf("a mistyped API path received the shell: %s", body)
	}

	// A real client route does get the SPA: the shell when built, the
	// placeholder's 503 in a binary without one (a fresh clone).
	want := http.StatusServiceUnavailable
	if web.HasBuiltIndex() {
		want = http.StatusOK
	}
	if code := h.get(t, "/metrics").Code; code != want {
		t.Errorf("/metrics = %d, want %d (built: %v)", code, want, web.HasBuiltIndex())
	}
}

// /teams/{slug} is both the team's API and the Teams page's link to the
// team. Through the whole mux (the SPA's own test never saw the API route
// win): a browser navigation gets the page, a client without a bearer the
// API's 401, a client with one the API.
func TestTeamLinkIsThePageForABrowser(t *testing.T) {
	h := newDashHarness(t)
	serve := func(accept, auth string) *httptest.ResponseRecorder {
		w := httptest.NewRecorder()
		req := httptest.NewRequest(http.MethodGet, "/teams/platform", nil)
		if accept != "" {
			req.Header.Set("Accept", accept)
		}
		if auth != "" {
			req.Header.Set("Authorization", auth)
		}
		h.srv.Handler().ServeHTTP(w, req)
		return w
	}
	page := http.StatusServiceUnavailable
	if web.HasBuiltIndex() {
		page = http.StatusOK
	}
	isPage := func(w *httptest.ResponseRecorder) bool {
		return w.Code == page && strings.Contains(w.Header().Get("Content-Type"), "text/html")
	}
	isAPI := func(w *httptest.ResponseRecorder) bool {
		return w.Code == http.StatusUnauthorized && strings.Contains(w.Header().Get("Content-Type"), "json")
	}
	nav := "text/html,application/xhtml+xml,*/*;q=0.8"
	for _, c := range []struct {
		name, accept, auth string
		page               bool
	}{
		{"browser", nav, "", true},
		{"browser behind an ingress's Basic auth", nav, "Basic dXNlcjpwdw==", true},
		{"client without a bearer", "", "", false},
		{"client that refuses html", "application/json, text/html;q=0", "", false},
		{"client with a bearer asking for html", "text/html", "Bearer wrong", false},
	} {
		w := serve(c.accept, c.auth)
		if c.page && !isPage(w) || !c.page && !isAPI(w) {
			t.Errorf("%s: %d %q, want the page: %v", c.name, w.Code, w.Header().Get("Content-Type"), c.page)
		}
		if !strings.Contains(w.Header().Get("Vary"), "Accept") || !strings.Contains(w.Header().Get("Vary"), "Authorization") {
			t.Errorf("%s: Vary %q, want Accept and Authorization", c.name, w.Header().Get("Vary"))
		}
	}
}

// Without the dashboard there is no page to serve: a browser on a team's
// URL gets the API, as before.
func TestTeamLinkWithoutTheDashboardIsTheAPI(t *testing.T) {
	h := newHarness(t, nil)
	w := httptest.NewRecorder()
	req := httptest.NewRequest(http.MethodGet, "/teams/"+testSlug, nil)
	req.Header.Set("Accept", "text/html")
	h.srv.Handler().ServeHTTP(w, req)
	if w.Code != http.StatusUnauthorized {
		t.Errorf("no dashboard: %d, want the API's 401", w.Code)
	}
}

// A team can pass every startup check and still review nothing, because it
// owns no repositories. Before this state that team showed a green "ready"
// pill beside an idle instance, which is the case an operator opens the
// dashboard to find.
func TestScopeSizeDistinguishesNoneFromWholeProject(t *testing.T) {
	cases := []struct {
		name   string
		scopes []config.ProjectScope
		want   int
	}{
		{"no projects and no repos", nil, 0},
		{"empty list", []config.ProjectScope{}, 0},
		// A project with no repo list is the WHOLE project: unbounded
		// coverage, reported as -1 rather than counted.
		{"whole project", []config.ProjectScope{{Key: "PAY"}}, -1},
		{"named repos", []config.ProjectScope{{Key: "PAY", Repos: []string{"a", "b"}}}, 2},
		{
			"across projects",
			[]config.ProjectScope{
				{Key: "PAY", Repos: []string{"a"}},
				{Key: "INF", Repos: []string{"b", "c"}},
			},
			3,
		},
		// One whole-project claim wins: the team's coverage is unbounded
		// regardless of what else it names.
		{
			"whole project beside named repos",
			[]config.ProjectScope{{Key: "PAY", Repos: []string{"a"}}, {Key: "INF"}},
			-1,
		},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			if got := scopeSize(c.scopes); got != c.want {
				t.Errorf("scopeSize = %d, want %d", got, c.want)
			}
		})
	}
}

func TestTeamStateFoldsStartupAndScope(t *testing.T) {
	cases := []struct {
		enabled bool
		repos   int
		want    string
	}{
		{true, 3, teamReady},
		{true, -1, teamReady},
		// Configured, started, owns nothing: not ready, and not disabled
		// either. The distinction is the whole point.
		{true, 0, teamNoScope},
		// A disabled team's scope is irrelevant: it takes no traffic at all.
		{false, 0, teamDisabled},
		{false, 5, teamDisabled},
	}
	for _, c := range cases {
		if got := teamState(c.enabled, c.repos); got != c.want {
			t.Errorf("teamState(%v, %d) = %q, want %q", c.enabled, c.repos, got, c.want)
		}
	}
}

// Live lists enabled teams that have run, nothing else: mobile is disabled
// and is left out although it has history. The last run is a skip newer than
// the last success, so the two times differ and the outcome is the skip's.
func TestLiveListsOnlyEnabledTeamsThatRan(t *testing.T) {
	h := newDashHarness(t)
	reviewed := time.Date(2026, 9, 23, 8, 0, 0, 0, time.UTC)
	ran := reviewed.Add(time.Hour)
	h.st.activity = []store.TeamActivity{
		{TeamSlug: "mobile", PRs: 2, LastReviewed: &reviewed, LastRun: &ran, LastOutcome: "ok"},
		{TeamSlug: "platform", PRs: 4, LastReviewed: &reviewed, LastRun: &ran,
			LastOutcome: "skipped", LastReason: "head_unchanged"},
	}

	var got liveBody
	if err := json.Unmarshal(h.get(t, "/api/dashboard/live").Body.Bytes(), &got); err != nil {
		t.Fatal(err)
	}
	if len(got.Teams) != 1 || got.Teams[0].Slug != "platform" {
		t.Fatalf("teams = %+v, want platform alone", got.Teams)
	}
	p := got.Teams[0]
	if p.PRs != 4 || p.LastReviewed == nil || !p.LastReviewed.Equal(reviewed) || !p.LastRun.Equal(ran) {
		t.Errorf("platform = %+v, want 4 PRs, reviewed %s, ran %s", p, reviewed, ran)
	}
	if p.LastOutcome != "skipped" || p.LastReason != "head_unchanged" || p.LastReasonMsg == "" {
		t.Errorf("platform outcome = %q/%q/%q, want the skip and its label", p.LastOutcome, p.LastReason, p.LastReasonMsg)
	}
}

// An enabled team with PRs but no run at all has nothing live to show.
func TestLiveOmitsTeamsThatNeverRan(t *testing.T) {
	h := newDashHarness(t)
	h.st.activity = []store.TeamActivity{{TeamSlug: "platform", PRs: 1}}

	var got liveBody
	if err := json.Unmarshal(h.get(t, "/api/dashboard/live").Body.Bytes(), &got); err != nil {
		t.Fatal(err)
	}
	if len(got.Teams) != 0 {
		t.Errorf("teams = %+v, want none", got.Teams)
	}
}

// Both helpers work in t's own location. time.Truncate(24h) works on
// absolute time, so it lands on UTC midnight whatever the location, and the
// two boundaries would then disagree for any location but UTC.
func TestWindowBoundariesAreMidnightInTheirLocation(t *testing.T) {
	at := time.Date(2026, 3, 17, 14, 30, 0, 0, time.Local)

	if got := dayStart(at); got.Hour() != 0 || got.Minute() != 0 || got.Day() != 17 {
		t.Errorf("dayStart = %s, want local midnight on the 17th", got)
	}
	if got := dayStart(at); got.Location() != time.Local {
		t.Errorf("dayStart location = %s, want local", got.Location())
	}
	if got := monthStart(at); got.Hour() != 0 || got.Day() != 1 {
		t.Errorf("monthStart = %s, want local midnight on the 1st", got)
	}

	// Both boundaries agree on what a day starts at, which is the property
	// the two code paths share.
	if dayStart(monthStart(at)) != monthStart(at) {
		t.Error("monthStart is not itself a day start")
	}
}

// A rolling window's ends are day starts too, not whatever moment the
// request happened to arrive at.
func TestRollingWindowStartsAtMidnight(t *testing.T) {
	h := newDashHarness(t)

	var got metricsBody
	if err := json.Unmarshal(h.get(t, "/api/dashboard/metrics?days=7").Body.Bytes(), &got); err != nil {
		t.Fatal(err)
	}
	if got.Since.Hour() != 0 || got.Since.Minute() != 0 || got.Since.Second() != 0 {
		t.Errorf("since = %s, want midnight", got.Since)
	}
	if got.Until.Hour() != 0 {
		t.Errorf("until = %s, want midnight", got.Until)
	}
}

// ?days=N is N days, today included: the page lays out since..until and
// titles itself "the last N days".
func TestRollingWindowIsExactlyNDays(t *testing.T) {
	h := newDashHarness(t)
	for _, n := range []int{1, 7, 30} {
		var got metricsBody
		q := fmt.Sprintf("/api/dashboard/metrics?days=%d", n)
		before := dayStart(time.Now().UTC())
		raw := h.get(t, q).Body.Bytes()
		today := dayStart(time.Now().UTC())
		if !today.Equal(before) {
			t.Skip("the request crossed midnight UTC")
		}
		if err := json.Unmarshal(raw, &got); err != nil {
			t.Fatal(err)
		}
		if got.Window != "rolling" {
			t.Errorf("%s: window = %q, want rolling", q, got.Window)
		}
		if !got.Until.Equal(today.AddDate(0, 0, 1)) {
			t.Errorf("%s: until = %s, want tomorrow", q, got.Until)
		}
		if !got.Since.Equal(today.AddDate(0, 0, -(n - 1))) {
			t.Errorf("%s: since = %s, want %d days including today", q, got.Since, n)
		}
	}
}

// The route is unauthenticated and cross-team. It says HOW MANY authors a
// team has configured, never who they are: the same response deliberately
// withholds a disable reason, and a username roster on it would be the same
// mistake in a different field.
func TestTeamsCountsAuthorsWithoutNamingThem(t *testing.T) {
	h := newDashHarness(t)

	raw := h.get(t, "/api/dashboard/teams").Body.String()
	if strings.Contains(raw, "renovate") {
		t.Errorf("author names reached an unauthenticated route: %s", raw)
	}

	var got teamsBody
	if err := json.Unmarshal([]byte(raw), &got); err != nil {
		t.Fatal(err)
	}
	var platform teamBody
	for _, tm := range got.Teams {
		if tm.Slug == "platform" {
			platform = tm
		}
	}
	if platform.IgnoreAuthors != 1 {
		t.Errorf("ignore_authors = %d, want the count 1", platform.IgnoreAuthors)
	}
	// A repo glob is not a person, so exclude_repos stays whole.
	if len(platform.ExcludeRepos) != 1 || platform.ExcludeRepos[0] != "*-infra" {
		t.Errorf("exclude_repos = %v, want the globs themselves", platform.ExcludeRepos)
	}
}
