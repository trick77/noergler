package api

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"log/slog"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/trick77/noergler/internal/config"
	"github.com/trick77/noergler/internal/httpapi"
	"github.com/trick77/noergler/internal/logging"
	"github.com/trick77/noergler/internal/queue"
	"github.com/trick77/noergler/internal/store"
	"github.com/trick77/noergler/internal/teams"
)

type fakeDashQueue struct{ snap queue.Snapshot }

func (f *fakeDashQueue) Snapshot() queue.Snapshot { return f.snap }

type fakeDashStore struct {
	attempts  []store.AttemptRow
	breakdown []store.OutcomeCount
	totals    store.Totals
	byTeam    []store.TeamTotals
	daily     []store.Bucket
	dailyAtt  []store.AttemptBucket
	activity  []store.TeamActivity

	err error
	// lastTeam and lastLimit record what the handler asked for.
	lastTeam  string
	lastLimit int
}

func (f *fakeDashStore) RecentAttempts(_ context.Context, team string, limit int) ([]store.AttemptRow, error) {
	f.lastTeam, f.lastLimit = team, limit
	return f.attempts, f.err
}
func (f *fakeDashStore) OutcomeBreakdown(context.Context, time.Time) ([]store.OutcomeCount, error) {
	return f.breakdown, f.err
}
func (f *fakeDashStore) TotalsSince(context.Context, time.Time) (store.Totals, error) {
	return f.totals, f.err
}
func (f *fakeDashStore) TotalsByTeam(context.Context, time.Time) ([]store.TeamTotals, error) {
	return f.byTeam, f.err
}
func (f *fakeDashStore) DailyByTeam(context.Context, time.Time) ([]store.Bucket, error) {
	return f.daily, f.err
}
func (f *fakeDashStore) DailyAttempts(context.Context, time.Time) ([]store.AttemptBucket, error) {
	return f.dailyAtt, f.err
}
func (f *fakeDashStore) ActivityByTeam(context.Context) ([]store.TeamActivity, error) {
	return f.activity, f.err
}

type dashHarness struct {
	srv *httpapi.Server
	q   *fakeDashQueue
	st  *fakeDashStore
}

func newDashHarness(t *testing.T) *dashHarness {
	t.Helper()
	team := &config.Team{
		Slug:     "platform",
		Projects: []config.ProjectScope{{Key: "INF"}, {Key: "SHARED", Repos: []string{"lib"}}},
	}
	team.Review.ExcludeRepos = []string{"*-infra"}
	team.Review.IgnoreAuthors = []string{"renovate"}
	rt := teams.NewRuntime(team, &fakeReviewer{}, nil, nil, nil)

	var buf bytes.Buffer
	log := slog.New(logging.NewHandler(&buf, slog.LevelDebug, "test"))
	reg := teams.NewRegistry(
		map[string]*teams.Runtime{"platform": rt},
		map[string]string{"mobile": "TEAM_MOBILE_OPENAI_API_KEY is unset"},
		log,
	)
	q := &fakeDashQueue{}
	st := &fakeDashStore{}
	srv := httpapi.New(reg.Status, log)
	Register(srv, Deps{
		Teams: reg, Queue: &fakeQueue{}, Log: log,
		Dashboard: q, DashboardStore: st,
	})
	return &dashHarness{srv: srv, q: q, st: st}
}

func (h *dashHarness) get(t *testing.T, path string) *httptest.ResponseRecorder {
	t.Helper()
	w := httptest.NewRecorder()
	h.srv.Handler().ServeHTTP(w, httptest.NewRequest(http.MethodGet, path, nil))
	return w
}

// The dashboard is unauthenticated, so anything it serves is public. A
// disable reason can name an env var, which is why /health withholds them
// and why this must too.
func TestDashboardNeverServesADisableReason(t *testing.T) {
	h := newDashHarness(t)

	for _, path := range []string{"/api/dashboard/live", "/api/dashboard/teams"} {
		body := h.get(t, path).Body.String()
		if strings.Contains(body, "TEAM_MOBILE_OPENAI_API_KEY") {
			t.Errorf("%s leaked the disable reason: %s", path, body)
		}
		if !strings.Contains(body, "mobile") {
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
// them unguarded. A null scalar is a different thing and stays: last_run is
// genuinely absent for a team that has never run, and "never" must not
// render as a date.
func TestLiveSerialisesEmptyListsAsArrays(t *testing.T) {
	h := newDashHarness(t)
	body := h.get(t, "/api/dashboard/live").Body.String()
	for _, field := range []string{`"running":[]`, `"waiting":[]`} {
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

// An unbounded window would seq-scan however much history exists.
func TestMetricsClampsTheWindow(t *testing.T) {
	h := newDashHarness(t)
	for _, q := range []string{"", "?days=0", "?days=9999", "?days=abc"} {
		var got metricsBody
		if err := json.Unmarshal(h.get(t, "/api/dashboard/metrics"+q).Body.Bytes(), &got); err != nil {
			t.Fatalf("%s: %v", q, err)
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
	if len(got.Teams) != 2 {
		t.Errorf("teams = %d, want the roster from the registry", len(got.Teams))
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
		{Day: day, TeamSlug: "platform", Runs: 4, CostNanoUSD: &cost},
		{Day: day, TeamSlug: "mobile", Runs: 1},
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
	if len(got.Daily) != 2 || got.Daily[0].Day != day.Format(time.DateOnly) {
		t.Errorf("daily = %+v", got.Daily)
	}
	if got.Daily[0].Cost == nil || *got.Daily[0].Cost != "1.250" {
		t.Errorf("cost = %v, want 1.250", got.Daily[0].Cost)
	}
	if got.Daily[1].Cost != nil {
		t.Errorf("an unpriced day must stay null, got %q", *got.Daily[1].Cost)
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
}

// The routes exist only when their dependencies are wired, so an instance
// that does not want a dashboard does not serve one.
func TestDashboardRoutesAbsentWithoutDeps(t *testing.T) {
	var buf bytes.Buffer
	log := slog.New(logging.NewHandler(&buf, slog.LevelDebug, "test"))
	reg := teams.NewRegistry(nil, nil, log)
	srv := httpapi.New(reg.Status, log)
	Register(srv, Deps{Teams: reg, Queue: &fakeQueue{}, Log: log})

	w := httptest.NewRecorder()
	srv.Handler().ServeHTTP(w, httptest.NewRequest(http.MethodGet, "/api/dashboard/live", nil))
	if w.Code != http.StatusNotFound {
		t.Errorf("status = %d, want 404 when the dashboard is not wired", w.Code)
	}
}
