package api

import (
	"net/http"
	"sort"
	"time"

	"github.com/trick77/noergler/internal/review"
	"github.com/trick77/noergler/internal/store"
	"github.com/trick77/ship/kit/httpx"
)

// prPageBody is GET /api/dashboard/prs/{token}: the PR, every run of it,
// every finding posted on it, the files its reviews saw, and what it cost.
// The summary comment links here; each run's page is under it by number.
//
// Sums only where they add up (decided): cost is PRCost (the cap's figure,
// never a sum of the rows), tokens and time add over runs, lines and files
// are the whole PR's at the latest run that read the whole diff. Waits are
// one total, never a summed timeline.
type prPageBody struct {
	Tag      string `json:"tag"`
	Title    string `json:"title"`
	Team     string `json:"team"`
	TeamName string `json:"team_name"`
	// State is open, merged or declined.
	State string `json:"state"`

	Cost *string `json:"cost_usd"`
	// Cap is the team's per-PR cost cap, nil when the team is no longer
	// configured.
	Cap *string `json:"cap_usd"`
	// Scope is the whole PR's reviewable diff; null when no run read it.
	Scope *store.PRScope `json:"scope"`
	// Tokens add over the runs.
	PromptTokens     int64 `json:"prompt_tokens"`
	CachedTokens     int64 `json:"cached_tokens"`
	CompletionTokens int64 `json:"completion_tokens"`
	// ElapsedMS adds the runs' wall clocks; WaitMS is the part of the
	// recorded stages spent waiting.
	ElapsedMS int64 `json:"elapsed_ms"`
	WaitMS    int64 `json:"wait_ms"`

	Runs     []prRunBody     `json:"runs"`
	MoreRuns bool            `json:"more_runs"`
	Findings []prFindingBody `json:"findings"`
	Files    []prFileBody    `json:"files"`
	// Calls are the PR's reply and Q&A calls, newest first: they count
	// toward Cost and the cap. MoreCalls: the list was cut.
	Calls     []commentCallBody `json:"calls"`
	MoreCalls bool              `json:"more_calls"`
	// CallsFailed: the list could not be read, so an empty one means
	// nothing; the total above still counts the calls.
	CallsFailed bool `json:"calls_failed"`

	BitbucketURL string `json:"bitbucket_url"`
}

type prRunBody struct {
	N           int           `json:"n"`
	Kind        string        `json:"kind"`
	Outcome     string        `json:"outcome"`
	Reason      string        `json:"reason,omitempty"`
	ReasonLabel string        `json:"reason_label,omitempty"`
	ElapsedMS   *int64        `json:"elapsed_ms"`
	CreatedAt   time.Time     `json:"created_at"`
	Cost        *string       `json:"cost_usd"`
	Run         *runFactsBody `json:"run"`
	Fixed       int           `json:"fixed"`
	// Time is the recorded stages in three parts; null without a record.
	Time *stageSplitBody `json:"time"`
}

type stageSplitBody struct {
	WorkMS    int64 `json:"work_ms"`
	WaitMS    int64 `json:"wait_ms"`
	GatewayMS int64 `json:"gateway_ms"`
}

// prFindingBody is a posted finding on the PR page: headline only, the
// text is on the PR (as on the run page), and the runs that posted and
// retired it by number (null when that run is not in the history shown).
type prFindingBody struct {
	Severity  string `json:"severity"`
	File      string `json:"file"`
	Line      int    `json:"line"`
	Headline  string `json:"headline"`
	CommentID *int64 `json:"comment_id"`
	Replied   bool   `json:"replied"`
	Fixed     bool   `json:"fixed"`
	PostedIn  *int   `json:"posted_in"`
	FixedIn   *int   `json:"fixed_in"`
}

// prFileBody is one file any run's prompt record lists, with how each run
// treated it, oldest run first.
type prFileBody struct {
	Path     string       `json:"path"`
	Runs     []prFileSent `json:"runs"`
	Findings int          `json:"findings"`
}

type prFileSent struct {
	N    int    `json:"n"`
	Sent string `json:"sent"`
}

type commentCallBody struct {
	// Kind is reply or question.
	Kind      string    `json:"kind"`
	CommentID *int64    `json:"comment_id"`
	Model     string    `json:"model"`
	Cost      *string   `json:"cost_usd"`
	CreatedAt time.Time `json:"created_at"`
}

// prPage serves the PR behind a link token. One 404 for a malformed token
// and an unknown one: no oracle.
func (d Deps) prPage(w http.ResponseWriter, r *http.Request) {
	token := r.PathValue("token")
	if !store.LinkTokenRE.MatchString(token) {
		httpx.WriteError(w, http.StatusNotFound, "no such pull request")
		return
	}
	pg, ok, err := d.DashboardStore.PRPage(r.Context(), token)
	if err != nil {
		d.Log.ErrorContext(r.Context(), "dashboard: PRPage failed: "+err.Error())
		httpx.WriteError(w, http.StatusInternalServerError, "could not read the pull request")
		return
	}
	if !ok {
		httpx.WriteError(w, http.StatusNotFound, "no such pull request")
		return
	}

	body := prPageBody{
		Tag: pg.Key.Tag(), Title: pg.Title, Team: pg.TeamSlug, TeamName: d.teamNamer()(pg.TeamSlug),
		State: "open", BitbucketURL: d.BitbucketURL,
		Runs: make([]prRunBody, 0, len(pg.Runs)), MoreRuns: pg.More,
	}
	switch {
	case pg.MergedAt != nil:
		body.State = "merged"
	case pg.DeclinedAt != nil:
		body.State = "declined"
	}

	numberOf := map[int64]int{}
	files := map[string]*prFileBody{}
	for _, run := range pg.Runs {
		rb := prRunBody{
			N: run.N, Kind: string(run.Kind), Outcome: run.Outcome, Reason: run.Reason,
			ReasonLabel: review.SkipReason(run.Reason).Label(),
			ElapsedMS:   run.ElapsedMS, CreatedAt: run.CreatedAt, Cost: usd(run.CostNanoUSD),
			Run: runFacts(run.Run), Fixed: run.Fixed,
		}
		if run.ElapsedMS != nil {
			body.ElapsedMS += *run.ElapsedMS
		}
		if f := run.Run; f != nil {
			numberOf[f.ID] = run.N
			body.PromptTokens += f.PromptTokens
			body.CachedTokens += f.CachedTokens
			body.CompletionTokens += f.CompletionTokens
		}
		if p := run.Detail; p != nil {
			if len(p.Stages) > 0 {
				work, wait, gw := review.StageSplit(p.Stages)
				rb.Time = &stageSplitBody{WorkMS: work, WaitMS: wait, GatewayMS: gw}
				body.WaitMS += wait
			}
			// Oldest first, so the last one read is the latest.
			if p.PRScope != nil {
				body.Scope = p.PRScope
			}
			for _, f := range p.Files {
				fb := files[f.Path]
				if fb == nil {
					fb = &prFileBody{Path: f.Path}
					files[f.Path] = fb
				}
				fb.Runs = append(fb.Runs, prFileSent{N: run.N, Sent: f.Sent})
			}
		}
		body.Runs = append(body.Runs, rb)
	}

	body.Findings = make([]prFindingBody, 0, len(pg.Findings))
	for _, f := range pg.Findings {
		fb := prFindingBody{
			Severity: f.Severity, File: f.FilePath, Line: f.Line, Headline: withheld(f.Headline),
			CommentID: f.CommentID, Replied: f.Replied, Fixed: f.Fixed,
		}
		if n, ok := numberOf[f.RunID]; ok {
			fb.PostedIn = &n
		}
		if f.FixedRunID != nil {
			if n, ok := numberOf[*f.FixedRunID]; ok {
				fb.FixedIn = &n
			}
		}
		body.Findings = append(body.Findings, fb)
		if file := files[f.FilePath]; file != nil {
			file.Findings++
		}
	}

	body.Files = make([]prFileBody, 0, len(files))
	for _, f := range files {
		body.Files = append(body.Files, *f)
	}
	sort.Slice(body.Files, func(i, j int) bool { return body.Files[i].Path < body.Files[j].Path })

	// The PR total is the figure the cap enforces (PRCost), not a sum of
	// the rows, which a cut history would undercount. A failed read is a
	// failed page: null here would read as "unpriced".
	c, err := d.DashboardStore.PRCost(r.Context(), pg.Key)
	if err != nil {
		d.Log.ErrorContext(r.Context(), "dashboard: PRCost failed: "+err.Error())
		httpx.WriteError(w, http.StatusInternalServerError, "could not read the pull request")
		return
	}
	body.Cost = usd(c)
	// A failed read leaves the list empty; the total above (which counts
	// the calls) still stands.
	calls, more, err := d.DashboardStore.PRCommentCalls(r.Context(), pg.Key)
	if err != nil {
		d.Log.ErrorContext(r.Context(), "dashboard: PRCommentCalls failed: "+err.Error())
		calls, more = nil, false
		body.CallsFailed = true
	}
	body.Calls, body.MoreCalls = make([]commentCallBody, 0, len(calls)), more
	for _, c := range calls {
		body.Calls = append(body.Calls, commentCallBody{Kind: c.Kind, CommentID: c.CommentID,
			Model: c.Model, Cost: usd(c.CostNanoUSD), CreatedAt: c.CreatedAt})
	}
	if rt, _, ok := d.Teams.Lookup(pg.TeamSlug); ok && rt != nil {
		if team := rt.Team(); team != nil && team.Review.MaxPRCostUSD > 0 {
			nano := int64(team.Review.MaxPRCostUSD*1e9 + 0.5)
			body.Cap = usd(&nano)
		}
	}
	// Finding headlines and code paths behind a token: no cache keeps a copy.
	w.Header().Set("Cache-Control", "no-store")
	httpx.WriteJSON(w, http.StatusOK, body)
}
