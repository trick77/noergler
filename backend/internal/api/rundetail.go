package api

import (
	"net/http"
	"strconv"
	"time"

	"github.com/trick77/noergler/internal/inference"
	"github.com/trick77/noergler/internal/review"
	"github.com/trick77/noergler/internal/store"
	"github.com/trick77/ship/kit/httpx"
)

// runDetailBody is GET /api/dashboard/runs/{id}: one attempt and everything
// recorded about it.
//
// Headlines only (decided): no finding text, suggestion, quoted code or
// dispute, and no gateway discount or margin. The dashboard is
// unauthenticated; the full text is one link away in Bitbucket.
//
// Money as on the rest of the dashboard: 3-decimal strings, nil for
// unpriced. The cost lanes are 6-decimal, because a cache read on a small
// call is a tenth of a milli-dollar.
type runDetailBody struct {
	ID          int64     `json:"id"`
	Tag         string    `json:"tag"`
	Title       string    `json:"title"`
	Team        string    `json:"team"`
	TeamName    string    `json:"team_name"`
	Kind        string    `json:"kind"`
	Outcome     string    `json:"outcome"`
	Reason      string    `json:"reason,omitempty"`
	ReasonLabel string    `json:"reason_label,omitempty"`
	ElapsedMS   *int64    `json:"elapsed_ms"`
	CreatedAt   time.Time `json:"created_at"`
	// Cost is the run's, or the attempt's for a billed call with no run.
	Cost *string `json:"cost_usd"`

	Run      *runFactsBody `json:"run"`
	Call     *callBody     `json:"call"`
	Findings []findingBody `json:"findings"`
	Fixed    []findingBody `json:"fixed"`

	PR PRBody `json:"pr"`
	// BitbucketURL as on the other bodies, for the PR and comment links.
	BitbucketURL string `json:"bitbucket_url"`
}

type runFactsBody struct {
	Incremental      bool   `json:"incremental"`
	FromCommit       string `json:"from_commit,omitempty"`
	ToCommit         string `json:"to_commit,omitempty"`
	Model            string `json:"model"`
	PromptTokens     int64  `json:"prompt_tokens"`
	CachedTokens     int64  `json:"cached_tokens"`
	CompletionTokens int64  `json:"completion_tokens"`
	FindingsPosted   int    `json:"findings_posted"`
	LinesAdded       int    `json:"lines_added"`
	LinesRemoved     int    `json:"lines_removed"`
	FilesChanged     int    `json:"files_changed"`
}

// callBody is the llm_calls row. Every nil is "the gateway did not send it",
// which the page shows as absent, never as zero.
type callBody struct {
	Profile             string   `json:"profile"`
	Model               string   `json:"model,omitempty"`
	Deployment          string   `json:"deployment,omitempty"`
	ReasoningSent       string   `json:"reasoning_sent,omitempty"`
	FinishReason        string   `json:"finish_reason,omitempty"`
	CallID              string   `json:"call_id,omitempty"`
	LiteLLMVersion      string   `json:"litellm_version,omitempty"`
	InputTokens         *int64   `json:"input_tokens"`
	InputUncachedTokens *int64   `json:"input_uncached_tokens"`
	CacheReadTokens     *int64   `json:"cache_read_tokens"`
	CacheWriteTokens    *int64   `json:"cache_write_tokens"`
	OutputTokens        *int64   `json:"output_tokens"`
	ReasoningTokens     *int64   `json:"reasoning_tokens"`
	Cost                *string  `json:"cost_usd"`
	CostInput           *string  `json:"cost_input_usd"`
	CostCacheRead       *string  `json:"cost_cache_read_usd"`
	CostCacheWrite      *string  `json:"cost_cache_write_usd"`
	CostOutput          *string  `json:"cost_output_usd"`
	CostReasoning       *string  `json:"cost_reasoning_usd"`
	CostToolUsage       *string  `json:"cost_tool_usage_usd"`
	KeySpend            *string  `json:"key_spend_usd"`
	HeadersMS           *int64   `json:"headers_ms"`
	TotalMS             *int64   `json:"total_ms"`
	Warnings            []string `json:"warnings"`
}

type findingBody struct {
	Severity   string `json:"severity"`
	Confidence *int   `json:"confidence"`
	File       string `json:"file"`
	Line       int    `json:"line"`
	Headline   string `json:"headline"`
	CommentID  *int64 `json:"comment_id"`
	Replied    bool   `json:"replied"`
	Fixed      bool   `json:"fixed"`
	// FromAttempt is the page of the run that posted it.
	FromAttempt *int64 `json:"from_attempt_id"`
}

// PRBody is the PR's side of the page: its history and what it has cost
// against the team's cap.
type PRBody struct {
	Attempts []prAttemptBody `json:"attempts"`
	Cost     *string         `json:"cost_usd"`
	// Cap is the team's per-PR cost cap, nil when the team is no longer
	// configured.
	Cap *string `json:"cap_usd"`
}

type prAttemptBody struct {
	ID          int64     `json:"id"`
	Kind        string    `json:"kind"`
	Outcome     string    `json:"outcome"`
	Reason      string    `json:"reason,omitempty"`
	ReasonLabel string    `json:"reason_label,omitempty"`
	ElapsedMS   *int64    `json:"elapsed_ms"`
	CreatedAt   time.Time `json:"created_at"`
	Findings    *int      `json:"findings"`
	Cost        *string   `json:"cost_usd"`
}

func usd6(nano *int64) *string {
	if nano == nil {
		return nil
	}
	s := inference.FormatUSD6(*nano)
	return &s
}

// runDetail serves one attempt. 404 for an id that is not a number or not
// an attempt: the page then says so rather than rendering an empty run.
func (d Deps) runDetail(w http.ResponseWriter, r *http.Request) {
	id, err := strconv.ParseInt(r.PathValue("id"), 10, 64)
	if err != nil || id <= 0 {
		httpx.WriteError(w, http.StatusNotFound, "no such run")
		return
	}
	det, ok, err := d.DashboardStore.RunDetail(r.Context(), id)
	if err != nil {
		d.Log.ErrorContext(r.Context(), "dashboard: RunDetail failed: "+err.Error())
		httpx.WriteError(w, http.StatusInternalServerError, "could not read the run")
		return
	}
	if !ok {
		httpx.WriteError(w, http.StatusNotFound, "no such run")
		return
	}

	nameOf := d.teamNamer()
	body := runDetailBody{
		ID: det.ID, Tag: det.Key.Tag(), Title: det.PRTitle,
		Team: det.TeamSlug, TeamName: nameOf(det.TeamSlug), Kind: string(det.Kind),
		Outcome: det.Outcome, Reason: det.Reason, ReasonLabel: review.SkipReason(det.Reason).Label(),
		ElapsedMS: det.ElapsedMS, CreatedAt: det.CreatedAt,
		Cost:         usd(det.AttemptCostNanoUSD),
		Findings:     findingBodies(det.Findings),
		Fixed:        findingBodies(det.Fixed),
		BitbucketURL: d.BitbucketURL,
	}
	if f := det.Run; f != nil {
		body.Cost = usd(f.CostNanoUSD)
		body.Run = &runFactsBody{
			Incremental: f.Incremental, FromCommit: f.FromCommit, ToCommit: f.ToCommit, Model: f.ModelLabel,
			PromptTokens: f.PromptTokens, CachedTokens: f.CachedTokens, CompletionTokens: f.CompletionTokens,
			FindingsPosted: f.FindingsPosted, LinesAdded: f.LinesAdded, LinesRemoved: f.LinesRemoved,
			FilesChanged: f.FilesChanged,
		}
	}
	if c := det.Call; c != nil {
		body.Call = &callBody{
			Profile: c.Profile, Model: c.Model, Deployment: c.Deployment, ReasoningSent: c.ReasoningSent,
			FinishReason: c.FinishReason, CallID: c.CallID, LiteLLMVersion: c.GatewayVer,
			InputTokens: c.InputTokens, InputUncachedTokens: c.InputUncachedTokens,
			CacheReadTokens: c.CacheReadTokens, CacheWriteTokens: c.CacheWriteTokens,
			OutputTokens: c.OutputTokens, ReasoningTokens: c.ReasoningTokens,
			Cost: usd(c.CostNanoUSD), CostInput: usd6(c.CostInputNanoUSD),
			CostCacheRead: usd6(c.CostCacheReadNanoUSD), CostCacheWrite: usd6(c.CostCacheWriteNanoUSD),
			CostOutput: usd6(c.CostOutputNanoUSD), CostReasoning: usd6(c.CostReasoningNanoUSD),
			CostToolUsage: usd6(c.CostToolUsageNanoUSD), KeySpend: usd(c.KeySpendNanoUSD),
			HeadersMS: c.HeadersMS, TotalMS: c.TotalMS, Warnings: c.Warnings,
		}
		if body.Call.Warnings == nil {
			body.Call.Warnings = []string{}
		}
	}

	body.PR.Attempts = make([]prAttemptBody, 0, len(det.PRAttempts))
	for _, a := range det.PRAttempts {
		body.PR.Attempts = append(body.PR.Attempts, prAttemptBody{
			ID: a.ID, Kind: string(a.Kind), Outcome: a.Outcome, Reason: a.Reason,
			ReasonLabel: review.SkipReason(a.Reason).Label(),
			ElapsedMS:   a.ElapsedMS, CreatedAt: a.CreatedAt, Findings: a.Findings, Cost: usd(a.CostNanoUSD),
		})
	}
	// The PR total is the same figure the cap enforces (PRCost), not a sum
	// of the rows above, which a history longer than the page would cut. A
	// failed read is a failed page: null here would read as "unpriced".
	c, err := d.DashboardStore.PRCost(r.Context(), det.Key)
	if err != nil {
		d.Log.ErrorContext(r.Context(), "dashboard: PRCost failed: "+err.Error())
		httpx.WriteError(w, http.StatusInternalServerError, "could not read the run")
		return
	}
	body.PR.Cost = usd(c)
	if rt, _, ok := d.Teams.Lookup(det.TeamSlug); ok && rt != nil {
		if team := rt.Team(); team != nil && team.Review.MaxPRCostUSD > 0 {
			nano := int64(team.Review.MaxPRCostUSD*1e9 + 0.5)
			body.PR.Cap = usd(&nano)
		}
	}
	httpx.WriteJSON(w, http.StatusOK, body)
}

func findingBodies(fs []store.FindingHeadline) []findingBody {
	out := make([]findingBody, 0, len(fs))
	for _, f := range fs {
		out = append(out, findingBody{
			Severity: f.Severity, Confidence: f.Confidence, File: f.FilePath, Line: f.Line,
			Headline: f.Headline, CommentID: f.CommentID, Replied: f.Replied, Fixed: f.Fixed,
			FromAttempt: f.FromAttemptID,
		})
	}
	return out
}
