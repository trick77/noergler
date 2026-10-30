package api

import (
	"fmt"
	"net/http"
	"strconv"
	"time"

	"github.com/trick77/noergler/internal/inference"
	"github.com/trick77/noergler/internal/review"
	"github.com/trick77/noergler/internal/store"
	"github.com/trick77/ship/kit/httpx"
	"github.com/trick77/ship/kit/logging"
)

// runDetailBody is GET /api/dashboard/prs/{token}/runs/{n}: one attempt of
// the PR and everything recorded about it. The PR's history, cost and calls
// are the PR page's (prPageBody).
//
// Full finding text, suggestion, dispute and quoted code (decided): the
// token is handed out only in the PR's summary comment, so the page's
// audience is the PR's. Never a gateway discount or margin.
//
// Money as on the rest of the dashboard: 3-decimal strings, nil for
// unpriced. The cost lanes are 6-decimal, because a cache read on a small
// call is a tenth of a milli-dollar. The page rounds all of it to cents.
type runDetailBody struct {
	N           int       `json:"n"`
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

	// Detail is the run's pipeline record (stages, prompt parts, files,
	// unposted findings, verdict, HTTP counts); null when none was written.
	Detail *store.Pipeline `json:"detail"`
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

func runFacts(f *store.RunFacts) *runFactsBody {
	if f == nil {
		return nil
	}
	return &runFactsBody{
		Incremental: f.Incremental, FromCommit: f.FromCommit, ToCommit: f.ToCommit, Model: f.ModelLabel,
		PromptTokens: f.PromptTokens, CachedTokens: f.CachedTokens, CompletionTokens: f.CompletionTokens,
		FindingsPosted: f.FindingsPosted, LinesAdded: f.LinesAdded, LinesRemoved: f.LinesRemoved,
		FilesChanged: f.FilesChanged,
	}
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
	Severity   string  `json:"severity"`
	Confidence *int    `json:"confidence"`
	File       string  `json:"file"`
	Line       int     `json:"line"`
	Headline   string  `json:"headline"`
	Comment    string  `json:"comment"`
	Suggestion *string `json:"suggestion"`
	Dispute    *string `json:"dispute"`
	CommentID  *int64  `json:"comment_id"`
	Replied    bool    `json:"replied"`
	Fixed      bool    `json:"fixed"`
}

// runNumber is a run's path segment: a plain positive decimal, nothing else
// (no sign, no leading zero), so one run has one path.
func runNumber(s string) (int, bool) {
	if s == "" || len(s) > 9 || s[0] == '0' {
		return 0, false
	}
	n, err := strconv.Atoi(s)
	if err != nil || n < 1 || strconv.Itoa(n) != s {
		return 0, false
	}
	return n, true
}

// prRunDetail serves run n of the PR behind a link token. One 404 for a
// malformed token, an unknown one and a run the PR does not have: no
// oracle, and the page says so rather than rendering an empty run.
func (d Deps) prRunDetail(w http.ResponseWriter, r *http.Request) {
	token := r.PathValue("token")
	n, okN := runNumber(r.PathValue("n"))
	if !store.LinkTokenRE.MatchString(token) || !okN {
		httpx.WriteError(w, http.StatusNotFound, "no such run")
		return
	}
	det, ok, err := d.DashboardStore.PRRun(r.Context(), token, n)
	if err != nil {
		httpx.ServerError(w, r, fmt.Errorf("dashboard: PRRun: %w", err), "could not read the run")
		return
	}
	if !ok {
		httpx.WriteError(w, http.StatusNotFound, "no such run")
		return
	}

	nameOf := d.teamNamer()
	body := runDetailBody{
		N: det.N, Tag: det.Key.Tag(), Title: det.PRTitle,
		Team: det.TeamSlug, TeamName: nameOf(det.TeamSlug), Kind: string(det.Kind),
		Outcome: det.Outcome, Reason: det.Reason, ReasonLabel: review.SkipReason(det.Reason).Label(),
		ElapsedMS: det.ElapsedMS, CreatedAt: det.CreatedAt,
		Cost:         usd(det.AttemptCostNanoUSD),
		Run:          runFacts(det.Run),
		Findings:     findingBodies(det.Findings),
		Fixed:        findingBodies(det.Fixed),
		BitbucketURL: d.BitbucketURL,
	}
	if det.Detail != nil {
		// The whole PR's counts are the PR page's figure, not this run's.
		p := *det.Detail
		p.PRScope = nil
		body.Detail = &p
	}
	if det.DetailErr != nil {
		// Debug: a record an older binary wrote stays undecodable, and every
		// view of its page would log again.
		d.Log.DebugContext(logging.With(r.Context(), "team", det.TeamSlug), fmt.Sprintf(
			"dashboard: run %d detail record not decoded, page shown without it: %v", det.ID, det.DetailErr))
	}
	if det.Run != nil {
		body.Cost = usd(det.Run.CostNanoUSD)
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
	// Full finding text behind a token: no cache keeps a copy.
	w.Header().Set("Cache-Control", "no-store")
	httpx.WriteJSON(w, http.StatusOK, body)
}

// findingBodies are the posted findings with their text. A text with a
// secret-like token (inference.HasSecret), the author's dispute included,
// is withheld: never a secret on the page, though Bitbucket shows it.
func findingBodies(fs []store.RunFinding) []findingBody {
	out := make([]findingBody, 0, len(fs))
	for _, f := range fs {
		b := findingBody{
			Severity: f.Severity, Confidence: f.Confidence, File: f.FilePath, Line: f.Line,
			Headline: withheld(f.Headline), Comment: withheld(f.CommentText),
			CommentID: f.CommentID, Replied: f.Replied, Fixed: f.Fixed,
		}
		if f.Suggestion != nil && !inference.HasSecret(*f.Suggestion) {
			b.Suggestion = f.Suggestion
		}
		if f.DisputeText != nil && !inference.HasSecret(*f.DisputeText) {
			b.Dispute = f.DisputeText
		}
		out = append(out, b)
	}
	return out
}

// withheld is s, or empty when it carries a secret-like token.
func withheld(s string) string {
	if inference.HasSecret(s) {
		return ""
	}
	return s
}
