// Package riptide forwards a per-PR rollup to the riptide FinOps service once a
// pull request reaches a terminal state.
//
// Forwarding is best-effort and must never fail a webhook: the review is the
// product, the rollup is telemetry. Emitting is therefore silent about its own
// failures, never retries, and is claimed in the database before the POST so a
// crash cannot produce a second one.
package riptide

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"strconv"
	"strings"
	"time"
)

const (
	emitPath = "/webhooks/noergler"
	pingPath = "/auth/ping"
	// requestTimeout is deliberately short: a slow riptide must not hold a
	// review worker, and a missed rollup is cheaper than a stalled queue.
	requestTimeout = 2 * time.Second
	// errBodyLimit caps how much of a rejection is logged.
	errBodyLimit = 200
	// nanoPerUSD is llmwire's cost unit, which the store also uses.
	nanoPerUSD = 1_000_000_000
)

// ErrAuth means riptide rejected the token (HTTP 401). Fatal at startup: it
// disables the team rather than emitting into the void for weeks.
var ErrAuth = errors.New("riptide token rejected")

// Rollup is the per-PR summary riptide expects.
type Rollup struct {
	// Outcome is "merged", "declined" or "deleted". Riptide rejects anything
	// else with a 422; it is not validated here.
	Outcome               string
	PRKey                 string
	Repo                  string
	SourceCommitSHA       string
	MergeCommitSHA        string
	LinesAdded            int
	LinesRemoved          int
	FilesChanged          int
	TotalRuns             int
	TotalPromptTokens     int64
	TotalCompletionTokens int64
	TotalElapsedMS        int64
	TotalFindingsCount    int
	// TotalCostNanoUSD is nil when no run was priced. It is then omitted from
	// the payload rather than sent as zero: a zero understates spend silently,
	// an absent field is countable on the riptide side.
	TotalCostNanoUSD *int64
	ModelsUsed       []string
	FirstReviewAt    time.Time
	ClosedAt         time.Time
	ReviewerHandle   string
}

// Emitter posts rollups for one team. A team without both a URL and a token has
// forwarding switched off, and every method is then a no-op.
type Emitter struct {
	url     string
	token   string
	enabled bool
	http    *http.Client
	log     *slog.Logger
}

// New builds an emitter. Forwarding is off unless both url and token are set.
func New(rawURL, token string, log *slog.Logger) *Emitter {
	url := strings.TrimSuffix(strings.TrimSpace(rawURL), "/")
	token = strings.TrimSpace(token)
	e := &Emitter{
		url:     url,
		token:   token,
		enabled: url != "" && token != "",
		log:     log,
	}
	if e.enabled {
		e.http = &http.Client{
			Timeout: requestTimeout,
			// Matches httpx, which does not follow redirects.
			CheckRedirect: func(*http.Request, []*http.Request) error { return http.ErrUseLastResponse },
		}
	}
	return e
}

// Enabled reports whether this team forwards to riptide.
func (e *Emitter) Enabled() bool { return e.enabled }

// VerifyAtStartup checks the token and returns the team riptide knows us as.
//
// A rejected token (401) is fatal and comes back as ErrAuth: emitting for weeks
// with a dead token would lose the data silently. Anything else, including an
// unreachable riptide, is a warning only, because riptide being down is not a
// reason to stop reviewing.
func (e *Emitter) VerifyAtStartup(ctx context.Context) (string, error) {
	if !e.enabled {
		return "", nil
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, e.url+pingPath, nil)
	if err != nil {
		return "", nil
	}
	req.Header.Set("Authorization", "Bearer "+e.token)

	resp, err := e.http.Do(req)
	if err != nil {
		e.log.WarnContext(ctx, fmt.Sprintf("Riptide unreachable at startup (url=%s): %v", e.url, err))
		return "", nil
	}
	defer func() { _ = resp.Body.Close() }()

	if resp.StatusCode == http.StatusUnauthorized {
		return "", fmt.Errorf("riptide token rejected by %s (HTTP 401); check that the token matches the team's entry in team-keys.json: %w", e.url, ErrAuth)
	}
	// 3xx included: a redirected URL is a misconfiguration, and reporting it as
	// an unreadable body would send the reader looking at riptide instead.
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		body, _ := io.ReadAll(io.LimitReader(resp.Body, errBodyLimit))
		e.log.WarnContext(ctx, fmt.Sprintf("Riptide ping returned unexpected status %d: %s",
			resp.StatusCode, strings.TrimSpace(string(body))))
		return "", nil
	}

	var out struct {
		Team string `json:"team"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&out); err != nil {
		e.log.WarnContext(ctx, "Riptide ping returned an unreadable body: "+err.Error())
		return "", nil
	}
	e.log.InfoContext(ctx, "Riptide: OK (team="+out.Team+")")
	return out.Team, nil
}

// EmitPRCompleted posts the rollup. It never returns an error: a transport
// failure or a rejection is logged and dropped, because telemetry must not fail
// a webhook, and it is never retried, because the claim that gated this call was
// already taken.
func (e *Emitter) EmitPRCompleted(ctx context.Context, r Rollup) {
	if !e.enabled {
		return
	}
	if r.TotalCostNanoUSD == nil {
		e.log.WarnContext(ctx, fmt.Sprintf(
			"Riptide: emitting %s (outcome=%s) without cost, no price for models=%v; FinOps will undercount until pricing is configured",
			r.PRKey, r.Outcome, r.ModelsUsed))
	}

	body := map[string]any{
		"event_type":              "pr_completed",
		"outcome":                 r.Outcome,
		"pr_key":                  r.PRKey,
		"repo":                    r.Repo,
		"source_commit_sha":       r.SourceCommitSHA,
		"merge_commit_sha":        nullableString(r.MergeCommitSHA),
		"lines_added":             r.LinesAdded,
		"lines_removed":           r.LinesRemoved,
		"files_changed":           r.FilesChanged,
		"total_runs":              r.TotalRuns,
		"total_prompt_tokens":     r.TotalPromptTokens,
		"total_completion_tokens": r.TotalCompletionTokens,
		"total_elapsed_ms":        r.TotalElapsedMS,
		"total_findings_count":    r.TotalFindingsCount,
		"models_used":             modelsOrEmpty(r.ModelsUsed),
		"first_review_at":         formatTime(r.FirstReviewAt),
		"closed_at":               formatTime(r.ClosedAt),
	}
	// Cost is a decimal string, never a float: riptide stores money and a
	// float64 cannot hold every nano-USD value exactly.
	if r.TotalCostNanoUSD != nil {
		body["total_cost_usd"] = FormatNanoUSD(*r.TotalCostNanoUSD)
	}
	// The handle and the account kind travel together or not at all.
	if r.ReviewerHandle != "" {
		body["reviewer_handle"] = r.ReviewerHandle
		body["reviewer_account_kind"] = "bot"
	}

	raw, err := json.Marshal(body)
	if err != nil {
		e.log.WarnContext(ctx, "Riptide emit failed to encode: "+err.Error())
		return
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, e.url+emitPath, bytes.NewReader(raw))
	if err != nil {
		e.log.WarnContext(ctx, "Riptide emit failed: "+err.Error())
		return
	}
	req.Header.Set("Authorization", "Bearer "+e.token)
	req.Header.Set("Content-Type", "application/json")

	resp, err := e.http.Do(req)
	if err != nil {
		e.log.WarnContext(ctx, fmt.Sprintf("Riptide emit failed (event_type=pr_completed): %v", err))
		return
	}
	defer func() { _ = resp.Body.Close() }()
	// Anything but a 2xx is a miss, 3xx included: redirects are not followed, so
	// a redirected endpoint would otherwise drop every rollup without a word,
	// and the DB claim taken before this call means it is never retried.
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		msg, _ := io.ReadAll(io.LimitReader(resp.Body, errBodyLimit))
		e.log.WarnContext(ctx, fmt.Sprintf("Riptide emit rejected (event_type=pr_completed, status=%d): %s",
			resp.StatusCode, strings.TrimSpace(string(msg))))
		return
	}
	_, _ = io.Copy(io.Discard, resp.Body)
}

// nullableString renders an empty string as JSON null. merge_commit_sha is
// always present in the payload and null for a declined or deleted PR.
func nullableString(s string) any {
	if s == "" {
		return nil
	}
	return s
}

// modelsOrEmpty keeps models_used an array rather than null.
func modelsOrEmpty(m []string) []string {
	if m == nil {
		return []string{}
	}
	return m
}

// FormatNanoUSD renders nano-USD as a plain decimal string: no exponent, no
// trailing zeros, always at least one digit either side of the point. Money
// crosses this boundary as text so neither side rounds it through a float.
func FormatNanoUSD(nano int64) string {
	sign := ""
	if nano < 0 {
		sign = "-"
		nano = -nano
	}
	whole := nano / nanoPerUSD
	frac := nano % nanoPerUSD
	if frac == 0 {
		return sign + strconv.FormatInt(whole, 10)
	}
	digits := strings.TrimRight(fmt.Sprintf("%09d", frac), "0")
	return sign + strconv.FormatInt(whole, 10) + "." + digits
}

// formatTime renders a timestamp as UTC in the shape riptide reads: second
// precision when there are no microseconds, six digits when there are.
func formatTime(t time.Time) string {
	utc := t.UTC()
	if utc.Nanosecond() == 0 {
		return utc.Format("2006-01-02T15:04:05Z")
	}
	return utc.Format("2006-01-02T15:04:05.000000Z")
}
