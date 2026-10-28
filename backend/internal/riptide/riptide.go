// Package riptide forwards a per-PR rollup to the riptide FinOps service once a
// pull request reaches a terminal state.
//
// Forwarding is best-effort and must never fail a webhook: the review is the
// product, the rollup is telemetry. Emitting is therefore silent about its own
// failures, never retries, and is claimed in the database before the POST so a
// crash cannot produce a second one. The wire (bearer, timeout, redirects,
// the ping, the status word) is internal/forward's, shared with Almanach.
package riptide

import (
	"context"
	"errors"
	"fmt"
	"log/slog"
	"time"

	"github.com/trick77/noergler/internal/forward"
)

const (
	emitPath = "/webhooks/noergler"
	pingPath = "/auth/ping"
)

// Status words, as the dashboard shows them: forward's.
const (
	StatusOff      = forward.StatusOff
	StatusOK       = forward.StatusOK
	StatusFailing  = forward.StatusFailing
	StatusRejected = forward.StatusRejected
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
	c   *forward.Client
	log *slog.Logger
}

// New builds an emitter. Forwarding is off unless both url and token are set.
func New(rawURL, token string, log *slog.Logger) *Emitter {
	return &Emitter{c: forward.New("Riptide", rawURL, token, log), log: log}
}

// Enabled reports whether this team forwards to riptide.
func (e *Emitter) Enabled() bool { return e != nil && e.c.Enabled() }

// Status is the forwarding state as of the last call (the startup ping, then
// each emit): off, ok, failing or rejected. A fixed word, never a reason.
func (e *Emitter) Status() string {
	if e == nil {
		return StatusOff
	}
	return e.c.Status()
}

// set records a status word; the tests drive the transitions with it.
func (e *Emitter) set(status string) { e.c.Set(status) }

// VerifyAtStartup checks the token and returns the team riptide knows us as.
//
// A rejected token (401) is fatal and comes back as ErrAuth: emitting for weeks
// with a dead token would lose the data silently. Anything else, including an
// unreachable riptide, is a warning only, because riptide being down is not a
// reason to stop reviewing; a 403 shows as rejected on the dashboard.
func (e *Emitter) VerifyAtStartup(ctx context.Context) (string, error) {
	if !e.Enabled() {
		return "", nil
	}
	team, unauthorized := e.c.Ping(ctx, pingPath)
	if unauthorized {
		return "", fmt.Errorf("riptide token rejected by %s (HTTP 401); check that the token matches the team's entry in team-keys.json: %w", e.c.URL(), ErrAuth)
	}
	return team, nil
}

// EmitPRCompleted posts the rollup. It never returns an error: a transport
// failure or a rejection is logged and dropped, because telemetry must not fail
// a webhook, and it is never retried, because the claim that gated this call was
// already taken.
func (e *Emitter) EmitPRCompleted(ctx context.Context, r Rollup) {
	if !e.Enabled() {
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
	e.c.Post(ctx, emitPath, body, "event_type=pr_completed")
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

// FormatNanoUSD is forward.FormatNanoUSD: money crosses this boundary as an
// exact decimal string, never a float.
func FormatNanoUSD(nano int64) string { return forward.FormatNanoUSD(nano) }

// formatTime renders a timestamp as UTC in the shape riptide reads: second
// precision when there are no microseconds, six digits when there are.
func formatTime(t time.Time) string {
	utc := t.UTC()
	if utc.Nanosecond() == 0 {
		return utc.Format("2006-01-02T15:04:05Z")
	}
	return utc.Format("2006-01-02T15:04:05.000000Z")
}
