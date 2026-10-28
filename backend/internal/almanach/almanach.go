// Package almanach forwards a merged PR's note to Almanach, the living wiki
// that is built from what noergler learns while reviewing.
//
// Forwarding is best effort and never fails a webhook: the review is the
// product, the note is a by-product. Unlike riptide's rollup, a note is NOT
// claimed before the POST: Almanach dedupes on the PR and its source commit,
// so a duplicate is harmless and a lost note is the real cost. The caller
// marks a note emitted only when EmitNote reports that Almanach took it.
package almanach

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"strings"
	"sync/atomic"
	"time"

	"github.com/trick77/noergler/internal/riptide"
)

const (
	emitPath = "/webhooks/noergler"
	pingPath = "/auth/ping"
	// requestTimeout is short on purpose: a slow Almanach must not hold the
	// review worker, and a note is sent again on the team's next merge.
	requestTimeout = 2 * time.Second
	errBodyLimit   = 200
	// EventType is the payload's event_type.
	EventType = "pr_merged_note"
)

// Status words, as the dashboard shows them: fixed, never a reason.
const (
	StatusOff      = "off"
	StatusOK       = "ok"
	StatusFailing  = "failing"
	StatusRejected = "rejected"
)

// ErrAuth means Almanach rejected the token (HTTP 401). Fatal at startup:
// it disables the team rather than losing every note for weeks.
var ErrAuth = errors.New("almanach token rejected")

// Claim is one row of the team's ownership as Almanach needs it to place
// the note: a whole project when RepoSlug is empty.
type Claim struct {
	ProjectKey string
	RepoSlug   string
}

// Call is what the gateway reported for the round's note call: every lane
// llmwire gave noergler, nil where it gave none.
type Call struct {
	Outcome        string
	NoteStatus     string
	Profile        string
	Model          string
	Deployment     string
	ReasoningSent  string
	FinishReason   string
	CallID         string
	GatewayVersion string

	InputTokens         *int64
	InputUncachedTokens *int64
	CacheReadTokens     *int64
	CacheWriteTokens    *int64
	OutputTokens        *int64
	ReasoningTokens     *int64

	CostNanoUSD           *int64
	CostInputNanoUSD      *int64
	CostCacheReadNanoUSD  *int64
	CostCacheWriteNanoUSD *int64
	CostOutputNanoUSD     *int64
	CostReasoningNanoUSD  *int64
	CostToolUsageNanoUSD  *int64
	KeySpendNanoUSD       *int64

	HeadersMS *int64
	TotalMS   *int64
	Warnings  []string
}

// Rounds sums every note call the PR made across its review rounds.
type Rounds struct {
	Count        int
	InputTokens  int64
	OutputTokens int64
	CostNanoUSD  *int64
}

// ReviewTotals is the PR's review aggregate, the same figures riptide gets.
type ReviewTotals struct {
	Runs             int
	PromptTokens     int64
	CachedTokens     int64
	CompletionTokens int64
	ElapsedMS        int64
	CostNanoUSD      *int64
	Models           []string
}

// Note is one merged PR's note with everything Almanach needs beside it:
// where it belongs, what the PR was, what the review found and what the
// note call cost. The team is not in the payload: the bearer token is the
// team's identity at Almanach.
type Note struct {
	ProjectKey string
	RepoSlug   string
	PRID       int
	// PRKey is PROJECT/repo#id, the tag noergler logs under.
	PRKey           string
	PRTitle         string
	TicketKey       string
	JiraIssueType   string
	ChangeKind      string
	ReviewedCommit  string
	SourceCommitSHA string
	MergeCommitSHA  string
	MergedAt        time.Time
	RunID           int64
	ModelLabel      string
	// IssuesPosted and SuggestionsPosted count the findings noergler posted
	// on the PR, by severity.
	IssuesPosted      int
	SuggestionsPosted int
	// Note is the validated note object as the model wrote it.
	Note json.RawMessage
	// Claims is the team's ownership of the note's project.
	Claims []Claim
	// NoteCall is nil when no call row was recorded for the round.
	NoteCall *Call
	Rounds   Rounds
	Review   ReviewTotals
}

// Emitter posts notes for one team. A team without both a URL and a token
// has forwarding switched off, and every method is then a no-op.
type Emitter struct {
	url     string
	token   string
	enabled bool
	// last is the last call's outcome, a Status word, as riptide's.
	last atomic.Pointer[string]
	http *http.Client
	log  *slog.Logger
}

func (e *Emitter) set(status string) { e.last.Store(&status) }

// New builds an emitter. Forwarding is off unless both url and token are set.
func New(rawURL, token string, log *slog.Logger) *Emitter {
	url := strings.TrimSuffix(strings.TrimSpace(rawURL), "/")
	token = strings.TrimSpace(token)
	e := &Emitter{url: url, token: token, enabled: url != "" && token != "", log: log}
	if e.enabled {
		e.http = &http.Client{
			Timeout: requestTimeout,
			// Redirects are not followed: a 3xx would replay the bearer at
			// the new host, and a redirect must never read as success.
			CheckRedirect: func(*http.Request, []*http.Request) error { return http.ErrUseLastResponse },
		}
	}
	return e
}

// Enabled reports whether this team forwards to Almanach.
func (e *Emitter) Enabled() bool { return e != nil && e.enabled }

// Status is the forwarding state as of the last call: off, ok, failing or
// rejected. A fixed word, never a reason.
func (e *Emitter) Status() string {
	if e == nil || !e.enabled {
		return StatusOff
	}
	if s := e.last.Load(); s != nil {
		return *s
	}
	return StatusFailing
}

// VerifyAtStartup checks the token and returns the team Almanach knows us
// as. A rejected token (401) is fatal and comes back as ErrAuth; anything
// else, an unreachable Almanach included, is a warning only.
func (e *Emitter) VerifyAtStartup(ctx context.Context) (string, error) {
	if !e.Enabled() {
		return "", nil
	}
	e.set(StatusFailing)
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, e.url+pingPath, nil)
	if err != nil {
		e.log.WarnContext(ctx, "Almanach ping failed: "+err.Error())
		return "", nil
	}
	req.Header.Set("Authorization", "Bearer "+e.token)
	resp, err := e.http.Do(req)
	if err != nil {
		e.log.WarnContext(ctx, fmt.Sprintf("Almanach unreachable at startup (url=%s): %v", e.url, err))
		return "", nil
	}
	defer func() { _ = resp.Body.Close() }()

	if resp.StatusCode == http.StatusUnauthorized {
		e.set(StatusRejected)
		return "", fmt.Errorf("almanach token rejected by %s (HTTP 401); check the team's token: %w", e.url, ErrAuth)
	}
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		if resp.StatusCode == http.StatusForbidden {
			e.set(StatusRejected)
		}
		body, _ := io.ReadAll(io.LimitReader(resp.Body, errBodyLimit))
		e.log.WarnContext(ctx, fmt.Sprintf("Almanach ping returned unexpected status %d: %s",
			resp.StatusCode, strings.TrimSpace(string(body))))
		return "", nil
	}
	var out struct {
		Team string `json:"team"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&out); err != nil {
		e.log.WarnContext(ctx, "Almanach ping returned an unreadable body: "+err.Error())
		return "", nil
	}
	if out.Team == "" {
		e.log.WarnContext(ctx, "Almanach ping named no team")
		return "", nil
	}
	e.set(StatusOK)
	e.log.InfoContext(ctx, "Almanach: OK (team="+out.Team+")")
	return out.Team, nil
}

// EmitNote posts one note and reports whether Almanach took it (2xx). A
// transport failure or a rejection is logged, the status word set, and the
// note left for the next drain. Never an error: a note must not fail a
// webhook.
func (e *Emitter) EmitNote(ctx context.Context, n Note) bool {
	if !e.Enabled() {
		return false
	}
	raw, err := json.Marshal(payload(n))
	if err != nil {
		e.set(StatusFailing)
		e.log.WarnContext(ctx, "Almanach emit failed to encode: "+err.Error())
		return false
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, e.url+emitPath, bytes.NewReader(raw))
	if err != nil {
		e.set(StatusFailing)
		e.log.WarnContext(ctx, "Almanach emit failed: "+err.Error())
		return false
	}
	req.Header.Set("Authorization", "Bearer "+e.token)
	req.Header.Set("Content-Type", "application/json")

	resp, err := e.http.Do(req)
	if err != nil {
		if ctx.Err() != nil {
			e.log.WarnContext(ctx, fmt.Sprintf("Almanach emit cancelled, note %s stays pending: %v", n.PRKey, err))
			return false
		}
		e.set(StatusFailing)
		e.log.WarnContext(ctx, fmt.Sprintf("Almanach emit failed, note %s stays pending: %v", n.PRKey, err))
		return false
	}
	defer func() { _ = resp.Body.Close() }()
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		word := "failed"
		if resp.StatusCode == http.StatusUnauthorized || resp.StatusCode == http.StatusForbidden {
			e.set(StatusRejected)
			word = "rejected"
		} else {
			e.set(StatusFailing)
		}
		msg, _ := io.ReadAll(io.LimitReader(resp.Body, errBodyLimit))
		e.log.WarnContext(ctx, fmt.Sprintf("Almanach emit %s (note %s, status=%d), note stays pending: %s",
			word, n.PRKey, resp.StatusCode, strings.TrimSpace(string(msg))))
		return false
	}
	e.set(StatusOK)
	_, _ = io.Copy(io.Discard, resp.Body)
	e.log.InfoContext(ctx, fmt.Sprintf("Almanach: note %s taken (status=%d)", n.PRKey, resp.StatusCode))
	return true
}

// payload is the wire shape. Money travels twice: as an exact decimal
// string (never a float) and as nano-USD integers, so Almanach can store
// the integer without parsing. A lane the gateway did not send is null.
func payload(n Note) map[string]any {
	claims := make([]map[string]any, 0, len(n.Claims))
	for _, c := range n.Claims {
		claims = append(claims, map[string]any{"project_key": c.ProjectKey, "repo_slug": nullable(c.RepoSlug)})
	}
	note := n.Note
	if len(note) == 0 {
		note = json.RawMessage("null")
	}
	body := map[string]any{
		"event_type":         EventType,
		"project_key":        n.ProjectKey,
		"repo_slug":          n.RepoSlug,
		"pr_id":              n.PRID,
		"pr_key":             n.PRKey,
		"pr_title":           n.PRTitle,
		"ticket_key":         nullable(n.TicketKey),
		"jira_issue_type":    nullable(n.JiraIssueType),
		"change_kind":        nullable(n.ChangeKind),
		"reviewed_commit":    nullable(n.ReviewedCommit),
		"source_commit_sha":  n.SourceCommitSHA,
		"merge_commit_sha":   nullable(n.MergeCommitSHA),
		"merged_at":          n.MergedAt.UTC().Format(time.RFC3339),
		"run_id":             n.RunID,
		"model_label":        nullable(n.ModelLabel),
		"issues_posted":      n.IssuesPosted,
		"suggestions_posted": n.SuggestionsPosted,
		"note":               note,
		"claims":             claims,
		"note_call":          callBody(n.NoteCall),
		"rounds": map[string]any{
			"count":         n.Rounds.Count,
			"input_tokens":  n.Rounds.InputTokens,
			"output_tokens": n.Rounds.OutputTokens,
			"cost_usd":      usd(n.Rounds.CostNanoUSD),
			"cost_nano_usd": n.Rounds.CostNanoUSD,
		},
		"review_totals": map[string]any{
			"runs":              n.Review.Runs,
			"prompt_tokens":     n.Review.PromptTokens,
			"cached_tokens":     n.Review.CachedTokens,
			"completion_tokens": n.Review.CompletionTokens,
			"elapsed_ms":        n.Review.ElapsedMS,
			"cost_usd":          usd(n.Review.CostNanoUSD),
			"cost_nano_usd":     n.Review.CostNanoUSD,
			"models_used":       orEmpty(n.Review.Models),
		},
	}
	return body
}

func callBody(c *Call) any {
	if c == nil {
		return nil
	}
	return map[string]any{
		"outcome":                   c.Outcome,
		"note_status":               c.NoteStatus,
		"profile":                   nullable(c.Profile),
		"model":                     nullable(c.Model),
		"deployment":                nullable(c.Deployment),
		"reasoning_sent":            nullable(c.ReasoningSent),
		"finish_reason":             nullable(c.FinishReason),
		"call_id":                   nullable(c.CallID),
		"litellm_version":           nullable(c.GatewayVersion),
		"input_tokens":              c.InputTokens,
		"input_uncached_tokens":     c.InputUncachedTokens,
		"cache_read_tokens":         c.CacheReadTokens,
		"cache_write_tokens":        c.CacheWriteTokens,
		"output_tokens":             c.OutputTokens,
		"reasoning_tokens":          c.ReasoningTokens,
		"cost_usd":                  usd(c.CostNanoUSD),
		"cost_nano_usd":             c.CostNanoUSD,
		"cost_input_nano_usd":       c.CostInputNanoUSD,
		"cost_cache_read_nano_usd":  c.CostCacheReadNanoUSD,
		"cost_cache_write_nano_usd": c.CostCacheWriteNanoUSD,
		"cost_output_nano_usd":      c.CostOutputNanoUSD,
		"cost_reasoning_nano_usd":   c.CostReasoningNanoUSD,
		"cost_tool_usage_nano_usd":  c.CostToolUsageNanoUSD,
		"key_spend_usd":             usd(c.KeySpendNanoUSD),
		"key_spend_nano_usd":        c.KeySpendNanoUSD,
		"headers_ms":                c.HeadersMS,
		"total_ms":                  c.TotalMS,
		"warnings":                  orEmpty(c.Warnings),
	}
}

func usd(nano *int64) any {
	if nano == nil {
		return nil
	}
	return riptide.FormatNanoUSD(*nano)
}

func nullable(s string) any {
	if s == "" {
		return nil
	}
	return s
}

func orEmpty(s []string) []string {
	if s == nil {
		return []string{}
	}
	return s
}
