// Package almanach forwards a merged PR's note to Almanach, the living wiki
// that is built from what noergler learns while reviewing.
//
// Forwarding is best effort and never fails a webhook: the review is the
// product, the note is a by-product. Unlike riptide's rollup, a note is NOT
// claimed before the POST: Almanach dedupes on the PR and its source commit,
// so a duplicate is harmless and a lost note is the real cost. The caller
// marks a note emitted only when EmitNote reports that Almanach took it.
// The wire (bearer, timeout, redirects, the ping, the status word) is
// internal/forward's, shared with riptide.
package almanach

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"log/slog"
	"time"

	"github.com/trick77/noergler/internal/forward"
	"github.com/trick77/noergler/internal/store"
)

const (
	emitPath = "/webhooks/noergler"
	pingPath = "/auth/ping"
	// EventType is the payload's event_type.
	EventType = "pr_merged_note"
)

// Status words, as the dashboard shows them: forward's.
const (
	StatusOff      = forward.StatusOff
	StatusOK       = forward.StatusOK
	StatusFailing  = forward.StatusFailing
	StatusRejected = forward.StatusRejected
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

// Call is the round's note call as recorded: what became of the note and
// every lane the gateway reported (store.LLMCall, so a lane added there
// reaches Almanach without a second list).
type Call struct {
	Outcome    string
	NoteStatus string
	store.LLMCall
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
	c   *forward.Client
	log *slog.Logger
}

// New builds an emitter. Forwarding is off unless both url and token are set.
func New(rawURL, token string, log *slog.Logger) *Emitter {
	return &Emitter{c: forward.New("Almanach", rawURL, token, log), log: log}
}

// Enabled reports whether this team forwards to Almanach.
func (e *Emitter) Enabled() bool { return e != nil && e.c.Enabled() }

// Status is the forwarding state as of the last call: off, ok, failing or
// rejected. A fixed word, never a reason.
func (e *Emitter) Status() string {
	if e == nil {
		return StatusOff
	}
	return e.c.Status()
}

// VerifyAtStartup checks the token and returns the team Almanach knows us
// as. A rejected token (401) is fatal and comes back as ErrAuth; anything
// else, an unreachable Almanach included, is a warning only.
func (e *Emitter) VerifyAtStartup(ctx context.Context) (string, error) {
	if !e.Enabled() {
		return "", nil
	}
	team, unauthorized := e.c.Ping(ctx, pingPath)
	if unauthorized {
		return "", fmt.Errorf("almanach token rejected by %s (HTTP 401); check the team's token: %w", e.c.URL(), ErrAuth)
	}
	return team, nil
}

// EmitNote posts one note and reports whether Almanach took it (2xx). A
// miss is logged by the wire and the note left for the next drain. Never
// an error: a note must not fail a webhook.
func (e *Emitter) EmitNote(ctx context.Context, n Note) bool {
	if !e.Enabled() {
		return false
	}
	if !e.c.Post(ctx, emitPath, payload(n), "note "+n.PRKey) {
		return false
	}
	e.log.InfoContext(ctx, "Almanach: note "+n.PRKey+" taken")
	return true
}

// payload is the wire shape. Money travels twice: as an exact decimal
// string (never a float) and as nano-USD integers, so Almanach can store
// the integer without parsing. A lane the gateway did not send is null.
func payload(n Note) map[string]any {
	claims := make([]map[string]any, 0, len(n.Claims))
	for _, c := range n.Claims {
		claims = append(claims, map[string]any{"project_key": c.ProjectKey, "repo_slug": forward.NullableString(c.RepoSlug)})
	}
	note := n.Note
	if len(note) == 0 {
		note = json.RawMessage("null")
	}
	return map[string]any{
		"event_type":         EventType,
		"project_key":        n.ProjectKey,
		"repo_slug":          n.RepoSlug,
		"pr_id":              n.PRID,
		"pr_key":             n.PRKey,
		"pr_title":           n.PRTitle,
		"ticket_key":         forward.NullableString(n.TicketKey),
		"jira_issue_type":    forward.NullableString(n.JiraIssueType),
		"change_kind":        forward.NullableString(n.ChangeKind),
		"reviewed_commit":    forward.NullableString(n.ReviewedCommit),
		"source_commit_sha":  n.SourceCommitSHA,
		"merge_commit_sha":   forward.NullableString(n.MergeCommitSHA),
		"merged_at":          n.MergedAt.UTC().Format(time.RFC3339),
		"run_id":             n.RunID,
		"model_label":        forward.NullableString(n.ModelLabel),
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
			"models_used":       forward.OrEmpty(n.Review.Models),
		},
	}
}

func callBody(c *Call) any {
	if c == nil {
		return nil
	}
	return map[string]any{
		"outcome":                   c.Outcome,
		"note_status":               c.NoteStatus,
		"profile":                   forward.NullableString(c.Profile),
		"model":                     forward.NullableString(c.Model),
		"deployment":                forward.NullableString(c.Deployment),
		"reasoning_sent":            forward.NullableString(c.ReasoningSent),
		"finish_reason":             forward.NullableString(c.FinishReason),
		"call_id":                   forward.NullableString(c.CallID),
		"litellm_version":           forward.NullableString(c.GatewayVer),
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
		"warnings":                  forward.OrEmpty(c.Warnings),
	}
}

func usd(nano *int64) any {
	if nano == nil {
		return nil
	}
	return forward.FormatNanoUSD(*nano)
}
