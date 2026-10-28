package store

import (
	"context"
	"errors"
	"fmt"
	"strconv"
	"strings"
	"time"

	"github.com/jackc/pgx/v5"
)

// LLMCall is one gateway call as the AI Gateway described it. See
// migrations/0009_llm_calls.sql and 0012: every pointer is nil when the
// gateway did not send that figure, which is not zero. A review call's
// CostNanoUSD is detail beside the run or attempt that carries the cost; a
// reply's or a question's is the only record of it, and the PR cost sums it.
type LLMCall struct {
	// Profile is the label asked for (`<profile>-<effort>`).
	Profile       string
	Model         string
	Deployment    string
	ReasoningSent string
	FinishReason  string
	CallID        string
	GatewayVer    string

	InputTokens         *int64
	InputUncachedTokens *int64
	CacheReadTokens     *int64
	CacheWriteTokens    *int64
	// OutputTokens includes ReasoningTokens.
	OutputTokens    *int64
	ReasoningTokens *int64

	CostNanoUSD           *int64
	CostInputNanoUSD      *int64
	CostCacheReadNanoUSD  *int64
	CostCacheWriteNanoUSD *int64
	CostOutputNanoUSD     *int64
	CostReasoningNanoUSD  *int64
	CostToolUsageNanoUSD  *int64
	// KeySpendNanoUSD is a gauge of the whole key: shown, never summed.
	KeySpendNanoUSD *int64

	HeadersMS *int64
	TotalMS   *int64
	Warnings  []string
}

// Call kinds. A review call hangs off its attempt; a reply or a question
// writes none, and its row is the ledger.
const (
	CallReview   = "review"
	CallReply    = "reply"
	CallQuestion = "question"
)

// insertLLMCall writes one review call for an attempt that is already
// committed, as a write of its own.
func (s *Store) insertLLMCall(ctx context.Context, attemptID int64, a Attempt, c LLMCall) error {
	return s.writeCall(ctx, &attemptID, a.TeamSlug, a.Key, CallReview, nil, c)
}

// InsertCommentCall records a reply's or a question's gateway call, the only
// record of it: the PR's cost counts it. commentID is the comment answered.
func (s *Store) InsertCommentCall(ctx context.Context, team string, k PRKey, kind string, commentID int64, c LLMCall) error {
	if kind != CallReply && kind != CallQuestion {
		return fmt.Errorf("comment call kind %q", kind)
	}
	return s.writeCall(ctx, nil, team, k, kind, &commentID, c)
}

// writeCall is the one INSERT. The strings are the gateway's own (header
// values, the response's model, llmwire's warning text), so they are made
// storable first: Postgres refuses invalid UTF-8 and NUL bytes, and a
// refused row would lose the whole call.
func (s *Store) writeCall(ctx context.Context, attemptID *int64, team string, k PRKey, kind string, commentID *int64, c LLMCall) error {
	args := append([]any{attemptID, team, k.Project, k.Repo, k.PRID, kind, commentID}, callArgs(c)...)
	_, err := s.pool.Exec(ctx, `
		INSERT INTO llm_calls (attempt_id, team_slug, project_key, repo_slug, pr_id, kind, comment_id, `+callColumns+`)
		VALUES (`+placeholders(len(args))+`)`, args...)
	return err
}

// The gateway's columns, shared by llm_calls and almanach_calls: one list
// for the INSERT, one for the SELECT, one set of arguments and one of scan
// targets, so the next lane is added in one place. callArgs and callScan
// follow callColumns' order exactly.
const callColumns = `profile, model, deployment, reasoning_sent, finish_reason, call_id, litellm_version,
		     input_tokens, input_uncached_tokens, cache_read_tokens, cache_write_tokens,
		     output_tokens, reasoning_tokens,
		     cost_nano_usd, cost_input_nano_usd, cost_cache_read_nano_usd, cost_cache_write_nano_usd,
		     cost_output_nano_usd, cost_reasoning_nano_usd, cost_tool_usage_nano_usd,
		     key_spend_nano_usd, headers_ms, total_ms, warnings`

// callSelect reads callColumns back, the strings COALESCEd for a scan into
// LLMCall. p is the table alias. llm_calls.profile is NOT NULL; a table
// where it is not (almanach_calls, on a LEFT JOIN) reads
// COALESCE(p.profile, ”) and then callSelectTail.
func callSelect(p string) string {
	return p + `.profile, ` + callSelectTail(p)
}

func callSelectTail(p string) string {
	return `COALESCE(` + p + `.model, ''), COALESCE(` + p + `.deployment, ''), COALESCE(` + p + `.reasoning_sent, ''),
		       COALESCE(` + p + `.finish_reason, ''), COALESCE(` + p + `.call_id, ''), COALESCE(` + p + `.litellm_version, ''),
		       ` + p + `.input_tokens, ` + p + `.input_uncached_tokens, ` + p + `.cache_read_tokens, ` + p + `.cache_write_tokens,
		       ` + p + `.output_tokens, ` + p + `.reasoning_tokens,
		       ` + p + `.cost_nano_usd, ` + p + `.cost_input_nano_usd, ` + p + `.cost_cache_read_nano_usd, ` + p + `.cost_cache_write_nano_usd,
		       ` + p + `.cost_output_nano_usd, ` + p + `.cost_reasoning_nano_usd, ` + p + `.cost_tool_usage_nano_usd,
		       ` + p + `.key_spend_nano_usd, ` + p + `.headers_ms, ` + p + `.total_ms, ` + p + `.warnings`
}

// callArgs is one call's values for callColumns. The strings are the
// gateway's own (header values, the response's model, llmwire's warning
// text), so they are made storable first: Postgres refuses invalid UTF-8
// and NUL bytes, and a refused row would lose the whole call. An empty
// string but the profile is NULL.
func callArgs(c LLMCall) []any {
	warnings := make([]string, 0, len(c.Warnings))
	for _, w := range c.Warnings {
		warnings = append(warnings, storable(w))
	}
	return []any{
		storable(c.Profile), nullIfEmpty(c.Model), nullIfEmpty(c.Deployment), nullIfEmpty(c.ReasoningSent),
		nullIfEmpty(c.FinishReason), nullIfEmpty(c.CallID), nullIfEmpty(c.GatewayVer),
		c.InputTokens, c.InputUncachedTokens, c.CacheReadTokens, c.CacheWriteTokens,
		c.OutputTokens, c.ReasoningTokens,
		c.CostNanoUSD, c.CostInputNanoUSD, c.CostCacheReadNanoUSD, c.CostCacheWriteNanoUSD,
		c.CostOutputNanoUSD, c.CostReasoningNanoUSD, c.CostToolUsageNanoUSD,
		c.KeySpendNanoUSD, c.HeadersMS, c.TotalMS, warnings,
	}
}

// callScan is the scan targets for callSelect.
func callScan(c *LLMCall) []any {
	return []any{
		&c.Profile, &c.Model, &c.Deployment, &c.ReasoningSent, &c.FinishReason, &c.CallID, &c.GatewayVer,
		&c.InputTokens, &c.InputUncachedTokens, &c.CacheReadTokens, &c.CacheWriteTokens,
		&c.OutputTokens, &c.ReasoningTokens,
		&c.CostNanoUSD, &c.CostInputNanoUSD, &c.CostCacheReadNanoUSD, &c.CostCacheWriteNanoUSD,
		&c.CostOutputNanoUSD, &c.CostReasoningNanoUSD, &c.CostToolUsageNanoUSD,
		&c.KeySpendNanoUSD, &c.HeadersMS, &c.TotalMS, &c.Warnings,
	}
}

// placeholders is "$1, $2, ..., $n".
func placeholders(n int) string {
	var b strings.Builder
	for i := 1; i <= n; i++ {
		if i > 1 {
			b.WriteString(", ")
		}
		b.WriteString("$")
		b.WriteString(strconv.Itoa(i))
	}
	return b.String()
}

func nullIfEmpty(s string) any {
	if s == "" {
		return nil
	}
	return storable(s)
}

// storable is s as Postgres TEXT accepts it: invalid UTF-8 replaced, NUL
// bytes dropped.
func storable(s string) string {
	return strings.ReplaceAll(strings.ToValidUTF8(s, "�"), "\x00", "")
}

// AttemptCall reads the gateway call recorded under an attempt. ok is false
// when there is no row: the gateway never answered (a pre-flight skip, a
// timeout, a pre-flight too-large), or its call write was refused, which
// was logged at the time. The page says "no call recorded", never "no call
// made".
func (s *Store) AttemptCall(ctx context.Context, attemptID int64) (c LLMCall, ok bool, err error) {
	err = s.pool.QueryRow(ctx, `SELECT `+callSelect("c")+` FROM llm_calls c WHERE c.attempt_id = $1`, attemptID).
		Scan(callScan(&c)...)
	if errors.Is(err, pgx.ErrNoRows) {
		return LLMCall{}, false, nil
	}
	if err != nil {
		return LLMCall{}, false, err
	}
	return c, true, nil
}

// CommentCall is one reply or question call on a PR, for the run page's
// list. Nil cost = unpriced.
type CommentCall struct {
	Kind        string
	CommentID   *int64
	Model       string
	CostNanoUSD *int64
	CreatedAt   time.Time
}

// commentCallsLimit bounds the run page's list; the PR's cost counts them
// all either way.
const commentCallsLimit = 200

// PRCommentCalls lists the PR's reply and question calls, newest first, from
// its current incarnation (the PR row's created_at, as prCostSQL counts).
// more is true when the list was cut.
func (s *Store) PRCommentCalls(ctx context.Context, k PRKey) (calls []CommentCall, more bool, err error) {
	rows, err := s.pool.Query(ctx, `
		SELECT c.kind, c.comment_id, COALESCE(c.model, c.profile), c.cost_nano_usd, c.created_at
		  FROM llm_calls c
		  JOIN pull_requests p
		    ON p.project_key = c.project_key AND p.repo_slug = c.repo_slug AND p.pr_id = c.pr_id
		 WHERE c.project_key = $1 AND c.repo_slug = $2 AND c.pr_id = $3
		   AND c.attempt_id IS NULL AND c.created_at >= p.created_at
		 ORDER BY c.created_at DESC, c.id DESC
		 LIMIT $4`, k.Project, k.Repo, k.PRID, commentCallsLimit+1)
	if err != nil {
		return nil, false, err
	}
	defer rows.Close()
	for rows.Next() {
		var c CommentCall
		if err := rows.Scan(&c.Kind, &c.CommentID, &c.Model, &c.CostNanoUSD, &c.CreatedAt); err != nil {
			return nil, false, err
		}
		calls = append(calls, c)
	}
	if err := rows.Err(); err != nil {
		return nil, false, err
	}
	if len(calls) > commentCallsLimit {
		return calls[:commentCallsLimit], true, nil
	}
	return calls, false, nil
}
