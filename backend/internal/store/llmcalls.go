package store

import (
	"context"
	"errors"

	"github.com/jackc/pgx/v5"
)

// LLMCall is one gateway call as the AI Gateway described it. See
// migrations/0009_llm_calls.sql: every pointer is nil when the gateway did
// not send that figure, which is not zero, and CostNanoUSD is detail, never
// a second ledger beside the run or attempt that carries the review's cost.
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

// insertLLMCall writes one review call under its attempt, inside the
// attempt's transaction.
func insertLLMCall(ctx context.Context, tx pgx.Tx, attemptID int64, a Attempt, c LLMCall) error {
	warnings := c.Warnings
	if warnings == nil {
		warnings = []string{}
	}
	_, err := tx.Exec(ctx, `
		INSERT INTO llm_calls
		    (attempt_id, team_slug, project_key, repo_slug, pr_id, kind,
		     profile, model, deployment, reasoning_sent, finish_reason, call_id, litellm_version,
		     input_tokens, input_uncached_tokens, cache_read_tokens, cache_write_tokens,
		     output_tokens, reasoning_tokens,
		     cost_nano_usd, cost_input_nano_usd, cost_cache_read_nano_usd, cost_cache_write_nano_usd,
		     cost_output_nano_usd, cost_reasoning_nano_usd, cost_tool_usage_nano_usd,
		     key_spend_nano_usd, headers_ms, total_ms, warnings)
		VALUES ($1, $2, $3, $4, $5, 'review',
		        $6, NULLIF($7, ''), NULLIF($8, ''), NULLIF($9, ''), NULLIF($10, ''), NULLIF($11, ''), NULLIF($12, ''),
		        $13, $14, $15, $16, $17, $18,
		        $19, $20, $21, $22, $23, $24, $25,
		        $26, $27, $28, $29)`,
		attemptID, a.TeamSlug, a.Key.Project, a.Key.Repo, a.Key.PRID,
		c.Profile, c.Model, c.Deployment, c.ReasoningSent, c.FinishReason, c.CallID, c.GatewayVer,
		c.InputTokens, c.InputUncachedTokens, c.CacheReadTokens, c.CacheWriteTokens,
		c.OutputTokens, c.ReasoningTokens,
		c.CostNanoUSD, c.CostInputNanoUSD, c.CostCacheReadNanoUSD, c.CostCacheWriteNanoUSD,
		c.CostOutputNanoUSD, c.CostReasoningNanoUSD, c.CostToolUsageNanoUSD,
		c.KeySpendNanoUSD, c.HeadersMS, c.TotalMS, warnings)
	return err
}

// AttemptCall reads the gateway call recorded under an attempt. ok is false
// when the attempt made none: a skip, a timeout, a pre-flight too-large.
func (s *Store) AttemptCall(ctx context.Context, attemptID int64) (c LLMCall, ok bool, err error) {
	err = s.pool.QueryRow(ctx, `
		SELECT profile, COALESCE(model, ''), COALESCE(deployment, ''), COALESCE(reasoning_sent, ''),
		       COALESCE(finish_reason, ''), COALESCE(call_id, ''), COALESCE(litellm_version, ''),
		       input_tokens, input_uncached_tokens, cache_read_tokens, cache_write_tokens,
		       output_tokens, reasoning_tokens,
		       cost_nano_usd, cost_input_nano_usd, cost_cache_read_nano_usd, cost_cache_write_nano_usd,
		       cost_output_nano_usd, cost_reasoning_nano_usd, cost_tool_usage_nano_usd,
		       key_spend_nano_usd, headers_ms, total_ms, warnings
		  FROM llm_calls
		 WHERE attempt_id = $1
		 ORDER BY id
		 LIMIT 1`, attemptID).Scan(
		&c.Profile, &c.Model, &c.Deployment, &c.ReasoningSent, &c.FinishReason, &c.CallID, &c.GatewayVer,
		&c.InputTokens, &c.InputUncachedTokens, &c.CacheReadTokens, &c.CacheWriteTokens,
		&c.OutputTokens, &c.ReasoningTokens,
		&c.CostNanoUSD, &c.CostInputNanoUSD, &c.CostCacheReadNanoUSD, &c.CostCacheWriteNanoUSD,
		&c.CostOutputNanoUSD, &c.CostReasoningNanoUSD, &c.CostToolUsageNanoUSD,
		&c.KeySpendNanoUSD, &c.HeadersMS, &c.TotalMS, &c.Warnings)
	if errors.Is(err, pgx.ErrNoRows) {
		return LLMCall{}, false, nil
	}
	if err != nil {
		return LLMCall{}, false, err
	}
	return c, true, nil
}
