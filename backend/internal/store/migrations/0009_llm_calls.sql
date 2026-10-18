-- One row per gateway call, with everything the AI Gateway reported about it:
-- the token lanes, the cost lanes, which model and deployment answered, the
-- call id, the LiteLLM version and the timing. Until now all of that was a
-- log line, so the run detail page had nothing to read.
--
-- A review's call hangs off its attempt (every outcome has one, a run only
-- the successes). attempt_id is nullable and the PR is kept flat because a
-- reply or a mention answer writes no attempt.
--
-- cost_nano_usd here is DETAIL, not a second ledger: the review's cost stays
-- on review_runs / review_attempts, which prCostSQL and spendCTE sum. Never
-- add this column to those totals, or a review is counted twice.
--
-- No discount, margin or original-price column (decided): the dashboard is
-- unauthenticated and the operator's margin is not its business. NULL in a
-- lane means the gateway did not send it, which is not zero.
--
-- Written fail-open, after its attempt and apart from it: a refused call row
-- never costs the attempt (and the billed cost on it) its own write.

CREATE TABLE llm_calls (
    id BIGSERIAL PRIMARY KEY,
    attempt_id BIGINT REFERENCES review_attempts(id) ON DELETE CASCADE,
    team_slug TEXT NOT NULL,
    project_key TEXT NOT NULL,
    repo_slug TEXT NOT NULL,
    pr_id INTEGER NOT NULL,
    kind TEXT NOT NULL CHECK (kind IN ('review')),
    -- The profile label (`<profile>-<effort>`), what was asked for.
    profile TEXT NOT NULL,
    -- What actually answered, and on which deployment. Behind a gateway the
    -- two differ from the profile.
    model TEXT,
    deployment TEXT,
    reasoning_sent TEXT,
    finish_reason TEXT,
    call_id TEXT,
    litellm_version TEXT,
    input_tokens BIGINT,
    input_uncached_tokens BIGINT,
    cache_read_tokens BIGINT,
    cache_write_tokens BIGINT,
    -- output_tokens includes reasoning_tokens; never add the two.
    output_tokens BIGINT,
    reasoning_tokens BIGINT,
    -- NULL = unpriced. Only a Reported cost is priced.
    cost_nano_usd BIGINT,
    -- The gateway's split. input is the uncached part only; input +
    -- cache_read + cache_write + output + tool_usage = the total, and
    -- reasoning is inside output.
    cost_input_nano_usd BIGINT,
    cost_cache_read_nano_usd BIGINT,
    cost_cache_write_nano_usd BIGINT,
    cost_output_nano_usd BIGINT,
    cost_reasoning_nano_usd BIGINT,
    cost_tool_usage_nano_usd BIGINT,
    -- A gauge of the whole key at call time: shown, never summed.
    key_spend_nano_usd BIGINT,
    -- Request to response headers. Never streaming, so this is the model's
    -- whole latency; total adds reading the body.
    headers_ms BIGINT,
    total_ms BIGINT,
    -- llmwire warnings, "<kind> <feature>: <details>".
    warnings TEXT[] NOT NULL DEFAULT '{}',
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- One call per attempt: a review makes a single gateway call.
CREATE UNIQUE INDEX idx_llm_calls_attempt ON llm_calls (attempt_id) WHERE attempt_id IS NOT NULL;
CREATE INDEX idx_llm_calls_pr ON llm_calls (project_key, repo_slug, pr_id);
