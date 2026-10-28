-- The Almanach feed: what noergler prepares for Almanach, the living wiki
-- that receives one note per merged PR (like riptide receives one rollup).
--
-- almanach_notes: one row per PR, the model's scratch note of what the PR
-- teaches about its repository. Every review round replaces it (run_id
-- orders the rounds); pr:merged stamps it final with the merge facts and
-- the posted-findings counts; the forwarder marks it emitted after Almanach
-- answered 2xx. A finalize that arrives before any note leaves a stub (note
-- NULL) for the late round to fill. Declined and deleted PRs delete it.
--
-- No FK to pull_requests: a final note must survive PurgeProject (a team
-- releasing the project) until it is emitted. Plain text columns carry the
-- structure Almanach rebuilds the Bitbucket tree from: team, project, repo,
-- PR. team_slug is the slug the webhook route authenticated, never a payload
-- value.
CREATE TABLE almanach_notes (
    project_key TEXT NOT NULL,
    repo_slug TEXT NOT NULL,
    pr_id INTEGER NOT NULL,
    team_slug TEXT NOT NULL,
    -- The review run whose round wrote the note; the ordering guard. NULL on
    -- a stub written by finalize before any round landed. No FK: a purged
    -- run must not take the note with it.
    run_id BIGINT,
    reviewed_commit TEXT,
    pr_title TEXT NOT NULL,
    ticket_key TEXT,
    jira_issue_type TEXT,
    change_kind TEXT,
    -- The validated note object, NULL on a stub. chars is the rune count the
    -- cap was checked on.
    note JSONB,
    chars INTEGER,
    model_label TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    -- Set once by pr:merged.
    finalized_at TIMESTAMPTZ,
    merged_at TIMESTAMPTZ,
    merge_commit TEXT,
    source_commit TEXT,
    issues_posted INTEGER,
    suggestions_posted INTEGER,
    -- Set after Almanach took the note (2xx), never before: Almanach dedupes,
    -- so a duplicate is harmless and a lost note is not.
    almanach_emitted_at TIMESTAMPTZ,
    PRIMARY KEY (project_key, repo_slug, pr_id)
);

-- What the forwarder drains: final, with a note, not yet taken.
CREATE INDEX almanach_notes_pending_idx ON almanach_notes (team_slug, finalized_at)
    WHERE finalized_at IS NOT NULL AND note IS NOT NULL AND almanach_emitted_at IS NULL;

-- One row per note call, the ONLY ledger of its cost. Never llm_calls: its
-- CHECK ((kind = 'review') = (attempt_id IS NOT NULL)) puts every non-review
-- row into prCostSQL, spendCTE, ClaimRollup's EXISTS and PRCommentCalls, and
-- the note call counts toward none of them (decided: not the PR's cost, not
-- the team's dashboard spend). The gateway columns are llm_calls' (0009).
CREATE TABLE almanach_calls (
    id BIGSERIAL PRIMARY KEY,
    team_slug TEXT NOT NULL,
    project_key TEXT NOT NULL,
    repo_slug TEXT NOT NULL,
    pr_id INTEGER NOT NULL,
    -- The review run the call continued. No FK, as above.
    run_id BIGINT,
    -- inference.Outcome's String(): ok, unparseable, too_large, timed_out, error.
    outcome TEXT NOT NULL,
    -- What became of the note: written, superseded (a later round had
    -- already landed), nothing_learned (the model returned an empty note),
    -- dropped_too_long, dropped_secret, dropped_invalid, no_answer (the
    -- gateway never answered).
    note_status TEXT NOT NULL CHECK (note_status IN
        ('written', 'superseded', 'nothing_learned', 'dropped_too_long', 'dropped_secret', 'dropped_invalid', 'no_answer')),
    profile TEXT,
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
    output_tokens BIGINT,
    reasoning_tokens BIGINT,
    cost_nano_usd BIGINT,
    cost_input_nano_usd BIGINT,
    cost_cache_read_nano_usd BIGINT,
    cost_cache_write_nano_usd BIGINT,
    cost_output_nano_usd BIGINT,
    cost_reasoning_nano_usd BIGINT,
    cost_tool_usage_nano_usd BIGINT,
    key_spend_nano_usd BIGINT,
    headers_ms BIGINT,
    total_ms BIGINT,
    warnings TEXT[] NOT NULL DEFAULT '{}',
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX almanach_calls_pr_idx ON almanach_calls (project_key, repo_slug, pr_id, created_at);
