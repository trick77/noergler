-- Redesigned around runs, not accumulators: pull_requests carries identity and
-- lifecycle, review_runs one row per completed run, findings per run. Totals
-- for the cost cap, the summary and the riptide rollup are aggregates over
-- review_runs. Cost is BIGINT nano-USD (llmwire's unit), NULL = unpriced.

CREATE TABLE pull_requests (
    id BIGSERIAL PRIMARY KEY,
    project_key TEXT NOT NULL,
    repo_slug TEXT NOT NULL,
    pr_id INTEGER NOT NULL,
    -- The slug the webhook route authenticated, never a payload value.
    team_slug TEXT NOT NULL,
    author TEXT,
    title TEXT,
    last_reviewed_commit TEXT,
    summary_comment_id INTEGER,
    summary_comment_version INTEGER,
    opened_at TIMESTAMPTZ,
    merged_at TIMESTAMPTZ,
    declined_at TIMESTAMPTZ,
    deleted_at TIMESTAMPTZ,
    -- Set when the summary comment was removed by a human: the PR is skipped
    -- until an @mention reactivates it.
    ignored_at TIMESTAMPTZ,
    final_cost_nano_usd BIGINT,
    final_source_commit TEXT,
    final_merge_commit TEXT,
    final_lines_added INTEGER,
    final_lines_removed INTEGER,
    final_files_changed INTEGER,
    -- Stamped by the same statement that reads the rollup snapshot, so a
    -- failed emission is never retried.
    riptide_emitted_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (project_key, repo_slug, pr_id)
);
CREATE INDEX idx_pull_requests_team ON pull_requests (team_slug);
CREATE INDEX idx_pull_requests_lifecycle ON pull_requests (merged_at, deleted_at);

CREATE TABLE review_runs (
    id BIGSERIAL PRIMARY KEY,
    pull_request_id BIGINT NOT NULL REFERENCES pull_requests(id) ON DELETE CASCADE,
    kind TEXT NOT NULL CHECK (kind IN ('auto', 'mention')),
    incremental BOOLEAN NOT NULL,
    from_commit TEXT,
    to_commit TEXT NOT NULL,
    model_label TEXT NOT NULL,
    prompt_tokens BIGINT NOT NULL,
    cached_tokens BIGINT NOT NULL,
    completion_tokens BIGINT NOT NULL,
    -- NULL = the gateway reported no cost; the run counts, its price does not.
    cost_nano_usd BIGINT,
    elapsed_ms BIGINT NOT NULL,
    findings_posted INTEGER NOT NULL,
    lines_added INTEGER NOT NULL,
    lines_removed INTEGER NOT NULL,
    files_changed INTEGER NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX idx_review_runs_pr ON review_runs (pull_request_id);

CREATE TABLE findings (
    id BIGSERIAL PRIMARY KEY,
    pull_request_id BIGINT NOT NULL REFERENCES pull_requests(id) ON DELETE CASCADE,
    review_run_id BIGINT NOT NULL REFERENCES review_runs(id) ON DELETE CASCADE,
    file_path TEXT NOT NULL,
    line_number INTEGER NOT NULL,
    severity TEXT NOT NULL,
    confidence INTEGER,
    headline TEXT,
    comment_text TEXT NOT NULL,
    suggestion TEXT,
    bitbucket_comment_id INTEGER,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX idx_findings_dedup ON findings (pull_request_id, file_path, line_number, severity);

-- What a team changes on its own. teams.yaml only seeds a slug the DB does
-- not know. A project or a repo belongs to exactly one team: the two partial
-- unique indexes hold that, the one overlap they cannot express (a whole
-- project against another team's repo claims) is checked in a transaction.
CREATE TABLE team_claims (
    id BIGSERIAL PRIMARY KEY,
    team_slug TEXT NOT NULL,
    project_key TEXT NOT NULL,
    -- NULL: the whole project (one project webhook)
    repo_slug TEXT,
    claimed_by TEXT NOT NULL,
    claimed_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE UNIQUE INDEX uq_team_claims_repo ON team_claims (project_key, repo_slug) WHERE repo_slug IS NOT NULL;
CREATE UNIQUE INDEX uq_team_claims_project ON team_claims (project_key) WHERE repo_slug IS NULL;
CREATE INDEX idx_team_claims_team ON team_claims (team_slug);

CREATE TABLE team_settings (
    team_slug TEXT PRIMARY KEY,
    auto_review_authors TEXT[] NOT NULL DEFAULT '{}',
    ignore_authors TEXT[] NOT NULL DEFAULT '{}',
    exclude_repos TEXT[] NOT NULL DEFAULT '{"*-infra"}',
    updated_by TEXT,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
