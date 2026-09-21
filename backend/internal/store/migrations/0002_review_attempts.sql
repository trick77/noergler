-- review_runs holds SUCCESSES ONLY. Every failure (timed_out, unparseable,
-- too_large, error) and every pre-flight skip writes no row there and lives
-- only as log text, so nothing can report what the pipeline decided not to do.
--
-- review_attempts is that record: one row per terminal outcome, including the
-- ones that produce no run. It is deliberately a SEPARATE table rather than a
-- status column on review_runs, because "a non-ok outcome writes no run row"
-- is pinned behaviour (TestTerminalOutcomes) and stays true.
--
-- The write fails open: an insert error here is logged and swallowed, never
-- returned. A dashboard row must not turn a successful review into an error or
-- block the commit pointer. Same posture as cost.

CREATE TABLE review_attempts (
    id BIGSERIAL PRIMARY KEY,
    -- The slug the webhook route authenticated, never a payload value. Kept
    -- flat rather than joined through pull_requests: a skip can be decided
    -- before any PR row exists.
    team_slug TEXT NOT NULL,
    project_key TEXT NOT NULL,
    repo_slug TEXT NOT NULL,
    pr_id INTEGER NOT NULL,
    kind TEXT NOT NULL CHECK (kind IN ('auto', 'mention')),
    -- inference.Outcome's String(), plus 'skipped' for the pre-flight exits
    -- that never reach inference at all.
    outcome TEXT NOT NULL CHECK (outcome IN (
        'ok', 'timed_out', 'unparseable', 'too_large', 'error', 'skipped'
    )),
    -- review.SkipReason. NULL unless outcome = 'skipped'.
    reason TEXT,
    elapsed_ms BIGINT,
    -- Set only when the attempt produced a run. ON DELETE SET NULL, not
    -- CASCADE: deleting a PR's runs must not erase the record that it was
    -- attempted.
    review_run_id BIGINT REFERENCES review_runs(id) ON DELETE SET NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- The feed reads newest-first across every team; the per-team pages filter
-- first. Two indexes, because a single (team, created_at) one cannot serve
-- the unfiltered feed's ordering.
CREATE INDEX idx_attempts_recent ON review_attempts (created_at DESC);
CREATE INDEX idx_attempts_team ON review_attempts (team_slug, created_at DESC);

-- review_runs carries only idx_review_runs_pr (pull_request_id), so every
-- time-windowed dashboard aggregate would seq-scan the table.
CREATE INDEX idx_review_runs_created ON review_runs (created_at DESC);
