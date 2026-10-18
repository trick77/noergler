-- What the run detail page shows beyond the run and call rows: stage
-- timings, the prompt's parts, how each file was sent, the findings that
-- were not posted and why, the verdict before and after adjustment, the
-- HTTP call counts. Display-only, one JSON record per attempt, written
-- fail-open after the summary is posted.
--
-- Headlines only (decided): no finding comment, suggestion or quoted code,
-- because the dashboard is unauthenticated. Nothing here is summed or
-- billed; money stays on review_runs, review_attempts and llm_calls.
CREATE TABLE run_details (
    attempt_id BIGINT PRIMARY KEY REFERENCES review_attempts(id) ON DELETE CASCADE,
    detail JSONB NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
