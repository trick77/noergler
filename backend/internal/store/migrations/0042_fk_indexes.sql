-- Foreign keys Postgres does not index by itself, and one time-window read
-- (the llm_calls index at the end, no foreign key).
--
-- findings.review_run_id and fixed_run_id: the run page lists a run's
-- findings and the ones it fixed by run id (PRRun), and every review_runs
-- delete (PurgeProject's cascade) looks both up. Without an index each is a
-- scan of every finding ever posted.
CREATE INDEX idx_findings_run ON findings (review_run_id);
CREATE INDEX idx_findings_fixed_run ON findings (fixed_run_id) WHERE fixed_run_id IS NOT NULL;

-- finding_fixes cascades from findings and from pull_requests. The partial
-- unique index on finding_id covers committed rows only, so a cascade still
-- scanned the table for every deleted finding or PR.
CREATE INDEX idx_finding_fixes_finding ON finding_fixes (finding_id) WHERE finding_id IS NOT NULL;
CREATE INDEX idx_finding_fixes_pr ON finding_fixes (pull_request_id) WHERE pull_request_id IS NOT NULL;

-- spendCTE's reply and question branch (attempt_id IS NULL, created_at in
-- the window) runs on every metrics request; the per-PR index leads with the
-- PR key and cannot serve a time window.
CREATE INDEX idx_llm_calls_comment_time ON llm_calls (created_at) WHERE attempt_id IS NULL;
