-- The run detail page finds the attempt behind a fixed finding's run, by
-- review_run_id. Without an index that is a scan of review_attempts per
-- finding, on every load of the page.
CREATE INDEX idx_attempts_run ON review_attempts (review_run_id) WHERE review_run_id IS NOT NULL;
