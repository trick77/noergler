-- A later review that finds an earlier finding fixed retires it: the thread
-- gets a reply and is resolved, and the row leaves the previously-posted
-- block and dedup, so a regression is posted again.
ALTER TABLE findings ADD COLUMN fixed_at TIMESTAMPTZ;
ALTER TABLE findings ADD COLUMN fixed_run_id BIGINT REFERENCES review_runs(id) ON DELETE SET NULL;
