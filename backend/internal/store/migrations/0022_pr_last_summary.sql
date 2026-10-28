-- The last completed review's PR summary prose, fed back to the next review
-- so its overview builds on the earlier one instead of starting over from
-- what fits this run's prompt (user, 2026-10-28). JSONB: only `overview` is
-- carried today; the other summary fields can join without a migration.
-- Written with the run (InsertRun), cleared by Reactivate.
ALTER TABLE pull_requests ADD COLUMN last_summary JSONB;
