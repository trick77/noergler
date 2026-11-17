-- The metrics page counts tasks off review_runs.tasks_posted now (0037,
-- ActivitySince); nothing reads findings by task time any more, and the
-- index only cost every finding insert.
DROP INDEX IF EXISTS idx_findings_task_time;
