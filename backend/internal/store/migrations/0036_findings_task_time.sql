-- The metrics page counts posted blocking tasks per day over its window
-- (TaskCountsSince). findings only grows; without this every page load
-- would scan all of it. Partial: tasks are a small share of findings.
CREATE INDEX idx_findings_task_time ON findings (created_at)
    WHERE blocking_outcome IN ('task', 'rule_removed');
