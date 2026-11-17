-- A post is a finding or a blocking task, never both (user, 2026-11-17):
-- findings_posted counted a task too, and the metrics page then showed it
-- as a finding and as a task. findings_posted is now the inline comments
-- only, tasks_posted the tasks the same run posted (one Bitbucket refused
-- went up as a comment and stays a finding).
--
-- Backfill from the run's findings rows, exact: a task still open is
-- 'task', one whose rule left AGENTS.md since is 'rule_removed'.
ALTER TABLE review_runs ADD COLUMN tasks_posted INTEGER NOT NULL DEFAULT 0;

UPDATE review_runs r
   SET tasks_posted = t.n,
       findings_posted = GREATEST(r.findings_posted - t.n, 0)
  FROM (SELECT review_run_id, COUNT(*) AS n
          FROM findings
         WHERE blocking_outcome IN ('task', 'rule_removed')
         GROUP BY review_run_id) t
 WHERE t.review_run_id = r.id;
