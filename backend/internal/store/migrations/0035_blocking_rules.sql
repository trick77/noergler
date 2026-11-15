-- Blocking rules (user, 2026-11-15): a finding the model says violates a
-- rule AGENTS.md marks [blocking] is posted as a Bitbucket task, or
-- downgraded to a plain comment by code. blocking_rule is the rule's text
-- (never its B<n> id, which shifts when AGENTS.md changes), NULL when the
-- model named no rule or one this review did not have; blocking_outcome is
-- what became of the claim, NULL when there was none, rule_removed once
-- noergler resolved the task because its rule left AGENTS.md. Kept per finding so
-- the confidence threshold can be tuned on real data later (user).
ALTER TABLE findings ADD COLUMN blocking_rule TEXT;
ALTER TABLE findings ADD COLUMN blocking_outcome TEXT
    CHECK (blocking_outcome IN ('task', 'downgraded_unknown_rule', 'downgraded_severity',
        'downgraded_confidence', 'downgraded_cap', 'downgraded_tasks_off', 'downgraded_task_failed',
        'rule_removed'));

-- The team switch covers every task noergler opens now, the SNAPSHOT task
-- and a blocking rule's (user, 2026-11-15: renamed, no alias). Its value
-- and default carry over.
ALTER TABLE team_settings RENAME COLUMN snapshot_task TO create_tasks;
