-- A PR into the repo's default branch that adds an unreleased Maven/Gradle
-- build (-SNAPSHOT) to a build file gets a Bitbucket task (user,
-- 2026-11-15). Every task noergler posted on the PR, as a JSON array of
-- {id, keys, open}: the comment id, the keys of the lines it names, and
-- whether it was open when last read. A task the author resolved stays
-- listed, so its lines never get another while they stay in the delta.
ALTER TABLE pull_requests ADD COLUMN snapshot_tasks JSONB NOT NULL DEFAULT '[]';

-- Per-team switch, on for every team, existing ones included (user,
-- 2026-11-15).
ALTER TABLE team_settings ADD COLUMN snapshot_task BOOLEAN NOT NULL DEFAULT TRUE;
