-- The runs feed reads the latest fix requests of every team, newest first,
-- on every poll (RecentAttempts); finding_fixes_team_time_idx leads with
-- the team, so without this the branch sorts every request ever made.
CREATE INDEX idx_finding_fixes_time ON finding_fixes (created_at);
