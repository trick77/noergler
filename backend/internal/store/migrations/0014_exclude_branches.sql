-- Per-team branch-name globs: a PR whose source or target branch matches is
-- never auto-reviewed (an @mention still is). Empty for every existing row
-- (decided): no team gets an exclude it did not set.
ALTER TABLE team_settings ADD COLUMN exclude_source_branches TEXT[] NOT NULL DEFAULT '{}';
ALTER TABLE team_settings ADD COLUMN exclude_target_branches TEXT[] NOT NULL DEFAULT '{}';
