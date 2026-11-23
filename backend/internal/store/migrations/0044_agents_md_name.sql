-- The file a team keeps its review rules under, root and subdirectories
-- alike: AGENTS.md unless the team chose another, e.g. REVIEW.md (user,
-- 2026-11-23). A plain file name, never a path: the same name is looked up
-- in every directory.
ALTER TABLE team_settings ADD COLUMN agents_md_name TEXT NOT NULL DEFAULT 'AGENTS.md'
    CHECK (agents_md_name <> '' AND position('/' in agents_md_name) = 0);
