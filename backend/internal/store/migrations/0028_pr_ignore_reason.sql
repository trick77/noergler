-- Why a PR is ignored: its summary comment was deleted, or someone told
-- noergler to stop reviewing it (user, 2026-11-04). Resuming a stopped PR
-- keeps its summary and overview; a deleted summary is forgotten. NULL
-- while ignored_at is NULL. stopped_by is the commenter's Bitbucket name,
-- shown on the summary's banner.
ALTER TABLE pull_requests ADD COLUMN ignore_reason TEXT
    CHECK (ignore_reason IN ('summary_deleted', 'stopped'));
ALTER TABLE pull_requests ADD COLUMN stopped_by TEXT;
UPDATE pull_requests SET ignore_reason = 'summary_deleted' WHERE ignored_at IS NOT NULL;
