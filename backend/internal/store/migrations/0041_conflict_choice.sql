-- A merge-conflict request that left conflicts to the PR author (user,
-- 2026-11-18): two whole versions of the same lines, neither carrying the
-- other's change. noergler asks "T or S?" and keeps, until the answer,
-- the question's comment, the two heads it merged and every conflict with
-- its resolution, so the answer settles the merge with no second call.
-- NULL when no question is open.
ALTER TABLE pull_requests ADD COLUMN conflict_choice JSONB;
