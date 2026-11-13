-- The PR author can have noergler resolve a simple merge conflict (user,
-- 2026-11-13): a merge commit of the target into the source branch. It
-- counts under auto-fixes on the metrics page, so each request is a
-- finding_fixes row, sourced from the author's comment that asked.
--
-- Not unique when committed: a later conflict on the same PR is a new
-- request on a new comment.
ALTER TABLE finding_fixes ADD COLUMN conflict_comment_id BIGINT;
ALTER TABLE finding_fixes DROP CONSTRAINT finding_fixes_source_check;
ALTER TABLE finding_fixes ADD CONSTRAINT finding_fixes_source_check
    CHECK (num_nonnulls(finding_id, answer_comment_id, conflict_comment_id) = 1
       AND (finding_id IS NULL) = (pull_request_id IS NOT NULL));
