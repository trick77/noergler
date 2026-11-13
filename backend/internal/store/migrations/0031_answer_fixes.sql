-- The PR author can have noergler apply the suggestion of its own Q&A
-- answer, as with a finding (user, 2026-11-13), when the question sat on a
-- line of code.
--
-- noergler's Q&A answer is its own kind, 'answer', apart from notices and
-- confirmations ('question'): only an answer carries a suggestion to apply.
-- It keeps the code line its question was on: Bitbucket anchors only a
-- thread's root and sends a reply's immediate parent only, so the line is
-- resolved when the answer is posted, not when the author asks for the fix.
-- anchor_commit is the commit the line belongs to.
--
-- A reply noergler posts in the fix flow (an answer to "apply it": done,
-- not possible, branch moved, only the PR author) names the answer it is
-- about in source_comment_id and keeps that answer's line, so "apply it"
-- under it applies the answer, not the reply.
ALTER TABLE bot_replies DROP CONSTRAINT bot_replies_kind_check;
ALTER TABLE bot_replies ADD CONSTRAINT bot_replies_kind_check
    CHECK (kind IN ('question', 'reply', 'fixed', 'answer'));
ALTER TABLE bot_replies ADD COLUMN anchor_path TEXT;
ALTER TABLE bot_replies ADD COLUMN anchor_line INT;
ALTER TABLE bot_replies ADD COLUMN anchor_commit TEXT;
ALTER TABLE bot_replies ADD COLUMN source_comment_id BIGINT;
ALTER TABLE bot_replies ADD CONSTRAINT bot_replies_anchor_check
    CHECK ((anchor_path IS NULL) = (anchor_line IS NULL) AND (anchor_path IS NULL) = (anchor_commit IS NULL));

-- A fix request comes from a finding or from such an answer, never both.
-- An answer is committed once, like a finding. Bitbucket comment ids are
-- the instance's, unique across PRs. An answer's row goes with its PR, as a
-- finding's goes with its finding.
ALTER TABLE finding_fixes ALTER COLUMN finding_id DROP NOT NULL;
ALTER TABLE finding_fixes ADD COLUMN answer_comment_id BIGINT;
ALTER TABLE finding_fixes ADD COLUMN pull_request_id BIGINT REFERENCES pull_requests(id) ON DELETE CASCADE;
ALTER TABLE finding_fixes ADD CONSTRAINT finding_fixes_source_check
    CHECK ((finding_id IS NULL) <> (answer_comment_id IS NULL)
       AND (answer_comment_id IS NULL) = (pull_request_id IS NULL));
CREATE UNIQUE INDEX finding_fixes_answer_committed_idx ON finding_fixes (answer_comment_id)
    WHERE outcome = 'committed';
