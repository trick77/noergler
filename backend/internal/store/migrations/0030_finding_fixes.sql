-- The PR author can ask noergler to apply a finding itself ("flick das",
-- "fix it"): noergler commits the change to the PR's source branch and
-- confirms it in the thread (user, 2026-11-13). Every such request is one row
-- here, whatever came of it, so the metrics page can count requests against
-- commits and say why the rest did not land.
--
-- A finding is committed once: the partial unique index. A failed attempt
-- (no write permission, the branch moved) may be asked again.
CREATE TABLE finding_fixes (
    id BIGSERIAL PRIMARY KEY,
    finding_id BIGINT NOT NULL REFERENCES findings(id) ON DELETE CASCADE,
    -- The route's team, never a payload value.
    team_slug TEXT NOT NULL,
    outcome TEXT NOT NULL CHECK (outcome IN
        ('committed', 'not_feasible', 'rejected', 'no_write', 'conflict', 'refused', 'error')),
    commit_sha TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    CHECK ((outcome = 'committed') = (commit_sha IS NOT NULL))
);
CREATE UNIQUE INDEX finding_fixes_committed_idx ON finding_fixes (finding_id) WHERE outcome = 'committed';
CREATE INDEX finding_fixes_team_time_idx ON finding_fixes (team_slug, created_at);

-- The fix call is a run-less gateway call like a reply, counted toward the
-- PR's cost.
ALTER TABLE llm_calls DROP CONSTRAINT llm_calls_kind_check;
ALTER TABLE llm_calls ADD CONSTRAINT llm_calls_kind_check
    CHECK (kind IN ('review', 'reply', 'question', 'fix'));

-- noergler's answer on a finding's thread names its finding: the author's
-- "ok, fix it" usually lands under that answer, and Bitbucket sends the
-- immediate parent only.
ALTER TABLE bot_replies ADD COLUMN finding_id BIGINT REFERENCES findings(id) ON DELETE CASCADE;
CREATE INDEX bot_replies_finding_idx ON bot_replies (finding_id) WHERE finding_id IS NOT NULL;
