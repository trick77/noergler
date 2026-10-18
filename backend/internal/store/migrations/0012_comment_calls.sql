-- A reply to a finding and a mention's Q&A answer are gateway calls too, and
-- now they are recorded, shown and counted toward the PR's cost (decided:
-- until now they were a log line, so a PR could talk its way past its cap).
--
-- Such a call writes no attempt: its row here is its only record, so for
-- these kinds cost_nano_usd IS the ledger, which prCostSQL and spendCTE sum.
-- A review call (kind 'review', attempt_id set) stays detail: its cost is on
-- its run or attempt, and summing it here too would count it twice.
--
-- comment_id is the Bitbucket comment answered, for linking the call to its
-- thread. A Q&A tool loop writes one row per round.

ALTER TABLE llm_calls DROP CONSTRAINT llm_calls_kind_check;
ALTER TABLE llm_calls ADD CONSTRAINT llm_calls_kind_check
    CHECK (kind IN ('review', 'reply', 'question'));
-- The ledger split rests on this: a review call always has its attempt, the
-- others never do.
ALTER TABLE llm_calls ADD CONSTRAINT llm_calls_attempt_kind_check
    CHECK ((kind = 'review') = (attempt_id IS NOT NULL));

ALTER TABLE llm_calls ADD COLUMN comment_id BIGINT;

CREATE INDEX llm_calls_comment_pr_idx ON llm_calls (project_key, repo_slug, pr_id, created_at)
    WHERE attempt_id IS NULL;
