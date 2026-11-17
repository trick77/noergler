-- Acts noergler counts that have no table of their own (user, 2026-11-17):
-- a SNAPSHOT task it posted (snapshot_tasks keeps no time and drops a task
-- once its lines leave the delta) and an engagement, a comment that tags
-- noergler or replies straight to one of its comments, by what came of it.
-- One row per Bitbucket comment: comment_id is the task's for a SNAPSHOT
-- task, the person's comment for an engagement, so a replayed event or a
-- task taken over after a lost write never counts twice.
--
-- A fix or merge-conflict request is a finding_fixes row and a review asked
-- by mention a review_runs row, never an engagement here: each act counts
-- under one kind. No backfill (user): neither kind kept a time before.
CREATE TABLE activity_events (
    pull_request_id BIGINT NOT NULL REFERENCES pull_requests(id) ON DELETE CASCADE,
    kind TEXT NOT NULL CHECK (kind IN ('snapshot_task', 'engagement')),
    comment_id BIGINT NOT NULL,
    -- Engagements only: tagged ('mention') or a reply to noergler's comment
    -- ('reply'), and answered, refused to anyone but the PR's author
    -- ('author_only'), refused at the cost cap ('refused') or failed.
    trigger TEXT CHECK (trigger IN ('mention', 'reply')),
    outcome TEXT CHECK (outcome IN ('answered', 'author_only', 'refused', 'failed')),
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (pull_request_id, kind, comment_id),
    CHECK ((kind = 'engagement') = (trigger IS NOT NULL)),
    CHECK ((kind = 'engagement') = (outcome IS NOT NULL))
);

-- The metrics window and the runs feed read by time.
CREATE INDEX idx_activity_events_time ON activity_events (created_at);
