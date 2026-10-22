-- Every reply noergler posts, by its Bitbucket comment id: a Q&A answer,
-- a finding-thread answer, a "Fixed in" reply, a language confirmation.
-- A plain reply directly under one of them is addressed to the bot and is
-- answered like a reply under the summary (decided, 2026-10-22). Until now
-- the answer's id was thrown away, so such a reply matched neither the
-- summary nor a finding and was dropped.
--
-- Matched from the DB like the summary and the findings, never by fetching
-- the parent: a fetch per reply was rejected (review/reply.go). Rows go
-- with their PR, as findings do.
CREATE TABLE bot_replies (
    pull_request_id BIGINT NOT NULL REFERENCES pull_requests(id) ON DELETE CASCADE,
    comment_id BIGINT NOT NULL,
    kind TEXT NOT NULL CHECK (kind IN ('question', 'reply', 'fixed')),
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (pull_request_id, comment_id)
);
