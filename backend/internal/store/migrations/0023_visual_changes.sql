-- What the model said about one screenshot pair of a PR, so the summary's
-- "Visual changes" section stays the whole PR's on every run without sending
-- the same images again (user, 2026-10-28). Keyed by the pair's content:
-- a later push that changes the screenshot again is a new pair. The PR row's
-- delete or purge takes these with it.
CREATE TABLE visual_changes (
    pull_request_id BIGINT NOT NULL REFERENCES pull_requests(id) ON DELETE CASCADE,
    path TEXT NOT NULL,
    -- Each version's git blob id from the PR's changes, or "sha256:<hex>" of
    -- the bytes when Bitbucket gives none; '' for the before side of an
    -- added screenshot.
    before_id TEXT NOT NULL,
    after_id TEXT NOT NULL,
    -- '' marks a pair whose pixels did not change: never sent, never listed.
    change_text TEXT NOT NULL,
    -- The llmwire profile id that wrote it.
    model TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (pull_request_id, path, before_id, after_id)
);
