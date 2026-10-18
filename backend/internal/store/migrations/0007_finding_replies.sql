-- A reply on a finding's thread is answered once. replied_at is that
-- answer; disputed_at and dispute_text are the author's case against the
-- finding, rendered beside it in later prompts so it is not raised again.
-- The thread stays open: the author resolves it, noergler never does.
ALTER TABLE findings ADD COLUMN replied_at TIMESTAMPTZ;
ALTER TABLE findings ADD COLUMN disputed_at TIMESTAMPTZ;
ALTER TABLE findings ADD COLUMN dispute_text TEXT;
-- The reply job looks a finding up by the comment a reply hangs under; the
-- dedup index does not lead with it.
CREATE INDEX idx_findings_comment ON findings (bitbucket_comment_id);
