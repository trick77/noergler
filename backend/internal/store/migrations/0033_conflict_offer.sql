-- A review that finds a merge conflict noergler could resolve offers it in
-- a comment of its own (user, 2026-11-14). The open offer's comment id, so
-- a later review neither posts it twice nor leaves it open once the offer
-- no longer holds. NULL: no offer open.
ALTER TABLE pull_requests ADD COLUMN conflict_offer_comment_id INTEGER;
