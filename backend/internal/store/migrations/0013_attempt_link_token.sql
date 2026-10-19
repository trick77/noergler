-- The run detail page is reached only through the link in the PR's summary
-- comment (decided): /runs/{link_token}, 16 random bytes, base64url, 22
-- characters. The numeric attempt id is enumerable on an unauthenticated
-- dashboard; the token is not, so the page shows the full finding text,
-- which lifts 0011's headlines-only rule.
--
-- NULL for every attempt from before, and for one that posted no summary:
-- no link was ever handed out, so no page.
ALTER TABLE review_attempts ADD COLUMN link_token TEXT UNIQUE;
