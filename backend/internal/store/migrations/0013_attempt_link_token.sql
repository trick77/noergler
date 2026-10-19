-- The run detail page is reached only by its link token (decided):
-- /runs/{link_token}, 16 random bytes, base64url, 22 characters. The token
-- leaves the server only in the PR summary comment's link: never in an API
-- response, a log or another page. The numeric attempt id was enumerable on
-- the unauthenticated dashboard, the token is not, so the page shows the
-- full finding text, which lifts 0011's headlines-only rule.
--
-- Every attempt has one. The store draws it (store.NewLinkToken); the
-- default, a v4 UUID's random bits in the same alphabet (122 random bits,
-- 22 characters), fills earlier rows (volatile, so one per row) and any
-- insert that names none. No link to those was ever handed out, so they
-- stay unreachable.
ALTER TABLE review_attempts ADD COLUMN link_token TEXT NOT NULL
    DEFAULT rtrim(translate(encode(uuid_send(gen_random_uuid()), 'base64'), '+/', '-_'), '=')
    CONSTRAINT review_attempts_link_token_key UNIQUE;
