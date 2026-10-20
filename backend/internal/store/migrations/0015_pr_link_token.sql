-- The summary links the PR's page, not one run's (decided): /prs/{link_token},
-- and each run under it at /prs/{link_token}/runs/{n}. Same format and rules
-- as 0013's attempt token: base64url, 22 characters, handed out only in the
-- PR summary comment's link, never in an API response or a log. 0013's
-- attempt token is no longer read: its /runs/{token} links 404 (decided).
--
-- The store draws it on insert (store.NewLinkToken, 16 random bytes). The
-- default, as in 0013 a v4 UUID's random bits (122, not 128), fills the
-- existing rows (volatile, so one per row) and any insert that names none.
-- A row deleted with its team's claim takes its token with it.
ALTER TABLE pull_requests ADD COLUMN link_token TEXT NOT NULL
    DEFAULT rtrim(translate(encode(uuid_send(gen_random_uuid()), 'base64'), '+/', '-_'), '=')
    CONSTRAINT pull_requests_link_token_key UNIQUE;
