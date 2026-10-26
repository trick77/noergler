-- A team's own Bitbucket and Jira accounts, pushed through the team API
-- (PUT /teams/{team}/credentials). No row = the shared account from the
-- instance env. The token is AES-GCM ciphertext under NOERGLER_SECRETS_KEY
-- with "<slug>|<kind>" as associated data, so a row copied to another team
-- does not decrypt. The username is not secret: it is the team bot's
-- @mention handle.
CREATE TABLE team_credentials (
    team_slug TEXT NOT NULL,
    kind TEXT NOT NULL CHECK (kind IN ('bitbucket', 'jira')),
    username TEXT,
    nonce BYTEA NOT NULL,
    ciphertext BYTEA NOT NULL,
    updated_by TEXT,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (team_slug, kind)
);

-- One row: a fixed plaintext sealed under NOERGLER_SECRETS_KEY. A wrong key
-- fails here, once, as the shared fault it is, instead of as every team's
-- row failing to decrypt.
CREATE TABLE secrets_key_check (
    id BOOLEAN PRIMARY KEY DEFAULT TRUE CHECK (id),
    nonce BYTEA NOT NULL,
    ciphertext BYTEA NOT NULL
);
