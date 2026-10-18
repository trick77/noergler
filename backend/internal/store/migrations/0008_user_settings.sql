-- A Bitbucket user's own choices, set from a PR comment. Instance-wide, not
-- per team: the language is the person's, whichever team reviews them.
-- Keyed by the lower-cased user name (the slug noergler matches authors on);
-- never the display name, which is not stored anywhere.
CREATE TABLE user_settings (
    user_name TEXT PRIMARY KEY,
    language TEXT NOT NULL CHECK (language IN ('de', 'en')),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
