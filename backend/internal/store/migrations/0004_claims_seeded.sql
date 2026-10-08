-- Startup reconcile seeds a slug's claims from teams.yaml only when the DB
-- has never held claims for it. "No team_claims rows" could not tell that
-- apart from "removed its last claim via /onboard", so a restart handed the
-- removed projects back. A slug lands here on its first claim and stays,
-- whatever is removed later. Not team_settings: the exclude_repos default
-- gives every team a settings row on first boot, claims or not.
CREATE TABLE team_claims_seeded (
    team_slug TEXT PRIMARY KEY
);

INSERT INTO team_claims_seeded (team_slug)
SELECT DISTINCT team_slug FROM team_claims;
