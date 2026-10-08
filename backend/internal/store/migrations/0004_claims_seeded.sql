-- Startup reconcile seeds a slug's claims from teams.yaml only when the DB
-- has never held claims for it. "No team_claims rows" could not tell that
-- apart from "removed its last claim via /onboard", so a restart handed the
-- removed projects back. A slug lands here on its first claim and stays,
-- whatever is removed later. Not team_settings: the exclude_repos default
-- gives every team a settings row on first boot, claims or not.
CREATE TABLE team_claims_seeded (
    team_slug TEXT PRIMARY KEY
);

-- Backfill: a slug with claims now, or with PR or attempt rows (a review
-- passed the ownership check, so it held a claim then). A team that removed
-- its last claim before this upgrade has no claim row left, and without the
-- other two its first boot here re-seeded it. Its teams.yaml projects stay a
-- seed only if it never reviewed anything.
INSERT INTO team_claims_seeded (team_slug)
SELECT team_slug FROM team_claims
UNION SELECT team_slug FROM pull_requests
UNION SELECT team_slug FROM review_attempts;
