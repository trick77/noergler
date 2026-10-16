-- Per-team switch for feeding the PR title and description to the model.
-- Default on: an existing row keeps reviewing with it, as a new one does.
ALTER TABLE team_settings ADD COLUMN use_pr_description BOOLEAN NOT NULL DEFAULT TRUE;
