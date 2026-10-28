-- A finding whose Bitbucket thread was resolved or deleted on the PR is
-- settled (user, 2026-10-28): never posted again, out of the PR summary's
-- list and its verdict, still shown to the model as closed so it is not
-- raised again. Stamped once, like fixed_at, so the thread is not read
-- from Bitbucket on every push; a thread reopened later is not noticed
-- (accepted).
ALTER TABLE findings ADD COLUMN closed_at TIMESTAMPTZ;
