-- A finding whose Bitbucket thread was resolved on the PR is settled
-- (user, 2026-10-28): never posted again, out of the PR summary's list and
-- its verdict, still shown to the model as closed so it is not raised
-- again. Stamped like fixed_at, so a review does not read the thread from
-- Bitbucket on every push. Cleared again only by a read that finds the
-- thread reopened (review/fixed.go, closeThreads).
ALTER TABLE findings ADD COLUMN closed_at TIMESTAMPTZ;
