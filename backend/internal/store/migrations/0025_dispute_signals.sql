-- A dispute signal is one project-wide rule PR authors keep giving as the
-- reason a finding is wrong ("validation lives in the UI"). Each dispute
-- that names such a rule is an occurrence of a signal; the finding-thread
-- answer counts the distinct PRs in the team's window and, at the
-- threshold, proposes the AGENTS.md bullet (user, 2026-10-31).
--
-- Read on the finding-reply path only, never by a review prompt. An
-- occurrence goes with its finding and so with its PR; a signal left without
-- one in the window is dormant, never deleted. Expiry is a WHERE, no job.
CREATE TABLE dispute_signals (
    id BIGSERIAL PRIMARY KEY,
    -- The route's team, never a payload value.
    team_slug TEXT NOT NULL,
    project_key TEXT NOT NULL,
    repo_slug TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX idx_dispute_signals_repo ON dispute_signals (team_slug, project_key, repo_slug);

CREATE TABLE dispute_occurrences (
    signal_id BIGINT NOT NULL REFERENCES dispute_signals(id) ON DELETE CASCADE,
    finding_id BIGINT NOT NULL REFERENCES findings(id) ON DELETE CASCADE,
    -- The checked AGENTS.md bullet the model wrote for this dispute.
    rule TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (signal_id, finding_id)
);
CREATE INDEX idx_dispute_occurrences_time ON dispute_occurrences (signal_id, created_at);
CREATE INDEX idx_dispute_occurrences_finding ON dispute_occurrences (finding_id);
