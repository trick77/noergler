"""Initial schema

Revision ID: 001
Revises:
Create Date: 2026-09-11

`team_slug` is the team the webhook route authenticated (the per-team path
and HMAC secret), never a value read from the payload. It is NOT NULL: every
row is written by a team, and riptide rollups are per team.

`team_claims` and `team_settings` hold what a team changes on its own: which
projects and repos it owns and its review author lists. The unique indexes on
`team_claims` are the ownership guarantee: a project or repo is held by
exactly one team; the whole-project-vs-repo overlap is checked in code inside
a transaction (`app/team_store.py`).
"""
from typing import Sequence, Union

from alembic import op

revision: str = "001"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute("""
        CREATE TABLE pr_reviews (
            id BIGSERIAL PRIMARY KEY,
            project_key TEXT NOT NULL,
            repo_slug TEXT NOT NULL,
            pr_id INTEGER NOT NULL,
            team_slug TEXT NOT NULL,
            last_reviewed_commit TEXT,
            summary_comment_id INTEGER,
            summary_comment_version INTEGER,
            author TEXT,
            pr_title TEXT,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            -- lifecycle
            opened_at TIMESTAMPTZ,
            merged_at TIMESTAMPTZ,
            declined_at TIMESTAMPTZ,
            deleted_at TIMESTAMPTZ,
            ignored_at TIMESTAMPTZ,
            -- cost cap + frozen final figure
            total_cost_usd NUMERIC(10,6),
            final_cost_usd NUMERIC(10,6),
            -- per-PR rollup accumulators shipped to riptide at close
            total_prompt_tokens BIGINT NOT NULL DEFAULT 0,
            total_completion_tokens BIGINT NOT NULL DEFAULT 0,
            total_elapsed_ms BIGINT NOT NULL DEFAULT 0,
            total_findings_count INTEGER NOT NULL DEFAULT 0,
            total_runs INTEGER NOT NULL DEFAULT 0,
            models_used TEXT[] NOT NULL DEFAULT '{}',
            first_review_at TIMESTAMPTZ,
            final_source_commit_sha TEXT,
            final_merge_commit_sha TEXT,
            final_lines_added INTEGER,
            final_lines_removed INTEGER,
            final_files_changed INTEGER,
            riptide_emitted_at TIMESTAMPTZ,
            UNIQUE (project_key, repo_slug, pr_id)
        )
    """)

    op.execute("CREATE INDEX idx_pr_reviews_lifecycle ON pr_reviews (merged_at, deleted_at)")
    op.execute("CREATE INDEX idx_pr_reviews_team ON pr_reviews (team_slug)")

    op.execute("""
        CREATE TABLE review_findings (
            id BIGSERIAL PRIMARY KEY,
            pr_review_id BIGINT NOT NULL REFERENCES pr_reviews(id) ON DELETE CASCADE,
            file_path TEXT,
            line_number INTEGER,
            severity TEXT,
            comment_text TEXT,
            suggestion TEXT,
            bitbucket_comment_id INTEGER,
            commit_sha TEXT,
            is_incremental BOOLEAN DEFAULT FALSE,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
        )
    """)

    op.execute("""
        CREATE INDEX idx_review_findings_dedup
        ON review_findings (pr_review_id, file_path, line_number, severity)
    """)


    op.execute("""
        CREATE TABLE team_claims (
            id BIGSERIAL PRIMARY KEY,
            team_slug TEXT NOT NULL,
            project_key TEXT NOT NULL,
            -- NULL: the whole project (one project webhook)
            repo_slug TEXT,
            claimed_by TEXT NOT NULL,
            claimed_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
        )
    """)
    op.execute(
        "CREATE UNIQUE INDEX uq_team_claims_repo ON team_claims (project_key, repo_slug) "
        "WHERE repo_slug IS NOT NULL"
    )
    op.execute(
        "CREATE UNIQUE INDEX uq_team_claims_project ON team_claims (project_key) "
        "WHERE repo_slug IS NULL"
    )
    op.execute("CREATE INDEX idx_team_claims_team ON team_claims (team_slug)")
    op.execute("""
        CREATE TABLE team_settings (
            team_slug TEXT PRIMARY KEY,
            auto_review_authors TEXT[] NOT NULL DEFAULT '{}',
            ignore_authors TEXT[] NOT NULL DEFAULT '{}',
            updated_by TEXT,
            updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
        )
    """)



def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS team_settings")
    op.execute("DROP TABLE IF EXISTS team_claims")
    op.execute("DROP TABLE IF EXISTS review_findings")
    op.execute("DROP TABLE IF EXISTS pr_reviews")
