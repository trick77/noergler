"""Team claims and settings

Revision ID: 002
Revises: 001
Create Date: 2026-09-14

What a team changes on its own moves out of teams.yaml: which projects and
repos it owns (`team_claims`) and its review author lists (`team_settings`).
teams.yaml only seeds an empty DB. The unique indexes are the ownership
guarantee: a project or repo is held by exactly one team, whoever wins the
race; the whole-project-vs-repo overlap is checked in code inside a
transaction (`app/team_store.py`).
"""
from typing import Sequence, Union

from alembic import op

revision: str = "002"
down_revision: Union[str, None] = "001"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
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
