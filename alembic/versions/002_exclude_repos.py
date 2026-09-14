"""team_settings.exclude_repos

Revision ID: 002
Revises: 001
Create Date: 2026-09-14

Repo slug patterns a team wants left alone although its project webhook
delivers their events. Default for existing and new rows: every *-infra repo.
"""
from typing import Sequence, Union

from alembic import op

revision: str = "002"
down_revision: Union[str, None] = "001"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute("""ALTER TABLE team_settings ADD COLUMN exclude_repos TEXT[] NOT NULL DEFAULT '{"*-infra"}'""")


def downgrade() -> None:
    op.execute("ALTER TABLE team_settings DROP COLUMN exclude_repos")
