"""Ignore a PR once its summary comment has been removed

Revision ID: 008
Revises: 007
Create Date: 2026-06-19

Adds a single lifecycle timestamp to pr_reviews:
    ignored_at  — set when the user deleted noergler's summary comment. While
                  this is non-NULL, the review path skips the PR entirely (no
                  review, no re-posting). An explicit @mention clears it again.

Mirrors the existing merged_at / declined_at / deleted_at lifecycle columns.
No index needed — lookups go through the existing unique key
(project_key, repo_slug, pr_id).
"""
from typing import Sequence, Union

from alembic import op

revision: str = "008"
down_revision: Union[str, None] = "007"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute("ALTER TABLE pr_reviews ADD COLUMN ignored_at TIMESTAMPTZ")


def downgrade() -> None:
    op.execute("ALTER TABLE pr_reviews DROP COLUMN IF EXISTS ignored_at")
