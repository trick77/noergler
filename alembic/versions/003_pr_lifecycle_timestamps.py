"""Add PR lifecycle timestamps to pr_reviews

Revision ID: 003
Revises: 002
Create Date: 2026-04-24

"""
from typing import Sequence, Union

from alembic import op

revision: str = "003"
down_revision: Union[str, None] = "002"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute("ALTER TABLE pr_reviews ADD COLUMN opened_at TIMESTAMPTZ")
    op.execute("ALTER TABLE pr_reviews ADD COLUMN merged_at TIMESTAMPTZ")
    op.execute("ALTER TABLE pr_reviews ADD COLUMN deleted_at TIMESTAMPTZ")
    op.execute("CREATE INDEX idx_pr_reviews_lifecycle ON pr_reviews (merged_at, deleted_at)")


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS idx_pr_reviews_lifecycle")
    op.execute("ALTER TABLE pr_reviews DROP COLUMN IF EXISTS deleted_at")
    op.execute("ALTER TABLE pr_reviews DROP COLUMN IF EXISTS merged_at")
    op.execute("ALTER TABLE pr_reviews DROP COLUMN IF EXISTS opened_at")
