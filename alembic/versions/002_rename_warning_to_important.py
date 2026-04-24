"""Rename review_statistics.warning_count to important_count

Revision ID: 002
Revises: 001
Create Date: 2026-04-24

"""
from typing import Sequence, Union

from alembic import op

revision: str = "002"
down_revision: Union[str, None] = "001"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute("ALTER TABLE review_statistics RENAME COLUMN warning_count TO important_count")


def downgrade() -> None:
    op.execute("ALTER TABLE review_statistics RENAME COLUMN important_count TO warning_count")
