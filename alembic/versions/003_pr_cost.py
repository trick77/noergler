"""Per-PR LLM cost tracking in USD

Revision ID: 003
Revises: 002
Create Date: 2026-04-28

Adds two columns to pr_reviews:
    total_cost_usd  — running upper-bound total across all review runs.
                      Updated on every review run. NULL when the model has
                      no entry in app.config._MODEL_PRICING.
    final_cost_usd  — frozen copy of total_cost_usd when the PR is merged.
                      Lets analytics distinguish "this PR is done" from
                      "still accumulating".

NUMERIC(10,6) covers up to ~$10k per PR with sub-cent precision.
"""
from typing import Sequence, Union

from alembic import op

revision: str = "003"
down_revision: Union[str, None] = "002"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute("ALTER TABLE pr_reviews ADD COLUMN total_cost_usd NUMERIC(10,6)")
    op.execute("ALTER TABLE pr_reviews ADD COLUMN final_cost_usd NUMERIC(10,6)")


def downgrade() -> None:
    op.execute("ALTER TABLE pr_reviews DROP COLUMN IF EXISTS final_cost_usd")
    op.execute("ALTER TABLE pr_reviews DROP COLUMN IF EXISTS total_cost_usd")
