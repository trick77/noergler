"""Drop the model_pricing cache

Revision ID: 010
Revises: 009
Create Date: 2026-07-27

noergler no longer caches LLM pricing. The model catalog is fetched over the
network at startup and re-fetched every 24h in memory; the configured model must
resolve against it or startup aborts. The catalog's own rates are the only
fallback for a call the endpoint doesn't price, so a persisted copy could only
ever serve stale prices alongside it — the table goes.
"""
from typing import Sequence, Union

from alembic import op

revision: str = "010"
down_revision: Union[str, None] = "009"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute("DROP TABLE IF EXISTS model_pricing")


def downgrade() -> None:
    # Recreated empty: the rows were a disposable cache, and the code that
    # populated them is gone. Restoring the shape is enough to roll back.
    op.execute(
        """
        CREATE TABLE model_pricing (
            model_id TEXT PRIMARY KEY,
            input_per_mtok NUMERIC(10,6) NOT NULL,
            cached_input_per_mtok NUMERIC(10,6) NOT NULL,
            output_per_mtok NUMERIC(10,6) NOT NULL,
            updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
        )
        """
    )
