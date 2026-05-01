"""Per-model token pricing cache

Revision ID: 006
Revises: 005
Create Date: 2026-05-01

Adds `model_pricing` — a cache of LLM per-model token pricing fetched from
the LiteLLM public pricing JSON every 24h. Loaded into memory at startup so
cost estimates survive a LiteLLM outage. Static fallback in
app.config._STATIC_MODEL_PRICING covers cold-start with an empty table.
"""
from typing import Sequence, Union

from alembic import op

revision: str = "006"
down_revision: Union[str, None] = "005"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
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


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS model_pricing")
