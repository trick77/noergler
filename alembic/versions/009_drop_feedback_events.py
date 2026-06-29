"""Drop the feedback_events table

Revision ID: 009
Revises: 008
Create Date: 2026-06-29

Removes the "disagree" feedback mechanic. The text-keyword feedback path
(developers replying "disagree" to an inline finding) proved to be noise:
replies flag false positives the reviewer could never have prevented without
context it never had. With the keyword detection, acknowledgement, riptide
reviewer-precision emission, and merge-time "useful %" stats all removed, the
`feedback_events` table has no remaining reader or writer, so it is dropped.

Irreversible data loss: any recorded disagree events are deleted on upgrade.
The downgrade recreates the table and its index (schema as of migration 001)
but cannot restore the rows.
"""
from typing import Sequence, Union

from alembic import op

revision: str = "009"
down_revision: Union[str, None] = "008"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute("DROP INDEX IF EXISTS idx_feedback_events_pr")
    op.execute("DROP TABLE IF EXISTS feedback_events")


def downgrade() -> None:
    op.execute(
        """
        CREATE TABLE feedback_events (
            id BIGSERIAL PRIMARY KEY,
            project_key TEXT NOT NULL,
            repo_slug TEXT NOT NULL,
            pr_id INTEGER NOT NULL,
            bitbucket_comment_id INTEGER,
            feedback_author TEXT,
            classification TEXT,
            file_path TEXT,
            severity TEXT,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
        )
        """
    )
    op.execute(
        """
        CREATE INDEX idx_feedback_events_pr
        ON feedback_events (project_key, repo_slug, pr_id)
        """
    )
