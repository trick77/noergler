"""Initial schema

Revision ID: 001
Revises:
Create Date: 2026-04-04

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
            last_reviewed_commit TEXT,
            summary_comment_id INTEGER,
            summary_comment_version INTEGER,
            author TEXT,
            pr_title TEXT,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            UNIQUE (project_key, repo_slug, pr_id)
        )
    """)

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
        CREATE TABLE review_statistics (
            id BIGSERIAL PRIMARY KEY,
            project_key TEXT NOT NULL,
            repo_slug TEXT NOT NULL,
            pr_id INTEGER NOT NULL,
            author TEXT,
            is_incremental BOOLEAN DEFAULT FALSE,
            reviewed_commit TEXT,
            diff_added INTEGER,
            diff_removed INTEGER,
            files_reviewed INTEGER,
            total_files INTEGER,
            critical_count INTEGER DEFAULT 0,
            warning_count INTEGER DEFAULT 0,
            security_count INTEGER DEFAULT 0,
            review_effort INTEGER,
            prompt_tokens INTEGER,
            completion_tokens INTEGER,
            model_name TEXT,
            elapsed_seconds REAL,
            cross_file_deps INTEGER DEFAULT 0,
            skipped_files INTEGER DEFAULT 0,
            content_skipped INTEGER DEFAULT 0,
            findings_posted INTEGER DEFAULT 0,
            findings_deduplicated INTEGER DEFAULT 0,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
        )
    """)

    op.execute("""
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
    """)

    op.execute("""
        CREATE INDEX idx_review_statistics_pr
        ON review_statistics (project_key, repo_slug, pr_id)
    """)

    op.execute("""
        CREATE INDEX idx_feedback_events_pr
        ON feedback_events (project_key, repo_slug, pr_id)
    """)


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS feedback_events")
    op.execute("DROP TABLE IF EXISTS review_statistics")
    op.execute("DROP TABLE IF EXISTS review_findings")
    op.execute("DROP TABLE IF EXISTS pr_reviews")
