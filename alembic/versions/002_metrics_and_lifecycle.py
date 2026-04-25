"""Metrics: rename warning_count, add PR lifecycle timestamps, create metric views

Revision ID: 002
Revises: 001
Create Date: 2026-04-25

Consolidates three previously-separate revisions (warning rename, lifecycle
timestamps, metric views) into one. None of them ever ran against the
production database, so no downstream migration depended on the intermediate
states.

Schema changes:
    - review_statistics.warning_count -> important_count
    - review_findings.severity: backfill 'warning' -> 'important'
    - pr_reviews: add opened_at / merged_at / deleted_at + lifecycle index
Views (one question each):
    v_reviewer_precision  — how useful is the LLM review?        (higher = better)
    v_lead_time           — DORA lead-time for changes
    v_activity_weekly     — SPACE Activity: PRs / runs per author per week
    v_cost_by_model       — LLM token spend per model per week
"""
from typing import Sequence, Union

from alembic import op

revision: str = "002"
down_revision: Union[str, None] = "001"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # --- review_statistics: rename warning_count -> important_count ---
    op.execute("ALTER TABLE review_statistics RENAME COLUMN warning_count TO important_count")

    # --- review_findings: backfill historical severity strings to match new naming ---
    op.execute("UPDATE review_findings SET severity = 'important' WHERE severity = 'warning'")

    # --- pr_reviews: add lifecycle timestamps + supporting index ---
    op.execute("ALTER TABLE pr_reviews ADD COLUMN opened_at TIMESTAMPTZ")
    op.execute("ALTER TABLE pr_reviews ADD COLUMN merged_at TIMESTAMPTZ")
    op.execute("ALTER TABLE pr_reviews ADD COLUMN deleted_at TIMESTAMPTZ")
    op.execute("CREATE INDEX idx_pr_reviews_lifecycle ON pr_reviews (merged_at, deleted_at)")

    # --- Reviewer precision = 1 - disagree_rate. Positive framing.
    # Both sides bucket by review_findings.created_at so a finding posted
    # in week N and disagreed in week N+1 lands in the same row (the
    # posting week). Otherwise a quiet week with old findings still being
    # disagreed would yield a negative precision_score.
    op.execute("""
        CREATE VIEW v_reviewer_precision AS
        WITH posted AS (
            SELECT
                pr.project_key,
                pr.repo_slug,
                DATE_TRUNC('week', rf.created_at) AS week,
                COUNT(*) AS n_posted
            FROM review_findings rf
            JOIN pr_reviews pr ON pr.id = rf.pr_review_id
            WHERE pr.deleted_at IS NULL
              AND rf.bitbucket_comment_id IS NOT NULL
            GROUP BY pr.project_key, pr.repo_slug, DATE_TRUNC('week', rf.created_at)
        ),
        disagreed AS (
            SELECT
                pr.project_key,
                pr.repo_slug,
                DATE_TRUNC('week', rf.created_at) AS week,
                COUNT(*) AS n_disagreed
            FROM feedback_events fe
            JOIN review_findings rf
                ON rf.bitbucket_comment_id = fe.bitbucket_comment_id
            JOIN pr_reviews pr ON pr.id = rf.pr_review_id
            WHERE fe.classification = 'negative'
              AND pr.deleted_at IS NULL
            GROUP BY pr.project_key, pr.repo_slug, DATE_TRUNC('week', rf.created_at)
        )
        SELECT
            p.project_key,
            p.repo_slug,
            p.week,
            p.n_posted,
            COALESCE(d.n_disagreed, 0) AS n_disagreed,
            CASE
                WHEN p.n_posted > 0 THEN
                    ROUND(1 - COALESCE(d.n_disagreed, 0)::numeric / p.n_posted, 3)::float8
                ELSE NULL
            END AS precision_score
        FROM posted p
        LEFT JOIN disagreed d
            ON d.project_key = p.project_key
           AND d.repo_slug   = p.repo_slug
           AND d.week        = p.week
    """)

    # Activity per author per week — PRs observed and review runs.
    op.execute("""
        CREATE VIEW v_activity_weekly AS
        SELECT
            rs.author,
            DATE_TRUNC('week', rs.created_at) AS week,
            COUNT(DISTINCT (rs.project_key, rs.repo_slug, rs.pr_id)) AS prs,
            COUNT(*) AS review_runs
        FROM review_statistics rs
        LEFT JOIN pr_reviews pr
            ON pr.project_key = rs.project_key
           AND pr.repo_slug   = rs.repo_slug
           AND pr.pr_id       = rs.pr_id
        WHERE rs.author IS NOT NULL
          AND (pr.deleted_at IS NULL OR pr.id IS NULL)
        GROUP BY rs.author, DATE_TRUNC('week', rs.created_at)
    """)

    # LLM cost per model per week.
    op.execute("""
        CREATE VIEW v_cost_by_model AS
        SELECT
            rs.model_name,
            DATE_TRUNC('week', rs.created_at) AS week,
            COUNT(*)                      AS runs,
            SUM(rs.prompt_tokens)         AS prompt_tokens,
            SUM(rs.completion_tokens)     AS completion_tokens,
            SUM(rs.prompt_tokens + rs.completion_tokens) AS total_tokens,
            AVG(rs.elapsed_seconds)       AS avg_elapsed_seconds
        FROM review_statistics rs
        WHERE rs.model_name IS NOT NULL
        GROUP BY rs.model_name, DATE_TRUNC('week', rs.created_at)
    """)

    # Lead-time per merged PR (DORA).
    op.execute("""
        CREATE VIEW v_lead_time AS
        SELECT
            project_key,
            repo_slug,
            pr_id,
            author,
            opened_at,
            merged_at,
            EXTRACT(EPOCH FROM (merged_at - opened_at))::bigint AS lead_time_seconds
        FROM pr_reviews
        WHERE merged_at IS NOT NULL
          AND opened_at IS NOT NULL
          AND deleted_at IS NULL
    """)

    # Supports v_lead_time's ORDER BY merged_at DESC and the deleted_at
    # filter. Partial index keeps it small — only live rows.
    op.execute("""
        CREATE INDEX idx_pr_reviews_lead_time
        ON pr_reviews (merged_at DESC)
        WHERE deleted_at IS NULL AND merged_at IS NOT NULL
    """)


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS idx_pr_reviews_lead_time")
    op.execute("DROP VIEW IF EXISTS v_lead_time")
    op.execute("DROP VIEW IF EXISTS v_cost_by_model")
    op.execute("DROP VIEW IF EXISTS v_activity_weekly")
    op.execute("DROP VIEW IF EXISTS v_reviewer_precision")
    op.execute("DROP INDEX IF EXISTS idx_pr_reviews_lifecycle")
    op.execute("ALTER TABLE pr_reviews DROP COLUMN IF EXISTS deleted_at")
    op.execute("ALTER TABLE pr_reviews DROP COLUMN IF EXISTS merged_at")
    op.execute("ALTER TABLE pr_reviews DROP COLUMN IF EXISTS opened_at")
    op.execute("UPDATE review_findings SET severity = 'warning' WHERE severity = 'important'")
    op.execute("ALTER TABLE review_statistics RENAME COLUMN important_count TO warning_count")
