"""Drop the metrics layer: views, lead-time index, and review_statistics

Revision ID: 005
Revises: 004
Create Date: 2026-04-29

Phase B of the noergler → riptide migration. Once riptide collects review
cost (model + tokens + cost_usd) and reviewer-precision (per-finding
verdict) directly via webhooks, noergler keeps:

- `pr_reviews` (incl. lifecycle timestamps + cost columns) — feeds the
  user-facing summary comment posted on each PR.
- `review_findings` — feeds inline-comment dedup on incremental reviews.
- `feedback_events` — feeds `has_negative_feedback` dedup so the bot
  doesn't react to the same disagree comment twice.

What this migration drops (purely metrics, nothing else reads it):

- The four read-only views (`v_lead_time`, `v_activity_weekly`,
  `v_cost_by_model`, `v_reviewer_precision`).
- `idx_pr_reviews_lead_time` — only feeds `v_lead_time`.
- `review_statistics` — only the views aggregated it; no review-posting
  code path queries it.

The lifecycle index `idx_pr_reviews_lifecycle` stays because
`mark_pr_merged` / `mark_pr_deleted` use it.
"""

from typing import Sequence, Union

from alembic import op

revision: str = "005"
down_revision: Union[str, None] = "004"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Views first — they may reference the table we're about to drop.
    op.execute("DROP VIEW IF EXISTS v_lead_time")
    op.execute("DROP VIEW IF EXISTS v_cost_by_model")
    op.execute("DROP VIEW IF EXISTS v_activity_weekly")
    op.execute("DROP VIEW IF EXISTS v_reviewer_precision")
    op.execute("DROP INDEX IF EXISTS idx_pr_reviews_lead_time")
    op.execute("DROP TABLE IF EXISTS review_statistics")


def downgrade() -> None:
    # Recreate review_statistics with the schema as of migration 004
    # (issue_count / suggestion_count rename already applied).
    op.execute(
        """
        CREATE TABLE review_statistics (
            id BIGSERIAL PRIMARY KEY,
            project_key TEXT NOT NULL,
            repo_slug TEXT NOT NULL,
            pr_id BIGINT NOT NULL,
            author TEXT,
            is_incremental BOOLEAN NOT NULL DEFAULT FALSE,
            reviewed_commit TEXT,
            diff_added INT NOT NULL DEFAULT 0,
            diff_removed INT NOT NULL DEFAULT 0,
            files_reviewed INT NOT NULL DEFAULT 0,
            total_files INT NOT NULL DEFAULT 0,
            issue_count INT NOT NULL DEFAULT 0,
            suggestion_count INT NOT NULL DEFAULT 0,
            security_count INT NOT NULL DEFAULT 0,
            review_effort INT,
            prompt_tokens INT NOT NULL DEFAULT 0,
            completion_tokens INT NOT NULL DEFAULT 0,
            model_name TEXT,
            elapsed_seconds DOUBLE PRECISION,
            cross_file_deps INT NOT NULL DEFAULT 0,
            skipped_files INT NOT NULL DEFAULT 0,
            content_skipped INT NOT NULL DEFAULT 0,
            findings_posted INT NOT NULL DEFAULT 0,
            findings_deduplicated INT NOT NULL DEFAULT 0,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
        )
        """
    )

    op.execute(
        """
        CREATE INDEX idx_pr_reviews_lead_time
        ON pr_reviews (merged_at DESC)
        WHERE deleted_at IS NULL AND merged_at IS NOT NULL
        """
    )

    op.execute(
        """
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
        """
    )

    op.execute(
        """
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
        """
    )

    op.execute(
        """
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
        """
    )

    op.execute(
        """
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
        """
    )
