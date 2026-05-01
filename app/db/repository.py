from datetime import datetime

import asyncpg


async def upsert_pr_review(
    pool: asyncpg.Pool,
    project_key: str,
    repo_slug: str,
    pr_id: int,
    last_reviewed_commit: str | None = None,
    author: str | None = None,
    pr_title: str | None = None,
    opened_at: datetime | None = None,
) -> int:
    """Insert or update PR review record. Returns the pr_review_id."""
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            INSERT INTO pr_reviews (project_key, repo_slug, pr_id, last_reviewed_commit, author, pr_title, opened_at)
            VALUES ($1, $2, $3, $4, $5, $6, $7)
            ON CONFLICT (project_key, repo_slug, pr_id)
            DO UPDATE SET
                last_reviewed_commit = EXCLUDED.last_reviewed_commit,
                author = EXCLUDED.author,
                pr_title = EXCLUDED.pr_title,
                opened_at = COALESCE(pr_reviews.opened_at, EXCLUDED.opened_at),
                updated_at = NOW()
            RETURNING id
            """,
            project_key, repo_slug, pr_id, last_reviewed_commit, author, pr_title, opened_at,
        )
        return row["id"]


async def get_last_reviewed_commit(
    pool: asyncpg.Pool, project_key: str, repo_slug: str, pr_id: int
) -> str | None:
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            SELECT last_reviewed_commit FROM pr_reviews
            WHERE project_key = $1 AND repo_slug = $2 AND pr_id = $3
              AND merged_at IS NULL AND deleted_at IS NULL
            """,
            project_key, repo_slug, pr_id,
        )
        return row["last_reviewed_commit"] if row else None


async def update_summary_comment(
    pool: asyncpg.Pool,
    pr_review_id: int,
    summary_comment_id: int,
    summary_comment_version: int,
) -> None:
    async with pool.acquire() as conn:
        await conn.execute(
            """
            UPDATE pr_reviews
            SET summary_comment_id = $2, summary_comment_version = $3, updated_at = NOW()
            WHERE id = $1
            """,
            pr_review_id, summary_comment_id, summary_comment_version,
        )


async def get_summary_comment_info(
    pool: asyncpg.Pool, pr_review_id: int
) -> dict | None:
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT summary_comment_id, summary_comment_version FROM pr_reviews WHERE id = $1",
            pr_review_id,
        )
        if row and row["summary_comment_id"]:
            return {
                "summary_comment_id": row["summary_comment_id"],
                "summary_comment_version": row["summary_comment_version"],
            }
        return None


async def add_pr_cost(
    pool: asyncpg.Pool,
    project_key: str,
    repo_slug: str,
    pr_id: int,
    delta_usd: float,
) -> float | None:
    """Add this run's cost to pr_reviews.total_cost_usd and return the new total.

    Returns None if no matching PR row exists (shouldn't happen — upsert runs
    first — but keeps the helper safe).
    """
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            UPDATE pr_reviews
            SET total_cost_usd = COALESCE(total_cost_usd, 0) + $4,
                updated_at = NOW()
            WHERE project_key = $1 AND repo_slug = $2 AND pr_id = $3
            RETURNING total_cost_usd
            """,
            project_key, repo_slug, pr_id, delta_usd,
        )
        return float(row["total_cost_usd"]) if row else None


async def freeze_pr_cost(
    pool: asyncpg.Pool, project_key: str, repo_slug: str, pr_id: int
) -> float | None:
    """Copy total_cost_usd into final_cost_usd and return the frozen value."""
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            UPDATE pr_reviews
            SET final_cost_usd = total_cost_usd, updated_at = NOW()
            WHERE project_key = $1 AND repo_slug = $2 AND pr_id = $3
            RETURNING final_cost_usd
            """,
            project_key, repo_slug, pr_id,
        )
        if row and row["final_cost_usd"] is not None:
            return float(row["final_cost_usd"])
        return None


async def mark_pr_merged(
    pool: asyncpg.Pool, project_key: str, repo_slug: str, pr_id: int
) -> None:
    async with pool.acquire() as conn:
        await conn.execute(
            """
            UPDATE pr_reviews SET merged_at = NOW(), updated_at = NOW()
            WHERE project_key = $1 AND repo_slug = $2 AND pr_id = $3
              AND merged_at IS NULL
            """,
            project_key, repo_slug, pr_id,
        )


async def mark_pr_deleted(
    pool: asyncpg.Pool, project_key: str, repo_slug: str, pr_id: int
) -> None:
    async with pool.acquire() as conn:
        await conn.execute(
            """
            UPDATE pr_reviews SET deleted_at = NOW(), updated_at = NOW()
            WHERE project_key = $1 AND repo_slug = $2 AND pr_id = $3
              AND deleted_at IS NULL
            """,
            project_key, repo_slug, pr_id,
        )


async def insert_finding(
    pool: asyncpg.Pool,
    pr_review_id: int,
    file_path: str | None,
    line_number: int | None,
    severity: str,
    comment_text: str,
    suggestion: str | None,
    bitbucket_comment_id: int | None,
    commit_sha: str | None,
    is_incremental: bool,
) -> None:
    async with pool.acquire() as conn:
        await conn.execute(
            """
            INSERT INTO review_findings
                (pr_review_id, file_path, line_number, severity, comment_text,
                 suggestion, bitbucket_comment_id, commit_sha, is_incremental)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
            """,
            pr_review_id, file_path, line_number, severity, comment_text,
            suggestion, bitbucket_comment_id, commit_sha, is_incremental,
        )


async def get_existing_finding_keys(
    pool: asyncpg.Pool, project_key: str, repo_slug: str, pr_id: int
) -> set[tuple[str | None, int | None, str]]:
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT f.file_path, f.line_number, f.severity
            FROM review_findings f
            JOIN pr_reviews r ON f.pr_review_id = r.id
            WHERE r.project_key = $1 AND r.repo_slug = $2 AND r.pr_id = $3
              AND r.merged_at IS NULL AND r.deleted_at IS NULL
            """,
            project_key, repo_slug, pr_id,
        )
        return {(row["file_path"], row["line_number"], row["severity"]) for row in rows}


async def get_existing_findings_for_prompt(
    pool: asyncpg.Pool, project_key: str, repo_slug: str, pr_id: int
) -> list[dict]:
    """Return previously posted findings on this PR for inclusion in the review prompt."""
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT f.file_path, f.line_number, f.severity, f.comment_text, f.id
            FROM review_findings f
            JOIN pr_reviews r ON f.pr_review_id = r.id
            WHERE r.project_key = $1 AND r.repo_slug = $2 AND r.pr_id = $3
              AND r.merged_at IS NULL AND r.deleted_at IS NULL
            ORDER BY f.id ASC
            """,
            project_key, repo_slug, pr_id,
        )
        return [
            {
                "file_path": row["file_path"],
                "line_number": row["line_number"],
                "severity": row["severity"],
                "comment_text": row["comment_text"],
            }
            for row in rows
        ]


async def get_finding_by_comment_id(
    pool: asyncpg.Pool, bitbucket_comment_id: int
) -> dict | None:
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT file_path, line_number, severity, commit_sha FROM review_findings WHERE bitbucket_comment_id = $1",
            bitbucket_comment_id,
        )
        return dict(row) if row else None


async def has_negative_feedback(
    pool: asyncpg.Pool,
    project_key: str,
    repo_slug: str,
    pr_id: int,
    bitbucket_comment_id: int,
) -> bool:
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            SELECT 1 FROM feedback_events
            WHERE project_key = $1 AND repo_slug = $2 AND pr_id = $3
              AND bitbucket_comment_id = $4 AND classification = 'negative'
            LIMIT 1
            """,
            project_key, repo_slug, pr_id, bitbucket_comment_id,
        )
        return row is not None


async def insert_feedback(
    pool: asyncpg.Pool,
    project_key: str,
    repo_slug: str,
    pr_id: int,
    bitbucket_comment_id: int,
    feedback_author: str,
    classification: str,
    file_path: str | None,
    severity: str | None,
) -> None:
    async with pool.acquire() as conn:
        await conn.execute(
            """
            INSERT INTO feedback_events
                (project_key, repo_slug, pr_id, bitbucket_comment_id,
                 feedback_author, classification, file_path, severity)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
            """,
            project_key, repo_slug, pr_id, bitbucket_comment_id,
            feedback_author, classification, file_path, severity,
        )


async def upsert_model_pricing(
    pool: asyncpg.Pool,
    entries: dict[str, tuple[float, float, float]],
) -> None:
    """Upsert (input, cached_input, output) per 1M tokens for each model id."""
    if not entries:
        return
    rows = [
        (model_id, inp, cached, out)
        for model_id, (inp, cached, out) in entries.items()
    ]
    async with pool.acquire() as conn:
        await conn.executemany(
            """
            INSERT INTO model_pricing
                (model_id, input_per_mtok, cached_input_per_mtok, output_per_mtok, updated_at)
            VALUES ($1, $2, $3, $4, NOW())
            ON CONFLICT (model_id) DO UPDATE SET
                input_per_mtok = EXCLUDED.input_per_mtok,
                cached_input_per_mtok = EXCLUDED.cached_input_per_mtok,
                output_per_mtok = EXCLUDED.output_per_mtok,
                updated_at = NOW()
            """,
            rows,
        )


async def load_model_pricing(
    pool: asyncpg.Pool,
) -> dict[str, tuple[float, float, float]]:
    """Return cached pricing as `{model_id: (input, cached_input, output)}`."""
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            "SELECT model_id, input_per_mtok, cached_input_per_mtok, output_per_mtok FROM model_pricing"
        )
    return {
        r["model_id"]: (
            float(r["input_per_mtok"]),
            float(r["cached_input_per_mtok"]),
            float(r["output_per_mtok"]),
        )
        for r in rows
    }
