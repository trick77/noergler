"""Read-only metrics API over the alembic 004 views.

Gated by an X-API-Key header. The key is compared with
`config.metrics.api_key` via `hmac.compare_digest`. Empty configured
key -> 503 (disabled).
"""
from __future__ import annotations

import hmac
import logging
from datetime import datetime
from typing import Any

import asyncpg
from fastapi import APIRouter, Depends, Header, HTTPException, Query, Request

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/metrics", tags=["metrics"])

_DEFAULT_LIMIT = 1000
_MAX_LIMIT = 10_000


def _require_api_key(request: Request, x_api_key: str | None = Header(None)) -> None:
    configured = request.app.state.config.metrics.api_key
    if not configured:
        raise HTTPException(status_code=503, detail="metrics API disabled")
    if not x_api_key or not hmac.compare_digest(x_api_key, configured):
        raise HTTPException(status_code=401, detail="invalid API key")


def _pool(request: Request) -> asyncpg.Pool:
    pool = request.app.state.db_pool
    if pool is None:
        raise HTTPException(status_code=503, detail="database unavailable")
    return pool


def _row_to_jsonable(row: asyncpg.Record) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in dict(row).items():
        if isinstance(value, datetime):
            out[key] = value.isoformat()
        else:
            out[key] = value
    return out


async def _query(
    pool: asyncpg.Pool,
    sql: str,
    params: list[Any],
) -> list[dict[str, Any]]:
    async with pool.acquire() as conn:
        rows = await conn.fetch(sql, *params)
    return [_row_to_jsonable(r) for r in rows]


def _bounded_limit(limit: int) -> int:
    return max(1, min(limit, _MAX_LIMIT))


@router.get("/reviewer-precision", dependencies=[Depends(_require_api_key)])
async def reviewer_precision(
    request: Request,
    project_key: str | None = Query(None),
    repo_slug: str | None = Query(None),
    since: datetime | None = Query(None, description="inclusive lower bound on week"),
    until: datetime | None = Query(None, description="exclusive upper bound on week"),
    limit: int = Query(_DEFAULT_LIMIT, ge=1, le=_MAX_LIMIT),
) -> dict[str, Any]:
    clauses: list[str] = []
    params: list[Any] = []
    if project_key is not None:
        params.append(project_key); clauses.append(f"project_key = ${len(params)}")
    if repo_slug is not None:
        params.append(repo_slug); clauses.append(f"repo_slug = ${len(params)}")
    if since is not None:
        params.append(since); clauses.append(f"week >= ${len(params)}")
    if until is not None:
        params.append(until); clauses.append(f"week < ${len(params)}")
    where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
    params.append(_bounded_limit(limit))
    sql = f"""
        SELECT project_key, repo_slug, week, n_posted, n_disagreed, precision
        FROM v_reviewer_precision
        {where}
        ORDER BY week DESC, project_key, repo_slug
        LIMIT ${len(params)}
    """
    rows = await _query(_pool(request), sql, params)
    return {"count": len(rows), "rows": rows}


@router.get("/lead-time", dependencies=[Depends(_require_api_key)])
async def lead_time(
    request: Request,
    project_key: str | None = Query(None),
    repo_slug: str | None = Query(None),
    author: str | None = Query(None),
    since: datetime | None = Query(None, description="inclusive lower bound on merged_at"),
    until: datetime | None = Query(None, description="exclusive upper bound on merged_at"),
    limit: int = Query(_DEFAULT_LIMIT, ge=1, le=_MAX_LIMIT),
) -> dict[str, Any]:
    clauses: list[str] = []
    params: list[Any] = []
    if project_key is not None:
        params.append(project_key); clauses.append(f"project_key = ${len(params)}")
    if repo_slug is not None:
        params.append(repo_slug); clauses.append(f"repo_slug = ${len(params)}")
    if author is not None:
        params.append(author); clauses.append(f"author = ${len(params)}")
    if since is not None:
        params.append(since); clauses.append(f"merged_at >= ${len(params)}")
    if until is not None:
        params.append(until); clauses.append(f"merged_at < ${len(params)}")
    where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
    params.append(_bounded_limit(limit))
    sql = f"""
        SELECT project_key, repo_slug, pr_id, author, opened_at, merged_at, lead_time_seconds
        FROM v_lead_time
        {where}
        ORDER BY merged_at DESC
        LIMIT ${len(params)}
    """
    rows = await _query(_pool(request), sql, params)
    return {"count": len(rows), "rows": rows}


@router.get("/activity", dependencies=[Depends(_require_api_key)])
async def activity(
    request: Request,
    author: str | None = Query(None),
    since: datetime | None = Query(None, description="inclusive lower bound on week"),
    until: datetime | None = Query(None, description="exclusive upper bound on week"),
    limit: int = Query(_DEFAULT_LIMIT, ge=1, le=_MAX_LIMIT),
) -> dict[str, Any]:
    clauses: list[str] = []
    params: list[Any] = []
    if author is not None:
        params.append(author); clauses.append(f"author = ${len(params)}")
    if since is not None:
        params.append(since); clauses.append(f"week >= ${len(params)}")
    if until is not None:
        params.append(until); clauses.append(f"week < ${len(params)}")
    where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
    params.append(_bounded_limit(limit))
    sql = f"""
        SELECT author, week, prs, review_runs
        FROM v_activity_weekly
        {where}
        ORDER BY week DESC, author
        LIMIT ${len(params)}
    """
    rows = await _query(_pool(request), sql, params)
    return {"count": len(rows), "rows": rows}


@router.get("/cost-by-model", dependencies=[Depends(_require_api_key)])
async def cost_by_model(
    request: Request,
    model: str | None = Query(None, description="filter by model_name"),
    since: datetime | None = Query(None, description="inclusive lower bound on week"),
    until: datetime | None = Query(None, description="exclusive upper bound on week"),
    limit: int = Query(_DEFAULT_LIMIT, ge=1, le=_MAX_LIMIT),
) -> dict[str, Any]:
    clauses: list[str] = []
    params: list[Any] = []
    if model is not None:
        params.append(model); clauses.append(f"model_name = ${len(params)}")
    if since is not None:
        params.append(since); clauses.append(f"week >= ${len(params)}")
    if until is not None:
        params.append(until); clauses.append(f"week < ${len(params)}")
    where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
    params.append(_bounded_limit(limit))
    sql = f"""
        SELECT model_name, week, runs, prompt_tokens, completion_tokens,
               total_tokens, avg_elapsed_seconds
        FROM v_cost_by_model
        {where}
        ORDER BY week DESC, model_name
        LIMIT ${len(params)}
    """
    rows = await _query(_pool(request), sql, params)
    return {"count": len(rows), "rows": rows}
