"""Read-only analytics API over the alembic 004 views.

Gated by an X-API-Key header. The key is compared with
`config.analytics.api_key` via `hmac.compare_digest`. Empty configured
key -> 503 (disabled). The /analytics path was chosen to avoid a
collision with the Prometheus /metrics convention.
"""
from __future__ import annotations

import hmac
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Sequence

import asyncpg
from fastapi import APIRouter, Depends, Header, HTTPException, Query, Request

router = APIRouter(prefix="/analytics", tags=["analytics"])

_DEFAULT_LIMIT = 1000
_MAX_LIMIT = 10_000


def _require_api_key(request: Request, x_api_key: str | None = Header(None)) -> None:
    configured = request.app.state.config.analytics.api_key
    if not configured:
        raise HTTPException(status_code=503, detail="analytics API disabled")
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


def _bounded_limit(limit: int) -> int:
    return max(1, min(limit, _MAX_LIMIT))


@dataclass(frozen=True)
class _Filter:
    column: str
    value: Any | None


async def _run_view_query(
    request: Request,
    *,
    view: str,
    columns: str,
    filters: Sequence[_Filter],
    time_column: str,
    since: datetime,
    until: datetime | None,
    order_by: str,
    limit: int,
) -> dict[str, Any]:
    """SELECT … FROM <view> WHERE <eq filters AND time bounds> ORDER BY … LIMIT …

    Placeholder positions are deterministic: equality filters in the order
    given, then since, then until, then limit. Tests rely on this ordering.
    """
    clauses: list[str] = []
    params: list[Any] = []
    for f in filters:
        if f.value is None:
            continue
        params.append(f.value)
        clauses.append(f"{f.column} = ${len(params)}")
    params.append(since)
    clauses.append(f"{time_column} >= ${len(params)}")
    if until is not None:
        params.append(until)
        clauses.append(f"{time_column} < ${len(params)}")
    where = f"WHERE {' AND '.join(clauses)}"
    params.append(_bounded_limit(limit))
    sql = f"""
        SELECT {columns}
        FROM {view}
        {where}
        ORDER BY {order_by}
        LIMIT ${len(params)}
    """
    async with _pool(request).acquire() as conn:
        rows = await conn.fetch(sql, *params)
    rows_out = [_row_to_jsonable(r) for r in rows]
    return {"count": len(rows_out), "rows": rows_out}


@router.get("/reviewer-precision", dependencies=[Depends(_require_api_key)])
async def reviewer_precision(
    request: Request,
    project_key: str | None = Query(None),
    repo_slug: str | None = Query(None),
    since: datetime = Query(..., description="required inclusive lower bound on week"),
    until: datetime | None = Query(None, description="exclusive upper bound on week"),
    limit: int = Query(_DEFAULT_LIMIT, ge=1, le=_MAX_LIMIT),
) -> dict[str, Any]:
    return await _run_view_query(
        request,
        view="v_reviewer_precision",
        columns="project_key, repo_slug, week, n_posted, n_disagreed, precision_score",
        filters=(_Filter("project_key", project_key), _Filter("repo_slug", repo_slug)),
        time_column="week",
        since=since,
        until=until,
        order_by="week DESC, project_key, repo_slug",
        limit=limit,
    )


@router.get("/lead-time", dependencies=[Depends(_require_api_key)])
async def lead_time(
    request: Request,
    project_key: str | None = Query(None),
    repo_slug: str | None = Query(None),
    author: str | None = Query(None),
    since: datetime = Query(..., description="required inclusive lower bound on merged_at"),
    until: datetime | None = Query(None, description="exclusive upper bound on merged_at"),
    limit: int = Query(_DEFAULT_LIMIT, ge=1, le=_MAX_LIMIT),
) -> dict[str, Any]:
    return await _run_view_query(
        request,
        view="v_lead_time",
        columns="project_key, repo_slug, pr_id, author, opened_at, merged_at, lead_time_seconds",
        filters=(
            _Filter("project_key", project_key),
            _Filter("repo_slug", repo_slug),
            _Filter("author", author),
        ),
        time_column="merged_at",
        since=since,
        until=until,
        order_by="merged_at DESC",
        limit=limit,
    )


@router.get("/activity", dependencies=[Depends(_require_api_key)])
async def activity(
    request: Request,
    author: str | None = Query(None),
    since: datetime = Query(..., description="required inclusive lower bound on week"),
    until: datetime | None = Query(None, description="exclusive upper bound on week"),
    limit: int = Query(_DEFAULT_LIMIT, ge=1, le=_MAX_LIMIT),
) -> dict[str, Any]:
    return await _run_view_query(
        request,
        view="v_activity_weekly",
        columns="author, week, prs, review_runs",
        filters=(_Filter("author", author),),
        time_column="week",
        since=since,
        until=until,
        order_by="week DESC, author",
        limit=limit,
    )


@router.get("/cost-by-model", dependencies=[Depends(_require_api_key)])
async def cost_by_model(
    request: Request,
    model: str | None = Query(None, description="filter by model_name"),
    since: datetime = Query(..., description="required inclusive lower bound on week"),
    until: datetime | None = Query(None, description="exclusive upper bound on week"),
    limit: int = Query(_DEFAULT_LIMIT, ge=1, le=_MAX_LIMIT),
) -> dict[str, Any]:
    return await _run_view_query(
        request,
        view="v_cost_by_model",
        columns=(
            "model_name, week, runs, prompt_tokens, completion_tokens, "
            "total_tokens, avg_elapsed_seconds"
        ),
        filters=(_Filter("model_name", model),),
        time_column="week",
        since=since,
        until=until,
        order_by="week DESC, model_name",
        limit=limit,
    )
