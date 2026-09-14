"""`app/team_store.py` against a scripted fake asyncpg connection."""

from contextlib import asynccontextmanager
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.config import ProjectScope
from app.team_store import (
    ClaimConflict,
    TeamSettings,
    add_claims,
    get_settings,
    list_all_claims,
    list_claims,
    purge_project,
    remove_claims,
)


def _row(**kw: Any) -> dict[str, Any]:
    return kw


def _pool(fetch_side_effect=None, fetchrow_return=None):
    conn = AsyncMock()
    conn.fetch = AsyncMock(side_effect=fetch_side_effect)
    conn.fetchrow = AsyncMock(return_value=fetchrow_return)
    conn.execute = AsyncMock()

    @asynccontextmanager
    async def _tx():
        yield

    conn.transaction = _tx
    pool = MagicMock()

    @asynccontextmanager
    async def _acquire():
        yield conn

    pool.acquire = _acquire
    pool._conn = conn
    return pool


@pytest.mark.asyncio
async def test_list_claims_groups_repo_claims_and_keeps_whole_projects_first():
    rows = [
        _row(project_key="INFRA", repo_slug="a"),
        _row(project_key="PLAT", repo_slug=None),
        _row(project_key="INFRA", repo_slug="b"),
    ]
    pool = _pool(fetch_side_effect=[rows])
    assert await list_claims(pool, "platform") == [
        ProjectScope(key="PLAT"), ProjectScope(key="INFRA", repos=["a", "b"]),
    ]


@pytest.mark.asyncio
async def test_list_all_claims_by_team():
    rows = [
        _row(team_slug="platform", project_key="PLAT", repo_slug=None),
        _row(team_slug="payments", project_key="PAY", repo_slug="billing"),
    ]
    pool = _pool(fetch_side_effect=[rows])
    assert await list_all_claims(pool) == {
        "platform": [ProjectScope(key="PLAT")],
        "payments": [ProjectScope(key="PAY", repos=["billing"])],
    }


@pytest.mark.asyncio
async def test_add_whole_project_conflicts_with_another_teams_repo_claim():
    pool = _pool(fetch_side_effect=[[_row(id=1, team_slug="payments", repo_slug="billing")]])
    with pytest.raises(ClaimConflict) as exc:
        await add_claims(pool, "platform", [ProjectScope(key="PAY")], "jan")
    assert str(exc.value) == "PAY/billing is claimed by team payments"
    pool._conn.execute.assert_not_called()


@pytest.mark.asyncio
async def test_add_whole_project_replaces_own_repo_claims():
    pool = _pool(fetch_side_effect=[[_row(id=1, team_slug="platform", repo_slug="a")]])
    assert await add_claims(pool, "platform", [ProjectScope(key="PLAT")], "jan") == ["PLAT"]
    sql = [c.args[0] for c in pool._conn.execute.await_args_list]
    assert sql[0].startswith("DELETE FROM team_claims WHERE project_key = $1 AND team_slug = $2")
    assert "INSERT INTO team_claims" in sql[1]


@pytest.mark.asyncio
async def test_add_repo_claims_skips_held_and_rejects_foreign():
    # our whole-project claim already covers it: nothing to add
    pool = _pool(fetch_side_effect=[[_row(id=1, team_slug="platform", repo_slug=None)]])
    assert await add_claims(pool, "platform", [ProjectScope(key="PLAT", repos=["x"])], "jan") == []

    # another team holds the whole project
    pool = _pool(fetch_side_effect=[[_row(id=1, team_slug="payments", repo_slug=None)]])
    with pytest.raises(ClaimConflict, match="PLAT is claimed by team payments"):
        await add_claims(pool, "platform", [ProjectScope(key="PLAT", repos=["x"])], "jan")

    # one repo ours already, one new, one foreign -> all or nothing
    rows = [_row(id=1, team_slug="platform", repo_slug="a"), _row(id=2, team_slug="payments", repo_slug="c")]
    pool = _pool(fetch_side_effect=[rows])
    with pytest.raises(ClaimConflict, match="PLAT/c is claimed by team payments"):
        await add_claims(pool, "platform", [ProjectScope(key="PLAT", repos=["a", "b", "c"])], "jan")

    pool = _pool(fetch_side_effect=[[_row(id=1, team_slug="platform", repo_slug="a")]])
    assert await add_claims(pool, "platform", [ProjectScope(key="PLAT", repos=["a", "b"])], "jan") == ["PLAT/b"]
    assert pool._conn.execute.await_count == 1


@pytest.mark.asyncio
async def test_remove_claims_reports_what_went():
    pool = _pool(fetch_side_effect=[
        [_row(repo_slug=None), _row(repo_slug="a")],  # whole project scope drops both
        [_row(id=5)],                                  # repo scope hit
        [],                                            # repo scope miss
    ])
    removed = await remove_claims(
        pool, "platform", [ProjectScope(key="PLAT"), ProjectScope(key="INFRA", repos=["x", "y"])]
    )
    assert removed == ["PLAT", "PLAT/a", "INFRA/x"]


@pytest.mark.asyncio
async def test_purge_is_scoped_to_the_team():
    pool = _pool(fetch_side_effect=[[_row(id=1), _row(id=2)]])
    assert await purge_project(pool, "platform", "PLAT", None) == 2
    sql, *args = pool._conn.fetch.await_args.args
    assert "WHERE team_slug = $1 AND project_key = $2" in sql
    assert args == ["platform", "PLAT"]

    pool = _pool(fetch_side_effect=[[]])
    assert await purge_project(pool, "platform", "PLAT", "repo") == 0
    assert "AND repo_slug = $3" in pool._conn.fetch.await_args.args[0]


@pytest.mark.asyncio
async def test_get_settings():
    pool = _pool(fetchrow_return=_row(auto_review_authors=["a"], ignore_authors=[]))
    assert await get_settings(pool, "platform") == TeamSettings(["a"], [])
    assert await get_settings(_pool(fetchrow_return=None), "platform") is None


@pytest.mark.asyncio
async def test_unique_index_race_is_a_conflict():
    import asyncpg
    pool = _pool(fetch_side_effect=[[]])
    pool._conn.execute = AsyncMock(side_effect=asyncpg.UniqueViolationError("dup"))
    with pytest.raises(ClaimConflict, match="concurrent claim"):
        await add_claims(pool, "platform", [ProjectScope(key="PLAT")], "jan")
