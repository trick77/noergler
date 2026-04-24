"""Basic tests for app/db/repository.py using mocked asyncpg connections."""
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.db import repository


def _make_pool(fetchrow_return=None, fetch_return=None):
    """Build a fake asyncpg pool whose acquire() context manager returns a mock connection."""
    conn = AsyncMock()
    conn.fetchrow = AsyncMock(return_value=fetchrow_return)
    conn.fetch = AsyncMock(return_value=fetch_return or [])
    conn.execute = AsyncMock()

    pool = MagicMock()

    @asynccontextmanager
    async def _acquire():
        yield conn

    pool.acquire = _acquire
    pool._conn = conn  # expose for assertions
    return pool


@pytest.mark.asyncio
async def test_upsert_pr_review_returns_id():
    fake_row = {"id": 99}
    pool = _make_pool(fetchrow_return=fake_row)

    result = await repository.upsert_pr_review(pool, "PROJ", "my-repo", 42, "abc123", "alice", "My PR")

    assert result == 99
    pool._conn.fetchrow.assert_awaited_once()
    sql, *args = pool._conn.fetchrow.call_args.args
    assert "INSERT INTO pr_reviews" in sql
    assert "PROJ" in args
    assert "my-repo" in args
    assert 42 in args


@pytest.mark.asyncio
async def test_get_last_reviewed_commit_returns_value():
    fake_row = {"last_reviewed_commit": "deadbeef"}
    pool = _make_pool(fetchrow_return=fake_row)

    result = await repository.get_last_reviewed_commit(pool, "PROJ", "my-repo", 1)

    assert result == "deadbeef"


@pytest.mark.asyncio
async def test_get_last_reviewed_commit_returns_none_when_no_row():
    pool = _make_pool(fetchrow_return=None)

    result = await repository.get_last_reviewed_commit(pool, "PROJ", "my-repo", 999)

    assert result is None


@pytest.mark.asyncio
async def test_update_summary_comment_executes_update():
    pool = _make_pool()

    await repository.update_summary_comment(pool, pr_review_id=1, summary_comment_id=55, summary_comment_version=3)

    pool._conn.execute.assert_awaited_once()
    sql, *args = pool._conn.execute.call_args.args
    assert "UPDATE pr_reviews" in sql
    assert 1 in args
    assert 55 in args
    assert 3 in args


@pytest.mark.asyncio
async def test_get_summary_comment_info_returns_dict():
    fake_row = {"summary_comment_id": 55, "summary_comment_version": 3}
    pool = _make_pool(fetchrow_return=fake_row)

    result = await repository.get_summary_comment_info(pool, pr_review_id=1)

    assert result == {"summary_comment_id": 55, "summary_comment_version": 3}


@pytest.mark.asyncio
async def test_get_summary_comment_info_returns_none_when_no_comment_id():
    fake_row = {"summary_comment_id": None, "summary_comment_version": None}
    pool = _make_pool(fetchrow_return=fake_row)

    result = await repository.get_summary_comment_info(pool, pr_review_id=1)

    assert result is None


@pytest.mark.asyncio
async def test_delete_pr_review_executes_delete():
    pool = _make_pool()

    await repository.delete_pr_review(pool, "PROJ", "my-repo", 42)

    pool._conn.execute.assert_awaited_once()
    sql, *_args = pool._conn.execute.call_args.args
    assert "DELETE FROM pr_reviews" in sql


@pytest.mark.asyncio
async def test_insert_finding_executes_insert():
    pool = _make_pool()

    await repository.insert_finding(
        pool,
        pr_review_id=1,
        file_path="src/foo.py",
        line_number=10,
        severity="important",
        comment_text="Fix this",
        suggestion=None,
        bitbucket_comment_id=77,
        commit_sha="abc123",
        is_incremental=False,
    )

    pool._conn.execute.assert_awaited_once()
    sql, *args = pool._conn.execute.call_args.args
    assert "INSERT INTO review_findings" in sql
    assert 1 in args
    assert "src/foo.py" in args
    assert 10 in args


@pytest.mark.asyncio
async def test_get_existing_finding_keys_returns_set():
    fake_rows = [
        {"file_path": "a.py", "line_number": 1, "severity": "important"},
        {"file_path": "b.py", "line_number": 2, "severity": "critical"},
    ]
    pool = _make_pool(fetch_return=fake_rows)

    result = await repository.get_existing_finding_keys(pool, "PROJ", "my-repo", 42)

    assert result == {("a.py", 1, "important"), ("b.py", 2, "critical")}


@pytest.mark.asyncio
async def test_get_existing_finding_keys_empty():
    pool = _make_pool(fetch_return=[])

    result = await repository.get_existing_finding_keys(pool, "PROJ", "my-repo", 42)

    assert result == set()


@pytest.mark.asyncio
async def test_get_finding_by_comment_id_returns_dict():
    fake_row = {"file_path": "foo.py", "line_number": 5, "severity": "critical"}
    pool = _make_pool(fetchrow_return=fake_row)

    result = await repository.get_finding_by_comment_id(pool, 77)

    assert result == {"file_path": "foo.py", "line_number": 5, "severity": "critical"}


@pytest.mark.asyncio
async def test_get_finding_by_comment_id_returns_none():
    pool = _make_pool(fetchrow_return=None)

    result = await repository.get_finding_by_comment_id(pool, 999)

    assert result is None


@pytest.mark.asyncio
async def test_insert_review_stats_executes_insert_with_correct_param_count():
    pool = _make_pool()

    await repository.insert_review_stats(
        pool,
        project_key="PROJ",
        repo_slug="my-repo",
        pr_id=42,
        author="alice",
        is_incremental=False,
        reviewed_commit="abc123",
        diff_added=10,
        diff_removed=5,
        files_reviewed=3,
        total_files=4,
        critical_count=1,
        important_count=2,
        security_count=0,
        review_effort=15,
        prompt_tokens=1000,
        completion_tokens=200,
        model_name="gpt-5.3-codex",
        elapsed_seconds=3.14,
        cross_file_deps=0,
        skipped_files=1,
        content_skipped=0,
        findings_posted=3,
        findings_deduplicated=1,
    )

    pool._conn.execute.assert_awaited_once()
    sql, *args = pool._conn.execute.call_args.args
    assert "INSERT INTO review_statistics" in sql
    # 23 positional parameters: $1 through $23
    assert len(args) == 23
    assert args[0] == "PROJ"
    assert args[1] == "my-repo"
    assert args[2] == 42


def _make_tx_pool(execute_side_effect):
    """Build a fake pool whose connection supports both execute() and transaction()."""
    conn = AsyncMock()
    conn.execute = AsyncMock(side_effect=execute_side_effect)

    @asynccontextmanager
    async def _transaction():
        yield

    conn.transaction = _transaction

    pool = MagicMock()

    @asynccontextmanager
    async def _acquire():
        yield conn

    pool.acquire = _acquire
    pool._conn = conn
    return pool


@pytest.mark.asyncio
async def test_purge_pr_data_deletes_review_data_but_keeps_stats():
    pool = _make_tx_pool(["DELETE 3", "DELETE 1", "DELETE 1"])

    result = await repository.purge_pr_data(pool, "PROJ", "my-repo", 42)

    assert result == {
        "review_findings": 3,
        "pr_reviews": 1,
        "feedback_events": 1,
    }
    assert "review_statistics" not in result
    assert pool._conn.execute.await_count == 3
    calls = [call.args[0] for call in pool._conn.execute.call_args_list]
    assert "DELETE FROM review_findings" in calls[0]
    assert "DELETE FROM pr_reviews" in calls[1]
    assert "DELETE FROM feedback_events" in calls[2]


@pytest.mark.asyncio
async def test_purge_pr_data_zero_rows():
    pool = _make_tx_pool(["DELETE 0", "DELETE 0", "DELETE 0"])

    result = await repository.purge_pr_data(pool, "PROJ", "my-repo", 999)

    assert all(v == 0 for v in result.values())


@pytest.mark.asyncio
async def test_insert_feedback_executes_insert():
    pool = _make_pool()

    await repository.insert_feedback(
        pool,
        project_key="PROJ",
        repo_slug="my-repo",
        pr_id=42,
        bitbucket_comment_id=55,
        feedback_author="bob",
        classification="disagree",
        file_path="foo.py",
        severity="important",
    )

    pool._conn.execute.assert_awaited_once()
    sql, *args = pool._conn.execute.call_args.args
    assert "INSERT INTO feedback_events" in sql
    assert "bob" in args
    assert "disagree" in args
