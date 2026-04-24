"""Tests for the /metrics API in app/metrics.py."""
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any
from unittest.mock import AsyncMock, MagicMock

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.config import AppConfig, BitbucketConfig, DatabaseConfig, JiraConfig, LLMConfig, MetricsConfig
from app.metrics import router as metrics_router


def _app_config(*, api_key: str = "secret-key") -> AppConfig:
    return AppConfig(
        bitbucket=BitbucketConfig(base_url="u", token="t", webhook_secret="s", username="bot"),
        llm=LLMConfig(oauth_token="tok"),
        jira=JiraConfig(url="u", token="t"),
        database=DatabaseConfig(url="postgres://x"),
        metrics=MetricsConfig(api_key=api_key),
    )


def _mock_pool(rows: list[dict]) -> MagicMock:
    """Mock asyncpg pool returning `rows` from fetch()."""
    conn = AsyncMock()
    conn.fetch = AsyncMock(return_value=rows)

    @asynccontextmanager
    async def _acquire():
        yield conn

    pool = MagicMock()
    pool.acquire = _acquire
    pool._conn = conn
    return pool


_UNSET: Any = object()


def _make_app(*, api_key: str = "secret-key", rows: list[dict] | None = None, db_pool: Any = _UNSET) -> FastAPI:
    app = FastAPI()
    app.include_router(metrics_router)
    app.state.config = _app_config(api_key=api_key)
    app.state.db_pool = _mock_pool(rows or []) if db_pool is _UNSET else db_pool
    return app


def _make_client(*, api_key: str = "secret-key", rows: list[dict] | None = None, db_pool: Any = _UNSET) -> TestClient:
    return TestClient(_make_app(api_key=api_key, rows=rows, db_pool=db_pool))


class TestAuth:
    def test_missing_header_401(self):
        c = _make_client()
        r = c.get("/metrics/reviewer-precision")
        assert r.status_code == 401
        assert "invalid API key" in r.json()["detail"]

    def test_wrong_key_401(self):
        c = _make_client()
        r = c.get("/metrics/reviewer-precision", headers={"X-API-Key": "nope"})
        assert r.status_code == 401

    def test_empty_configured_key_503(self):
        c = _make_client(api_key="")
        r = c.get("/metrics/reviewer-precision", headers={"X-API-Key": "anything"})
        assert r.status_code == 503
        assert "disabled" in r.json()["detail"]

    def test_correct_key_200(self):
        c = _make_client()
        r = c.get("/metrics/reviewer-precision", headers={"X-API-Key": "secret-key"})
        assert r.status_code == 200

    def test_db_missing_503(self):
        c = _make_client(db_pool=None)
        r = c.get("/metrics/reviewer-precision", headers={"X-API-Key": "secret-key"})
        assert r.status_code == 503
        assert "database" in r.json()["detail"]


class TestReviewerPrecision:
    def test_returns_rows(self):
        rows = [
            {"project_key": "PROJ", "repo_slug": "r", "week": datetime(2026, 4, 20, tzinfo=timezone.utc),
             "n_posted": 10, "n_disagreed": 2, "precision": 0.8},
        ]
        c = _make_client(rows=rows)
        r = c.get("/metrics/reviewer-precision", headers={"X-API-Key": "secret-key"})
        assert r.status_code == 200
        body = r.json()
        assert body["count"] == 1
        assert body["rows"][0]["precision"] == 0.8
        assert body["rows"][0]["week"].startswith("2026-04-20")

    def test_filters_compose(self):
        app = _make_app(rows=[])
        c = TestClient(app)
        r = c.get(
            "/metrics/reviewer-precision",
            params={"project_key": "PROJ", "repo_slug": "r",
                    "since": "2026-04-01T00:00:00Z", "until": "2026-05-01T00:00:00Z"},
            headers={"X-API-Key": "secret-key"},
        )
        assert r.status_code == 200
        conn = app.state.db_pool._conn
        conn.fetch.assert_awaited_once()
        sql, *args = conn.fetch.call_args.args
        assert "project_key = $1" in sql
        assert "repo_slug = $2" in sql
        assert "week >= $3" in sql
        assert "week < $4" in sql
        assert args[0] == "PROJ"
        assert args[1] == "r"

    def test_no_filters_produces_clean_sql(self):
        app = _make_app(rows=[])
        c = TestClient(app)
        r = c.get("/metrics/reviewer-precision", headers={"X-API-Key": "secret-key"})
        assert r.status_code == 200
        sql, *args = app.state.db_pool._conn.fetch.call_args.args
        assert "WHERE" not in sql.upper().replace("WHERE  LIMIT", "LIMIT")
        # Only one positional arg (the limit)
        assert len(args) == 1
        assert args[0] == 1000  # default limit

    def test_empty_result_returns_zero_count(self):
        c = _make_client(rows=[])
        r = c.get("/metrics/reviewer-precision", headers={"X-API-Key": "secret-key"})
        assert r.status_code == 200
        assert r.json() == {"count": 0, "rows": []}

    def test_limit_bounded(self):
        c = _make_client(rows=[])
        r = c.get(
            "/metrics/reviewer-precision",
            params={"limit": 99999},
            headers={"X-API-Key": "secret-key"},
        )
        # FastAPI validates le=10000 and returns 422 for too-large limit
        assert r.status_code == 422


class TestLeadTime:
    def test_returns_rows(self):
        rows = [
            {"project_key": "PROJ", "repo_slug": "r", "pr_id": 42, "author": "dev",
             "opened_at": datetime(2026, 4, 22, tzinfo=timezone.utc),
             "merged_at": datetime(2026, 4, 23, tzinfo=timezone.utc),
             "lead_time_seconds": 86400},
        ]
        c = _make_client(rows=rows)
        r = c.get("/metrics/lead-time", headers={"X-API-Key": "secret-key"})
        assert r.status_code == 200
        assert r.json()["rows"][0]["lead_time_seconds"] == 86400
        assert r.json()["rows"][0]["opened_at"].startswith("2026-04-22")

    def test_filters_by_author(self):
        app = _make_app(rows=[])
        c = TestClient(app)
        r = c.get(
            "/metrics/lead-time",
            params={"author": "jan"},
            headers={"X-API-Key": "secret-key"},
        )
        assert r.status_code == 200
        sql, *args = app.state.db_pool._conn.fetch.call_args.args
        assert "author = $1" in sql
        assert args[0] == "jan"


class TestActivity:
    def test_returns_rows(self):
        rows = [{"author": "dev", "week": datetime(2026, 4, 20, tzinfo=timezone.utc),
                 "prs": 3, "review_runs": 7}]
        c = _make_client(rows=rows)
        r = c.get("/metrics/activity", headers={"X-API-Key": "secret-key"})
        assert r.status_code == 200
        assert r.json()["rows"][0]["prs"] == 3
        assert r.json()["rows"][0]["week"].startswith("2026-04-20")

    def test_author_and_date_filters(self):
        app = _make_app(rows=[])
        c = TestClient(app)
        r = c.get(
            "/metrics/activity",
            params={"author": "jan", "since": "2026-04-01T00:00:00Z"},
            headers={"X-API-Key": "secret-key"},
        )
        assert r.status_code == 200
        sql, *args = app.state.db_pool._conn.fetch.call_args.args
        assert "author = $1" in sql
        assert "week >= $2" in sql


class TestCostByModel:
    def test_returns_rows(self):
        rows = [{"model_name": "gpt-5.4", "week": datetime(2026, 4, 20, tzinfo=timezone.utc),
                 "runs": 50, "prompt_tokens": 100000, "completion_tokens": 20000,
                 "total_tokens": 120000, "avg_elapsed_seconds": 2.5}]
        c = _make_client(rows=rows)
        r = c.get("/metrics/cost-by-model", headers={"X-API-Key": "secret-key"})
        assert r.status_code == 200
        assert r.json()["rows"][0]["model_name"] == "gpt-5.4"

    def test_model_filter(self):
        app = _make_app(rows=[])
        c = TestClient(app)
        r = c.get(
            "/metrics/cost-by-model",
            params={"model": "gpt-5.4"},
            headers={"X-API-Key": "secret-key"},
        )
        assert r.status_code == 200
        sql, *args = app.state.db_pool._conn.fetch.call_args.args
        assert "model_name = $1" in sql
        assert args[0] == "gpt-5.4"


class TestLimit:
    def test_default_limit_is_1000(self):
        app = _make_app(rows=[])
        c = TestClient(app)
        r = c.get("/metrics/activity", headers={"X-API-Key": "secret-key"})
        assert r.status_code == 200
        _sql, *args = app.state.db_pool._conn.fetch.call_args.args
        assert args[-1] == 1000

    def test_custom_limit_passed_through(self):
        app = _make_app(rows=[])
        c = TestClient(app)
        r = c.get(
            "/metrics/activity",
            params={"limit": 50},
            headers={"X-API-Key": "secret-key"},
        )
        assert r.status_code == 200
        _sql, *args = app.state.db_pool._conn.fetch.call_args.args
        assert args[-1] == 50

    def test_limit_zero_rejected(self):
        c = _make_client(rows=[])
        r = c.get(
            "/metrics/activity",
            params={"limit": 0},
            headers={"X-API-Key": "secret-key"},
        )
        assert r.status_code == 422

    def test_limit_over_max_rejected(self):
        c = _make_client(rows=[])
        r = c.get(
            "/metrics/lead-time",
            params={"limit": 999999},
            headers={"X-API-Key": "secret-key"},
        )
        assert r.status_code == 422


class TestConfigConstant:
    def test_metrics_config_default_empty(self):
        from app.config import MetricsConfig
        assert MetricsConfig().api_key == ""

    def test_metrics_is_in_secret_fields(self):
        from app.config import _SECRET_FIELDS
        assert "api_key" in _SECRET_FIELDS["metrics"]
