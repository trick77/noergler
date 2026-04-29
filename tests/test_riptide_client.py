"""Tests for app/riptide_client.py."""

from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal

import httpx
import pytest
import respx

from app.riptide_client import RiptideAuthError, RiptideClient


def _completed_kwargs():
    return dict(
        pr_key="PROJ/repo#1",
        repo="org/repo",
        commit_sha="abc1234567890abc1234567890abc1234567890a",
        run_id="42",
        model="gpt-4o",
        prompt_tokens=10,
        completion_tokens=20,
        elapsed_ms=1500,
        findings_count=2,
        cost_usd=Decimal("0.0123"),
        finished_at=datetime(2026, 4, 29, 12, 0, 0, tzinfo=timezone.utc),
    )


class TestEnabledFlag:
    def test_disabled_when_url_missing(self):
        assert RiptideClient(url="", token="t").enabled is False

    def test_disabled_when_token_missing(self):
        assert RiptideClient(url="http://x", token="").enabled is False

    def test_disabled_when_both_missing(self):
        assert RiptideClient.from_env(url="", token="").enabled is False

    def test_enabled_when_both_set(self):
        client = RiptideClient(url="http://x", token="t")
        assert client.enabled is True


class TestNoOpWhenDisabled:
    @pytest.mark.asyncio
    async def test_emit_completed_is_noop(self):
        client = RiptideClient(url="", token="")
        # Should not raise, should not require any HTTP mock.
        await client.emit_completed(**_completed_kwargs())

    @pytest.mark.asyncio
    async def test_emit_feedback_is_noop(self):
        client = RiptideClient(url="", token="")
        await client.emit_feedback(
            pr_key="PROJ/repo#1",
            finding_id="f1",
            verdict="disagreed",
            actor="alice",
            repo="org/repo",
            commit_sha=None,
            occurred_at=datetime(2026, 4, 29, tzinfo=timezone.utc),
        )

    @pytest.mark.asyncio
    async def test_verify_at_startup_is_noop_when_disabled(self):
        client = RiptideClient(url="", token="")
        assert await client.verify_at_startup() is None


class TestStartupVerification:
    @pytest.mark.asyncio
    @respx.mock
    async def test_returns_team_on_200(self):
        respx.get("http://r/auth/ping").mock(
            return_value=httpx.Response(200, json={"status": "ok", "team": "checkout"})
        )
        client = RiptideClient(url="http://r", token="t")
        try:
            assert await client.verify_at_startup() == "checkout"
        finally:
            await client.close()

    @pytest.mark.asyncio
    @respx.mock
    async def test_raises_on_401(self):
        respx.get("http://r/auth/ping").mock(
            return_value=httpx.Response(401, json={"detail": "no"})
        )
        client = RiptideClient(url="http://r", token="t")
        try:
            with pytest.raises(RiptideAuthError):
                await client.verify_at_startup()
        finally:
            await client.close()

    @pytest.mark.asyncio
    @respx.mock
    async def test_returns_none_on_connect_error(self):
        respx.get("http://r/auth/ping").mock(side_effect=httpx.ConnectError("boom"))
        client = RiptideClient(url="http://r", token="t")
        try:
            # Connection error is non-fatal — caller continues startup.
            assert await client.verify_at_startup() is None
        finally:
            await client.close()


class TestEmit:
    @pytest.mark.asyncio
    @respx.mock
    async def test_emit_completed_posts_expected_body(self):
        route = respx.post("http://r/webhooks/noergler").mock(
            return_value=httpx.Response(202, json={"status": "accepted"})
        )
        client = RiptideClient(url="http://r", token="t")
        try:
            await client.emit_completed(**_completed_kwargs())
        finally:
            await client.close()

        assert route.called
        sent = route.calls[0].request
        assert sent.headers["authorization"] == "Bearer t"
        assert sent.headers["content-type"] == "application/json"
        body = httpx.Response(200, content=sent.content).json()
        assert body["event_type"] == "completed"
        assert body["run_id"] == "42"
        assert body["cost_usd"] == "0.0123"
        assert body["finished_at"].endswith("Z")

    @pytest.mark.asyncio
    @respx.mock
    async def test_emit_completed_skipped_when_cost_none(self):
        route = respx.post("http://r/webhooks/noergler").mock(
            return_value=httpx.Response(202, json={"status": "accepted"})
        )
        client = RiptideClient(url="http://r", token="t")
        try:
            kwargs = _completed_kwargs()
            kwargs["cost_usd"] = None
            await client.emit_completed(**kwargs)
        finally:
            await client.close()

        assert not route.called

    @pytest.mark.asyncio
    @respx.mock
    async def test_emit_feedback_posts_expected_body(self):
        route = respx.post("http://r/webhooks/noergler").mock(
            return_value=httpx.Response(202)
        )
        client = RiptideClient(url="http://r", token="t")
        try:
            await client.emit_feedback(
                pr_key="PROJ/repo#1",
                finding_id="42",
                verdict="disagreed",
                actor="alice",
                repo="org/repo",
                commit_sha="abc1234567890abc1234567890abc1234567890a",
                occurred_at=datetime(2026, 4, 29, 12, 0, 0, tzinfo=timezone.utc),
            )
        finally:
            await client.close()

        assert route.called
        body = httpx.Response(200, content=route.calls[0].request.content).json()
        assert body["event_type"] == "feedback"
        assert body["finding_id"] == "42"
        assert body["verdict"] == "disagreed"
        assert body["commit_sha"] == "abc1234567890abc1234567890abc1234567890a"

    @pytest.mark.asyncio
    @respx.mock
    async def test_emit_swallows_connection_errors(self):
        respx.post("http://r/webhooks/noergler").mock(side_effect=httpx.ConnectError("nope"))
        client = RiptideClient(url="http://r", token="t")
        try:
            # Must not raise; emission is best-effort.
            await client.emit_completed(**_completed_kwargs())
        finally:
            await client.close()

    @pytest.mark.asyncio
    @respx.mock
    async def test_emit_logs_on_4xx_does_not_raise(self):
        respx.post("http://r/webhooks/noergler").mock(
            return_value=httpx.Response(422, json={"detail": "bad"})
        )
        client = RiptideClient(url="http://r", token="t")
        try:
            await client.emit_completed(**_completed_kwargs())
        finally:
            await client.close()
