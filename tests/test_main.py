import hashlib
import hmac
import json
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock

import pytest
from fastapi.testclient import TestClient

import app.main as main_module
from app.main import app


WEBHOOK_SECRET = "test-secret"

PR_PAYLOAD = {
    "eventKey": "pr:opened",
    "pullRequest": {
        "id": 1,
        "title": "Test PR",
        "fromRef": {
            "id": "refs/heads/feature",
            "displayId": "feature",
            "latestCommit": "abc123",
            "repository": {"slug": "repo", "project": {"key": "PROJ"}},
        },
        "toRef": {
            "id": "refs/heads/main",
            "displayId": "main",
            "latestCommit": "def456",
            "repository": {"slug": "repo", "project": {"key": "PROJ"}},
        },
        "author": {"user": {"name": "jan"}},
    },
}

COMMENT_MENTION_PAYLOAD = {
    "eventKey": "pr:comment:added",
    "comment": {"id": 100, "text": "@noergler explain this", "author": {"name": "someone"}},
    "pullRequest": PR_PAYLOAD["pullRequest"],
}

COMMENT_NO_MENTION_PAYLOAD = {
    "eventKey": "pr:comment:added",
    "comment": {"id": 101, "text": "just a regular comment", "author": {"name": "someone"}},
    "pullRequest": PR_PAYLOAD["pullRequest"],
}

COMMENT_REPLY_PAYLOAD = {
    "eventKey": "pr:comment:added",
    "commentParentId": 100,
    "comment": {
        "id": 200, "text": "\U0001f44d", "author": {"name": "dev"},
    },
    "pullRequest": PR_PAYLOAD["pullRequest"],
}

COMMENT_REPLY_WITH_MENTION_PAYLOAD = {
    "eventKey": "pr:comment:added",
    "commentParentId": 100,
    "comment": {
        "id": 201, "text": "@noergler explain this", "author": {"name": "dev"},
    },
    "pullRequest": PR_PAYLOAD["pullRequest"],
}

NON_PR_PAYLOAD = {"eventKey": "repo:refs_changed"}


def _sign(body: bytes, secret: str = WEBHOOK_SECRET) -> str:
    digest = hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()
    return f"sha256={digest}"


@pytest.fixture()
def client():
    mock_config = type("C", (), {
        "bitbucket": type("B", (), {"webhook_secret": WEBHOOK_SECRET, "username": "noergler"})(),
        "review": type("R", (), {"debounce_seconds": 0.0})(),
    })()
    mock_reviewer = AsyncMock()
    mock_reviewer.review_pull_request = AsyncMock()
    mock_reviewer.handle_mention = AsyncMock()
    mock_reviewer.handle_feedback = AsyncMock()
    mock_reviewer.handle_pr_merged = AsyncMock()
    mock_reviewer.handle_pr_deleted = AsyncMock()

    from app.debounce import PRDebouncer
    mock_debouncer = PRDebouncer(delay_seconds=0.0)

    original_config = main_module.config
    original_reviewer = main_module.reviewer
    original_debouncer = main_module.pr_debouncer
    original_lifespan = app.router.lifespan_context

    @asynccontextmanager
    async def noop_lifespan(_a):
        yield

    main_module.config = mock_config
    main_module.reviewer = mock_reviewer
    main_module.pr_debouncer = mock_debouncer
    app.router.lifespan_context = noop_lifespan
    try:
        with TestClient(app, raise_server_exceptions=False) as c:
            yield c
    finally:
        main_module.config = original_config
        main_module.reviewer = original_reviewer
        main_module.pr_debouncer = original_debouncer
        app.router.lifespan_context = original_lifespan


class TestWebhookSignature:
    def test_valid_signature_pr_event(self, client):
        body = json.dumps(PR_PAYLOAD).encode()
        resp = client.post(
            "/webhook",
            content=body,
            headers={"X-Hub-Signature": _sign(body), "Content-Type": "application/json"},
        )
        assert resp.status_code == 200
        assert resp.json()["status"] == "accepted"

    def test_missing_signature_header_returns_401(self, client):
        body = json.dumps(PR_PAYLOAD).encode()
        resp = client.post(
            "/webhook",
            content=body,
            headers={"Content-Type": "application/json"},
        )
        assert resp.status_code == 401

    def test_invalid_signature_returns_401(self, client):
        body = json.dumps(PR_PAYLOAD).encode()
        resp = client.post(
            "/webhook",
            content=body,
            headers={"X-Hub-Signature": "sha256=bad", "Content-Type": "application/json"},
        )
        assert resp.status_code == 401

    def test_diagnostic_ping_returns_200_without_signature(self, client):
        resp = client.post(
            "/webhook",
            content=b"{}",
            headers={"X-Event-Key": "diagnostics:ping", "Content-Type": "application/json"},
        )
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"

    def test_test_connection_returns_200_without_signature_or_event_key(self, client):
        resp = client.post(
            "/webhook",
            content=b"{}",
            headers={"Content-Type": "application/json"},
        )
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"

    def test_non_pr_event_ignored(self, client):
        body = json.dumps(NON_PR_PAYLOAD).encode()
        resp = client.post(
            "/webhook",
            content=body,
            headers={"X-Hub-Signature": _sign(body), "Content-Type": "application/json"},
        )
        assert resp.status_code == 200
        assert resp.json()["status"] == "ignored"


class TestMentionRouting:
    def test_comment_with_mention_accepted(self, client):
        body = json.dumps(COMMENT_MENTION_PAYLOAD).encode()
        resp = client.post(
            "/webhook",
            content=body,
            headers={"X-Hub-Signature": _sign(body), "Content-Type": "application/json"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "accepted"
        assert data["reason"] == "mention"

    def test_comment_without_mention_ignored(self, client):
        body = json.dumps(COMMENT_NO_MENTION_PAYLOAD).encode()
        resp = client.post(
            "/webhook",
            content=body,
            headers={"X-Hub-Signature": _sign(body), "Content-Type": "application/json"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ignored"
        assert data["reason"] == "comment without mention"


class TestMergedRouting:
    def test_pr_merged_routes_to_handle_pr_merged(self, client):
        payload = {**PR_PAYLOAD, "eventKey": "pr:merged"}
        body = json.dumps(payload).encode()
        resp = client.post(
            "/webhook",
            content=body,
            headers={"X-Hub-Signature": _sign(body), "Content-Type": "application/json"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "accepted"
        assert data["reason"] == "merged-stats"


class TestDeletedRouting:
    def test_pr_deleted_routes_to_handle_pr_deleted(self, client):
        payload = {**PR_PAYLOAD, "eventKey": "pr:deleted"}
        body = json.dumps(payload).encode()
        resp = client.post(
            "/webhook",
            content=body,
            headers={"X-Hub-Signature": _sign(body), "Content-Type": "application/json"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "accepted"
        assert data["reason"] == "deleted-purge"


class TestEventKeyAllowList:
    def test_unhandled_pr_event_returns_ignored_with_warning(self, client):
        payload = {**PR_PAYLOAD, "eventKey": "pr:declined"}
        body = json.dumps(payload).encode()
        resp = client.post(
            "/webhook",
            content=body,
            headers={"X-Hub-Signature": _sign(body), "Content-Type": "application/json"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ignored"
        assert "unhandled event" in data["reason"]

    def test_pr_from_ref_updated_triggers_review(self, client):
        payload = {**PR_PAYLOAD, "eventKey": "pr:from_ref_updated"}
        body = json.dumps(payload).encode()
        resp = client.post(
            "/webhook",
            content=body,
            headers={"X-Hub-Signature": _sign(body), "Content-Type": "application/json"},
        )
        assert resp.status_code == 200
        assert resp.json()["status"] == "accepted"


class TestDebounceIntegration:
    def test_rapid_from_ref_updates_coalesce_to_single_review(self):
        """Two pr:from_ref_updated events within the debounce window → one review."""
        from app.debounce import PRDebouncer

        mock_config = type("C", (), {
            "bitbucket": type("B", (), {"webhook_secret": WEBHOOK_SECRET, "username": "noergler"})(),
            "review": type("R", (), {"debounce_seconds": 0.1})(),
        })()
        mock_reviewer = AsyncMock()
        mock_reviewer.review_pull_request = AsyncMock()

        original_config = main_module.config
        original_reviewer = main_module.reviewer
        original_debouncer = main_module.pr_debouncer
        original_lifespan = app.router.lifespan_context

        @asynccontextmanager
        async def noop_lifespan(_a):
            yield

        main_module.config = mock_config
        main_module.reviewer = mock_reviewer
        main_module.pr_debouncer = PRDebouncer(delay_seconds=0.1)
        app.router.lifespan_context = noop_lifespan
        import time
        try:
            with TestClient(app, raise_server_exceptions=False) as c:
                payload = {**PR_PAYLOAD, "eventKey": "pr:from_ref_updated"}
                body = json.dumps(payload).encode()
                headers = {"X-Hub-Signature": _sign(body), "Content-Type": "application/json"}
                r1 = c.post("/webhook", content=body, headers=headers)
                r2 = c.post("/webhook", content=body, headers=headers)
                assert r1.status_code == 200 and r2.status_code == 200
                # TestClient runs the app on a background loop in another thread,
                # so time.sleep lets that loop drain the debounced task.
                time.sleep(0.3)
                assert mock_reviewer.review_pull_request.await_count == 1
        finally:
            main_module.config = original_config
            main_module.reviewer = original_reviewer
            main_module.pr_debouncer = original_debouncer
            app.router.lifespan_context = original_lifespan


class TestFeedbackRouting:
    def test_reply_with_parent_routes_to_feedback(self, client):
        body = json.dumps(COMMENT_REPLY_PAYLOAD).encode()
        resp = client.post(
            "/webhook",
            content=body,
            headers={"X-Hub-Signature": _sign(body), "Content-Type": "application/json"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "accepted"
        assert data["reason"] == "feedback"

    def test_mention_takes_priority_over_parent(self, client):
        body = json.dumps(COMMENT_REPLY_WITH_MENTION_PAYLOAD).encode()
        resp = client.post(
            "/webhook",
            content=body,
            headers={"X-Hub-Signature": _sign(body), "Content-Type": "application/json"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "accepted"
        assert data["reason"] == "mention"

    def test_comment_without_parent_or_mention_ignored(self, client):
        body = json.dumps(COMMENT_NO_MENTION_PAYLOAD).encode()
        resp = client.post(
            "/webhook",
            content=body,
            headers={"X-Hub-Signature": _sign(body), "Content-Type": "application/json"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ignored"
        assert data["reason"] == "comment without mention"


class TestMentionRoutingCaseSensitivity:
    def test_comment_mention_case_insensitive(self, client):
        payload = {**COMMENT_MENTION_PAYLOAD, "comment": {
            "id": 102, "text": "@NOERGLER explain this", "author": {"name": "someone"},
        }}
        body = json.dumps(payload).encode()
        resp = client.post(
            "/webhook",
            content=body,
            headers={"X-Hub-Signature": _sign(body), "Content-Type": "application/json"},
        )
        assert resp.status_code == 200
        assert resp.json()["status"] == "accepted"
        assert resp.json()["reason"] == "mention"
