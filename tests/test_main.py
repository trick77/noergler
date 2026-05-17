import asyncio
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
    })()
    mock_reviewer = AsyncMock()
    mock_reviewer.review_pull_request = AsyncMock()
    mock_reviewer.handle_mention = AsyncMock()
    mock_reviewer.handle_feedback = AsyncMock()
    mock_reviewer.handle_pr_merged = AsyncMock()
    mock_reviewer.handle_pr_declined = AsyncMock()
    mock_reviewer.handle_pr_deleted = AsyncMock()

    # A stub review queue that invokes the reviewer synchronously on submit,
    # so existing assertions on review_pull_request.await_count keep working
    # without waiting on a real worker.
    class _StubQueue:
        def submit(self, key, payload):
            asyncio.ensure_future(mock_reviewer.review_pull_request(payload))
            return "queued"

    original_config = main_module.config
    original_reviewer = main_module.reviewer
    original_queue = main_module.review_queue
    original_lifespan = app.router.lifespan_context

    @asynccontextmanager
    async def noop_lifespan(_a):
        yield

    main_module.config = mock_config
    main_module.reviewer = mock_reviewer
    main_module.review_queue = _StubQueue()
    app.router.lifespan_context = noop_lifespan
    try:
        with TestClient(app, raise_server_exceptions=False) as c:
            yield c
    finally:
        main_module.config = original_config
        main_module.reviewer = original_reviewer
        main_module.review_queue = original_queue
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


class TestDeclinedRouting:
    def test_pr_declined_routes_to_handle_pr_declined(self, client):
        payload = {**PR_PAYLOAD, "eventKey": "pr:declined"}
        body = json.dumps(payload).encode()
        resp = client.post(
            "/webhook",
            content=body,
            headers={"X-Hub-Signature": _sign(body), "Content-Type": "application/json"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "accepted"
        assert data["reason"] == "declined-rollup"


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
        # 'pr:reviewed' is intentionally not handled — used here to exercise
        # the unknown-event branch now that pr:declined has a real route.
        payload = {**PR_PAYLOAD, "eventKey": "pr:reviewed"}
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


class TestQueueIntegration:
    def test_rapid_from_ref_updates_are_bounded_via_queue(self):
        """Rapid pr:from_ref_updated events produce at most 2 reviews total.

        First submit always returns "queued". Subsequent submits arriving
        while the key is still pending in the queue return "superseded"; if
        the worker has already dequeued the first entry, a subsequent submit
        returns "queued" for the next (deduped) slot. Either way, three
        rapid submits result in no more than two actual review runs.
        """
        import time

        from app.review_queue import ReviewQueue

        async def slow_review(payload):
            await asyncio.sleep(0.15)

        mock_config = type("C", (), {
            "bitbucket": type("B", (), {"webhook_secret": WEBHOOK_SECRET, "username": "noergler"})(),
        })()
        mock_reviewer = AsyncMock()
        mock_reviewer.review_pull_request = AsyncMock(side_effect=slow_review)

        original_config = main_module.config
        original_reviewer = main_module.reviewer
        original_queue = main_module.review_queue
        original_lifespan = app.router.lifespan_context

        @asynccontextmanager
        async def lifespan_with_queue(_a):
            q = ReviewQueue(mock_reviewer.review_pull_request)
            q.start()
            main_module.review_queue = q
            try:
                yield
            finally:
                await q.stop()

        main_module.config = mock_config
        main_module.reviewer = mock_reviewer
        app.router.lifespan_context = lifespan_with_queue
        try:
            with TestClient(app, raise_server_exceptions=False) as c:
                payload = {**PR_PAYLOAD, "eventKey": "pr:from_ref_updated"}
                body = json.dumps(payload).encode()
                headers = {"X-Hub-Signature": _sign(body), "Content-Type": "application/json"}
                outcomes = [
                    c.post("/webhook", content=body, headers=headers).json()["queue"]
                    for _ in range(3)
                ]
                assert outcomes[0] == "queued"
                assert "superseded" in outcomes
                time.sleep(0.6)
                assert mock_reviewer.review_pull_request.await_count <= 2
        finally:
            main_module.config = original_config
            main_module.reviewer = original_reviewer
            main_module.review_queue = original_queue
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
