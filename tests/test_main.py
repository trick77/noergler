import asyncio
import hashlib
import hmac
import json
from contextlib import ExitStack, asynccontextmanager, contextmanager
from typing import cast
from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

import app.main as main_module
from app.config import JiraConfig, LLMConfig, ProjectScope, ReviewConfig, TeamConfig
from app.jira import JiraClient
from app.llm_client import LLMClient
from app.main import TeamRuntime, app
from app.riptide_client import RiptideAuthError, RiptideClient


WEBHOOK_SECRET = "test-secret"
TEAM = "platform"
OTHER_TEAM = "payments"
OTHER_SECRET = "other-secret"
WEBHOOK = f"/webhook/{TEAM}"

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


def _team_config(slug: str, secret: str, projects: list[ProjectScope]) -> TeamConfig:
    llm = LLMConfig(api_key="k", api_url="https://llm.test/v1", catalog_url="https://catalog.test/c.json")
    return TeamConfig(
        slug=slug, name=slug, webhook_secret=secret, projects=projects,
        llm=llm, review=ReviewConfig(), jira=JiraConfig(url="https://jira.test", token="t"),
    )


def _runtime(slug: str, secret: str, projects: list[ProjectScope], reviewer) -> TeamRuntime:
    # Only `config` and `reviewer` are read by the webhook route; the clients
    # are never touched in these tests.
    return TeamRuntime(
        config=_team_config(slug, secret, projects),
        llm=cast(LLMClient, cast(object, None)),
        jira=cast(JiraClient, cast(object, None)),
        riptide=cast(RiptideClient, cast(object, None)),
        reviewer=reviewer,
    )


def _mock_reviewer():
    mock_reviewer = AsyncMock()
    mock_reviewer.review_pull_request = AsyncMock()
    mock_reviewer.handle_mention = AsyncMock()
    mock_reviewer.handle_pr_merged = AsyncMock()
    mock_reviewer.handle_pr_declined = AsyncMock()
    mock_reviewer.handle_pr_deleted = AsyncMock()
    mock_reviewer.handle_comment_deleted = AsyncMock()
    return mock_reviewer


@pytest.fixture()
def client():
    mock_config = type("C", (), {
        "bitbucket": type("B", (), {"username": "noergler"})(),
    })()
    mock_reviewer = _mock_reviewer()
    other_reviewer = _mock_reviewer()

    # A stub review queue that invokes the reviewer synchronously on submit,
    # so existing assertions on review_pull_request.await_count keep working
    # without waiting on a real worker.
    class _StubQueue:
        def submit(self, key, payload, team):
            asyncio.ensure_future(main_module.teams[team].reviewer.review_pull_request(payload))
            return "queued"

    original_config = main_module.config
    original_queue = main_module.review_queue
    original_lifespan = app.router.lifespan_context

    @asynccontextmanager
    async def noop_lifespan(_a):
        yield

    main_module.config = mock_config
    main_module.teams.clear()
    main_module.teams[TEAM] = _runtime(
        TEAM, WEBHOOK_SECRET, [ProjectScope(key="PROJ")], mock_reviewer,
    )
    main_module.teams[OTHER_TEAM] = _runtime(
        OTHER_TEAM, OTHER_SECRET, [ProjectScope(key="PAY", repos=["billing"])], other_reviewer,
    )
    main_module.disabled_teams.clear()
    main_module.disabled_teams["broken"] = "TEAM_BROKEN_OPENAI_API_KEY is not set"
    main_module.review_queue = _StubQueue()
    app.router.lifespan_context = noop_lifespan
    try:
        with TestClient(app, raise_server_exceptions=False) as c:
            c.reviewer = mock_reviewer  # pyright: ignore[reportAttributeAccessIssue]
            c.other_reviewer = other_reviewer  # pyright: ignore[reportAttributeAccessIssue]
            yield c
    finally:
        main_module.config = original_config
        main_module.teams.clear()
        main_module.disabled_teams.clear()
        main_module.review_queue = original_queue
        app.router.lifespan_context = original_lifespan


class TestTeamRouting:
    def test_unknown_team_returns_404(self, client):
        body = json.dumps(PR_PAYLOAD).encode()
        resp = client.post(
            "/webhook/nobody",
            content=body,
            headers={"X-Hub-Signature": _sign(body), "Content-Type": "application/json"},
        )
        assert resp.status_code == 404

    def test_disabled_team_returns_503_with_reason(self, client):
        body = json.dumps(PR_PAYLOAD).encode()
        resp = client.post(
            "/webhook/broken",
            content=body,
            headers={"X-Hub-Signature": _sign(body), "Content-Type": "application/json"},
        )
        assert resp.status_code == 503
        # the reason is for the log, not for an unauthenticated caller
        assert "TEAM_BROKEN_OPENAI_API_KEY" not in resp.json()["detail"]
        assert "disabled" in resp.json()["detail"]

    def test_signature_is_checked_against_the_path_team(self, client):
        # Signed with the other team's secret, sent to this team's path.
        body = json.dumps(PR_PAYLOAD).encode()
        resp = client.post(
            WEBHOOK,
            content=body,
            headers={"X-Hub-Signature": _sign(body, OTHER_SECRET), "Content-Type": "application/json"},
        )
        assert resp.status_code == 401

    def test_foreign_repository_returns_403(self, client):
        # Valid signature for `payments`, but the payload names PROJ/repo,
        # which belongs to `platform`.
        body = json.dumps(PR_PAYLOAD).encode()
        resp = client.post(
            f"/webhook/{OTHER_TEAM}",
            content=body,
            headers={"X-Hub-Signature": _sign(body, OTHER_SECRET), "Content-Type": "application/json"},
        )
        assert resp.status_code == 403
        assert client.other_reviewer.review_pull_request.await_count == 0

    def test_repo_outside_restricted_list_returns_403(self, client):
        payload = json.loads(json.dumps(PR_PAYLOAD))
        for ref in ("fromRef", "toRef"):
            payload["pullRequest"][ref]["repository"] = {"slug": "other", "project": {"key": "PAY"}}
        body = json.dumps(payload).encode()
        resp = client.post(
            f"/webhook/{OTHER_TEAM}",
            content=body,
            headers={"X-Hub-Signature": _sign(body, OTHER_SECRET), "Content-Type": "application/json"},
        )
        assert resp.status_code == 403

    def test_owned_repo_routes_to_that_team(self, client):
        payload = json.loads(json.dumps(PR_PAYLOAD))
        for ref in ("fromRef", "toRef"):
            payload["pullRequest"][ref]["repository"] = {"slug": "billing", "project": {"key": "PAY"}}
        body = json.dumps(payload).encode()
        resp = client.post(
            f"/webhook/{OTHER_TEAM}",
            content=body,
            headers={"X-Hub-Signature": _sign(body, OTHER_SECRET), "Content-Type": "application/json"},
        )
        assert resp.status_code == 200
        assert client.other_reviewer.review_pull_request.await_count == 1
        assert client.reviewer.review_pull_request.await_count == 0

    def test_mention_on_disabled_team_is_rejected_without_posting(self, client):
        body = json.dumps(COMMENT_MENTION_PAYLOAD).encode()
        resp = client.post(
            "/webhook/broken",
            content=body,
            headers={"X-Hub-Signature": _sign(body), "Content-Type": "application/json"},
        )
        assert resp.status_code == 503
        assert client.reviewer.handle_mention.await_count == 0


class TestProbes:
    def test_health_is_200_and_lists_teams(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["teams"]["enabled"] == [OTHER_TEAM, TEAM]
        assert resp.json()["teams"]["disabled"] == ["broken"]
        assert "TEAM_BROKEN_OPENAI_API_KEY" not in resp.text

    def test_ready_is_200_with_an_enabled_team(self, client):
        assert client.get("/ready").status_code == 200

    def test_ready_is_503_with_no_enabled_team(self, client):
        main_module.teams.clear()
        assert client.get("/ready").status_code == 503
        assert client.get("/health").status_code == 200


class TestWebhookSignature:
    def test_valid_signature_pr_event(self, client):
        body = json.dumps(PR_PAYLOAD).encode()
        resp = client.post(
            WEBHOOK,
            content=body,
            headers={"X-Hub-Signature": _sign(body), "Content-Type": "application/json"},
        )
        assert resp.status_code == 200
        assert resp.json()["status"] == "accepted"

    def test_missing_signature_header_returns_401(self, client):
        body = json.dumps(PR_PAYLOAD).encode()
        resp = client.post(
            WEBHOOK,
            content=body,
            headers={"Content-Type": "application/json"},
        )
        assert resp.status_code == 401

    def test_invalid_signature_returns_401(self, client):
        body = json.dumps(PR_PAYLOAD).encode()
        resp = client.post(
            WEBHOOK,
            content=body,
            headers={"X-Hub-Signature": "sha256=bad", "Content-Type": "application/json"},
        )
        assert resp.status_code == 401

    def test_diagnostic_ping_returns_200_without_signature(self, client):
        resp = client.post(
            WEBHOOK,
            content=b"{}",
            headers={"X-Event-Key": "diagnostics:ping", "Content-Type": "application/json"},
        )
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"

    def test_test_connection_returns_200_without_signature_or_event_key(self, client):
        resp = client.post(
            WEBHOOK,
            content=b"{}",
            headers={"Content-Type": "application/json"},
        )
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"

    def test_non_pr_event_ignored(self, client):
        body = json.dumps(NON_PR_PAYLOAD).encode()
        resp = client.post(
            WEBHOOK,
            content=body,
            headers={"X-Hub-Signature": _sign(body), "Content-Type": "application/json"},
        )
        assert resp.status_code == 200
        assert resp.json()["status"] == "ignored"


class TestMentionRouting:
    def test_comment_with_mention_accepted(self, client):
        body = json.dumps(COMMENT_MENTION_PAYLOAD).encode()
        resp = client.post(
            WEBHOOK,
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
            WEBHOOK,
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
            WEBHOOK,
            content=body,
            headers={"X-Hub-Signature": _sign(body), "Content-Type": "application/json"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "accepted"
        assert data["reason"] == "merged-rollup"


class TestDeclinedRouting:
    def test_pr_declined_routes_to_handle_pr_declined(self, client):
        payload = {**PR_PAYLOAD, "eventKey": "pr:declined"}
        body = json.dumps(payload).encode()
        resp = client.post(
            WEBHOOK,
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
            WEBHOOK,
            content=body,
            headers={"X-Hub-Signature": _sign(body), "Content-Type": "application/json"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "accepted"
        assert data["reason"] == "deleted-purge"

    def test_pr_comment_deleted_routes_to_handle_comment_deleted(self, client):
        payload = {**PR_PAYLOAD, "eventKey": "pr:comment:deleted",
                   "comment": {"id": 55, "text": "", "author": {"name": "user"}}}
        body = json.dumps(payload).encode()
        resp = client.post(
            WEBHOOK,
            content=body,
            headers={"X-Hub-Signature": _sign(body), "Content-Type": "application/json"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "accepted"
        assert data["reason"] == "comment-deleted"


class TestEventKeyAllowList:
    def test_unhandled_pr_event_returns_ignored_with_warning(self, client):
        # 'pr:reviewed' is intentionally not handled — used here to exercise
        # the unknown-event branch now that pr:declined has a real route.
        payload = {**PR_PAYLOAD, "eventKey": "pr:reviewed"}
        body = json.dumps(payload).encode()
        resp = client.post(
            WEBHOOK,
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
            WEBHOOK,
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
            "bitbucket": type("B", (), {"username": "noergler"})(),
        })()
        mock_reviewer = AsyncMock()
        mock_reviewer.review_pull_request = AsyncMock(side_effect=slow_review)

        original_config = main_module.config
        original_queue = main_module.review_queue
        original_lifespan = app.router.lifespan_context

        @asynccontextmanager
        async def lifespan_with_queue(_a):
            q = ReviewQueue(main_module._review_for_team)
            q.start()
            main_module.review_queue = q
            try:
                yield
            finally:
                await q.stop()

        main_module.config = mock_config
        main_module.teams.clear()
        main_module.teams[TEAM] = _runtime(
            TEAM, WEBHOOK_SECRET, [ProjectScope(key="PROJ")], mock_reviewer,
        )
        app.router.lifespan_context = lifespan_with_queue
        try:
            with TestClient(app, raise_server_exceptions=False) as c:
                payload = {**PR_PAYLOAD, "eventKey": "pr:from_ref_updated"}
                body = json.dumps(payload).encode()
                headers = {"X-Hub-Signature": _sign(body), "Content-Type": "application/json"}
                outcomes = [
                    c.post(WEBHOOK, content=body, headers=headers).json()["queue"]
                    for _ in range(3)
                ]
                assert outcomes[0] == "queued"
                assert "superseded" in outcomes
                time.sleep(0.6)
                assert mock_reviewer.review_pull_request.await_count <= 2
        finally:
            main_module.config = original_config
            main_module.teams.clear()
            main_module.review_queue = original_queue
            app.router.lifespan_context = original_lifespan


class TestCommentReplyRouting:
    def test_mention_takes_priority_over_parent(self, client):
        body = json.dumps(COMMENT_REPLY_WITH_MENTION_PAYLOAD).encode()
        resp = client.post(
            WEBHOOK,
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
            WEBHOOK,
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
            WEBHOOK,
            content=body,
            headers={"X-Hub-Signature": _sign(body), "Content-Type": "application/json"},
        )
        assert resp.status_code == 200
        assert resp.json()["status"] == "accepted"
        assert resp.json()["reason"] == "mention"


class TestLifespan:
    """The real startup with the shared layer mocked: a per-team failure
    disables that team only, and the LLM client each team gets carries that
    team's key."""

    TEAMS = """
teams:
  - slug: platform
    webhook_secret_env: TEAM_PLATFORM_WEBHOOK_SECRET
    projects: [{key: PLAT}]
    inference: {api_key_env: TEAM_PLATFORM_OPENAI_API_KEY}
    riptide: {url: https://riptide.test, token_env: TEAM_PLATFORM_RIPTIDE_TOKEN}
  - slug: payments
    webhook_secret_env: TEAM_PAYMENTS_WEBHOOK_SECRET
    projects: [{key: PAY}]
    inference: {api_key_env: TEAM_PAYMENTS_OPENAI_API_KEY}
"""

    @pytest.fixture
    def env(self, monkeypatch, tmp_path):
        teams = tmp_path / "teams.yaml"
        teams.write_text(self.TEAMS)
        for k, v in {
            "BITBUCKET_URL": "https://bb.test", "BITBUCKET_TOKEN": "t", "BITBUCKET_USERNAME": "noergler",
            "OPENAI_BASE_URL": "https://llm.test/v1", "MODEL_CATALOG_URL": "https://catalog.test/c.json",
            "JIRA_URL": "https://jira.test", "JIRA_TOKEN": "j", "DATABASE_URL": "postgresql://u:p@db/x",
            "TEAMS_CONFIG": str(teams),
            "TEAM_PLATFORM_WEBHOOK_SECRET": "plat-secret", "TEAM_PLATFORM_OPENAI_API_KEY": "plat-key",
            "TEAM_PLATFORM_RIPTIDE_TOKEN": "rt",
            "TEAM_PAYMENTS_WEBHOOK_SECRET": "pay-secret", "TEAM_PAYMENTS_OPENAI_API_KEY": "pay-key",
        }.items():
            monkeypatch.setenv(k, v)

    @contextmanager
    def _boot(self, llm_check, riptide_check):
        with ExitStack() as stack:
            stack.enter_context(patch("app.main.create_pool", new=AsyncMock(return_value=object())))
            stack.enter_context(patch("app.main.close_pool", new=AsyncMock()))
            stack.enter_context(patch("app.bitbucket.BitbucketClient.check_connectivity", new=AsyncMock()))
            stack.enter_context(patch("app.jira.JiraClient.check_connectivity", new=AsyncMock()))
            stack.enter_context(patch("app.llm_client.LLMClient.check_connectivity", new=llm_check))
            stack.enter_context(patch("app.riptide_client.RiptideClient.verify_at_startup", new=riptide_check))
            with TestClient(app) as c:
                yield c
            main_module.teams.clear()
            main_module.disabled_teams.clear()

    def test_both_teams_start_with_their_own_keys(self, env):
        keys: list[str] = []

        async def llm_check(self):
            keys.append(self.config.api_key)

        with self._boot(llm_check, AsyncMock(return_value="platform")) as c:
            assert sorted(main_module.teams) == ["payments", "platform"]
            assert main_module.disabled_teams == {}
            assert main_module.teams["platform"].llm.config.api_key == "plat-key"
            assert main_module.teams["payments"].llm.config.api_key == "pay-key"
            assert main_module.teams["platform"].riptide.enabled is True
            assert main_module.teams["payments"].riptide.enabled is False
            assert c.get("/ready").status_code == 200
        assert sorted(keys) == ["pay-key", "plat-key"]

    def test_llm_failure_disables_that_team_only(self, env):
        async def llm_check(self):
            if self.config.api_key == "pay-key":
                raise RuntimeError("HTTP 401 from gateway")

        with patch("app.llm_client.LLMClient.close", new_callable=AsyncMock) as close:
            with self._boot(llm_check, AsyncMock(return_value="platform")) as c:
                assert list(main_module.teams) == ["platform"]
                assert main_module.disabled_teams == {
                    "payments": "LLM check failed: HTTP 401 from gateway",
                }
                assert c.get("/ready").status_code == 200
                assert c.get("/health").json()["teams"]["disabled"] == ["payments"]
                assert "gateway" not in c.get("/health").text
                # the failed team's clients were torn down at startup
                assert close.await_count == 1
                body = json.dumps(PR_PAYLOAD).encode()
                resp = c.post(
                    "/webhook/payments", content=body,
                    headers={"X-Hub-Signature": _sign(body, "pay-secret"), "Content-Type": "application/json"},
                )
                assert resp.status_code == 503

    def test_riptide_401_disables_the_team_but_a_network_error_does_not(self, env):
        with self._boot(AsyncMock(), AsyncMock(side_effect=RiptideAuthError("token rejected"))):
            assert list(main_module.teams) == ["payments"]
            assert main_module.disabled_teams == {"platform": "riptide check failed: token rejected"}

        # verify_at_startup swallows transport errors itself and returns None
        with self._boot(AsyncMock(), AsyncMock(return_value=None)):
            assert sorted(main_module.teams) == ["payments", "platform"]

        # anything else the ping raises (e.g. a non-JSON 200 body) is not a
        # bad token: the team stays enabled
        with self._boot(AsyncMock(), AsyncMock(side_effect=ValueError("not json"))):
            assert sorted(main_module.teams) == ["payments", "platform"]

    def test_unexpected_exception_in_team_startup_disables_only_that_team(self, env):
        with patch("app.main.Reviewer", side_effect=[RuntimeError("boom"), AsyncMock()]):
            with self._boot(AsyncMock(), AsyncMock(return_value="platform")):
                assert list(main_module.teams) == ["payments"]
                assert main_module.disabled_teams == {"platform": "startup failed: boom"}

    def test_no_enabled_team_boots_but_is_not_ready(self, env):
        async def llm_check(self):
            raise RuntimeError("down")

        with self._boot(llm_check, AsyncMock()) as c:
            assert main_module.teams == {}
            assert sorted(main_module.disabled_teams) == ["payments", "platform"]
            assert c.get("/health").status_code == 200
            assert c.get("/ready").status_code == 503


class TestProbe:
    """Signed onboarding probe: same HMAC as an event, answers ownership and
    bot read access, never enqueues anything."""

    def _probe(self, client, team, secret, body):
        raw = json.dumps(body).encode()
        return client.post(
            f"/webhook/{team}", content=raw,
            headers={
                "X-Hub-Signature": _sign(raw, secret),
                "X-Event-Key": "noergler:probe",
                "Content-Type": "application/json",
            },
        )

    def test_unsigned_probe_is_401(self, client):
        raw = json.dumps({"project": "PROJ", "repo": None}).encode()
        resp = client.post(
            WEBHOOK, content=raw,
            headers={"X-Event-Key": "noergler:probe", "Content-Type": "application/json"},
        )
        assert resp.status_code == 401

    def test_wrong_team_secret_is_401(self, client):
        assert self._probe(client, TEAM, OTHER_SECRET, {"project": "PROJ", "repo": None}).status_code == 401

    def test_owned_project_with_bot_access(self, client):
        with patch.object(main_module, "bitbucket_client") as bb:
            bb.get_project = AsyncMock(return_value={"key": "PROJ"})
            resp = self._probe(client, TEAM, WEBHOOK_SECRET, {"project": "PROJ", "repo": None})
        assert resp.status_code == 200
        assert resp.json() == {
            "team": TEAM, "owned": True, "claim": "whole", "bot_can_read": True, "bot_username": "noergler",
        }
        bb.get_project.assert_awaited_once_with("PROJ")
        assert client.reviewer.review_pull_request.await_count == 0

    def test_bot_without_access(self, client):
        with patch.object(main_module, "bitbucket_client") as bb:
            bb.get_repo = AsyncMock(side_effect=RuntimeError("404"))
            resp = self._probe(client, TEAM, WEBHOOK_SECRET, {"project": "PROJ", "repo": "x"})
        assert resp.json()["owned"] is True
        assert resp.json()["bot_can_read"] is False
        bb.get_repo.assert_awaited_once_with("PROJ", "x")

    def test_unowned_target_skips_the_bitbucket_call(self, client):
        with patch.object(main_module, "bitbucket_client") as bb:
            bb.get_project = AsyncMock()
            bb.get_repo = AsyncMock()
            # payments owns PAY/billing only: the whole project is not owned,
            # and PROJ belongs to platform
            r1 = self._probe(client, OTHER_TEAM, OTHER_SECRET, {"project": "PAY", "repo": None})
            r2 = self._probe(client, OTHER_TEAM, OTHER_SECRET, {"project": "PROJ", "repo": "repo"})
            r3 = self._probe(client, OTHER_TEAM, OTHER_SECRET, {"project": "PAY", "repo": "billing"})
        assert (r1.json()["owned"], r1.json()["claim"]) == (False, "repos")
        assert (r2.json()["owned"], r2.json()["claim"]) == (False, "none")
        assert (r3.json()["owned"], r3.json()["claim"]) == (True, "repos")
        assert bb.get_project.await_count == 0
        assert bb.get_repo.await_count == 1

    def test_malformed_probe_is_400(self, client):
        assert self._probe(client, TEAM, WEBHOOK_SECRET, {"repo": "x"}).status_code == 400
        assert self._probe(client, TEAM, WEBHOOK_SECRET, {"project": "PROJ", "repo": 3}).status_code == 400

    def test_disabled_team_is_503(self, client):
        assert self._probe(client, "broken", WEBHOOK_SECRET, {"project": "PROJ", "repo": None}).status_code == 503
