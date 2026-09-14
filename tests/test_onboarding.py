"""`app/onboarding.py` against a respx-mocked Bitbucket, and `POST /onboard/{team}`."""

import logging
from contextlib import asynccontextmanager
from typing import Any, cast
from unittest.mock import AsyncMock

import httpx
import pytest
import respx
from fastapi.testclient import TestClient

import app.main as main_module
from app.bitbucket import BitbucketClient
from app.config import (
    REQUIRED_WEBHOOK_EVENTS,
    BitbucketConfig,
    JiraConfig,
    LLMConfig,
    ProjectScope,
    ReviewConfig,
    ServerConfig,
    TeamConfig,
)
from app.jira import JiraClient
from app.llm_client import LLMClient
from app.main import TeamRuntime, app
from app.onboarding import (
    Onboarder,
    StatusRow,
    Target,
    TargetResult,
    UnknownTarget,
    render_results,
    render_status,
    run,
    targets_for,
)
from app.riptide_client import RiptideClient

BASE_URL = "https://bitbucket.test"
PUBLIC_URL = "https://noergler.test"
TEAM = "platform"
SECRET = "whsec"
BOT = "noergler"
ADMIN_TOKEN = "BBDC-admin-token-never-logged"
WEBHOOK_URL = f"{PUBLIC_URL}/webhook/{TEAM}"

PROJ = Target("PROJ")
REPO = Target("PROJ", "my-repo")
PROJ_HOOKS = f"{BASE_URL}/rest/api/1.0/projects/PROJ/webhooks"
REPO_HOOKS = f"{BASE_URL}/rest/api/1.0/projects/PROJ/repos/my-repo/webhooks"


def _team(projects: list[ProjectScope]) -> TeamConfig:
    return TeamConfig(
        slug=TEAM, name=TEAM, webhook_secret=SECRET, projects=projects,
        llm=LLMConfig(model="m", api_key="k", api_url="https://llm.test/v1"),
        review=ReviewConfig(), jira=JiraConfig(url="https://jira.test", token="t"),
    )


def _good_hook(**overrides: Any) -> dict[str, Any]:
    hook = {
        "id": 7, "name": "noergler", "url": WEBHOOK_URL, "active": True,
        "events": list(REQUIRED_WEBHOOK_EVENTS), "configuration": {"secret": "x"},
        "sslVerificationRequired": True,
    }
    hook.update(overrides)
    return hook


def _page(values: list[dict[str, Any]]) -> dict[str, Any]:
    return {"values": values, "isLastPage": True}


def _onboarder(team: TeamConfig, **kw: Any) -> Onboarder:
    cfg = BitbucketConfig(base_url=BASE_URL, token="bot-token", username=BOT)
    return Onboarder(
        BitbucketClient(cfg, token=ADMIN_TOKEN), BitbucketClient(cfg), team, WEBHOOK_URL, **kw
    )


def _bot_can_read() -> None:
    respx.get(f"{BASE_URL}/rest/api/1.0/projects/PROJ").mock(return_value=httpx.Response(200, json={"key": "PROJ"}))
    respx.get(f"{BASE_URL}/rest/api/1.0/projects/PROJ/repos/my-repo").mock(
        return_value=httpx.Response(200, json={"slug": "my-repo"})
    )


class TestTargets:
    def test_whole_and_repos_claims(self):
        team = _team([ProjectScope(key="A"), ProjectScope(key="B", repos=["x", "y"])])
        assert targets_for(team) == [Target("A"), Target("B", "x"), Target("B", "y")]

    def test_subset_filters_and_rejects_unknown(self):
        team = _team([ProjectScope(key="A"), ProjectScope(key="B", repos=["x"])])
        assert targets_for(team, ["B/x"]) == [Target("B", "x")]
        with pytest.raises(UnknownTarget, match="B/y"):
            targets_for(team, ["A", "B/y"])


@pytest.mark.asyncio
class TestStatus:
    @respx.mock
    async def test_ok_and_stray_and_foreign(self):
        _bot_can_read()
        respx.get(PROJ_HOOKS).mock(return_value=httpx.Response(200, json=_page([_good_hook()])))
        respx.get(f"{BASE_URL}/rest/api/1.0/projects/PROJ/repos").mock(
            return_value=httpx.Response(200, json=_page([{"slug": "my-repo"}, {"slug": "other"}]))
        )
        respx.get(REPO_HOOKS).mock(return_value=httpx.Response(200, json=_page([_good_hook(id=8)])))
        respx.get(f"{BASE_URL}/rest/api/1.0/projects/PROJ/repos/other/webhooks").mock(
            return_value=httpx.Response(200, json=_page([_good_hook(id=9, url="https://old.test/webhook")]))
        )
        row = await _onboarder(_team([ProjectScope(key="PROJ")])).status(PROJ)
        assert (row.owned, row.bot_can_read, row.webhook) == (True, True, "ok")
        assert row.stray == ["PROJ/my-repo"]
        assert row.foreign == ["PROJ/other -> https://old.test/webhook"]

        # the hook verdict stands when only the repo listing fails; the row is
        # still not healthy, and says why
        respx.get(f"{BASE_URL}/rest/api/1.0/projects/PROJ/repos").mock(return_value=httpx.Response(500))
        row = await _onboarder(_team([ProjectScope(key="PROJ")])).status(PROJ)
        assert row.webhook == "ok (repo hooks unchecked: HTTP 500)"
        assert row.stray == []

    @respx.mock
    async def test_missing_stale_foreign_and_admin_401(self):
        _bot_can_read()
        team = _team([ProjectScope(key="PROJ", repos=["my-repo"])])
        ob = _onboarder(team)
        # no project hook of ours next to the repo target
        respx.get(PROJ_HOOKS).mock(return_value=httpx.Response(200, json=_page([])))

        respx.get(REPO_HOOKS).mock(return_value=httpx.Response(200, json=_page([])))
        assert (await ob.status(REPO)).webhook == "missing"

        respx.get(REPO_HOOKS).mock(return_value=httpx.Response(200, json=_page([_good_hook(url="https://old.test/webhook/x")])))
        assert (await ob.status(REPO)).webhook == "foreign: https://old.test/webhook/x"

        stale = _good_hook(events=["pr:opened"], configuration={})
        respx.get(REPO_HOOKS).mock(return_value=httpx.Response(200, json=_page([stale])))
        webhook = (await ob.status(REPO)).webhook
        assert webhook.startswith("stale: events: missing=")
        assert "configuration.secret: (unset) -> (set)" in webhook

        respx.get(REPO_HOOKS).mock(return_value=httpx.Response(401, text="not permitted"))
        assert (await ob.status(REPO)).webhook == "HTTP 401"

    @respx.mock
    async def test_project_hook_guard_only_yields_on_no_admin(self):
        _bot_can_read()
        team = _team([ProjectScope(key="PROJ", repos=["my-repo"])])
        ob = _onboarder(team)
        respx.get(REPO_HOOKS).mock(return_value=httpx.Response(200, json=_page([])))
        respx.get(PROJ_HOOKS).mock(return_value=httpx.Response(403))
        assert (await ob.status(REPO)).webhook == "missing"
        # a Bitbucket hiccup on the guard is an error, not a licence to double-hook
        respx.get(PROJ_HOOKS).mock(return_value=httpx.Response(503))
        result = await ob.onboard(REPO)
        assert result.status == "failed"
        assert "HTTP 503" in result.detail

    @respx.mock
    async def test_not_owned_and_whole_claim_block(self):
        team = _team([ProjectScope(key="PROJ")])
        ob = _onboarder(team)
        _bot_can_read()
        other = await ob.status(Target("OTHER"))
        assert other.owned is False
        assert "ask the noergler admin" in other.webhook
        # a repo target under a whole-project claim would double-deliver
        blocked = await ob.status(REPO)
        assert blocked.owned is True
        assert blocked.webhook.startswith("blocked: teams.yaml claims all of PROJ")


@pytest.mark.asyncio
class TestOnboard:
    @respx.mock
    async def test_create_with_grant_and_prune(self):
        # bot cannot read yet -> grant, then create the project hook, then prune the stray repo hook
        respx.get(f"{BASE_URL}/rest/api/1.0/projects/PROJ").mock(return_value=httpx.Response(404))
        grant = respx.put(f"{BASE_URL}/rest/api/1.0/projects/PROJ/permissions/users").mock(
            return_value=httpx.Response(204)
        )
        respx.get(PROJ_HOOKS).mock(return_value=httpx.Response(200, json=_page([])))
        create = respx.post(PROJ_HOOKS).mock(return_value=httpx.Response(201, json={"id": 42}))
        respx.get(f"{BASE_URL}/rest/api/1.0/projects/PROJ/repos").mock(
            return_value=httpx.Response(200, json=_page([{"slug": "my-repo"}]))
        )
        respx.get(REPO_HOOKS).mock(return_value=httpx.Response(200, json=_page([_good_hook(id=8)])))
        delete = respx.delete(f"{REPO_HOOKS}/8").mock(return_value=httpx.Response(204))

        result = await _onboarder(_team([ProjectScope(key="PROJ")]), grant_bot=True).onboard(PROJ)

        assert result.status == "ok", result.detail
        assert result.detail == f"webhook created (id=42), {BOT} granted PROJECT_WRITE, pruned 1 repo hook(s): PROJ/my-repo"
        assert grant.calls.last.request.url.params["name"] == BOT
        assert grant.calls.last.request.url.params["permission"] == "PROJECT_WRITE"
        body = create.calls.last.request.read()
        assert b'"secret": "whsec"' in body.replace(b"\n", b"") or b'"secret":"whsec"' in body
        assert delete.called
        # every admin-side call carried the admin token, the bot check the bot token
        for call in create.calls + grant.calls + delete.calls:
            assert call.request.headers["Authorization"] == f"Bearer {ADMIN_TOKEN}"

    @respx.mock
    async def test_without_grant_bot_is_skipped(self):
        respx.get(f"{BASE_URL}/rest/api/1.0/projects/PROJ").mock(return_value=httpx.Response(403))
        result = await _onboarder(_team([ProjectScope(key="PROJ")])).onboard(PROJ)
        assert result.status == "skipped"
        assert "run grant-bot" in result.detail

    @respx.mock
    async def test_up_to_date_update_and_foreign(self):
        _bot_can_read()
        team = _team([ProjectScope(key="PROJ", repos=["my-repo"])])
        ob = _onboarder(team, prune=False)
        respx.get(PROJ_HOOKS).mock(return_value=httpx.Response(200, json=_page([])))

        respx.get(REPO_HOOKS).mock(return_value=httpx.Response(200, json=_page([_good_hook()])))
        assert (await ob.onboard(REPO)).detail == "webhook already up to date"

        respx.get(REPO_HOOKS).mock(return_value=httpx.Response(200, json=_page([_good_hook(active=False)])))
        update = respx.put(f"{REPO_HOOKS}/7").mock(return_value=httpx.Response(200, json={"id": 7}))
        result = await ob.onboard(REPO)
        assert (result.status, result.detail, result.diff) == ("ok", "webhook updated (id=7)", ["active: False -> True"])
        assert update.called

        respx.get(REPO_HOOKS).mock(return_value=httpx.Response(200, json=_page([_good_hook(url="https://old.test/webhook")])))
        result = await ob.onboard(REPO)
        assert result.status == "failed"
        assert "points at another noergler" in result.detail

    @respx.mock
    async def test_dry_run_writes_nothing(self):
        respx.get(f"{BASE_URL}/rest/api/1.0/projects/PROJ").mock(return_value=httpx.Response(404))
        respx.get(PROJ_HOOKS).mock(return_value=httpx.Response(200, json=_page([])))
        respx.get(f"{BASE_URL}/rest/api/1.0/projects/PROJ/repos").mock(return_value=httpx.Response(200, json=_page([])))
        result = await _onboarder(_team([ProjectScope(key="PROJ")]), grant_bot=True, dry_run=True).onboard(PROJ)
        assert result.status == "ok"
        assert result.detail.startswith("dry-run: webhook created (id=-1)")
        assert not any(c.request.method in ("POST", "PUT", "DELETE") for c in respx.calls)


@pytest.mark.asyncio
class TestRemove:
    @respx.mock
    async def test_removes_own_leaves_foreign(self):
        ob = _onboarder(_team([ProjectScope(key="PROJ")]))
        respx.get(PROJ_HOOKS).mock(return_value=httpx.Response(200, json=_page([_good_hook()])))
        delete = respx.delete(f"{PROJ_HOOKS}/7").mock(return_value=httpx.Response(204))
        assert (await ob.remove(PROJ)).detail == "webhook removed: id=7"
        assert delete.called

        respx.get(PROJ_HOOKS).mock(return_value=httpx.Response(200, json=_page([_good_hook(url="https://old.test/webhook")])))
        result = await ob.remove(PROJ)
        assert result.status == "skipped"
        assert "left alone" in result.detail

        respx.get(PROJ_HOOKS).mock(return_value=httpx.Response(200, json=_page([])))
        assert (await ob.remove(PROJ)).status == "skipped"


@pytest.mark.asyncio
class TestRun:
    @respx.mock
    async def test_one_failure_does_not_abort(self):
        _bot_can_read()
        team = _team([ProjectScope(key="PROJ", repos=["my-repo", "other"])])
        respx.get(PROJ_HOOKS).mock(return_value=httpx.Response(200, json=_page([])))
        respx.get(f"{BASE_URL}/rest/api/1.0/projects/PROJ/repos/other").mock(return_value=httpx.Response(200, json={}))
        respx.get(REPO_HOOKS).mock(return_value=httpx.Response(401, text="no"))
        respx.get(f"{BASE_URL}/rest/api/1.0/projects/PROJ/repos/other/webhooks").mock(
            return_value=httpx.Response(200, json=_page([_good_hook()]))
        )
        rows, text, healthy = await run(_onboarder(team), "onboard", targets_for(team))
        results = cast(list[TargetResult], rows)
        assert [r.status for r in results] == ["failed", "ok"]
        assert healthy is False
        assert text == render_results(results)
        assert "PROJ/my-repo" in text and "HTTP 401" in text


def test_render_status_columns():
    text = render_status([StatusRow(PROJ, True, True, "ok", ["PROJ/x"], [])])
    assert text.splitlines()[0].startswith("target")
    assert "PROJ (project)" in text and "yes    yes   ok  stray repo hooks: PROJ/x" in text


# -- endpoint -- #

@pytest.fixture()
def api(monkeypatch):
    cfg = BitbucketConfig(base_url=BASE_URL, token="bot-token", username=BOT)
    mock_config = type("C", (), {"bitbucket": cfg, "server": ServerConfig(public_url=PUBLIC_URL)})()
    runtime = TeamRuntime(
        config=_team([ProjectScope(key="PROJ")]),
        llm=cast(LLMClient, cast(object, None)),
        jira=cast(JiraClient, cast(object, None)),
        riptide=cast(RiptideClient, cast(object, None)),
        reviewer=AsyncMock(),
    )

    @asynccontextmanager
    async def noop_lifespan(_a):
        yield

    original = (main_module.config, main_module.bitbucket_client, app.router.lifespan_context)
    main_module.config = mock_config
    main_module.bitbucket_client = BitbucketClient(cfg)
    main_module.teams.clear()
    main_module.teams[TEAM] = runtime
    main_module.disabled_teams.clear()
    main_module.disabled_teams["broken"] = "disabled"
    app.router.lifespan_context = noop_lifespan
    try:
        with TestClient(app, raise_server_exceptions=False) as c:
            yield c
    finally:
        main_module.config, main_module.bitbucket_client, app.router.lifespan_context = original
        main_module.teams.clear()
        main_module.disabled_teams.clear()


AUTH = {"Authorization": f"Bearer {ADMIN_TOKEN}"}
WHOAMI = f"{BASE_URL}/plugins/servlet/applinks/whoami"


def _token_ok() -> None:
    respx.get(WHOAMI).mock(return_value=httpx.Response(200, text="jan"))


class TestEndpoint:
    @respx.mock
    def test_auth_and_team_errors(self, api):
        _token_ok()
        assert api.post(f"/onboard/{TEAM}", json={}).status_code == 401
        assert api.post(f"/onboard/{TEAM}", json={}, headers={"Authorization": "Basic x"}).status_code == 401
        assert api.post("/onboard/nobody", json={}, headers=AUTH).status_code == 404
        assert api.post("/onboard/broken", json={}, headers=AUTH).status_code == 503
        r = api.post(f"/onboard/{TEAM}", json={"targets": ["NOPE"]}, headers=AUTH)
        assert r.status_code == 400 and "NOPE" in r.json()["detail"]
        assert api.post(f"/onboard/{TEAM}", json={"action": "explode"}, headers=AUTH).status_code == 422

    @respx.mock
    def test_rejected_token_reveals_nothing(self, api):
        # Bitbucket answers whoami with an empty body for a bad token: 401 here,
        # before targets or ownership are computed, and no other Bitbucket call.
        respx.get(WHOAMI).mock(return_value=httpx.Response(200, text=""))
        r = api.post(f"/onboard/{TEAM}", json={"targets": ["NOPE"]}, headers=AUTH)
        assert r.status_code == 401
        assert "NOPE" not in r.text and "PROJ" not in r.text
        assert len(respx.calls) == 1

    @respx.mock
    def test_bitbucket_down_on_token_check_is_502(self, api):
        respx.get(WHOAMI).mock(side_effect=httpx.ConnectError("down"))
        assert api.post(f"/onboard/{TEAM}", json={}, headers=AUTH).status_code == 502

    def test_public_url_required(self, api):
        main_module.config.server = ServerConfig()
        r = api.post(f"/onboard/{TEAM}", json={}, headers=AUTH)
        assert r.status_code == 503
        assert "NOERGLER_PUBLIC_URL" in r.json()["detail"]

    @respx.mock
    def test_status_and_grant_bot(self, api, caplog):
        _token_ok()
        _bot_can_read()
        respx.get(PROJ_HOOKS).mock(return_value=httpx.Response(200, json=_page([_good_hook()])))
        respx.get(f"{BASE_URL}/rest/api/1.0/projects/PROJ/repos").mock(return_value=httpx.Response(200, json=_page([])))

        with caplog.at_level(logging.DEBUG):
            r = api.post(f"/onboard/{TEAM}", json={"action": "status"}, headers=AUTH)
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["webhook_url"] == WEBHOOK_URL
        assert body["healthy"] is True
        assert body["rows"][0]["target"] == {"project": "PROJ", "repo": None}
        assert body["rows"][0]["webhook"] == "ok"
        assert "PROJ (project)" in body["text"]
        assert ADMIN_TOKEN not in caplog.text
        assert ADMIN_TOKEN not in r.text

        r = api.post(f"/onboard/{TEAM}", json={"action": "grant-bot"}, headers=AUTH)
        assert r.status_code == 200, r.text
        assert r.json()["rows"][0]["status"] == "ok"
        assert r.json()["rows"][0]["detail"] == "webhook already up to date"
        # the admin token went to Bitbucket on the hook listing, the bot token on the read check
        assert "onboard by=jan" in caplog.text
        hook_calls = [c for c in respx.calls if c.request.url.path.endswith("/PROJ/webhooks")]
        assert all(c.request.headers["Authorization"] == f"Bearer {ADMIN_TOKEN}" for c in hook_calls)
        read_calls = [c for c in respx.calls if c.request.url.path.endswith("/projects/PROJ")]
        assert all(c.request.headers["Authorization"] == "Bearer bot-token" for c in read_calls)
