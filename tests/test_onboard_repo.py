import json
from pathlib import Path

import httpx
import pytest
import respx

from app.bitbucket import BitbucketClient
from app.config import REQUIRED_WEBHOOK_EVENTS, BitbucketConfig
from scripts.onboard_repo import (
    DEFAULT_WEBHOOK_NAME,
    RepoOnboarder,
    RepoSpec,
    load_onboarding_input,
    main,
    resolve_secrets,
)

BASE_URL = "https://bitbucket.company.com"
WEBHOOK_URL = "https://noergler.internal/webhook"


@pytest.fixture
def bb_config():
    return BitbucketConfig(
        base_url=BASE_URL,
        token="test-token",
        webhook_secret="test-secret",
        username="bot",
    )


@pytest.fixture
def client(bb_config):
    c = BitbucketClient(bb_config)
    yield c


@pytest.fixture
def onboarder(client):
    return RepoOnboarder(
        client,
        webhook_url=WEBHOOK_URL,
        webhook_secret="whsec",
        webhook_name=DEFAULT_WEBHOOK_NAME,
    )


@pytest.fixture
def spec():
    return RepoSpec(project="PROJ", repo="my-repo")


# --------------------------------------------------------------------------- #
# JSON config validation
# --------------------------------------------------------------------------- #

class TestLoadOnboardingInput:
    def _write(self, tmp_path: Path, data) -> Path:
        p = tmp_path / "config.json"
        p.write_text(json.dumps(data))
        return p

    def test_valid_config(self, tmp_path):
        path = self._write(tmp_path, {
            "bitbucket_url": "https://bb.example.com",
            "webhook_url": "https://noergler/webhook",
            "projects": [
                {"project": "A", "repos": ["one", "two"]},
                {"project": "B", "repos": ["three"]},
            ],
        })
        result = load_onboarding_input(path)
        assert result.bitbucket_url == "https://bb.example.com"
        assert result.webhook_url == "https://noergler/webhook"
        assert [r.key for r in result.repos] == ["A/one", "A/two", "B/three"]

    def test_strips_trailing_slash_on_bitbucket_url(self, tmp_path):
        path = self._write(tmp_path, {
            "bitbucket_url": "https://bb.example.com/",
            "webhook_url": "https://x/webhook",
            "projects": [{"project": "A", "repos": ["one"]}],
        })
        assert load_onboarding_input(path).bitbucket_url == "https://bb.example.com"

    def test_rejects_http_bitbucket_url(self, tmp_path):
        path = self._write(tmp_path, {
            "bitbucket_url": "http://bb",
            "webhook_url": "https://x/webhook",
            "projects": [{"project": "A", "repos": ["one"]}],
        })
        with pytest.raises(SystemExit, match="bitbucket_url"):
            load_onboarding_input(path)

    def test_rejects_missing_webhook_url(self, tmp_path):
        path = self._write(tmp_path, {
            "bitbucket_url": "https://bb",
            "projects": [{"project": "A", "repos": ["one"]}],
        })
        with pytest.raises(SystemExit, match="webhook_url"):
            load_onboarding_input(path)

    def test_rejects_missing_projects(self, tmp_path):
        path = self._write(tmp_path, {
            "bitbucket_url": "https://bb",
            "webhook_url": "https://x/webhook",
        })
        with pytest.raises(SystemExit, match="projects"):
            load_onboarding_input(path)

    def test_rejects_empty_projects_list(self, tmp_path):
        path = self._write(tmp_path, {
            "bitbucket_url": "https://bb",
            "webhook_url": "https://x/webhook",
            "projects": [],
        })
        with pytest.raises(SystemExit, match="projects"):
            load_onboarding_input(path)

    def test_rejects_empty_repos_under_project(self, tmp_path):
        path = self._write(tmp_path, {
            "bitbucket_url": "https://bb",
            "webhook_url": "https://x/webhook",
            "projects": [{"project": "A", "repos": []}],
        })
        with pytest.raises(SystemExit, match="repos"):
            load_onboarding_input(path)

    def test_rejects_duplicate_repos(self, tmp_path):
        path = self._write(tmp_path, {
            "bitbucket_url": "https://bb",
            "webhook_url": "https://x/webhook",
            "projects": [
                {"project": "A", "repos": ["one", "one"]},
            ],
        })
        with pytest.raises(SystemExit, match="duplicate"):
            load_onboarding_input(path)

    def test_rejects_duplicate_across_project_entries(self, tmp_path):
        path = self._write(tmp_path, {
            "bitbucket_url": "https://bb",
            "webhook_url": "https://x/webhook",
            "projects": [
                {"project": "A", "repos": ["one"]},
                {"project": "A", "repos": ["one"]},
            ],
        })
        with pytest.raises(SystemExit, match="duplicate"):
            load_onboarding_input(path)

    def test_rejects_missing_project_field(self, tmp_path):
        path = self._write(tmp_path, {
            "bitbucket_url": "https://bb",
            "webhook_url": "https://x/webhook",
            "projects": [{"repos": ["one"]}],
        })
        with pytest.raises(SystemExit, match="project"):
            load_onboarding_input(path)


# --------------------------------------------------------------------------- #
# resolve_secrets
# --------------------------------------------------------------------------- #

class TestResolveSecrets:
    def test_env_vars_win(self, monkeypatch, tmp_path):
        monkeypatch.setenv("BITBUCKET_TOKEN", "from-env")
        monkeypatch.setenv("BITBUCKET_WEBHOOK_SECRET", "sec-env")
        monkeypatch.chdir(tmp_path)
        token, secret = resolve_secrets(None)
        assert (token, secret) == ("from-env", "sec-env")

    def test_env_file_fallback(self, monkeypatch, tmp_path):
        monkeypatch.delenv("BITBUCKET_TOKEN", raising=False)
        monkeypatch.delenv("BITBUCKET_WEBHOOK_SECRET", raising=False)
        monkeypatch.chdir(tmp_path)
        env_file = tmp_path / "extra.env"
        env_file.write_text(
            'BITBUCKET_TOKEN="file-token"\nBITBUCKET_WEBHOOK_SECRET=file-secret\n# ignored\n'
        )
        token, secret = resolve_secrets(env_file)
        assert (token, secret) == ("file-token", "file-secret")

    def test_cwd_dotenv_used(self, monkeypatch, tmp_path):
        monkeypatch.delenv("BITBUCKET_TOKEN", raising=False)
        monkeypatch.delenv("BITBUCKET_WEBHOOK_SECRET", raising=False)
        (tmp_path / ".env").write_text(
            "BITBUCKET_TOKEN=cwd-token\nBITBUCKET_WEBHOOK_SECRET=cwd-secret\n"
        )
        monkeypatch.chdir(tmp_path)
        token, secret = resolve_secrets(None)
        assert (token, secret) == ("cwd-token", "cwd-secret")

    def test_precedence_env_beats_cwd_beats_envfile(self, monkeypatch, tmp_path):
        """Pin down: process env > cwd .env > --env-file."""
        (tmp_path / ".env").write_text(
            "BITBUCKET_TOKEN=cwd-token\nBITBUCKET_WEBHOOK_SECRET=cwd-secret\n"
        )
        env_file = tmp_path / "lowest.env"
        env_file.write_text(
            "BITBUCKET_TOKEN=envfile-token\nBITBUCKET_WEBHOOK_SECRET=envfile-secret\n"
        )
        monkeypatch.chdir(tmp_path)

        monkeypatch.delenv("BITBUCKET_TOKEN", raising=False)
        monkeypatch.delenv("BITBUCKET_WEBHOOK_SECRET", raising=False)
        token, secret = resolve_secrets(env_file)
        assert (token, secret) == ("cwd-token", "cwd-secret")

        monkeypatch.setenv("BITBUCKET_TOKEN", "env-token")
        monkeypatch.setenv("BITBUCKET_WEBHOOK_SECRET", "env-secret")
        token, secret = resolve_secrets(env_file)
        assert (token, secret) == ("env-token", "env-secret")

    def test_missing_secrets_exits(self, monkeypatch, tmp_path):
        monkeypatch.delenv("BITBUCKET_TOKEN", raising=False)
        monkeypatch.delenv("BITBUCKET_WEBHOOK_SECRET", raising=False)
        monkeypatch.chdir(tmp_path)
        with pytest.raises(SystemExit):
            resolve_secrets(None)


# --------------------------------------------------------------------------- #
# RepoOnboarder
# --------------------------------------------------------------------------- #

class TestVerifyPermissions:
    @pytest.mark.asyncio
    @respx.mock
    async def test_ok(self, onboarder, spec, client):
        respx.get(f"{BASE_URL}/rest/api/1.0/projects/PROJ/repos/my-repo").mock(
            return_value=httpx.Response(200, json={"slug": "my-repo"})
        )
        respx.get(
            f"{BASE_URL}/rest/api/1.0/projects/PROJ/repos/my-repo/pull-requests"
        ).mock(return_value=httpx.Response(200, json={"values": []}))
        await onboarder.verify_permissions(spec)
        await client.close()

    @pytest.mark.asyncio
    @respx.mock
    async def test_403_raises(self, onboarder, spec, client):
        respx.get(f"{BASE_URL}/rest/api/1.0/projects/PROJ/repos/my-repo").mock(
            return_value=httpx.Response(403, text="Forbidden")
        )
        with pytest.raises(httpx.HTTPStatusError):
            await onboarder.verify_permissions(spec)
        await client.close()


class TestUpsertWebhook:
    @pytest.mark.asyncio
    @respx.mock
    async def test_creates_when_absent(self, onboarder, spec, client):
        respx.get(
            f"{BASE_URL}/rest/api/1.0/projects/PROJ/repos/my-repo/webhooks"
        ).mock(return_value=httpx.Response(200, json={"values": [], "isLastPage": True}))
        create = respx.post(
            f"{BASE_URL}/rest/api/1.0/projects/PROJ/repos/my-repo/webhooks"
        ).mock(return_value=httpx.Response(200, json={"id": 42}))

        webhook_id, diff = await onboarder.upsert_webhook(spec)

        assert webhook_id == 42
        assert diff == ["create"]
        body = json.loads(create.calls[0].request.content)
        assert body["name"] == DEFAULT_WEBHOOK_NAME
        assert body["url"] == WEBHOOK_URL
        assert set(body["events"]) == set(REQUIRED_WEBHOOK_EVENTS)
        assert body["configuration"]["secret"] == "whsec"
        assert body["active"] is True
        assert body["sslVerificationRequired"] is True
        await client.close()

    @pytest.mark.asyncio
    @respx.mock
    async def test_no_op_when_up_to_date(self, onboarder, spec, client):
        respx.get(
            f"{BASE_URL}/rest/api/1.0/projects/PROJ/repos/my-repo/webhooks"
        ).mock(return_value=httpx.Response(200, json={
            "values": [{
                "id": 7,
                "name": DEFAULT_WEBHOOK_NAME,
                "url": WEBHOOK_URL,
                "events": list(REQUIRED_WEBHOOK_EVENTS),
                "active": True,
                "configuration": {"secret": "hidden"},
            }],
            "isLastPage": True,
        }))

        webhook_id, diff = await onboarder.upsert_webhook(spec)
        assert webhook_id == 7
        assert diff == []
        await client.close()

    @pytest.mark.asyncio
    @respx.mock
    async def test_updates_on_drift(self, onboarder, spec, client):
        respx.get(
            f"{BASE_URL}/rest/api/1.0/projects/PROJ/repos/my-repo/webhooks"
        ).mock(return_value=httpx.Response(200, json={
            "values": [{
                "id": 7,
                "name": DEFAULT_WEBHOOK_NAME,
                "url": "https://stale.example/webhook",
                "events": ["pr:opened"],
                "active": True,
                "configuration": {"secret": "x"},
            }],
            "isLastPage": True,
        }))
        put = respx.put(
            f"{BASE_URL}/rest/api/1.0/projects/PROJ/repos/my-repo/webhooks/7"
        ).mock(return_value=httpx.Response(200, json={"id": 7}))

        webhook_id, diff = await onboarder.upsert_webhook(spec)
        assert webhook_id == 7
        assert any("url" in d for d in diff)
        assert any("events" in d for d in diff)
        assert put.called
        await client.close()

    @pytest.mark.asyncio
    @respx.mock
    async def test_dry_run_skips_writes(self, client, spec):
        p = RepoOnboarder(
            client, webhook_url=WEBHOOK_URL, webhook_secret="whsec", dry_run=True,
        )
        respx.get(
            f"{BASE_URL}/rest/api/1.0/projects/PROJ/repos/my-repo/webhooks"
        ).mock(return_value=httpx.Response(200, json={"values": [], "isLastPage": True}))
        post_route = respx.post(
            f"{BASE_URL}/rest/api/1.0/projects/PROJ/repos/my-repo/webhooks"
        ).mock(return_value=httpx.Response(200, json={"id": 1}))

        webhook_id, diff = await p.upsert_webhook(spec)
        assert webhook_id == -1
        assert diff == ["create"]
        assert not post_route.called
        await client.close()


class TestTestWebhook:
    @pytest.mark.asyncio
    @respx.mock
    async def test_success_returns_200(self, onboarder, spec, client):
        respx.post(
            f"{BASE_URL}/rest/api/1.0/projects/PROJ/repos/my-repo/webhooks/7/test"
        ).mock(return_value=httpx.Response(200, json={"statusCode": 200}))
        assert await onboarder.test_webhook(spec, 7) == 200
        await client.close()

    @pytest.mark.asyncio
    @respx.mock
    async def test_downstream_401_reported(self, onboarder, spec, client):
        respx.post(
            f"{BASE_URL}/rest/api/1.0/projects/PROJ/repos/my-repo/webhooks/7/test"
        ).mock(return_value=httpx.Response(200, json={"statusCode": 401}))
        assert await onboarder.test_webhook(spec, 7) == 401
        await client.close()

    @pytest.mark.asyncio
    @respx.mock
    async def test_bitbucket_side_failure(self, onboarder, spec, client):
        respx.post(
            f"{BASE_URL}/rest/api/1.0/projects/PROJ/repos/my-repo/webhooks/7/test"
        ).mock(return_value=httpx.Response(500, text="boom"))
        assert await onboarder.test_webhook(spec, 7) == 500
        await client.close()


# --------------------------------------------------------------------------- #
# End-to-end via main()
# --------------------------------------------------------------------------- #

class TestMain:
    def test_multi_repo_mixed_results(self, tmp_path, monkeypatch):
        cfg = tmp_path / "config.json"
        cfg.write_text(json.dumps({
            "bitbucket_url": BASE_URL,
            "webhook_url": WEBHOOK_URL,
            "projects": [
                {"project": "PROJ", "repos": ["good", "bad"]},
            ],
        }))
        monkeypatch.setenv("BITBUCKET_TOKEN", "t")
        monkeypatch.setenv("BITBUCKET_WEBHOOK_SECRET", "s")
        monkeypatch.chdir(tmp_path)

        with respx.mock(assert_all_called=False) as mock:
            mock.get(f"{BASE_URL}/rest/api/1.0/projects/PROJ/repos/good").mock(
                return_value=httpx.Response(200, json={})
            )
            mock.get(
                f"{BASE_URL}/rest/api/1.0/projects/PROJ/repos/good/pull-requests"
            ).mock(return_value=httpx.Response(200, json={"values": []}))
            mock.get(
                f"{BASE_URL}/rest/api/1.0/projects/PROJ/repos/good/webhooks"
            ).mock(return_value=httpx.Response(200, json={"values": [], "isLastPage": True}))
            mock.post(
                f"{BASE_URL}/rest/api/1.0/projects/PROJ/repos/good/webhooks"
            ).mock(return_value=httpx.Response(200, json={"id": 1}))
            mock.post(
                f"{BASE_URL}/rest/api/1.0/projects/PROJ/repos/good/webhooks/1/test"
            ).mock(return_value=httpx.Response(200, json={"statusCode": 200}))
            mock.get(f"{BASE_URL}/rest/api/1.0/projects/PROJ/repos/bad").mock(
                return_value=httpx.Response(404, text="no repo")
            )

            rc = main([str(cfg)])

        assert rc == 1

    def test_dry_run_issues_no_writes(self, tmp_path, monkeypatch, capsys):
        cfg = tmp_path / "config.json"
        cfg.write_text(json.dumps({
            "bitbucket_url": BASE_URL,
            "webhook_url": WEBHOOK_URL,
            "projects": [{"project": "PROJ", "repos": ["r"]}],
        }))
        monkeypatch.setenv("BITBUCKET_TOKEN", "t")
        monkeypatch.setenv("BITBUCKET_WEBHOOK_SECRET", "s")
        monkeypatch.chdir(tmp_path)

        with respx.mock(assert_all_called=False) as mock:
            mock.get(f"{BASE_URL}/rest/api/1.0/projects/PROJ/repos/r").mock(
                return_value=httpx.Response(200, json={})
            )
            mock.get(
                f"{BASE_URL}/rest/api/1.0/projects/PROJ/repos/r/pull-requests"
            ).mock(return_value=httpx.Response(200, json={"values": []}))
            mock.get(
                f"{BASE_URL}/rest/api/1.0/projects/PROJ/repos/r/webhooks"
            ).mock(return_value=httpx.Response(200, json={"values": [], "isLastPage": True}))
            create_route = mock.post(
                f"{BASE_URL}/rest/api/1.0/projects/PROJ/repos/r/webhooks"
            ).mock(return_value=httpx.Response(200, json={"id": 1}))
            test_route = mock.post(
                f"{BASE_URL}/rest/api/1.0/projects/PROJ/repos/r/webhooks/1/test"
            ).mock(return_value=httpx.Response(200, json={"statusCode": 200}))

            rc = main([str(cfg), "--dry-run"])

        assert rc == 0
        assert not create_route.called
        assert not test_route.called
        out = capsys.readouterr().out
        assert "PROJ/r" in out
        assert "ok" in out
