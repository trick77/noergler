import json
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Callable
from urllib.parse import urlsplit

import pytest

from scripts.onboard_repo import (
    DEFAULT_WEBHOOK_NAME,
    REQUIRED_WEBHOOK_EVENTS,
    BitbucketHTTP,
    HTTPStatusError,
    RepoOnboarder,
    RepoSpec,
    default_webhook_secret_var,
    load_onboarding_input,
    load_team_entry,
    team_owns,
    main,
    resolve_secrets,
)

BASE_URL = "https://bitbucket.company.com"
TEAM = "platform"
WEBHOOK_URL = f"https://noergler.internal/webhook/{TEAM}"
SECRET_VAR = "TEAM_PLATFORM_WEBHOOK_SECRET"


# --------------------------------------------------------------------------- #
# urlopen stub — dispatches by (method, path) → callable returning (status, body)
# --------------------------------------------------------------------------- #

class _FakeHTTPResponse:
    def __init__(self, status: int, body: bytes):
        self.status = status
        self._body = body

    def read(self) -> bytes:
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


RouteHandler = Callable[[dict[str, Any], bytes | None], tuple[int, dict[str, Any] | str]]


class FakeBitbucket:
    """Route registry; install into urllib.request.urlopen for the duration of a test."""

    def __init__(self):
        self.routes: dict[tuple[str, str], RouteHandler] = {}
        self.calls: list[dict[str, Any]] = []

    def route(self, method: str, path: str, handler: RouteHandler) -> None:
        self.routes[(method.upper(), path)] = handler

    def respond_json(self, method: str, path: str, status: int, body: dict[str, Any]) -> None:
        self.route(method, path, lambda q, b: (status, body))

    def respond_text(self, method: str, path: str, status: int, text: str) -> None:
        self.route(method, path, lambda q, b: (status, text))

    def urlopen(self, req: urllib.request.Request, timeout: float | None = None):
        parsed = urlsplit(req.full_url)
        path = parsed.path
        query_str = parsed.query
        method = req.get_method().upper()
        raw = req.data
        body: bytes | None = bytes(raw) if raw is not None else None  # pyright: ignore[reportArgumentType]
        self.calls.append({
            "method": method,
            "path": path,
            "query": query_str,
            "url": req.full_url,
            "body": body,
            "headers": dict(req.headers),
        })
        handler = self.routes.get((method, path))
        if handler is None:
            raise AssertionError(f"No fake route for {method} {path}")
        query = dict(
            kv.split("=", 1) for kv in query_str.split("&")
        ) if query_str else {}
        status, payload = handler(query, body)
        if isinstance(payload, (dict, list)):
            body_bytes = json.dumps(payload).encode("utf-8")
        else:
            body_bytes = str(payload).encode("utf-8")
        if status >= 400:
            raise _make_httperror(req.full_url, status, body_bytes)
        return _FakeHTTPResponse(status, body_bytes)


def _make_httperror(url: str, status: int, body_bytes: bytes) -> urllib.error.HTTPError:
    import io

    return urllib.error.HTTPError(
        url=url,
        code=status,
        msg=f"HTTP {status}",
        hdrs=None,  # pyright: ignore[reportArgumentType]
        fp=io.BytesIO(body_bytes),
    )


@pytest.fixture
def fake(monkeypatch):
    fb = FakeBitbucket()
    monkeypatch.setattr("scripts.onboard_repo.urllib.request.urlopen", fb.urlopen)
    return fb


@pytest.fixture
def client():
    return BitbucketHTTP(base_url=BASE_URL, token="test-token")


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
            "team": TEAM,
            "bitbucket_url": "https://bb.example.com",
            "webhook_url": f"https://noergler/webhook/{TEAM}",
            "projects": [
                {"project": "A", "repos": ["one", "two"]},
                {"project": "B", "repos": ["three"]},
            ],
        })
        result = load_onboarding_input(path)
        assert result.bitbucket_url == "https://bb.example.com"
        assert result.webhook_url == f"https://noergler/webhook/{TEAM}"
        assert result.team == TEAM
        assert [r.key for r in result.repos] == ["A/one", "A/two", "B/three"]

    def test_strips_trailing_slash_on_bitbucket_url(self, tmp_path):
        path = self._write(tmp_path, {
            "team": TEAM,
            "bitbucket_url": "https://bb.example.com/",
            "webhook_url": f"https://x/webhook/{TEAM}",
            "projects": [{"project": "A", "repos": ["one"]}],
        })
        assert load_onboarding_input(path).bitbucket_url == "https://bb.example.com"

    def test_rejects_http_bitbucket_url(self, tmp_path):
        path = self._write(tmp_path, {
            "team": TEAM,
            "bitbucket_url": "http://bb",
            "webhook_url": f"https://x/webhook/{TEAM}",
            "projects": [{"project": "A", "repos": ["one"]}],
        })
        with pytest.raises(SystemExit, match="bitbucket_url"):
            load_onboarding_input(path)

    def test_rejects_missing_webhook_url(self, tmp_path):
        path = self._write(tmp_path, {
            "team": TEAM,
            "bitbucket_url": "https://bb",
            "projects": [{"project": "A", "repos": ["one"]}],
        })
        with pytest.raises(SystemExit, match="webhook_url"):
            load_onboarding_input(path)

    def test_rejects_missing_projects(self, tmp_path):
        path = self._write(tmp_path, {
            "team": TEAM,
            "bitbucket_url": "https://bb",
            "webhook_url": f"https://x/webhook/{TEAM}",
        })
        with pytest.raises(SystemExit, match="projects"):
            load_onboarding_input(path)

    def test_rejects_empty_projects_list(self, tmp_path):
        path = self._write(tmp_path, {
            "team": TEAM,
            "bitbucket_url": "https://bb",
            "webhook_url": f"https://x/webhook/{TEAM}",
            "projects": [],
        })
        with pytest.raises(SystemExit, match="projects"):
            load_onboarding_input(path)

    def test_rejects_empty_repos_under_project(self, tmp_path):
        path = self._write(tmp_path, {
            "team": TEAM,
            "bitbucket_url": "https://bb",
            "webhook_url": f"https://x/webhook/{TEAM}",
            "projects": [{"project": "A", "repos": []}],
        })
        with pytest.raises(SystemExit, match="repos"):
            load_onboarding_input(path)

    def test_rejects_duplicate_repos(self, tmp_path):
        path = self._write(tmp_path, {
            "team": TEAM,
            "bitbucket_url": "https://bb",
            "webhook_url": f"https://x/webhook/{TEAM}",
            "projects": [
                {"project": "A", "repos": ["one", "one"]},
            ],
        })
        with pytest.raises(SystemExit, match="duplicate"):
            load_onboarding_input(path)

    def test_rejects_duplicate_across_project_entries(self, tmp_path):
        path = self._write(tmp_path, {
            "team": TEAM,
            "bitbucket_url": "https://bb",
            "webhook_url": f"https://x/webhook/{TEAM}",
            "projects": [
                {"project": "A", "repos": ["one"]},
                {"project": "A", "repos": ["one"]},
            ],
        })
        with pytest.raises(SystemExit, match="duplicate"):
            load_onboarding_input(path)

    def test_rejects_missing_project_field(self, tmp_path):
        path = self._write(tmp_path, {
            "team": TEAM,
            "bitbucket_url": "https://bb",
            "webhook_url": f"https://x/webhook/{TEAM}",
            "projects": [{"repos": ["one"]}],
        })
        with pytest.raises(SystemExit, match="project"):
            load_onboarding_input(path)


    def test_rejects_missing_team(self, tmp_path):
        path = self._write(tmp_path, {
            "bitbucket_url": "https://bb.example.com",
            "webhook_url": f"https://x/webhook/{TEAM}",
            "projects": [{"project": "A", "repos": ["one"]}],
        })
        with pytest.raises(SystemExit, match="'team'"):
            load_onboarding_input(path)

    def test_rejects_invalid_team_slug(self, tmp_path):
        path = self._write(tmp_path, {
            "team": "Platform Team",
            "bitbucket_url": "https://bb.example.com",
            "webhook_url": "https://x/webhook/platform",
            "projects": [{"project": "A", "repos": ["one"]}],
        })
        with pytest.raises(SystemExit, match="'team'"):
            load_onboarding_input(path)

    def test_rejects_webhook_url_without_the_team_path(self, tmp_path):
        # The service routes by /webhook/<team>; a bare /webhook is a 404.
        path = self._write(tmp_path, {
            "team": TEAM,
            "bitbucket_url": "https://bb.example.com",
            "webhook_url": "https://x/webhook",
            "projects": [{"project": "A", "repos": ["one"]}],
        })
        with pytest.raises(SystemExit, match="/webhook/platform"):
            load_onboarding_input(path)

    def test_rejects_webhook_url_for_another_team(self, tmp_path):
        path = self._write(tmp_path, {
            "team": TEAM,
            "bitbucket_url": "https://bb.example.com",
            "webhook_url": "https://x/webhook/payments",
            "projects": [{"project": "A", "repos": ["one"]}],
        })
        with pytest.raises(SystemExit, match="/webhook/platform"):
            load_onboarding_input(path)


# --------------------------------------------------------------------------- #
# teams.yaml lookup
# --------------------------------------------------------------------------- #

TEAMS_YAML = """
teams:
  - slug: platform
    webhook_secret_env: PLATFORM_HOOK
    projects:
      - key: PROJ
        repos: [r, good]
      - key: OPEN
    inference:
      api_key_env: X
"""


class TestTeamsLookup:
    def test_default_secret_var_follows_the_convention(self):
        assert default_webhook_secret_var("data-platform") == "TEAM_DATA_PLATFORM_WEBHOOK_SECRET"

    def test_load_team_entry_finds_the_slug(self, tmp_path):
        path = tmp_path / "teams.yaml"
        path.write_text(TEAMS_YAML)
        entry = load_team_entry(path, "platform")
        assert entry is not None
        assert entry["webhook_secret_env"] == "PLATFORM_HOOK"
        assert load_team_entry(path, "nobody") is None

    def test_load_team_entry_rejects_a_file_without_teams(self, tmp_path):
        path = tmp_path / "teams.yaml"
        path.write_text("foo: bar")
        with pytest.raises(SystemExit, match="teams:"):
            load_team_entry(path, "platform")

    def test_team_owns_respects_repo_lists(self, tmp_path):
        path = tmp_path / "teams.yaml"
        path.write_text(TEAMS_YAML)
        entry = load_team_entry(path, "platform")
        assert entry is not None
        assert team_owns(entry, RepoSpec("PROJ", "r"))
        assert not team_owns(entry, RepoSpec("PROJ", "other"))
        assert team_owns(entry, RepoSpec("OPEN", "anything"))
        assert not team_owns(entry, RepoSpec("NOPE", "x"))


# --------------------------------------------------------------------------- #
# resolve_secrets
# --------------------------------------------------------------------------- #

class TestResolveSecrets:
    def test_env_vars_win(self, monkeypatch, tmp_path):
        monkeypatch.setenv("BITBUCKET_TOKEN", "from-env")
        monkeypatch.setenv(SECRET_VAR, "sec-env")
        monkeypatch.chdir(tmp_path)
        token, secret = resolve_secrets(None, SECRET_VAR)
        assert (token, secret) == ("from-env", "sec-env")

    def test_env_file_fallback(self, monkeypatch, tmp_path):
        monkeypatch.delenv("BITBUCKET_TOKEN", raising=False)
        monkeypatch.delenv(SECRET_VAR, raising=False)
        monkeypatch.chdir(tmp_path)
        env_file = tmp_path / "extra.env"
        env_file.write_text(
            'BITBUCKET_TOKEN="file-token"\nTEAM_PLATFORM_WEBHOOK_SECRET=file-secret\n# ignored\n'
        )
        token, secret = resolve_secrets(env_file, SECRET_VAR)
        assert (token, secret) == ("file-token", "file-secret")

    def test_cwd_dotenv_used(self, monkeypatch, tmp_path):
        monkeypatch.delenv("BITBUCKET_TOKEN", raising=False)
        monkeypatch.delenv(SECRET_VAR, raising=False)
        (tmp_path / ".env").write_text(
            "BITBUCKET_TOKEN=cwd-token\nTEAM_PLATFORM_WEBHOOK_SECRET=cwd-secret\n"
        )
        monkeypatch.chdir(tmp_path)
        token, secret = resolve_secrets(None, SECRET_VAR)
        assert (token, secret) == ("cwd-token", "cwd-secret")

    def test_precedence_env_beats_cwd_beats_envfile(self, monkeypatch, tmp_path):
        """Pin down: process env > cwd .env > --env-file."""
        (tmp_path / ".env").write_text(
            "BITBUCKET_TOKEN=cwd-token\nTEAM_PLATFORM_WEBHOOK_SECRET=cwd-secret\n"
        )
        env_file = tmp_path / "lowest.env"
        env_file.write_text(
            "BITBUCKET_TOKEN=envfile-token\nTEAM_PLATFORM_WEBHOOK_SECRET=envfile-secret\n"
        )
        monkeypatch.chdir(tmp_path)

        monkeypatch.delenv("BITBUCKET_TOKEN", raising=False)
        monkeypatch.delenv(SECRET_VAR, raising=False)
        token, secret = resolve_secrets(env_file, SECRET_VAR)
        assert (token, secret) == ("cwd-token", "cwd-secret")

        monkeypatch.setenv("BITBUCKET_TOKEN", "env-token")
        monkeypatch.setenv(SECRET_VAR, "env-secret")
        token, secret = resolve_secrets(env_file, SECRET_VAR)
        assert (token, secret) == ("env-token", "env-secret")

    def test_missing_secrets_exits(self, monkeypatch, tmp_path):
        monkeypatch.delenv("BITBUCKET_TOKEN", raising=False)
        monkeypatch.delenv(SECRET_VAR, raising=False)
        monkeypatch.chdir(tmp_path)
        with pytest.raises(SystemExit):
            resolve_secrets(None, SECRET_VAR)


# --------------------------------------------------------------------------- #
# Drift guard: inlined constant must match app/config.py
# --------------------------------------------------------------------------- #

def test_required_webhook_events_match_app_config():
    from app.config import REQUIRED_WEBHOOK_EVENTS as canonical
    assert REQUIRED_WEBHOOK_EVENTS == canonical


# --------------------------------------------------------------------------- #
# RepoOnboarder
# --------------------------------------------------------------------------- #

class TestVerifyPermissions:
    def test_ok(self, onboarder, spec, fake):
        fake.respond_json("GET", "/rest/api/1.0/projects/PROJ/repos/my-repo", 200, {"slug": "my-repo"})
        fake.respond_json(
            "GET", "/rest/api/1.0/projects/PROJ/repos/my-repo/pull-requests", 200, {"values": []}
        )
        onboarder.verify_permissions(spec)

    def test_403_raises(self, onboarder, spec, fake):
        fake.respond_text("GET", "/rest/api/1.0/projects/PROJ/repos/my-repo", 403, "Forbidden")
        with pytest.raises(HTTPStatusError):
            onboarder.verify_permissions(spec)


class TestUpsertWebhook:
    def test_creates_when_absent(self, onboarder, spec, fake):
        fake.respond_json(
            "GET", "/rest/api/1.0/projects/PROJ/repos/my-repo/webhooks", 200,
            {"values": [], "isLastPage": True},
        )
        fake.respond_json(
            "POST", "/rest/api/1.0/projects/PROJ/repos/my-repo/webhooks", 200, {"id": 42},
        )

        webhook_id, diff = onboarder.upsert_webhook(spec)

        assert webhook_id == 42
        assert diff == ["create"]
        create_call = next(
            c for c in fake.calls
            if c["method"] == "POST" and c["path"].endswith("/webhooks")
        )
        body = json.loads(create_call["body"])
        assert body["name"] == DEFAULT_WEBHOOK_NAME
        assert body["url"] == WEBHOOK_URL
        assert set(body["events"]) == set(REQUIRED_WEBHOOK_EVENTS)
        assert body["configuration"]["secret"] == "whsec"
        assert body["active"] is True
        assert body["sslVerificationRequired"] is True

    def test_no_op_when_up_to_date(self, onboarder, spec, fake):
        fake.respond_json(
            "GET", "/rest/api/1.0/projects/PROJ/repos/my-repo/webhooks", 200,
            {
                "values": [{
                    "id": 7,
                    "name": DEFAULT_WEBHOOK_NAME,
                    "url": WEBHOOK_URL,
                    "events": list(REQUIRED_WEBHOOK_EVENTS),
                    "active": True,
                    "configuration": {"secret": "hidden"},
                }],
                "isLastPage": True,
            },
        )

        webhook_id, diff = onboarder.upsert_webhook(spec)
        assert webhook_id == 7
        assert diff == []

    def test_updates_on_drift(self, onboarder, spec, fake):
        fake.respond_json(
            "GET", "/rest/api/1.0/projects/PROJ/repos/my-repo/webhooks", 200,
            {
                "values": [{
                    "id": 7,
                    "name": DEFAULT_WEBHOOK_NAME,
                    "url": "https://stale.example/webhook",
                    "events": ["pr:opened"],
                    "active": True,
                    "configuration": {"secret": "x"},
                }],
                "isLastPage": True,
            },
        )
        fake.respond_json(
            "PUT", "/rest/api/1.0/projects/PROJ/repos/my-repo/webhooks/7", 200, {"id": 7},
        )

        webhook_id, diff = onboarder.upsert_webhook(spec)
        assert webhook_id == 7
        assert any("url" in d for d in diff)
        assert any("events" in d for d in diff)
        assert any(c["method"] == "PUT" for c in fake.calls)

    def test_dry_run_skips_writes(self, client, spec, fake):
        p = RepoOnboarder(
            client, webhook_url=WEBHOOK_URL, webhook_secret="whsec", dry_run=True,
        )
        fake.respond_json(
            "GET", "/rest/api/1.0/projects/PROJ/repos/my-repo/webhooks", 200,
            {"values": [], "isLastPage": True},
        )

        webhook_id, diff = p.upsert_webhook(spec)
        assert webhook_id == -1
        assert diff == ["create"]
        assert not any(c["method"] == "POST" for c in fake.calls)


class TestRemoveWebhook:
    def test_removes_existing(self, onboarder, spec, fake):
        fake.respond_json(
            "GET", "/rest/api/1.0/projects/PROJ/repos/my-repo/webhooks", 200,
            {
                "values": [{"id": 42, "name": DEFAULT_WEBHOOK_NAME}],
                "isLastPage": True,
            },
        )
        fake.respond_text(
            "DELETE", "/rest/api/1.0/projects/PROJ/repos/my-repo/webhooks/42", 204, "",
        )

        result = onboarder.remove_webhook(spec)
        assert result.status == "ok"
        assert "42" in result.detail
        assert any(c["method"] == "DELETE" for c in fake.calls)

    def test_skipped_when_absent(self, onboarder, spec, fake):
        fake.respond_json(
            "GET", "/rest/api/1.0/projects/PROJ/repos/my-repo/webhooks", 200,
            {"values": [], "isLastPage": True},
        )

        result = onboarder.remove_webhook(spec)
        assert result.status == "skipped"
        assert not any(c["method"] == "DELETE" for c in fake.calls)

    def test_dry_run_does_not_delete(self, client, spec, fake):
        p = RepoOnboarder(
            client, webhook_url=WEBHOOK_URL, webhook_secret="whsec", dry_run=True,
        )
        fake.respond_json(
            "GET", "/rest/api/1.0/projects/PROJ/repos/my-repo/webhooks", 200,
            {
                "values": [{"id": 42, "name": DEFAULT_WEBHOOK_NAME}],
                "isLastPage": True,
            },
        )

        result = p.remove_webhook(spec)
        assert result.status == "ok"
        assert "dry-run" in result.detail
        assert not any(c["method"] == "DELETE" for c in fake.calls)

    def test_ignores_other_webhook_names(self, onboarder, spec, fake):
        fake.respond_json(
            "GET", "/rest/api/1.0/projects/PROJ/repos/my-repo/webhooks", 200,
            {
                "values": [{"id": 99, "name": "other-hook"}],
                "isLastPage": True,
            },
        )

        result = onboarder.remove_webhook(spec)
        assert result.status == "skipped"
        assert not any(c["method"] == "DELETE" for c in fake.calls)


# --------------------------------------------------------------------------- #
# End-to-end via main()
# --------------------------------------------------------------------------- #

class TestMain:
    def test_multi_repo_mixed_results(self, tmp_path, monkeypatch, fake):
        cfg = tmp_path / "config.json"
        cfg.write_text(json.dumps({
            "team": TEAM,
            "bitbucket_url": BASE_URL,
            "webhook_url": WEBHOOK_URL,
            "projects": [
                {"project": "PROJ", "repos": ["good", "bad"]},
            ],
        }))
        monkeypatch.setenv("BITBUCKET_TOKEN", "t")
        monkeypatch.setenv(SECRET_VAR, "s")
        monkeypatch.chdir(tmp_path)

        fake.respond_json("GET", "/rest/api/1.0/projects/PROJ/repos/good", 200, {})
        fake.respond_json(
            "GET", "/rest/api/1.0/projects/PROJ/repos/good/pull-requests", 200, {"values": []},
        )
        fake.respond_json(
            "GET", "/rest/api/1.0/projects/PROJ/repos/good/webhooks", 200,
            {"values": [], "isLastPage": True},
        )
        fake.respond_json(
            "POST", "/rest/api/1.0/projects/PROJ/repos/good/webhooks", 200, {"id": 1},
        )
        fake.respond_text("GET", "/rest/api/1.0/projects/PROJ/repos/bad", 404, "no repo")

        rc = main([str(cfg)])
        assert rc == 1

    def test_dry_run_issues_no_writes(self, tmp_path, monkeypatch, capsys, fake):
        cfg = tmp_path / "config.json"
        cfg.write_text(json.dumps({
            "team": TEAM,
            "bitbucket_url": BASE_URL,
            "webhook_url": WEBHOOK_URL,
            "projects": [{"project": "PROJ", "repos": ["r"]}],
        }))
        monkeypatch.setenv("BITBUCKET_TOKEN", "t")
        monkeypatch.setenv(SECRET_VAR, "s")
        monkeypatch.chdir(tmp_path)

        fake.respond_json("GET", "/rest/api/1.0/projects/PROJ/repos/r", 200, {})
        fake.respond_json(
            "GET", "/rest/api/1.0/projects/PROJ/repos/r/pull-requests", 200, {"values": []},
        )
        fake.respond_json(
            "GET", "/rest/api/1.0/projects/PROJ/repos/r/webhooks", 200,
            {"values": [], "isLastPage": True},
        )

        rc = main([str(cfg), "--dry-run"])
        assert rc == 0
        assert not any(c["method"] == "POST" for c in fake.calls)
        out = capsys.readouterr().out
        assert "PROJ/r" in out
        assert "ok" in out

    def test_teams_file_names_the_secret_var_and_warns_on_foreign_repos(
        self, tmp_path, monkeypatch, fake, caplog,
    ):
        cfg = tmp_path / "config.json"
        cfg.write_text(json.dumps({
            "team": TEAM,
            "bitbucket_url": BASE_URL,
            "webhook_url": WEBHOOK_URL,
            "projects": [{"project": "PROJ", "repos": ["r", "foreign"]}],
        }))
        teams = tmp_path / "teams.yaml"
        teams.write_text(TEAMS_YAML)
        monkeypatch.setenv("BITBUCKET_TOKEN", "t")
        monkeypatch.setenv("PLATFORM_HOOK", "from-teams-file")
        monkeypatch.delenv(SECRET_VAR, raising=False)
        monkeypatch.chdir(tmp_path)

        for repo in ("r", "foreign"):
            fake.respond_json("GET", f"/rest/api/1.0/projects/PROJ/repos/{repo}", 200, {})
            fake.respond_json(
                "GET", f"/rest/api/1.0/projects/PROJ/repos/{repo}/pull-requests", 200, {"values": []},
            )
            fake.respond_json(
                "GET", f"/rest/api/1.0/projects/PROJ/repos/{repo}/webhooks", 200,
                {"values": [], "isLastPage": True},
            )

        rc = main([str(cfg), "--dry-run", "--teams", str(teams)])
        assert rc == 0
        assert "PROJ/foreign] not owned by team platform" in caplog.text
        assert "PROJ/r] not owned" not in caplog.text

    def test_teams_file_without_the_team_exits(self, tmp_path, monkeypatch, fake):
        cfg = tmp_path / "config.json"
        cfg.write_text(json.dumps({
            "team": "payments",
            "bitbucket_url": BASE_URL,
            "webhook_url": "https://noergler.internal/webhook/payments",
            "projects": [{"project": "PROJ", "repos": ["r"]}],
        }))
        teams = tmp_path / "teams.yaml"
        teams.write_text(TEAMS_YAML)
        monkeypatch.setenv("BITBUCKET_TOKEN", "t")
        monkeypatch.setenv("TEAMS_CONFIG", str(teams))
        monkeypatch.chdir(tmp_path)
        with pytest.raises(SystemExit, match="'payments' is not in"):
            main([str(cfg), "--dry-run"])

    def test_remove_via_main(self, tmp_path, monkeypatch, fake):
        cfg = tmp_path / "config.json"
        cfg.write_text(json.dumps({
            "team": TEAM,
            "bitbucket_url": BASE_URL,
            "webhook_url": WEBHOOK_URL,
            "projects": [{"project": "PROJ", "repos": ["r"]}],
        }))
        monkeypatch.setenv("BITBUCKET_TOKEN", "t")
        monkeypatch.setenv(SECRET_VAR, "s")
        monkeypatch.chdir(tmp_path)

        fake.respond_json(
            "GET", "/rest/api/1.0/projects/PROJ/repos/r/webhooks", 200,
            {
                "values": [{"id": 1, "name": DEFAULT_WEBHOOK_NAME}],
                "isLastPage": True,
            },
        )
        fake.respond_text(
            "DELETE", "/rest/api/1.0/projects/PROJ/repos/r/webhooks/1", 204, "",
        )

        rc = main([str(cfg), "--remove"])
        assert rc == 0
        assert sum(1 for c in fake.calls if c["method"] == "DELETE") == 1
        # Remove mode should skip permission check GETs (repo, pull-requests).
        assert not any(c["path"].endswith("/repos/r") for c in fake.calls)
