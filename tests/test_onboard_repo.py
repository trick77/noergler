import hashlib
import hmac
import io
import json
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Callable
from urllib.parse import urlsplit

import pytest

from scripts.onboard_repo import (
    DEFAULT_WEBHOOK_NAME,
    PROBE_EVENT_KEY,
    REQUIRED_WEBHOOK_EVENTS,
    BitbucketHTTP,
    HTTPStatusError,
    NoerglerProbe,
    Onboarder,
    Target,
    default_webhook_secret_var,
    load_onboarding_input,
    main,
    resolve_secrets,
)

BASE_URL = "https://bitbucket.company.com"
TEAM = "platform"
NOERGLER_URL = "https://noergler.internal"
WEBHOOK_URL = f"{NOERGLER_URL}/webhook/{TEAM}"
WEBHOOK_PATH = f"/webhook/{TEAM}"
SECRET = "whsec"
SECRET_VAR = "TEAM_PLATFORM_WEBHOOK_SECRET"
BOT = "noergler"

PROJ = Target(project="PROJ")
REPO = Target(project="PROJ", repo="my-repo")
PROJ_HOOKS = "/rest/api/1.0/projects/PROJ/webhooks"
REPO_HOOKS = "/rest/api/1.0/projects/PROJ/repos/my-repo/webhooks"


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


class FakeHTTP:
    """Route registry for both Bitbucket and noergler; installed into
    urllib.request.urlopen for the duration of a test."""

    def __init__(self):
        self.routes: dict[tuple[str, str], RouteHandler] = {}
        self.calls: list[dict[str, Any]] = []

    def route(self, method: str, path: str, handler: RouteHandler) -> None:
        self.routes[(method.upper(), path)] = handler

    def respond_json(self, method: str, path: str, status: int, body: Any) -> None:
        self.route(method, path, lambda q, b: (status, body))

    def respond_text(self, method: str, path: str, status: int, text: str) -> None:
        self.route(method, path, lambda q, b: (status, text))

    # -- noergler probe -- #
    def probe(
        self, owned: bool = True, bot_can_read: bool = True, status: int = 200, claim: str | None = None,
    ) -> None:
        """Answer the signed probe. Verifies the signature like the service does."""
        def handler(_q: dict[str, Any], body: bytes | None) -> tuple[int, Any]:
            assert body is not None
            expected = hmac.new(SECRET.encode(), body, hashlib.sha256).hexdigest()
            sig = self.calls[-1]["headers"].get("X-hub-signature", "")
            if sig != f"sha256={expected}":
                return 401, {"detail": "Invalid signature"}
            if status != 200:
                return status, {"detail": "nope"}
            req = json.loads(body)
            return 200, {
                "team": TEAM, "owned": owned, "bot_can_read": bot_can_read,
                "bot_username": BOT, "echo": req,
                # default: a teams.yaml consistent with the target (whole for a
                # project probe, repos for a repo probe)
                "claim": claim if claim is not None else (
                    "none" if not owned else ("whole" if req.get("repo") is None else "repos")
                ),
            }
        self.route("POST", WEBHOOK_PATH, handler)

    # -- Bitbucket shortcuts -- #
    def hooks(self, path: str, values: list[dict[str, Any]]) -> None:
        self.respond_json("GET", path, 200, {"values": values, "isLastPage": True})

    def repos(self, project: str, slugs: list[str]) -> None:
        self.respond_json(
            "GET", f"/rest/api/1.0/projects/{project}/repos", 200,
            {"values": [{"slug": s} for s in slugs], "isLastPage": True},
        )

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

    def calls_to(self, method: str, path: str) -> list[dict[str, Any]]:
        return [c for c in self.calls if c["method"] == method and c["path"] == path]


def _make_httperror(url: str, status: int, body_bytes: bytes) -> urllib.error.HTTPError:
    return urllib.error.HTTPError(
        url=url,
        code=status,
        msg=f"HTTP {status}",
        hdrs=None,  # pyright: ignore[reportArgumentType]
        fp=io.BytesIO(body_bytes),
    )


@pytest.fixture
def fake(monkeypatch):
    fb = FakeHTTP()
    monkeypatch.setattr("scripts.onboard_repo.urllib.request.urlopen", fb.urlopen)
    return fb


@pytest.fixture
def client():
    return BitbucketHTTP(base_url=BASE_URL, token="test-token")


def _onboarder(client, **kwargs) -> Onboarder:
    return Onboarder(
        client,
        NoerglerProbe(WEBHOOK_URL, SECRET),
        webhook_url=WEBHOOK_URL,
        webhook_secret=SECRET,
        webhook_name=DEFAULT_WEBHOOK_NAME,
        **kwargs,
    )


@pytest.fixture
def onboarder(client):
    return _onboarder(client)


def _good_hook(hook_id: int = 1) -> dict[str, Any]:
    return {
        "id": hook_id, "name": DEFAULT_WEBHOOK_NAME, "url": WEBHOOK_URL, "active": True,
        "events": list(REQUIRED_WEBHOOK_EVENTS), "configuration": {"secret": "x"},
    }


FOREIGN_URL = "https://noergler-intg.internal/webhook/platform"


def _foreign_hook(hook_id: int = 9) -> dict[str, Any]:
    """Same name, another noergler instance (intg next to prod)."""
    return _good_hook(hook_id) | {"url": FOREIGN_URL}


# --------------------------------------------------------------------------- #
# JSON config validation
# --------------------------------------------------------------------------- #

def _cfg(**overrides) -> dict[str, Any]:
    base = {
        "team": TEAM,
        "bitbucket_url": BASE_URL,
        "noergler_url": NOERGLER_URL,
        "projects": [{"key": "PLAT"}, {"key": "INFRA", "repos": ["terraform-core", "ansible"]}],
    }
    base.update(overrides)
    return base


class TestLoadOnboardingInput:
    def _write(self, tmp_path: Path, data) -> Path:
        p = tmp_path / "team.json"
        p.write_text(json.dumps(data))
        return p

    def test_whole_project_and_repo_list_become_targets(self, tmp_path):
        result = load_onboarding_input(self._write(tmp_path, _cfg()))
        assert result.team == TEAM
        assert result.webhook_url == WEBHOOK_URL
        assert result.targets == [
            Target("PLAT"), Target("INFRA", "terraform-core"), Target("INFRA", "ansible"),
        ]
        assert result.targets[0].is_project and not result.targets[1].is_project
        assert [t.label for t in result.targets][:2] == ["PLAT (project)", "INFRA/terraform-core"]

    def test_strips_trailing_slashes(self, tmp_path):
        cfg = _cfg(bitbucket_url=BASE_URL + "/", noergler_url=NOERGLER_URL + "/")
        result = load_onboarding_input(self._write(tmp_path, cfg))
        assert result.bitbucket_url == BASE_URL
        assert result.webhook_url == WEBHOOK_URL

    def test_rejects_old_format(self, tmp_path):
        with pytest.raises(SystemExit, match="old config format"):
            load_onboarding_input(self._write(tmp_path, {
                "team": TEAM, "bitbucket_url": BASE_URL, "webhook_url": WEBHOOK_URL,
                "projects": [{"project": "PROJ", "repos": ["r"]}],
            }))

    def test_rejects_noergler_url_with_a_path(self, tmp_path):
        with pytest.raises(SystemExit, match="base URL without a path"):
            load_onboarding_input(self._write(tmp_path, _cfg(noergler_url=WEBHOOK_URL)))

    @pytest.mark.parametrize("field", ["team", "bitbucket_url", "noergler_url", "projects"])
    def test_rejects_missing_field(self, tmp_path, field):
        cfg = _cfg()
        del cfg[field]
        with pytest.raises(SystemExit, match=field):
            load_onboarding_input(self._write(tmp_path, cfg))

    def test_rejects_invalid_team_slug(self, tmp_path):
        with pytest.raises(SystemExit, match="'team'"):
            load_onboarding_input(self._write(tmp_path, _cfg(team="Platform Team")))

    def test_rejects_http_bitbucket_url(self, tmp_path):
        with pytest.raises(SystemExit, match="bitbucket_url"):
            load_onboarding_input(self._write(tmp_path, _cfg(bitbucket_url="http://bb")))

    @pytest.mark.parametrize(
        ("projects", "message"),
        [
            ([], "'projects' must be a non-empty list"),
            (["PLAT"], "must be an object"),
            ([{"key": ""}], "key must be a non-empty string"),
            ([{"project": "PLAT"}], "old config format"),
            ([{"key": "PLAT", "extra": 1}], "unknown field"),
            ([{"key": "PLAT", "repos": []}], "non-empty list of repo slugs, or omitted"),
            ([{"key": "PLAT", "repos": [""]}], "must be a non-empty string"),
            ([{"key": "PLAT"}, {"key": "PLAT"}], "duplicate project entry"),
            ([{"key": "PLAT", "repos": ["a", "a"]}], "duplicate repo entry"),
            ([{"key": "PLAT", "repos": ["a"]}, {"key": "PLAT", "repos": ["a"]}], "duplicate repo entry"),
            ([{"key": "PLAT"}, {"key": "PLAT", "repos": ["a"]}], "both whole and with repos"),
        ],
    )
    def test_rejects_bad_projects(self, tmp_path, projects, message):
        with pytest.raises(SystemExit, match=message):
            load_onboarding_input(self._write(tmp_path, _cfg(projects=projects)))


# --------------------------------------------------------------------------- #
# resolve_secrets
# --------------------------------------------------------------------------- #

class TestResolveSecrets:
    def test_default_secret_var_follows_the_convention(self):
        assert default_webhook_secret_var("data-platform") == "TEAM_DATA_PLATFORM_WEBHOOK_SECRET"

    def test_env_vars_win(self, monkeypatch, tmp_path):
        monkeypatch.setenv("BITBUCKET_TOKEN", "from-env")
        monkeypatch.setenv(SECRET_VAR, "sec-env")
        monkeypatch.chdir(tmp_path)
        assert resolve_secrets(None, SECRET_VAR) == ("from-env", "sec-env")

    def test_env_file_fallback(self, monkeypatch, tmp_path):
        monkeypatch.delenv("BITBUCKET_TOKEN", raising=False)
        monkeypatch.delenv(SECRET_VAR, raising=False)
        monkeypatch.chdir(tmp_path)
        env_file = tmp_path / "extra.env"
        env_file.write_text(f'BITBUCKET_TOKEN="file-token"\n{SECRET_VAR}=file-secret\n# ignored\n')
        assert resolve_secrets(env_file, SECRET_VAR) == ("file-token", "file-secret")

    def test_precedence_env_beats_cwd_beats_envfile(self, monkeypatch, tmp_path):
        (tmp_path / ".env").write_text(f"BITBUCKET_TOKEN=cwd-token\n{SECRET_VAR}=cwd-secret\n")
        env_file = tmp_path / "lowest.env"
        env_file.write_text(f"BITBUCKET_TOKEN=envfile-token\n{SECRET_VAR}=envfile-secret\n")
        monkeypatch.chdir(tmp_path)
        monkeypatch.delenv("BITBUCKET_TOKEN", raising=False)
        monkeypatch.delenv(SECRET_VAR, raising=False)
        assert resolve_secrets(env_file, SECRET_VAR) == ("cwd-token", "cwd-secret")
        monkeypatch.setenv("BITBUCKET_TOKEN", "env-token")
        monkeypatch.setenv(SECRET_VAR, "env-secret")
        assert resolve_secrets(env_file, SECRET_VAR) == ("env-token", "env-secret")

    def test_custom_secret_var(self, monkeypatch, tmp_path):
        monkeypatch.setenv("BITBUCKET_TOKEN", "t")
        monkeypatch.setenv("PLATFORM_HOOK", "s")
        monkeypatch.chdir(tmp_path)
        assert resolve_secrets(None, "PLATFORM_HOOK") == ("t", "s")

    def test_missing_secrets_exits(self, monkeypatch, tmp_path, capsys):
        monkeypatch.delenv("BITBUCKET_TOKEN", raising=False)
        monkeypatch.delenv(SECRET_VAR, raising=False)
        monkeypatch.chdir(tmp_path)
        with pytest.raises(SystemExit):
            resolve_secrets(None, SECRET_VAR)
        assert SECRET_VAR in capsys.readouterr().err


# --------------------------------------------------------------------------- #
# Probe
# --------------------------------------------------------------------------- #

class TestProbe:
    def test_signs_body_and_reads_answer(self, fake):
        fake.probe(owned=True, bot_can_read=False)
        result = NoerglerProbe(WEBHOOK_URL, SECRET).probe(REPO)
        assert (result.owned, result.bot_can_read, result.bot_username) == (True, False, BOT)
        call = fake.calls[-1]
        assert call["headers"]["X-event-key"] == PROBE_EVENT_KEY
        assert json.loads(call["body"]) == {"project": "PROJ", "repo": "my-repo"}

    def test_project_probe_sends_null_repo(self, fake):
        fake.probe()
        NoerglerProbe(WEBHOOK_URL, SECRET).probe(PROJ)
        assert json.loads(fake.calls[-1]["body"]) == {"project": "PROJ", "repo": None}

    def test_wrong_secret_exits_with_hint(self, fake):
        fake.probe()
        with pytest.raises(SystemExit, match="secret does not match"):
            NoerglerProbe(WEBHOOK_URL, "other").probe(PROJ)

    @pytest.mark.parametrize(("status", "hint"), [(404, "no such team"), (503, "disabled this team")])
    def test_team_level_errors_exit(self, fake, status, hint):
        fake.probe(status=status)
        with pytest.raises(SystemExit, match=hint):
            NoerglerProbe(WEBHOOK_URL, SECRET).probe(PROJ)


# --------------------------------------------------------------------------- #
# Onboarding a project / a repo
# --------------------------------------------------------------------------- #

class TestOnboardProject:
    def test_creates_project_webhook_and_prunes_repo_hooks(self, onboarder, fake):
        fake.probe()
        fake.respond_json("GET", "/rest/api/1.0/projects/PROJ", 200, {"key": "PROJ"})
        fake.hooks(PROJ_HOOKS, [])
        fake.respond_json("POST", PROJ_HOOKS, 201, {"id": 7})
        fake.repos("PROJ", ["a", "b"])
        fake.hooks("/rest/api/1.0/projects/PROJ/repos/a/webhooks", [_good_hook(3)])
        fake.hooks("/rest/api/1.0/projects/PROJ/repos/b/webhooks", [{"id": 4, "name": "jenkins"}])
        fake.respond_text("DELETE", "/rest/api/1.0/projects/PROJ/repos/a/webhooks/3", 204, "")

        result = onboarder.onboard(PROJ)

        assert result.status == "ok"
        assert "webhook created (id=7)" in result.detail
        assert "pruned 1 repo hook(s): PROJ/a" in result.detail
        created = fake.calls_to("POST", PROJ_HOOKS)[0]
        body = json.loads(created["body"])
        assert body["url"] == WEBHOOK_URL
        assert body["configuration"] == {"secret": SECRET}
        assert set(body["events"]) == set(REQUIRED_WEBHOOK_EVENTS)
        assert len(fake.calls_to("DELETE", "/rest/api/1.0/projects/PROJ/repos/a/webhooks/3")) == 1
        # the jenkins hook on b is not ours
        assert not fake.calls_to("DELETE", "/rest/api/1.0/projects/PROJ/repos/b/webhooks/4")

    def test_no_prune_keeps_repo_hooks(self, client, fake):
        fake.probe()
        fake.respond_json("GET", "/rest/api/1.0/projects/PROJ", 200, {})
        fake.hooks(PROJ_HOOKS, [_good_hook()])
        result = _onboarder(client, prune=False).onboard(PROJ)
        assert result.status == "ok"
        assert result.detail == "webhook already up to date"
        assert not fake.calls_to("GET", "/rest/api/1.0/projects/PROJ/repos")

    def test_not_owned_is_skipped_before_touching_bitbucket(self, onboarder, fake):
        fake.probe(owned=False)
        result = onboarder.onboard(PROJ)
        assert result.status == "skipped"
        assert "ask the noergler admin" in result.detail
        assert all(c["path"] == WEBHOOK_PATH for c in fake.calls)

    def test_project_claimed_by_repos_tells_admin_to_list_repos(self, onboarder, fake):
        fake.probe(owned=False, claim="repos")
        result = onboarder.onboard(PROJ)
        assert result.status == "skipped"
        assert 'list them under "repos" in team.json' in result.detail

    def test_repo_under_whole_claim_tells_admin_to_use_project_form(self, onboarder, fake):
        # teams.yaml claims all of PROJ, so noergler owns the repo, but team.json
        # lists it: a repo hook next to the project hook would deliver twice.
        fake.probe(owned=True, claim="whole")
        result = onboarder.onboard(REPO)
        assert result.status == "skipped"
        assert '{"key": "PROJ"}' in result.detail
        assert "twice" in result.detail
        assert not fake.calls_to("POST", REPO_HOOKS)

    def test_repo_target_refused_while_own_project_hook_remains(self, onboarder, fake):
        # whole claim turned into a repos: list; the old project hook is still there
        fake.probe(claim="repos")
        fake.hooks(PROJ_HOOKS, [_good_hook(5)])
        result = onboarder.onboard(REPO)
        assert result.status == "skipped"
        assert "id=5" in result.detail and '--remove with {"key": "PROJ"}' in result.detail
        assert not fake.calls_to("POST", REPO_HOOKS)

    def test_repo_target_ignores_foreign_project_hook(self, onboarder, fake):
        fake.probe(claim="repos")
        fake.hooks(PROJ_HOOKS, [_foreign_hook(5)])
        fake.respond_json("GET", "/rest/api/1.0/projects/PROJ/repos/my-repo", 200, {})
        fake.hooks(REPO_HOOKS, [])
        fake.respond_json("POST", REPO_HOOKS, 201, {"id": 9})
        assert onboarder.onboard(REPO).status == "ok"

    def test_repo_target_without_project_admin_still_onboards(self, onboarder, fake):
        fake.probe(claim="repos")
        fake.respond_json("GET", PROJ_HOOKS, 401, {"errors": [{"message": "no"}]})
        fake.respond_json("GET", "/rest/api/1.0/projects/PROJ/repos/my-repo", 200, {})
        fake.hooks(REPO_HOOKS, [])
        fake.respond_json("POST", REPO_HOOKS, 201, {"id": 9})
        assert onboarder.onboard(REPO).status == "ok"

    def test_prune_leaves_hooks_of_another_instance_alone(self, onboarder, fake):
        fake.probe()
        fake.respond_json("GET", "/rest/api/1.0/projects/PROJ", 200, {})
        fake.hooks(PROJ_HOOKS, [])
        fake.respond_json("POST", PROJ_HOOKS, 201, {"id": 7})
        fake.repos("PROJ", ["a", "b"])
        fake.hooks("/rest/api/1.0/projects/PROJ/repos/a/webhooks", [_foreign_hook(3)])
        fake.hooks("/rest/api/1.0/projects/PROJ/repos/b/webhooks", [_good_hook(4)])
        fake.respond_text("DELETE", "/rest/api/1.0/projects/PROJ/repos/b/webhooks/4", 204, "")

        result = onboarder.onboard(PROJ)

        assert result.status == "ok"
        assert "pruned 1 repo hook(s): PROJ/b" in result.detail
        assert not [c for c in fake.calls if c["method"] == "DELETE" and "/repos/a/" in c["path"]]

    def test_hook_of_another_instance_is_never_rewritten(self, onboarder, fake):
        fake.probe()
        fake.respond_json("GET", "/rest/api/1.0/projects/PROJ", 200, {})
        fake.hooks(PROJ_HOOKS, [_foreign_hook(5)])
        result = onboarder.onboard(PROJ)
        assert result.status == "failed"
        assert FOREIGN_URL in result.detail
        assert "--name" in result.detail
        assert not fake.calls_to("PUT", PROJ_HOOKS + "/5")
        assert not fake.calls_to("POST", PROJ_HOOKS)

    def test_bot_without_access_is_skipped_unless_grant(self, onboarder, fake):
        fake.probe(bot_can_read=False)
        fake.respond_json("GET", "/rest/api/1.0/projects/PROJ", 200, {})
        result = onboarder.onboard(PROJ)
        assert result.status == "skipped"
        assert f"grant {BOT} PROJECT_WRITE" in result.detail
        assert not fake.calls_to("GET", PROJ_HOOKS)

    def test_grant_bot_grants_project_write_then_onboards(self, client, fake):
        fake.probe(bot_can_read=False)
        fake.respond_json("GET", "/rest/api/1.0/projects/PROJ", 200, {})
        fake.respond_text("PUT", "/rest/api/1.0/projects/PROJ/permissions/users", 204, "")
        fake.hooks(PROJ_HOOKS, [])
        fake.respond_json("POST", PROJ_HOOKS, 201, {"id": 1})
        fake.repos("PROJ", [])
        result = _onboarder(client, grant_bot=True).onboard(PROJ)
        assert result.status == "ok"
        assert f"{BOT} granted PROJECT_WRITE" in result.detail
        grant = fake.calls_to("PUT", "/rest/api/1.0/projects/PROJ/permissions/users")[0]
        assert grant["query"] == f"name={BOT}&permission=PROJECT_WRITE"

    def test_admin_without_access_fails(self, onboarder, fake):
        fake.probe()
        fake.respond_text("GET", "/rest/api/1.0/projects/PROJ", 403, "forbidden")
        result = onboarder.onboard(PROJ)
        assert result.status == "failed"
        assert "admin access check HTTP 403" in result.detail

    def test_dry_run_writes_nothing(self, client, fake):
        fake.probe(bot_can_read=False)
        fake.respond_json("GET", "/rest/api/1.0/projects/PROJ", 200, {})
        fake.hooks(PROJ_HOOKS, [])
        fake.repos("PROJ", ["a"])
        fake.hooks("/rest/api/1.0/projects/PROJ/repos/a/webhooks", [_good_hook(3)])
        result = _onboarder(client, dry_run=True, grant_bot=True).onboard(PROJ)
        assert result.status == "ok"
        assert result.detail.startswith("dry-run: ")
        assert "pruned 1 repo hook(s)" in result.detail
        assert not [c for c in fake.calls if c["method"] in ("POST", "PUT", "DELETE") and c["path"] != WEBHOOK_PATH]


class TestOnboardRepo:
    def test_creates_repo_webhook(self, onboarder, fake):
        fake.probe()
        fake.hooks(PROJ_HOOKS, [])  # no project hook next to the repo hook
        fake.respond_json("GET", "/rest/api/1.0/projects/PROJ/repos/my-repo", 200, {})
        fake.hooks(REPO_HOOKS, [])
        fake.respond_json("POST", REPO_HOOKS, 201, {"id": 9})
        result = onboarder.onboard(REPO)
        assert result.status == "ok"
        assert result.detail == "webhook created (id=9)"
        # a repo target never lists the project's repos (no prune)
        assert not fake.calls_to("GET", "/rest/api/1.0/projects/PROJ/repos")

    def test_updates_on_drift(self, onboarder, fake):
        stale = _good_hook(2) | {"active": False, "events": ["pr:opened"]}
        fake.probe()
        fake.hooks(PROJ_HOOKS, [])  # no project hook next to the repo hook
        fake.respond_json("GET", "/rest/api/1.0/projects/PROJ/repos/my-repo", 200, {})
        fake.hooks(REPO_HOOKS, [stale])
        fake.respond_json("PUT", REPO_HOOKS + "/2", 200, {"id": 2})
        result = onboarder.onboard(REPO)
        assert result.status == "ok"
        assert result.detail == "webhook updated (id=2)"
        assert any(d.startswith("active:") for d in result.diff)
        assert any(d.startswith("events:") for d in result.diff)

    def test_grant_bot_uses_repo_write(self, client, fake):
        fake.probe(bot_can_read=False)
        fake.hooks(PROJ_HOOKS, [])  # no project hook next to the repo hook
        fake.respond_json("GET", "/rest/api/1.0/projects/PROJ/repos/my-repo", 200, {})
        fake.respond_text("PUT", "/rest/api/1.0/projects/PROJ/repos/my-repo/permissions/users", 204, "")
        fake.hooks(REPO_HOOKS, [_good_hook()])
        result = _onboarder(client, grant_bot=True).onboard(REPO)
        assert result.status == "ok"
        grant = fake.calls_to("PUT", "/rest/api/1.0/projects/PROJ/repos/my-repo/permissions/users")[0]
        assert grant["query"] == f"name={BOT}&permission=REPO_WRITE"

    def test_upsert_failure_is_reported(self, onboarder, fake):
        fake.probe()
        fake.hooks(PROJ_HOOKS, [])  # no project hook next to the repo hook
        fake.respond_json("GET", "/rest/api/1.0/projects/PROJ/repos/my-repo", 200, {})
        fake.respond_text("GET", REPO_HOOKS, 500, "boom")
        result = onboarder.onboard(REPO)
        assert result.status == "failed"
        assert "upsert webhook HTTP 500" in result.detail


class TestPagination:
    def test_list_webhooks_follows_pages(self, client, fake):
        pages = {
            "0": {"values": [{"id": 1, "name": "x"}], "isLastPage": False, "nextPageStart": 1},
            "1": {"values": [{"id": 2, "name": "y"}], "isLastPage": True},
        }
        fake.route("GET", PROJ_HOOKS, lambda q, b: (200, pages[q["start"]]))
        assert [h["id"] for h in client.list_webhooks(PROJ)] == [1, 2]


# --------------------------------------------------------------------------- #
# Status
# --------------------------------------------------------------------------- #

class TestStatus:
    def test_reports_state_without_writing(self, onboarder, fake):
        fake.probe(bot_can_read=False)
        fake.hooks(PROJ_HOOKS, [_good_hook() | {"active": False}])
        fake.repos("PROJ", ["a"])
        fake.hooks("/rest/api/1.0/projects/PROJ/repos/a/webhooks", [_good_hook(3)])
        row = onboarder.status(PROJ)
        assert (row.owned, row.bot_can_read) == (True, False)
        assert row.webhook == "stale: active: False -> True"
        assert row.stray == ["PROJ/a"]
        assert all(c["method"] in ("GET", "POST") for c in fake.calls)
        assert not [c for c in fake.calls if c["method"] == "POST" and c["path"] != WEBHOOK_PATH]

    def test_missing_hook(self, onboarder, fake):
        fake.probe()
        fake.hooks(PROJ_HOOKS, [])  # no project hook next to the repo hook
        fake.hooks(REPO_HOOKS, [])
        row = onboarder.status(REPO)
        assert row.webhook == "missing"
        assert row.stray == []

    def test_foreign_hooks_are_reported_not_counted_as_stray(self, onboarder, fake):
        fake.probe()
        fake.hooks(PROJ_HOOKS, [_foreign_hook(1)])
        fake.repos("PROJ", ["a"])
        fake.hooks("/rest/api/1.0/projects/PROJ/repos/a/webhooks", [_foreign_hook(3)])
        row = onboarder.status(PROJ)
        assert row.webhook == f"foreign: {FOREIGN_URL}"
        assert row.stray == []
        assert row.foreign == [f"PROJ/a -> {FOREIGN_URL}"]

    def test_not_owned_row_carries_the_reason(self, onboarder, fake):
        fake.probe(owned=False, claim="repos")
        row = onboarder.status(PROJ)
        assert row.owned is False
        assert 'list them under "repos"' in row.webhook

    def test_repo_under_whole_claim_is_blocked(self, onboarder, fake):
        fake.probe(claim="whole")
        row = onboarder.status(REPO)
        assert row.owned is True
        assert row.webhook.startswith("blocked: ") and '{"key": "PROJ"}' in row.webhook

    def test_ssl_verification_off_is_stale(self, onboarder, fake):
        fake.probe()
        fake.hooks(PROJ_HOOKS, [_good_hook() | {"sslVerificationRequired": False}])
        fake.repos("PROJ", [])
        assert onboarder.status(PROJ).webhook == "stale: sslVerificationRequired: False -> True"


# --------------------------------------------------------------------------- #
# Remove
# --------------------------------------------------------------------------- #

class TestRemoveWebhook:
    def test_removes_project_hook(self, onboarder, fake):
        fake.hooks(PROJ_HOOKS, [_good_hook(5)])
        fake.respond_text("DELETE", PROJ_HOOKS + "/5", 204, "")
        result = onboarder.remove_webhook(PROJ)
        assert result.status == "ok"
        assert len(fake.calls_to("DELETE", PROJ_HOOKS + "/5")) == 1

    def test_skipped_when_absent(self, onboarder, fake):
        fake.hooks(REPO_HOOKS, [{"id": 1, "name": "jenkins"}])
        assert onboarder.remove_webhook(REPO).status == "skipped"

    def test_skips_hook_of_another_instance(self, onboarder, fake):
        fake.hooks(REPO_HOOKS, [_foreign_hook(5)])
        result = onboarder.remove_webhook(REPO)
        assert result.status == "skipped"
        assert FOREIGN_URL in result.detail
        assert not fake.calls_to("DELETE", REPO_HOOKS + "/5")

    def test_dry_run_does_not_delete(self, client, fake):
        fake.hooks(REPO_HOOKS, [_good_hook(5)])
        result = _onboarder(client, dry_run=True).remove_webhook(REPO)
        assert result.status == "ok"
        assert not fake.calls_to("DELETE", REPO_HOOKS + "/5")


# --------------------------------------------------------------------------- #
# End-to-end via main()
# --------------------------------------------------------------------------- #

class TestMain:
    def _config(self, tmp_path, monkeypatch, projects) -> Path:
        cfg = tmp_path / "team.json"
        cfg.write_text(json.dumps(_cfg(projects=projects)))
        monkeypatch.setenv("BITBUCKET_TOKEN", "t")
        monkeypatch.setenv(SECRET_VAR, SECRET)
        monkeypatch.chdir(tmp_path)
        return cfg

    def test_mixed_targets_and_results(self, tmp_path, monkeypatch, fake, capsys):
        cfg = self._config(tmp_path, monkeypatch, [{"key": "PROJ"}, {"key": "OTHER", "repos": ["r"]}])
        fake.probe()
        fake.respond_json("GET", "/rest/api/1.0/projects/PROJ", 200, {})
        fake.hooks(PROJ_HOOKS, [])
        fake.respond_json("POST", PROJ_HOOKS, 201, {"id": 1})
        fake.repos("PROJ", [])
        fake.respond_text("GET", "/rest/api/1.0/projects/OTHER/repos/r", 404, "no repo")

        rc = main([str(cfg)])

        assert rc == 1
        out = capsys.readouterr().out
        assert "PROJ (project)" in out and "ok" in out
        assert "OTHER/r" in out and "failed" in out

    def test_status_mode_exit_code_reflects_health(self, tmp_path, monkeypatch, fake, capsys):
        cfg = self._config(tmp_path, monkeypatch, [{"key": "PROJ"}])
        fake.probe()
        fake.hooks(PROJ_HOOKS, [_good_hook()])
        fake.repos("PROJ", [])
        assert main([str(cfg), "--status"]) == 0
        assert "PROJ (project)" in capsys.readouterr().out

        fake.hooks(PROJ_HOOKS, [])
        assert main([str(cfg), "--status"]) == 1
        assert "missing" in capsys.readouterr().out

    def test_secret_env_override(self, tmp_path, monkeypatch, fake):
        cfg = self._config(tmp_path, monkeypatch, [{"key": "PROJ"}])
        monkeypatch.delenv(SECRET_VAR)
        monkeypatch.setenv("PLATFORM_HOOK", SECRET)
        fake.probe()
        fake.hooks(PROJ_HOOKS, [_good_hook()])
        fake.repos("PROJ", [])
        assert main([str(cfg), "--status", "--secret-env", "PLATFORM_HOOK"]) == 0

    def test_wrong_secret_aborts_the_run(self, tmp_path, monkeypatch, fake):
        cfg = self._config(tmp_path, monkeypatch, [{"key": "PROJ"}])
        monkeypatch.setenv(SECRET_VAR, "wrong")
        fake.probe()
        with pytest.raises(SystemExit, match="secret does not match"):
            main([str(cfg)])

    def test_remove_via_main(self, tmp_path, monkeypatch, fake):
        cfg = self._config(tmp_path, monkeypatch, [{"key": "PROJ", "repos": ["my-repo"]}])
        fake.hooks(REPO_HOOKS, [_good_hook(1)])
        fake.respond_text("DELETE", REPO_HOOKS + "/1", 204, "")
        assert main([str(cfg), "--remove"]) == 0
        assert len(fake.calls_to("DELETE", REPO_HOOKS + "/1")) == 1
        # remove mode neither probes nor checks access
        assert not fake.calls_to("POST", WEBHOOK_PATH)
