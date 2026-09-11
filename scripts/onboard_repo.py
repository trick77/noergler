"""
Onboard a team's Bitbucket Server projects and repositories to noergler.

Usage:
    python -m scripts.onboard_repo team.json [--status | --remove] [--grant-bot] [--no-prune]
                                             [--dry-run] [--name noergler] [--env-file PATH]
                                             [--secret-env VAR]

Run by the team admin. Needs three things and nothing else:
    - BITBUCKET_TOKEN:               the admin's own personal access token
                                     (project/repo admin, it creates webhooks)
    - TEAM_<SLUG>_WEBHOOK_SECRET:    the team's webhook secret, handed over by
                                     the noergler admin (override the variable
                                     name with --secret-env)
    - team.json:                     see below
Resolved from, in order: process environment > .env in CWD > --env-file.

team.json mirrors the team's block in noergler's teams.yaml:

    {
      "team": "platform",
      "bitbucket_url": "https://git.company.com",
      "noergler_url": "https://noergler.company.com",
      "projects": [
        {"key": "PLAT"},                                  whole project: ONE project
                                                          webhook, covers every current
                                                          and future repo
        {"key": "INFRA", "repos": ["terraform-core"]}     only these repos: one repo
                                                          webhook each (a project
                                                          shared between teams)
      ]
    }

Per target the script asks noergler (a signed probe against
<noergler_url>/webhook/<team>) whether the team owns it and whether the
noergler bot can read it, then creates or updates the webhook idempotently
with the admin's token. Under a project webhook, leftover repo-level hooks
would deliver every event twice, so they are pruned (--no-prune keeps them).
--status only reports; --grant-bot gives the bot write access where it has
none. End-to-end delivery is verified by opening a real PR; Bitbucket's
"/test" endpoint is intentionally not used.

Project-level webhooks need Bitbucket Data Center 8.8 or newer.

This script is intentionally stdlib-only so it can run on any host with Python
3.10+ without a venv or `pip install` (it also ships in the noergler image as
`onboard`).
"""
from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import logging
import os
import re
import sys
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

logger = logging.getLogger("onboard_repo")

DEFAULT_WEBHOOK_NAME = "noergler"
PROBE_EVENT_KEY = "noergler:probe"

# Must match TEAM_SLUG_RE / team_env_prefix in app/config.py. Inlined to keep
# this script stdlib-only.
TEAM_SLUG_RE = re.compile(r"^[a-z0-9][a-z0-9-]*$")


def team_env_prefix(slug: str) -> str:
    return f"TEAM_{slug.upper().replace('-', '_')}_"


def default_webhook_secret_var(slug: str) -> str:
    return team_env_prefix(slug) + "WEBHOOK_SECRET"


# Must match REQUIRED_WEBHOOK_EVENTS in app/config.py. Inlined here to keep this
# script stdlib-only (no import of app.config, which pulls in pydantic).
REQUIRED_WEBHOOK_EVENTS: tuple[str, ...] = (
    "pr:opened",
    "pr:from_ref_updated",
    "pr:comment:added",
    "pr:comment:deleted",
    "pr:merged",
    "pr:declined",
    "pr:deleted",
)


# --------------------------------------------------------------------------- #
# Config loading
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class Target:
    """A project (repo=None: one project webhook) or a single repo."""

    project: str
    repo: str | None = None

    @property
    def is_project(self) -> bool:
        return self.repo is None

    @property
    def key(self) -> str:
        return self.project if self.repo is None else f"{self.project}/{self.repo}"

    @property
    def label(self) -> str:
        return f"{self.project} (project)" if self.repo is None else self.key

    @property
    def api_path(self) -> str:
        base = f"/rest/api/1.0/projects/{self.project}"
        return base if self.repo is None else f"{base}/repos/{self.repo}"

    @property
    def bot_permission(self) -> str:
        return "PROJECT_WRITE" if self.repo is None else "REPO_WRITE"


@dataclass(frozen=True)
class OnboardingInput:
    team: str
    bitbucket_url: str
    noergler_url: str
    targets: list[Target]

    @property
    def webhook_url(self) -> str:
        return f"{self.noergler_url}/webhook/{self.team}"


def _load_env_file(path: Path) -> dict[str, str]:
    """Minimal .env parser: KEY=VALUE lines, ignores comments and blanks."""
    if not path.exists():
        return {}
    env: dict[str, str] = {}
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key:
            env[key] = value
    return env


def resolve_secrets(env_file: Path | None, secret_var: str) -> tuple[str, str]:
    """Return (BITBUCKET_TOKEN, <the team's webhook secret>).

    `secret_var` names the env var holding the team's secret (see
    `default_webhook_secret_var` and `--secret-env`).
    Precedence (first match wins): process env > cwd .env > --env-file.
    Raises SystemExit on missing.
    """
    merged: dict[str, str] = {}
    if env_file is not None:
        merged.update(_load_env_file(env_file))
    merged.update(_load_env_file(Path.cwd() / ".env"))
    for k in ("BITBUCKET_TOKEN", secret_var):
        if k in os.environ and os.environ[k]:
            merged[k] = os.environ[k]

    missing = [k for k in ("BITBUCKET_TOKEN", secret_var) if not merged.get(k)]
    if missing:
        sys.stderr.write(
            "ERROR: missing required environment variable(s): "
            + ", ".join(missing)
            + "\nSet them in the process env, a .env in CWD, or pass --env-file.\n"
        )
        raise SystemExit(2)
    return merged["BITBUCKET_TOKEN"], merged[secret_var]


def _mask(secret: str) -> str:
    if len(secret) <= 4:
        return "****"
    return f"{secret[:4]}-****"


def load_onboarding_input(path: Path) -> OnboardingInput:
    try:
        data = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"ERROR: cannot read {path}: {exc}")

    if not isinstance(data, dict):
        raise SystemExit("ERROR: config root must be a JSON object")
    if "webhook_url" in data or any(
        isinstance(p, dict) and "project" in p for p in (data.get("projects") or [])
    ):
        raise SystemExit(
            "ERROR: old config format. Use 'noergler_url' (the service base URL) and "
            "'projects': [{\"key\": ...}, {\"key\": ..., \"repos\": [...]}] — the same "
            "shape as the team's block in teams.yaml"
        )

    team = data.get("team")
    if not isinstance(team, str) or not TEAM_SLUG_RE.match(team):
        raise SystemExit(
            f"ERROR: 'team' must be the team's slug ({TEAM_SLUG_RE.pattern}), "
            "exactly as in teams.yaml"
        )
    bitbucket_url = _require_https(data.get("bitbucket_url"), "bitbucket_url")
    noergler_url = _require_url(data.get("noergler_url"), "noergler_url").rstrip("/")
    if urlparse(noergler_url).path not in ("", "/"):
        raise SystemExit(
            "ERROR: 'noergler_url' is the service base URL without a path; the script "
            f"appends /webhook/{team} itself"
        )
    projects_raw = data.get("projects")
    if not isinstance(projects_raw, list) or not projects_raw:
        raise SystemExit("ERROR: 'projects' must be a non-empty list")

    seen: set[str] = set()
    targets: list[Target] = []
    for p_idx, entry in enumerate(projects_raw):
        if not isinstance(entry, dict):
            raise SystemExit(f"ERROR: projects[{p_idx}] must be an object")
        unknown = set(entry) - {"key", "repos"}
        if unknown:
            raise SystemExit(f"ERROR: projects[{p_idx}] has unknown field(s): {sorted(unknown)}")
        key = entry.get("key")
        if not isinstance(key, str) or not key.strip():
            raise SystemExit(f"ERROR: projects[{p_idx}].key must be a non-empty string")
        project_key = key.strip()
        repos_list = entry.get("repos")
        if repos_list is None:
            target = Target(project=project_key)
            if target.key in seen:
                raise SystemExit(f"ERROR: duplicate project entry: {project_key}")
            seen.add(target.key)
            targets.append(target)
            continue
        if not isinstance(repos_list, list) or not repos_list:
            raise SystemExit(
                f"ERROR: projects[{p_idx}].repos must be a non-empty list of repo slugs, "
                "or omitted for the whole project"
            )
        for r_idx, repo in enumerate(repos_list):
            if not isinstance(repo, str) or not repo.strip():
                raise SystemExit(
                    f"ERROR: projects[{p_idx}].repos[{r_idx}] must be a non-empty string"
                )
            target = Target(project=project_key, repo=repo.strip())
            if target.key in seen:
                raise SystemExit(f"ERROR: duplicate repo entry: {target.key}")
            seen.add(target.key)
            targets.append(target)
    whole = {t.project for t in targets if t.is_project}
    for t in targets:
        if not t.is_project and t.project in whole:
            raise SystemExit(
                f"ERROR: project {t.project} is listed both whole and with repos; pick one"
            )

    return OnboardingInput(
        team=team,
        bitbucket_url=bitbucket_url.rstrip("/"),
        noergler_url=noergler_url,
        targets=targets,
    )


def _require_https(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise SystemExit(f"ERROR: '{field_name}' must be a non-empty string")
    parsed = urlparse(value)
    if parsed.scheme != "https" or not parsed.netloc:
        raise SystemExit(f"ERROR: '{field_name}' must be an https URL")
    return value


def _require_url(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise SystemExit(f"ERROR: '{field_name}' must be a non-empty string")
    parsed = urlparse(value)
    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        raise SystemExit(f"ERROR: '{field_name}' must be an http(s) URL")
    return value


# --------------------------------------------------------------------------- #
# Minimal stdlib HTTP
# --------------------------------------------------------------------------- #

class HTTPStatusError(Exception):
    def __init__(self, status_code: int, text: str, url: str):
        super().__init__(f"HTTP {status_code} for {url}: {text[:200]}")
        self.status_code = status_code
        self.text = text
        self.url = url


@dataclass
class _Response:
    status_code: int
    text: str

    def json(self) -> Any:
        return json.loads(self.text) if self.text else {}


def _http(
    method: str,
    url: str,
    *,
    headers: dict[str, str],
    params: dict[str, Any] | None = None,
    body: Any | None = None,
    raw_body: bytes | None = None,
    timeout: float = 30.0,
) -> _Response:
    if params:
        url = url + "?" + urllib.parse.urlencode(params)
    data: bytes | None = raw_body
    headers = dict(headers)
    if body is not None:
        data = json.dumps(body).encode("utf-8")
        headers["Content-Type"] = "application/json"
    elif raw_body is not None:
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=data, method=method, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            text = resp.read().decode("utf-8", errors="replace")
            return _Response(status_code=resp.status, text=text)
    except urllib.error.HTTPError as exc:
        text = ""
        try:
            text = exc.read().decode("utf-8", errors="replace")
        except Exception:
            pass
        raise HTTPStatusError(exc.code, text, url) from exc


class BitbucketHTTP:
    """Stdlib-only synchronous Bitbucket Server REST client.

    Only implements the endpoints the onboarder needs. Every webhook call takes
    a `Target`: a project (`/projects/{key}/webhooks`, Bitbucket DC 8.8+) or a
    repo (`/projects/{key}/repos/{slug}/webhooks`) share the same body shape
    and pagination.
    """

    def __init__(self, base_url: str, token: str, timeout: float = 30.0):
        self.base_url = base_url.rstrip("/")
        self.token = token
        self.timeout = timeout

    def _request(
        self,
        method: str,
        path: str,
        *,
        params: dict[str, Any] | None = None,
        body: Any | None = None,
    ) -> _Response:
        return _http(
            method, self.base_url + path,
            headers={"Authorization": f"Bearer {self.token}", "Accept": "application/json"},
            params=params, body=body, timeout=self.timeout,
        )

    def _paged(self, path: str) -> list[dict[str, Any]]:
        values: list[dict[str, Any]] = []
        start = 0
        while True:
            page = self._request("GET", path, params={"start": start, "limit": 100}).json()
            values.extend(page.get("values") or [])
            if page.get("isLastPage", True):
                break
            next_start = page.get("nextPageStart")
            if not isinstance(next_start, int) or next_start <= start:
                break
            start = next_start
        return values

    def get(self, target: Target) -> dict[str, Any]:
        """The project or repo itself; proves the token can see it."""
        return self._request("GET", target.api_path).json()

    def list_repos(self, project: str) -> list[dict[str, Any]]:
        return self._paged(f"/rest/api/1.0/projects/{project}/repos")

    def list_webhooks(self, target: Target) -> list[dict[str, Any]]:
        return self._paged(target.api_path + "/webhooks")

    def create_webhook(self, target: Target, body: dict[str, Any]) -> dict[str, Any]:
        return self._request("POST", target.api_path + "/webhooks", body=body).json()

    def update_webhook(self, target: Target, webhook_id: int, body: dict[str, Any]) -> dict[str, Any]:
        return self._request("PUT", f"{target.api_path}/webhooks/{webhook_id}", body=body).json()

    def delete_webhook(self, target: Target, webhook_id: int) -> None:
        self._request("DELETE", f"{target.api_path}/webhooks/{webhook_id}")

    def grant_user_permission(self, target: Target, username: str, permission: str) -> None:
        self._request(
            "PUT", target.api_path + "/permissions/users",
            params={"name": username, "permission": permission},
        )


@dataclass(frozen=True)
class ProbeResult:
    owned: bool
    claim: str  # how teams.yaml claims the project: "whole", "repos" or "none"
    bot_can_read: bool
    bot_username: str


class NoerglerProbe:
    """Signed probe against the team's webhook route.

    noergler verifies the HMAC exactly as for a Bitbucket event, so a 401 means
    the secret does not match the team's, 404 an unknown team, 503 a team the
    service has disabled. Those are team-level faults and abort the run; a
    per-target answer says whether the team owns it and whether the bot can
    read it.
    """

    def __init__(self, webhook_url: str, secret: str, timeout: float = 15.0):
        self.webhook_url = webhook_url
        self.secret = secret
        self.timeout = timeout

    def probe(self, target: Target) -> ProbeResult:
        raw = json.dumps({"project": target.project, "repo": target.repo}).encode("utf-8")
        signature = hmac.new(self.secret.encode(), raw, hashlib.sha256).hexdigest()
        try:
            resp = _http(
                "POST", self.webhook_url,
                headers={
                    "X-Event-Key": PROBE_EVENT_KEY,
                    "X-Hub-Signature": f"sha256={signature}",
                    "Accept": "application/json",
                },
                raw_body=raw, timeout=self.timeout,
            )
        except HTTPStatusError as exc:
            hints = {
                401: "the webhook secret does not match the team's on the noergler side",
                404: "noergler has no such team (check 'team' against teams.yaml)",
                503: "noergler has disabled this team; see its startup log",
            }
            hint = hints.get(exc.status_code, exc.text[:200])
            raise SystemExit(f"ERROR: probe {self.webhook_url} -> HTTP {exc.status_code}: {hint}")
        except urllib.error.URLError as exc:
            raise SystemExit(f"ERROR: cannot reach noergler at {self.webhook_url}: {exc}")
        data = resp.json()
        return ProbeResult(
            owned=bool(data.get("owned")),
            claim=str(data.get("claim") or "none"),
            bot_can_read=bool(data.get("bot_can_read")),
            bot_username=str(data.get("bot_username") or ""),
        )


def _not_owned_reason(target: Target, probe: ProbeResult) -> str:
    """Why noergler will not take this target, in terms of what to change."""
    if target.is_project and probe.claim == "repos":
        return (
            f"noergler's teams.yaml lists specific repos of {target.project} for this team; "
            f"list them under \"repos\" in team.json, or ask the noergler admin to claim the whole project"
        )
    if not target.is_project and probe.claim == "whole":
        return (
            f"noergler's teams.yaml claims all of {target.project} for this team; use "
            f"{{\"key\": \"{target.project}\"}} in team.json (one project webhook), a repo webhook "
            "next to it would deliver every event twice"
        )
    return "not owned by this team in noergler's teams.yaml; ask the noergler admin"


# --------------------------------------------------------------------------- #
# Onboarder
# --------------------------------------------------------------------------- #

@dataclass
class TargetResult:
    target: Target
    status: str  # "ok", "failed", "skipped"
    detail: str = ""
    diff: list[str] = field(default_factory=list)


@dataclass
class StatusRow:
    target: Target
    owned: bool
    bot_can_read: bool
    webhook: str          # "ok", "missing", "stale: ...", or why it is not owned
    stray: list[str]      # this instance's repo-level hooks under a project target
    foreign: list[str]    # same-named hooks pointing at another noergler instance


def _hook_named(hooks: list[dict[str, Any]], name: str) -> dict[str, Any] | None:
    return next((h for h in hooks if h.get("name") == name), None)


class Onboarder:
    def __init__(
        self,
        client: BitbucketHTTP,
        probe: NoerglerProbe,
        webhook_url: str,
        webhook_secret: str,
        webhook_name: str = DEFAULT_WEBHOOK_NAME,
        dry_run: bool = False,
        grant_bot: bool = False,
        prune: bool = True,
    ):
        self.client = client
        self.probe = probe
        self.webhook_url = webhook_url
        self.webhook_secret = webhook_secret
        self.webhook_name = webhook_name
        self.dry_run = dry_run
        self.grant_bot = grant_bot
        self.prune = prune
        # Hooks are matched by name AND instance: a hook named `noergler` that
        # points at another instance (intg next to prod) is reported, never
        # pruned or rewritten.
        self.instance_url = webhook_url.rsplit("/webhook/", 1)[0]

    def _is_ours(self, hook: dict[str, Any]) -> bool:
        url = hook.get("url")
        return isinstance(url, str) and url.startswith(self.instance_url + "/")

    # -- building blocks -- #
    def verify_admin_access(self, target: Target) -> None:
        """Confirm the admin's token can see the target. Raises HTTPStatusError."""
        self.client.get(target)
        logger.info("[%s] admin token can access it", target.key)

    def _build_webhook_body(self) -> dict[str, Any]:
        return {
            "name": self.webhook_name,
            "url": self.webhook_url,
            "active": True,
            "events": list(REQUIRED_WEBHOOK_EVENTS),
            "configuration": {"secret": self.webhook_secret},
            "sslVerificationRequired": True,
        }

    def _diff_webhook(self, existing: dict[str, Any]) -> list[str]:
        diffs: list[str] = []
        if existing.get("url") != self.webhook_url:
            diffs.append(f"url: {existing.get('url')!r} -> {self.webhook_url!r}")
        existing_events = set(existing.get("events") or [])
        required = set(REQUIRED_WEBHOOK_EVENTS)
        if existing_events != required:
            missing = sorted(required - existing_events)
            extra = sorted(existing_events - required)
            diffs.append(f"events: missing={missing} extra={extra}")
        if not existing.get("active", True):
            diffs.append("active: False -> True")
        existing_cfg = existing.get("configuration") or {}
        if not existing_cfg:
            diffs.append("configuration.secret: (unset) -> (set)")
        return diffs

    def upsert_webhook(self, target: Target) -> tuple[int, list[str]]:
        """Create or update the webhook. Returns (webhook_id, diff)."""
        existing = _hook_named(self.client.list_webhooks(target), self.webhook_name)
        body = self._build_webhook_body()

        if existing is not None and not self._is_ours(existing):
            raise HTTPStatusError(
                409,
                f"{self.webhook_name!r} hook points at another noergler ({existing.get('url')}); "
                "gone? --remove with its noergler_url first, else pass --name",
                self.webhook_url,
            )
        if existing is None:
            logger.info("[%s] creating webhook %r", target.key, self.webhook_name)
            if self.dry_run:
                logger.info("[%s] DRY-RUN body=%s", target.key, _redact(body))
                return -1, ["create"]
            created = self.client.create_webhook(target, body)
            return int(created["id"]), ["create"]

        diff = self._diff_webhook(existing)
        if not diff:
            logger.info("[%s] webhook already up to date", target.key)
            return int(existing["id"]), []

        logger.info("[%s] updating webhook id=%s changes=%s", target.key, existing.get("id"), diff)
        if self.dry_run:
            logger.info("[%s] DRY-RUN body=%s", target.key, _redact(body))
            return int(existing["id"]), diff
        self.client.update_webhook(target, int(existing["id"]), body)
        return int(existing["id"]), diff

    def stray_repo_hooks(self, project: str) -> tuple[list[tuple[Target, int]], list[str]]:
        """(this instance's repo-level hooks under a project, foreign same-named
        hooks). With a project webhook in place the former deliver every event
        a second time; the latter belong to another noergler and are left alone."""
        stray: list[tuple[Target, int]] = []
        foreign: list[str] = []
        for repo in self.client.list_repos(project):
            slug = repo.get("slug")
            if not isinstance(slug, str):
                continue
            repo_target = Target(project=project, repo=slug)
            hook = _hook_named(self.client.list_webhooks(repo_target), self.webhook_name)
            if hook is None:
                continue
            if self._is_ours(hook):
                stray.append((repo_target, int(hook["id"])))
            else:
                foreign.append(f"{repo_target.key} -> {hook.get('url')}")
        return stray, foreign

    def prune_repo_hooks(self, project: str) -> list[str]:
        pruned: list[str] = []
        stray, foreign = self.stray_repo_hooks(project)
        for entry in foreign:
            logger.warning("[%s] hook named %r points at another noergler, left alone", entry, self.webhook_name)
        for repo_target, hook_id in stray:
            if self.dry_run:
                logger.info("[%s] DRY-RUN would delete stray repo webhook id=%d", repo_target.key, hook_id)
            else:
                self.client.delete_webhook(repo_target, hook_id)
                logger.info("[%s] deleted stray repo webhook id=%d", repo_target.key, hook_id)
            pruned.append(repo_target.key)
        return pruned

    # -- orchestrators -- #
    def status(self, target: Target) -> StatusRow:
        probe = self.probe.probe(target)
        webhook = "-"
        stray: list[str] = []
        foreign: list[str] = []
        if not probe.owned:
            # Nothing to configure for a target the team does not own.
            return StatusRow(target, False, False, _not_owned_reason(target, probe), stray, foreign)
        try:
            existing = _hook_named(self.client.list_webhooks(target), self.webhook_name)
            if existing is None:
                webhook = "missing"
            elif not self._is_ours(existing):
                webhook = f"foreign: {existing.get('url')}"
            else:
                diff = self._diff_webhook(existing)
                webhook = "ok" if not diff else "stale: " + "; ".join(diff)
            if target.is_project:
                found, foreign = self.stray_repo_hooks(target.project)
                stray = [t.key for t, _ in found]
        except HTTPStatusError as exc:
            webhook = f"HTTP {exc.status_code}"
        except urllib.error.URLError as exc:
            webhook = f"error: {exc}"
        return StatusRow(target, probe.owned, probe.bot_can_read, webhook, stray, foreign)

    def onboard(self, target: Target) -> TargetResult:
        probe = self.probe.probe(target)
        if not probe.owned:
            return TargetResult(target, "skipped", detail=_not_owned_reason(target, probe))

        try:
            self.verify_admin_access(target)
        except HTTPStatusError as exc:
            return TargetResult(
                target, "failed",
                detail=f"admin access check HTTP {exc.status_code}: {exc.text[:200]}",
            )
        except urllib.error.URLError as exc:
            return TargetResult(target, "failed", detail=f"admin access check: {exc}")

        notes: list[str] = []
        if not probe.bot_can_read:
            bot = probe.bot_username or "the noergler bot"
            if not self.grant_bot:
                return TargetResult(
                    target, "skipped",
                    detail=f"{bot} cannot read it; grant {bot} {target.bot_permission} "
                           f"in Bitbucket or re-run with --grant-bot",
                )
            if self.dry_run:
                logger.info("[%s] DRY-RUN would grant %s %s", target.key, bot, target.bot_permission)
            else:
                try:
                    self.client.grant_user_permission(target, bot, target.bot_permission)
                except HTTPStatusError as exc:
                    return TargetResult(
                        target, "failed",
                        detail=f"grant {target.bot_permission} to {bot}: HTTP {exc.status_code}: {exc.text[:200]}",
                    )
                logger.info("[%s] granted %s %s", target.key, bot, target.bot_permission)
            notes.append(f"{bot} granted {target.bot_permission}")

        try:
            webhook_id, diff = self.upsert_webhook(target)
        except HTTPStatusError as exc:
            return TargetResult(
                target, "failed",
                detail=f"upsert webhook HTTP {exc.status_code}: {exc.text[:200]}",
            )
        except urllib.error.URLError as exc:
            return TargetResult(target, "failed", detail=f"upsert webhook: {exc}")

        if diff == ["create"]:
            notes.insert(0, f"webhook created (id={webhook_id})")
        elif diff:
            notes.insert(0, f"webhook updated (id={webhook_id})")
        else:
            notes.insert(0, "webhook already up to date")

        if target.is_project and self.prune:
            try:
                pruned = self.prune_repo_hooks(target.project)
            except (HTTPStatusError, urllib.error.URLError) as exc:
                notes.append(f"prune of repo hooks failed: {exc}")
            else:
                if pruned:
                    notes.append(f"pruned {len(pruned)} repo hook(s): {', '.join(pruned)}")

        prefix = "dry-run: " if self.dry_run else ""
        return TargetResult(target, "ok", detail=prefix + ", ".join(notes), diff=diff)

    def remove_webhook(self, target: Target) -> TargetResult:
        """Delete the noergler webhook from the target. No-op if absent."""
        try:
            hooks = self.client.list_webhooks(target)
        except HTTPStatusError as exc:
            return TargetResult(
                target, "failed",
                detail=f"list webhooks HTTP {exc.status_code}: {exc.text[:200]}",
            )
        except urllib.error.URLError as exc:
            return TargetResult(target, "failed", detail=f"list webhooks: {exc}")

        existing = _hook_named(hooks, self.webhook_name)
        if existing is None:
            logger.info("[%s] no %r webhook found, nothing to remove", target.key, self.webhook_name)
            return TargetResult(target, "skipped", detail=f"no {self.webhook_name!r} webhook found")
        if not self._is_ours(existing):
            return TargetResult(
                target, "skipped",
                detail=f"{self.webhook_name!r} webhook points at another noergler ({existing.get('url')}), left alone",
            )

        webhook_id = int(existing["id"])
        if self.dry_run:
            logger.info("[%s] DRY-RUN would delete webhook id=%d", target.key, webhook_id)
            return TargetResult(target, "ok", detail=f"dry-run: would remove webhook id={webhook_id}")

        try:
            self.client.delete_webhook(target, webhook_id)
        except HTTPStatusError as exc:
            return TargetResult(
                target, "failed",
                detail=f"delete webhook HTTP {exc.status_code}: {exc.text[:200]}",
            )
        except urllib.error.URLError as exc:
            return TargetResult(target, "failed", detail=f"delete webhook: {exc}")

        logger.info("[%s] removed webhook id=%d", target.key, webhook_id)
        return TargetResult(target, "ok", detail=f"webhook removed: id={webhook_id}")


def _redact(body: dict[str, Any]) -> dict[str, Any]:
    copy = dict(body)
    cfg = dict(copy.get("configuration") or {})
    if "secret" in cfg:
        cfg["secret"] = "***"
    copy["configuration"] = cfg
    return copy


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m scripts.onboard_repo",
        description=(
            "Onboard a team's Bitbucket Server projects/repos to noergler by creating or "
            "updating their webhooks (idempotent). Run by the team admin with their own token."
        ),
        epilog=(
            "A whole project ({\"key\": \"PLAT\"}) gets one project webhook that covers every "
            "current and future repo (Bitbucket DC 8.8+); a repo list gets one webhook per repo. "
            "Creating webhooks needs project/repo admin on the token. The team's webhook secret "
            "must match what noergler resolves for the team; the probe checks that first. "
            "To verify end-to-end delivery, open a real PR after onboarding."
        ),
    )
    parser.add_argument("config", type=Path, help="Path to the team's onboarding JSON")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--status", action="store_true",
        help="Report only: ownership, bot access, webhook state and stray repo hooks. No writes",
    )
    mode.add_argument(
        "--remove", action="store_true",
        help="Deboard: delete the webhook from every target in the config",
    )
    parser.add_argument(
        "--grant-bot", action="store_true",
        help="Grant the noergler bot write access (PROJECT_WRITE / REPO_WRITE) where it has none",
    )
    parser.add_argument(
        "--no-prune", action="store_true",
        help="Keep repo-level noergler webhooks under a project webhook (they double every delivery)",
    )
    parser.add_argument("--name", default=DEFAULT_WEBHOOK_NAME, help=f"Webhook name (default: {DEFAULT_WEBHOOK_NAME})")
    parser.add_argument("--dry-run", action="store_true", help="Print planned changes without mutating Bitbucket")
    parser.add_argument("--env-file", type=Path, default=None, help="Additional .env file to read secrets from")
    parser.add_argument(
        "--secret-env", default=None,
        help="Env var holding the team's webhook secret (default: TEAM_<SLUG>_WEBHOOK_SECRET)",
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable DEBUG logging")
    return parser.parse_args(argv)


def _print_results(results: list[TargetResult]) -> None:
    width = max((len(r.target.label) for r in results), default=10)
    header = f"{'target'.ljust(width)}  status   detail"
    print()
    print(header)
    print("-" * len(header))
    for r in results:
        print(f"{r.target.label.ljust(width)}  {r.status.ljust(7)}  {r.detail}")


def _print_status(rows: list[StatusRow]) -> None:
    width = max((len(r.target.label) for r in rows), default=10)
    header = f"{'target'.ljust(width)}  owned  bot   webhook"
    print()
    print(header)
    print("-" * len(header))
    for r in rows:
        owned = "yes" if r.owned else "no"
        bot = ("yes" if r.bot_can_read else "no") if r.owned else "-"
        line = f"{r.target.label.ljust(width)}  {owned.ljust(5)}  {bot.ljust(4)}  {r.webhook}"
        if r.stray:
            line += f"  stray repo hooks: {', '.join(r.stray)}"
        if r.foreign:
            line += f"  foreign hooks: {', '.join(r.foreign)}"
        print(line)


def _run(args: argparse.Namespace) -> int:
    inp = load_onboarding_input(args.config)
    secret_var = args.secret_env or default_webhook_secret_var(inp.team)
    token, webhook_secret = resolve_secrets(args.env_file, secret_var)

    logger.info("Team:          %s", inp.team)
    logger.info("Bitbucket URL: %s", inp.bitbucket_url)
    logger.info("Webhook URL:   %s", inp.webhook_url)
    logger.info("Bitbucket token loaded: %s", _mask(token))
    logger.info("Webhook secret loaded:  %s (from %s)", _mask(webhook_secret), secret_var)
    logger.info("Targets (%d): %s", len(inp.targets), ", ".join(t.label for t in inp.targets))
    if args.status:
        logger.info("STATUS mode: no writes")
    if args.remove:
        logger.info("REMOVE mode: will delete the %r webhook from each target", args.name)
    if args.dry_run:
        logger.info("DRY-RUN: no writes will be issued")

    client = BitbucketHTTP(base_url=inp.bitbucket_url, token=token)
    onboarder = Onboarder(
        client,
        NoerglerProbe(inp.webhook_url, webhook_secret),
        webhook_url=inp.webhook_url,
        webhook_secret=webhook_secret,
        webhook_name=args.name,
        dry_run=args.dry_run,
        grant_bot=args.grant_bot,
        prune=not args.no_prune,
    )

    if args.status:
        rows = [onboarder.status(t) for t in inp.targets]
        _print_status(rows)
        healthy = all(r.owned and r.bot_can_read and r.webhook == "ok" and not r.stray for r in rows)
        return 0 if healthy else 1

    action = onboarder.remove_webhook if args.remove else onboarder.onboard
    results: list[TargetResult] = []
    for target in inp.targets:
        logger.info("--- %s ---", target.label)
        try:
            result = action(target)
        except SystemExit:
            raise
        except Exception as exc:  # noqa: BLE001 — per-target isolation
            logger.exception("[%s] unexpected error", target.key)
            result = TargetResult(target, "failed", detail=f"unexpected: {exc}")
        results.append(result)

    _print_results(results)
    failed = [r for r in results if r.status == "failed"]
    return 1 if failed else 0


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    return _run(args)


if __name__ == "__main__":
    sys.exit(main())
