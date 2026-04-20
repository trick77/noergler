"""
Onboard a Bitbucket Server repository to noergler webhook delivery.

Usage:
    python scripts/onboard_repo.py config.json [--name noergler] [--dry-run] [--env-file PATH]
    python scripts/onboard_repo.py config.json --remove [--dry-run] [--env-file PATH]

`--remove` deletes the noergler webhook from every repo in the config instead
of creating/updating it. Same config file drives both directions.

The JSON config describes target repos and URLs. Secrets (BITBUCKET_TOKEN,
BITBUCKET_WEBHOOK_SECRET) are resolved from, in order:
    1. Process environment
    2. .env in CWD
    3. --env-file <path>

BITBUCKET_WEBHOOK_SECRET used by this script MUST match the value the running
noergler service has configured — Bitbucket and noergler share the HMAC secret;
this script only programs Bitbucket's side.

The script does two things per repo: a read-permission check and an idempotent
webhook upsert. End-to-end delivery is verified by opening a real PR — Bitbucket's
"/test" endpoint is intentionally not used (behavior varies by server version).

This script is intentionally stdlib-only so it can run on any host with Python
3.10+ without a venv or `pip install`.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
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

# Must match REQUIRED_WEBHOOK_EVENTS in app/config.py. Inlined here to keep this
# script stdlib-only (no import of app.config, which pulls in pydantic).
REQUIRED_WEBHOOK_EVENTS: tuple[str, ...] = (
    "pr:opened",
    "pr:from_ref_updated",
    "pr:comment:added",
    "pr:merged",
    "pr:deleted",
)


# --------------------------------------------------------------------------- #
# Config loading
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class RepoSpec:
    project: str
    repo: str

    @property
    def key(self) -> str:
        return f"{self.project}/{self.repo}"


@dataclass(frozen=True)
class OnboardingInput:
    bitbucket_url: str
    webhook_url: str
    repos: list[RepoSpec]


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


def resolve_secrets(env_file: Path | None) -> tuple[str, str]:
    """Return (BITBUCKET_TOKEN, BITBUCKET_WEBHOOK_SECRET).

    Precedence (first match wins): process env > cwd .env > --env-file.
    Raises SystemExit on missing.
    """
    merged: dict[str, str] = {}
    if env_file is not None:
        merged.update(_load_env_file(env_file))
    merged.update(_load_env_file(Path.cwd() / ".env"))
    for k in ("BITBUCKET_TOKEN", "BITBUCKET_WEBHOOK_SECRET", "BITBUCKET_USERNAME"):
        if k in os.environ and os.environ[k]:
            merged[k] = os.environ[k]
    resolve_secrets._resolved = merged  # type: ignore[attr-defined]

    missing = [k for k in ("BITBUCKET_TOKEN", "BITBUCKET_WEBHOOK_SECRET") if not merged.get(k)]
    if missing:
        sys.stderr.write(
            "ERROR: missing required environment variable(s): "
            + ", ".join(missing)
            + "\nSet them in the process env, a .env in CWD, or pass --env-file.\n"
        )
        raise SystemExit(2)
    return merged["BITBUCKET_TOKEN"], merged["BITBUCKET_WEBHOOK_SECRET"]


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

    bitbucket_url = _require_https(data.get("bitbucket_url"), "bitbucket_url")
    webhook_url = _require_url(data.get("webhook_url"), "webhook_url")
    projects_raw = data.get("projects")

    if not isinstance(projects_raw, list) or not projects_raw:
        raise SystemExit("ERROR: 'projects' must be a non-empty list")

    seen: set[tuple[str, str]] = set()
    repos: list[RepoSpec] = []
    for p_idx, entry in enumerate(projects_raw):
        if not isinstance(entry, dict):
            raise SystemExit(f"ERROR: projects[{p_idx}] must be an object")
        project = entry.get("project")
        repos_list = entry.get("repos")
        if not isinstance(project, str) or not project.strip():
            raise SystemExit(f"ERROR: projects[{p_idx}].project must be a non-empty string")
        if not isinstance(repos_list, list) or not repos_list:
            raise SystemExit(
                f"ERROR: projects[{p_idx}].repos must be a non-empty list of repo slugs"
            )
        project_key = project.strip()
        for r_idx, repo in enumerate(repos_list):
            if not isinstance(repo, str) or not repo.strip():
                raise SystemExit(
                    f"ERROR: projects[{p_idx}].repos[{r_idx}] must be a non-empty string"
                )
            key = (project_key, repo.strip())
            if key in seen:
                raise SystemExit(f"ERROR: duplicate repo entry: {key[0]}/{key[1]}")
            seen.add(key)
            repos.append(RepoSpec(project=key[0], repo=key[1]))

    return OnboardingInput(
        bitbucket_url=bitbucket_url.rstrip("/"),
        webhook_url=webhook_url,
        repos=repos,
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
# Minimal stdlib HTTP client for Bitbucket Server
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


class BitbucketHTTP:
    """Stdlib-only synchronous Bitbucket Server REST client.

    Only implements the endpoints the onboarder needs.
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
        url = self.base_url + path
        if params:
            url = url + "?" + urllib.parse.urlencode(params)

        data: bytes | None = None
        headers = {
            "Authorization": f"Bearer {self.token}",
            "Accept": "application/json",
        }
        if body is not None:
            data = json.dumps(body).encode("utf-8")
            headers["Content-Type"] = "application/json"

        req = urllib.request.Request(url, data=data, method=method, headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                text = resp.read().decode("utf-8", errors="replace")
                return _Response(status_code=resp.status, text=text)
        except urllib.error.HTTPError as exc:
            text = ""
            try:
                text = exc.read().decode("utf-8", errors="replace")
            except Exception:
                pass
            raise HTTPStatusError(exc.code, text, url) from exc

    def get_repo(self, project: str, repo: str) -> dict:
        resp = self._request("GET", f"/rest/api/1.0/projects/{project}/repos/{repo}")
        return resp.json()

    def list_pull_requests(self, project: str, repo: str, limit: int = 1) -> dict:
        resp = self._request(
            "GET",
            f"/rest/api/1.0/projects/{project}/repos/{repo}/pull-requests",
            params={"limit": limit},
        )
        return resp.json()

    def list_webhooks(self, project: str, repo: str) -> list[dict]:
        """Return all webhooks (handles pagination)."""
        values: list[dict] = []
        start = 0
        while True:
            resp = self._request(
                "GET",
                f"/rest/api/1.0/projects/{project}/repos/{repo}/webhooks",
                params={"start": start, "limit": 100},
            )
            page = resp.json()
            values.extend(page.get("values") or [])
            if page.get("isLastPage", True):
                break
            next_start = page.get("nextPageStart")
            if not isinstance(next_start, int) or next_start <= start:
                break
            start = next_start
        return values

    def create_webhook(self, project: str, repo: str, body: dict) -> dict:
        resp = self._request(
            "POST",
            f"/rest/api/1.0/projects/{project}/repos/{repo}/webhooks",
            body=body,
        )
        return resp.json()

    def update_webhook(self, project: str, repo: str, webhook_id: int, body: dict) -> dict:
        resp = self._request(
            "PUT",
            f"/rest/api/1.0/projects/{project}/repos/{repo}/webhooks/{webhook_id}",
            body=body,
        )
        return resp.json()

    def delete_webhook(self, project: str, repo: str, webhook_id: int) -> None:
        self._request(
            "DELETE",
            f"/rest/api/1.0/projects/{project}/repos/{repo}/webhooks/{webhook_id}",
        )


# --------------------------------------------------------------------------- #
# Onboarder
# --------------------------------------------------------------------------- #

@dataclass
class RepoResult:
    repo: RepoSpec
    status: str  # "ok", "failed", "skipped"
    detail: str = ""
    diff: list[str] = field(default_factory=list)


class RepoOnboarder:
    def __init__(
        self,
        client: BitbucketHTTP,
        webhook_url: str,
        webhook_secret: str,
        webhook_name: str = DEFAULT_WEBHOOK_NAME,
        dry_run: bool = False,
    ):
        self.client = client
        self.webhook_url = webhook_url
        self.webhook_secret = webhook_secret
        self.webhook_name = webhook_name
        self.dry_run = dry_run

    # -- Step 1 -- #
    def verify_permissions(self, spec: RepoSpec) -> None:
        """Confirm read access to repo and its PRs. Raises HTTPStatusError on failure."""
        self.client.get_repo(spec.project, spec.repo)
        self.client.list_pull_requests(spec.project, spec.repo, limit=1)
        logger.info("[%s] read permissions OK", spec.key)

    # -- Step 2 -- #
    def _build_webhook_body(self) -> dict:
        return {
            "name": self.webhook_name,
            "url": self.webhook_url,
            "active": True,
            "events": list(REQUIRED_WEBHOOK_EVENTS),
            "configuration": {"secret": self.webhook_secret},
            "sslVerificationRequired": True,
        }

    def _diff_webhook(self, existing: dict) -> list[str]:
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

    def upsert_webhook(self, spec: RepoSpec) -> tuple[int, list[str]]:
        """Create or update the webhook. Returns (webhook_id, diff)."""
        hooks = self.client.list_webhooks(spec.project, spec.repo)
        existing = next((h for h in hooks if h.get("name") == self.webhook_name), None)
        body = self._build_webhook_body()

        if existing is None:
            logger.info("[%s] creating webhook %r", spec.key, self.webhook_name)
            if self.dry_run:
                logger.info("[%s] DRY-RUN body=%s", spec.key, _redact(body))
                return -1, ["create"]
            created = self.client.create_webhook(spec.project, spec.repo, body)
            return int(created["id"]), ["create"]

        diff = self._diff_webhook(existing)
        if not diff:
            logger.info("[%s] webhook already up to date", spec.key)
            return int(existing["id"]), []

        logger.info("[%s] updating webhook id=%s changes=%s", spec.key, existing.get("id"), diff)
        if self.dry_run:
            logger.info("[%s] DRY-RUN body=%s", spec.key, _redact(body))
            return int(existing["id"]), diff
        self.client.update_webhook(spec.project, spec.repo, int(existing["id"]), body)
        return int(existing["id"]), diff

    # -- Orchestrator -- #
    def onboard(self, spec: RepoSpec) -> RepoResult:
        try:
            self.verify_permissions(spec)
        except HTTPStatusError as exc:
            return RepoResult(
                spec, "failed",
                detail=f"permission check HTTP {exc.status_code}: {exc.text[:200]}",
            )
        except urllib.error.URLError as exc:
            return RepoResult(spec, "failed", detail=f"permission check: {exc}")

        try:
            _webhook_id, diff = self.upsert_webhook(spec)
        except HTTPStatusError as exc:
            return RepoResult(
                spec, "failed",
                detail=f"upsert webhook HTTP {exc.status_code}: {exc.text[:200]}",
            )
        except urllib.error.URLError as exc:
            return RepoResult(spec, "failed", detail=f"upsert webhook: {exc}")

        if self.dry_run:
            return RepoResult(spec, "ok", detail="dry-run", diff=diff)

        return RepoResult(spec, "ok", detail="webhook configured", diff=diff)

    # -- Deboarding -- #
    def remove_webhook(self, spec: RepoSpec) -> RepoResult:
        """Delete the noergler webhook from the repo. No-op if absent."""
        try:
            hooks = self.client.list_webhooks(spec.project, spec.repo)
        except HTTPStatusError as exc:
            return RepoResult(
                spec, "failed",
                detail=f"list webhooks HTTP {exc.status_code}: {exc.text[:200]}",
            )
        except urllib.error.URLError as exc:
            return RepoResult(spec, "failed", detail=f"list webhooks: {exc}")

        existing = next((h for h in hooks if h.get("name") == self.webhook_name), None)
        if existing is None:
            logger.info("[%s] no %r webhook found, nothing to remove", spec.key, self.webhook_name)
            return RepoResult(spec, "skipped", detail=f"no {self.webhook_name!r} webhook found")

        webhook_id = int(existing["id"])
        if self.dry_run:
            logger.info("[%s] DRY-RUN would delete webhook id=%d", spec.key, webhook_id)
            return RepoResult(spec, "ok", detail=f"dry-run: would remove webhook id={webhook_id}")

        try:
            self.client.delete_webhook(spec.project, spec.repo, webhook_id)
        except HTTPStatusError as exc:
            return RepoResult(
                spec, "failed",
                detail=f"delete webhook HTTP {exc.status_code}: {exc.text[:200]}",
            )
        except urllib.error.URLError as exc:
            return RepoResult(spec, "failed", detail=f"delete webhook: {exc}")

        logger.info("[%s] removed webhook id=%d", spec.key, webhook_id)
        return RepoResult(spec, "ok", detail=f"webhook removed: id={webhook_id}")


def _redact(body: dict) -> dict:
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
        prog="python scripts/onboard_repo.py",
        description="Onboard Bitbucket Server repos to noergler by creating/updating their webhooks (idempotent).",
        epilog=(
            "Write permission check is implicit: if the token lacks repo-write, the webhook "
            "create/update will surface a 403. BITBUCKET_WEBHOOK_SECRET must match the value "
            "configured on the running noergler service. To verify end-to-end delivery, open "
            "a real PR after onboarding."
        ),
    )
    parser.add_argument("config", type=Path, help="Path to onboarding JSON config")
    parser.add_argument("--name", default=DEFAULT_WEBHOOK_NAME, help=f"Webhook name (default: {DEFAULT_WEBHOOK_NAME})")
    parser.add_argument("--dry-run", action="store_true", help="Print planned changes without mutating Bitbucket")
    parser.add_argument(
        "--remove", action="store_true",
        help="Deboard: delete the webhook from every repo in the config instead of creating/updating it",
    )
    parser.add_argument("--env-file", type=Path, default=None, help="Additional .env file to read secrets from")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable DEBUG logging")
    return parser.parse_args(argv)


def _print_summary(results: list[RepoResult]) -> None:
    width_repo = max((len(r.repo.key) for r in results), default=10)
    header = f"{'repo'.ljust(width_repo)}  status   detail"
    print()
    print(header)
    print("-" * len(header))
    for r in results:
        print(f"{r.repo.key.ljust(width_repo)}  {r.status.ljust(7)}  {r.detail}")


def _run(args: argparse.Namespace) -> int:
    token, webhook_secret = resolve_secrets(args.env_file)
    inp = load_onboarding_input(args.config)

    logger.info("Bitbucket URL: %s", inp.bitbucket_url)
    logger.info("Webhook URL:   %s", inp.webhook_url)
    logger.info("Bitbucket token loaded: %s", _mask(token))
    logger.info("Webhook secret loaded:  %s", _mask(webhook_secret))
    logger.info("Target repos (%d): %s", len(inp.repos), ", ".join(r.key for r in inp.repos))
    if args.remove:
        logger.info("REMOVE mode: will delete the %r webhook from each repo", args.name)
    if args.dry_run:
        logger.info("DRY-RUN: no writes will be issued")

    client = BitbucketHTTP(base_url=inp.bitbucket_url, token=token)
    onboarder = RepoOnboarder(
        client,
        webhook_url=inp.webhook_url,
        webhook_secret=webhook_secret,
        webhook_name=args.name,
        dry_run=args.dry_run,
    )

    action = onboarder.remove_webhook if args.remove else onboarder.onboard
    results: list[RepoResult] = []
    for spec in inp.repos:
        logger.info("--- %s ---", spec.key)
        try:
            result = action(spec)
        except Exception as exc:  # noqa: BLE001 — per-repo isolation
            logger.exception("[%s] unexpected error", spec.key)
            result = RepoResult(spec, "failed", detail=f"unexpected: {exc}")
        results.append(result)

    _print_summary(results)
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
