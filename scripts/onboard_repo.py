"""
Onboard a Bitbucket Server repository to noergler webhook delivery.

Usage:
    python -m scripts.onboard_repo config.json [--name noergler] [--dry-run] [--env-file PATH]

The JSON config describes target repos and URLs. Secrets (BITBUCKET_TOKEN,
BITBUCKET_WEBHOOK_SECRET) are resolved from, in order:
    1. Process environment
    2. .env in CWD
    3. --env-file <path>

BITBUCKET_WEBHOOK_SECRET used by this script MUST match the value the running
noergler service has configured — Bitbucket and noergler share the HMAC secret;
this script only programs Bitbucket's side.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import httpx

from app.bitbucket import BitbucketClient
from app.config import REQUIRED_WEBHOOK_EVENTS, BitbucketConfig

logger = logging.getLogger("onboard_repo")

DEFAULT_WEBHOOK_NAME = "noergler"


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
    # Build the map in reverse-precedence order so later updates (= higher
    # priority) overwrite earlier ones.
    merged: dict[str, str] = {}
    if env_file is not None:
        merged.update(_load_env_file(env_file))
    merged.update(_load_env_file(Path.cwd() / ".env"))
    for k in ("BITBUCKET_TOKEN", "BITBUCKET_WEBHOOK_SECRET", "BITBUCKET_USERNAME"):
        if k in os.environ and os.environ[k]:
            merged[k] = os.environ[k]
    # Stash the resolved env for later use (e.g. BITBUCKET_USERNAME lookup).
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
# Onboarder
# --------------------------------------------------------------------------- #

@dataclass
class RepoResult:
    repo: RepoSpec
    status: str  # "ok", "failed", "skipped"
    detail: str = ""
    test_status: int | None = None
    diff: list[str] = field(default_factory=list)


class RepoOnboarder:
    def __init__(
        self,
        client: BitbucketClient,
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
    async def verify_permissions(self, spec: RepoSpec) -> None:
        """Confirm read access to repo and its PRs. Raises httpx.HTTPStatusError on failure."""
        await self.client.get_repo(spec.project, spec.repo)
        await self.client.list_pull_requests(spec.project, spec.repo, limit=1)
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
        # Secret is not returned by the API; assume it must be re-applied when other
        # changes exist, but never just for the secret (can't detect drift).
        existing_cfg = existing.get("configuration") or {}
        if not existing_cfg:
            diffs.append("configuration.secret: (unset) -> (set)")
        return diffs

    async def upsert_webhook(self, spec: RepoSpec) -> tuple[int, list[str]]:
        """Create or update the webhook. Returns (webhook_id, diff)."""
        hooks = await self.client.list_webhooks(spec.project, spec.repo)
        existing = next((h for h in hooks if h.get("name") == self.webhook_name), None)
        body = self._build_webhook_body()

        if existing is None:
            logger.info("[%s] creating webhook %r", spec.key, self.webhook_name)
            if self.dry_run:
                logger.info("[%s] DRY-RUN body=%s", spec.key, _redact(body))
                return -1, ["create"]
            created = await self.client.create_webhook(spec.project, spec.repo, body)
            return int(created["id"]), ["create"]

        diff = self._diff_webhook(existing)
        if not diff:
            logger.info("[%s] webhook already up to date", spec.key)
            return int(existing["id"]), []

        logger.info("[%s] updating webhook id=%s changes=%s", spec.key, existing.get("id"), diff)
        if self.dry_run:
            logger.info("[%s] DRY-RUN body=%s", spec.key, _redact(body))
            return int(existing["id"]), diff
        await self.client.update_webhook(spec.project, spec.repo, int(existing["id"]), body)
        return int(existing["id"]), diff

    # -- Step 3 -- #
    async def test_webhook(self, spec: RepoSpec, webhook_id: int) -> int:
        """Trigger Bitbucket's test-webhook endpoint. Returns the observed downstream status."""
        response = await self.client.test_webhook(spec.project, spec.repo, webhook_id)
        # Bitbucket returns 200 with a body describing the downstream attempt, or a 4xx if the
        # test couldn't be issued at all. Surface the raw status — callers decide what's OK.
        if response.status_code >= 400:
            logger.error(
                "[%s] test-webhook failed: HTTP %d — %s",
                spec.key, response.status_code, response.text,
            )
            return response.status_code
        try:
            data = response.json()
        except ValueError:
            data = {}
        downstream = data.get("statusCode") or data.get("status") or response.status_code
        logger.info("[%s] test-webhook downstream status=%s", spec.key, downstream)
        return int(downstream) if isinstance(downstream, int) else response.status_code

    # -- Orchestrator -- #
    async def onboard(self, spec: RepoSpec) -> RepoResult:
        try:
            await self.verify_permissions(spec)
        except httpx.HTTPStatusError as exc:
            return RepoResult(
                spec, "failed",
                detail=f"permission check HTTP {exc.response.status_code}: {exc.response.text[:200]}",
            )
        except httpx.HTTPError as exc:
            return RepoResult(spec, "failed", detail=f"permission check: {exc}")

        try:
            webhook_id, diff = await self.upsert_webhook(spec)
        except httpx.HTTPStatusError as exc:
            return RepoResult(
                spec, "failed",
                detail=f"upsert webhook HTTP {exc.response.status_code}: {exc.response.text[:200]}",
            )
        except httpx.HTTPError as exc:
            return RepoResult(spec, "failed", detail=f"upsert webhook: {exc}")

        if self.dry_run:
            return RepoResult(spec, "ok", detail="dry-run", diff=diff)

        try:
            status = await self.test_webhook(spec, webhook_id)
        except httpx.HTTPError as exc:
            # Webhook was created/updated, but connectivity test failed. Surface both.
            return RepoResult(
                spec, "failed",
                detail=f"webhook upserted; test failed: {exc}",
                diff=diff,
            )

        if 200 <= status < 300:
            return RepoResult(spec, "ok", detail=f"test status {status}", test_status=status, diff=diff)

        detail = f"test-webhook returned {status}"
        if status == 401:
            detail += " — likely BITBUCKET_WEBHOOK_SECRET mismatch with the running service"
        return RepoResult(spec, "failed", detail=detail, test_status=status, diff=diff)


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
        prog="python -m scripts.onboard_repo",
        description="Onboard Bitbucket Server repos to noergler by creating/updating their webhooks (idempotent).",
        epilog=(
            "Write permission check is implicit: if the token lacks repo-write, the webhook "
            "create/update will surface a 403. BITBUCKET_WEBHOOK_SECRET must match the value "
            "configured on the running noergler service."
        ),
    )
    parser.add_argument("config", type=Path, help="Path to onboarding JSON config")
    parser.add_argument("--name", default=DEFAULT_WEBHOOK_NAME, help=f"Webhook name (default: {DEFAULT_WEBHOOK_NAME})")
    parser.add_argument("--dry-run", action="store_true", help="Print planned changes without mutating Bitbucket")
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


async def _run(args: argparse.Namespace) -> int:
    token, webhook_secret = resolve_secrets(args.env_file)
    inp = load_onboarding_input(args.config)

    logger.info("Bitbucket URL: %s", inp.bitbucket_url)
    logger.info("Webhook URL:   %s", inp.webhook_url)
    logger.info("Bitbucket token loaded: %s", _mask(token))
    logger.info("Webhook secret loaded:  %s", _mask(webhook_secret))
    logger.info("Target repos (%d): %s", len(inp.repos), ", ".join(r.key for r in inp.repos))
    if args.dry_run:
        logger.info("DRY-RUN: no writes will be issued")

    bb_config = BitbucketConfig(
        base_url=inp.bitbucket_url,
        token=token,
        webhook_secret=webhook_secret,
        username=getattr(resolve_secrets, "_resolved", {}).get("BITBUCKET_USERNAME")
        or os.environ.get("BITBUCKET_USERNAME", "noergler-onboard"),
    )
    client = BitbucketClient(bb_config)
    onboarder = RepoOnboarder(
        client,
        webhook_url=inp.webhook_url,
        webhook_secret=webhook_secret,
        webhook_name=args.name,
        dry_run=args.dry_run,
    )

    results: list[RepoResult] = []
    try:
        for spec in inp.repos:
            logger.info("--- %s ---", spec.key)
            try:
                result = await onboarder.onboard(spec)
            except Exception as exc:  # noqa: BLE001 — per-repo isolation
                logger.exception("[%s] unexpected error", spec.key)
                result = RepoResult(spec, "failed", detail=f"unexpected: {exc}")
            results.append(result)
    finally:
        await client.close()

    _print_summary(results)
    failed = [r for r in results if r.status == "failed"]
    return 1 if failed else 0


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)
    return asyncio.run(_run(args))


if __name__ == "__main__":
    sys.exit(main())
