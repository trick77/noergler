"""Optional outbound emitter to riptide-collector.

Forwards per-PR review rollups (cost + diff size) and reviewer-precision
(feedback) events. The PR lifecycle itself is not forwarded directly —
riptide already gets that from Bitbucket — but the rollup is keyed off
the PR's terminal state (merged / declined / deleted) so finops can
distinguish review spend that shipped from review spend that didn't.

If `RIPTIDE_URL` or `RIPTIDE_TOKEN` is unset, every emit is a no-op and the
client is `enabled=False`. When configured, the client validates the token
against riptide's `GET /auth/ping` at startup; a 401 fails noergler boot,
while a network error is logged and noergler continues (riptide may be
temporarily down — emissions are best-effort anyway).
"""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from typing import Any, final

import httpx
import structlog

logger = structlog.stdlib.get_logger(__name__)


class RiptideAuthError(RuntimeError):
    """Raised at startup when riptide rejects the configured token."""


@final
class RiptideClient:
    """Best-effort emitter for noergler → riptide events.

    Construct from env via `RiptideClient.from_env()`. The instance is always
    usable — when `enabled` is False, `emit_*` and `verify_at_startup` are
    no-ops.
    """

    _PATH = "/webhooks/noergler"
    _PING_PATH = "/auth/ping"

    def __init__(
        self,
        url: str | None,
        token: str | None,
        timeout_seconds: float = 2.0,
    ):
        self._url = url.rstrip("/") if url else None
        self._token = token or None
        self.enabled = bool(self._url and self._token)
        self._timeout = timeout_seconds
        self._client: httpx.AsyncClient | None = None
        if self.enabled:
            self._client = httpx.AsyncClient(
                headers={
                    "Authorization": f"Bearer {self._token}",
                    "Content-Type": "application/json",
                },
                timeout=self._timeout,
            )

    @classmethod
    def from_env(cls, url: str, token: str) -> RiptideClient:
        """Build a client from explicit values; either may be empty.

        The config layer extracts the env vars; passing them in keeps this
        module test-friendly and free of `os.environ` reads.
        """
        return cls(url=url or None, token=token or None)

    async def close(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def verify_at_startup(self) -> str | None:
        """Validate reachability + token. Returns the resolved team or None.

        - `enabled=False` → returns None silently.
        - 200 → returns the team name from the ping response.
        - 401 → raises `RiptideAuthError` to fail noergler startup.
        - Any other failure → logs a warning and returns None; noergler
          starts and the runtime emit path will keep retrying.
        """
        if not self.enabled or self._client is None or self._url is None:
            return None
        try:
            response = await self._client.get(self._url + self._PING_PATH)
        except httpx.HTTPError as exc:
            logger.warning("Riptide unreachable at startup (url=%s): %s", self._url, exc)
            return None
        if response.status_code == 401:
            raise RiptideAuthError(
                f"RIPTIDE_TOKEN rejected by {self._url} (HTTP 401). "
                "Check that the token matches the team's entry in team-keys.json."
            )
        if response.status_code >= 400:
            logger.warning(
                "Riptide ping returned unexpected status %d: %s",
                response.status_code, response.text[:200],
            )
            return None
        team = (response.json() or {}).get("team")
        logger.info("Riptide: OK (team=%s)", team)
        return team if isinstance(team, str) else None

    async def emit_pr_completed(
        self,
        *,
        outcome: str,
        pr_key: str,
        repo: str,
        source_commit_sha: str,
        merge_commit_sha: str | None,
        lines_added: int,
        lines_removed: int,
        files_changed: int,
        total_runs: int,
        total_prompt_tokens: int,
        total_completion_tokens: int,
        total_elapsed_ms: int,
        total_findings_count: int,
        total_cost_usd: Decimal | float | None,
        models_used: list[str],
        first_review_at: datetime,
        closed_at: datetime,
    ) -> None:
        """Emit the per-PR rollup once the PR has reached a terminal state.

        `outcome` must be one of 'merged' / 'declined' / 'deleted' — riptide
        rejects anything else with HTTP 422.
        """
        if total_cost_usd is None:
            # Skip emission rather than send a meaningless 0; missing pricing
            # for a model is a config gap worth surfacing on the riptide side
            # via low row-counts, not silently filled with zeros.
            logger.debug(
                "Riptide emit skipped (no cost) for %s outcome=%s models=%s",
                pr_key, outcome, models_used,
            )
            return
        body: dict[str, Any] = {
            "event_type": "pr_completed",
            "outcome": outcome,
            "pr_key": pr_key,
            "repo": repo,
            "source_commit_sha": source_commit_sha,
            "merge_commit_sha": merge_commit_sha,
            "lines_added": lines_added,
            "lines_removed": lines_removed,
            "files_changed": files_changed,
            "total_runs": total_runs,
            "total_prompt_tokens": total_prompt_tokens,
            "total_completion_tokens": total_completion_tokens,
            "total_elapsed_ms": total_elapsed_ms,
            "total_findings_count": total_findings_count,
            "total_cost_usd": str(total_cost_usd),
            "models_used": models_used,
            "first_review_at": _isoformat_z(first_review_at),
            "closed_at": _isoformat_z(closed_at),
        }
        await self._post(body)

    async def emit_feedback(
        self,
        *,
        pr_key: str,
        finding_id: str,
        verdict: str,
        actor: str,
        repo: str | None,
        commit_sha: str | None,
        occurred_at: datetime,
    ) -> None:
        body: dict[str, Any] = {
            "event_type": "feedback",
            "pr_key": pr_key,
            "finding_id": finding_id,
            "verdict": verdict,
            "actor": actor,
            "repo": repo,
            "commit_sha": commit_sha,
            "occurred_at": _isoformat_z(occurred_at),
        }
        await self._post(body)

    async def _post(self, body: dict[str, Any]) -> None:
        """Best-effort POST. Never raises — at most logs."""
        if not self.enabled or self._client is None or self._url is None:
            return
        url = self._url + self._PATH
        try:
            response = await self._client.post(url, json=body)
        except httpx.HTTPError as exc:
            logger.warning(
                "Riptide emit failed (event_type=%s): %s",
                body.get("event_type"), exc,
            )
            return
        if response.status_code >= 400:
            logger.warning(
                "Riptide emit rejected (event_type=%s, status=%d): %s",
                body.get("event_type"), response.status_code, response.text[:200],
            )


def _isoformat_z(value: datetime) -> str:
    """ISO-8601 with explicit 'Z' for UTC (riptide expects UTC offsets)."""
    if value.tzinfo is None:
        return value.strftime("%Y-%m-%dT%H:%M:%SZ")
    return value.isoformat().replace("+00:00", "Z")
