"""Optional outbound emitter to riptide-collector.

Forwards finops (review-completion cost) and reviewer-precision (feedback)
events. PR lifecycle is not forwarded — riptide already gets that from
Bitbucket directly.

If `RIPTIDE_URL` or `RIPTIDE_TOKEN` is unset, every emit is a no-op and the
client is `enabled=False`. When configured, the client validates the token
against riptide's `GET /auth/ping` at startup; a 401 fails noergler boot,
while a network error is logged and noergler continues (riptide may be
temporarily down — emissions are best-effort anyway).
"""

from __future__ import annotations

import logging
from datetime import datetime
from decimal import Decimal
from typing import Any

import httpx

logger = logging.getLogger(__name__)


class RiptideAuthError(RuntimeError):
    """Raised at startup when riptide rejects the configured token."""


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
            logger.warning(
                "riptide_unreachable_at_startup: %s — continuing without forwarding",
                exc,
            )
            return None
        if response.status_code == 401:
            raise RiptideAuthError(
                f"RIPTIDE_TOKEN rejected by {self._url} (HTTP 401). "
                "Check that the token matches the team's entry in team-keys.json."
            )
        if response.status_code >= 400:
            logger.warning(
                "riptide_ping_unexpected_status status=%s body=%s",
                response.status_code,
                response.text[:200],
            )
            return None
        team = (response.json() or {}).get("team")
        logger.info("riptide_reachable team=%s url=%s", team, self._url)
        return team if isinstance(team, str) else None

    async def emit_completed(
        self,
        *,
        pr_key: str,
        repo: str,
        commit_sha: str,
        run_id: str,
        model: str,
        prompt_tokens: int,
        completion_tokens: int,
        elapsed_ms: int,
        findings_count: int,
        cost_usd: Decimal | float | None,
        finished_at: datetime,
    ) -> None:
        if cost_usd is None:
            # Skip emission rather than send a meaningless 0; missing pricing
            # for a model is a config gap worth surfacing on the riptide side
            # via low row-counts, not silently filled with zeros.
            logger.debug("riptide_emit_skipped_no_cost model=%s run_id=%s", model, run_id)
            return
        body: dict[str, Any] = {
            "event_type": "completed",
            "pr_key": pr_key,
            "repo": repo,
            "commit_sha": commit_sha,
            "run_id": run_id,
            "model": model,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "elapsed_ms": elapsed_ms,
            "findings_count": findings_count,
            "cost_usd": str(cost_usd),
            "finished_at": _isoformat_z(finished_at),
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
                "riptide_emit_failed event=%s err=%s", body.get("event_type"), exc
            )
            return
        if response.status_code >= 400:
            logger.warning(
                "riptide_emit_rejected event=%s status=%s body=%s",
                body.get("event_type"),
                response.status_code,
                response.text[:200],
            )


def _isoformat_z(value: datetime) -> str:
    """ISO-8601 with explicit 'Z' for UTC (riptide expects UTC offsets)."""
    if value.tzinfo is None:
        return value.strftime("%Y-%m-%dT%H:%M:%SZ")
    return value.isoformat().replace("+00:00", "Z")
