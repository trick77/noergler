import asyncio
import logging
import time

import httpx

logger = logging.getLogger(__name__)

TOKEN_EXCHANGE_URL = "https://api.github.com/copilot_internal/v2/token"
DEFAULT_API_URL = "https://api.business.githubcopilot.com"
REFRESH_LEEWAY_SECONDS = 60


class CopilotTokenProvider:
    """Exchanges a long-lived GitHub OAuth token for short-lived Copilot tokens.

    The OAuth token (provisioned via hack/copilot-provision-token.sh) stays valid
    until the user revokes it. Each inference call needs a short-lived Copilot
    token (~30 min TTL) obtained from /copilot_internal/v2/token. This provider
    caches the short-lived token and refreshes it before expiry.
    """

    def __init__(
        self,
        oauth_token: str,
        integration_id: str = "vscode-chat",
        editor_version: str = "vscode/1.109.2",
        editor_plugin_version: str = "copilot-chat/0.37.5",
        user_agent: str = "GitHubCopilotChat/0.37.5",
    ):
        self._oauth_token = oauth_token
        self._integration_id = integration_id
        self._editor_version = editor_version
        self._editor_plugin_version = editor_plugin_version
        self._user_agent = user_agent
        self._token: str | None = None
        self._expires_at: int = 0
        self._endpoints_api: str = DEFAULT_API_URL
        self._lock = asyncio.Lock()
        self._client = httpx.AsyncClient(timeout=30.0)

    @property
    def endpoints_api(self) -> str:
        return self._endpoints_api

    async def close(self) -> None:
        await self._client.aclose()

    async def get_token(self) -> tuple[str, str]:
        """Return a (copilot_token, endpoints_api) tuple, refreshing if needed."""
        if self._fresh():
            return self._token, self._endpoints_api  # type: ignore[return-value]
        async with self._lock:
            if self._fresh():
                return self._token, self._endpoints_api  # type: ignore[return-value]
            await self._exchange()
            return self._token, self._endpoints_api  # type: ignore[return-value]

    def _fresh(self) -> bool:
        return (
            self._token is not None
            and (self._expires_at - int(time.time())) > REFRESH_LEEWAY_SECONDS
        )

    async def _exchange(self) -> None:
        response = await self._client.get(
            TOKEN_EXCHANGE_URL,
            headers={
                "Authorization": f"token {self._oauth_token}",
                "Accept": "application/json",
                "Editor-Version": self._editor_version,
                "Editor-Plugin-Version": self._editor_plugin_version,
                "Copilot-Integration-Id": self._integration_id,
                "User-Agent": self._user_agent,
            },
        )
        response.raise_for_status()
        data = response.json()
        token = data.get("token")
        expires_at = data.get("expires_at")
        if not token or not expires_at:
            raise RuntimeError(
                f"Copilot token exchange returned unexpected payload: {data!r}"
            )
        endpoints = data.get("endpoints") or {}
        api = endpoints.get("api") or DEFAULT_API_URL
        self._token = token
        self._expires_at = int(expires_at)
        self._endpoints_api = api.rstrip("/")
        ttl = self._expires_at - int(time.time())
        logger.info(
            "Copilot token exchanged (expires in %ds, endpoints.api=%s)",
            ttl, self._endpoints_api,
        )
