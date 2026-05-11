import logging

logger = logging.getLogger(__name__)

DEFAULT_API_URL = "https://api.business.githubcopilot.com"


class CopilotTokenProvider:
    """Holds the long-lived GitHub OAuth token used directly against the Copilot API.

    Mirrors `sst/opencode@dev` `packages/opencode/src/plugin/github-copilot/copilot.ts`
    lines 92-168: opencode sends the OAuth access token from the device flow straight
    to api.business.githubcopilot.com — it does not exchange it for a short-lived
    Copilot token via `/copilot_internal/v2/token`, and does not stamp it with a
    `Copilot-Integration-Id`. Doing so causes "token not authorized for this
    integration" 403s when paired with `User-Agent: opencode/...`.
    """

    def __init__(self, oauth_token: str):
        self._oauth_token = oauth_token

    async def close(self) -> None:
        return None

    async def get_token(self) -> str:
        return self._oauth_token
