import asyncio
import time

import httpx
import pytest
import respx

from app.copilot_auth import CopilotTokenProvider, TOKEN_EXCHANGE_URL


_counter = 0


def _payload(expires_in: int = 1800, api: str = "https://api.business.githubcopilot.com"):
    global _counter
    _counter += 1
    return {
        "token": f"tid-{_counter}",
        "expires_at": int(time.time()) + expires_in,
        "endpoints": {"api": api},
    }


@pytest.mark.asyncio
@respx.mock
async def test_exchange_and_cache():
    route = respx.get(TOKEN_EXCHANGE_URL).mock(return_value=httpx.Response(200, json=_payload()))
    provider = CopilotTokenProvider(oauth_token="gho_test")
    try:
        tok1, api1 = await provider.get_token()
        tok2, api2 = await provider.get_token()
        assert tok1 == tok2
        assert api1 == api2 == "https://api.business.githubcopilot.com"
        assert route.call_count == 1  # second call is served from cache
    finally:
        await provider.close()


@pytest.mark.asyncio
@respx.mock
async def test_refreshes_when_near_expiry():
    # First exchange returns a token that is ~already expired (below 60 s leeway).
    first = _payload(expires_in=30)
    second = _payload(expires_in=1800)
    route = respx.get(TOKEN_EXCHANGE_URL).mock(
        side_effect=[
            httpx.Response(200, json=first),
            httpx.Response(200, json=second),
        ]
    )
    provider = CopilotTokenProvider(oauth_token="gho_test")
    try:
        tok1, _ = await provider.get_token()
        tok2, _ = await provider.get_token()
        assert tok1 != tok2
        assert route.call_count == 2
    finally:
        await provider.close()


@pytest.mark.asyncio
@respx.mock
async def test_concurrent_callers_share_single_exchange():
    route = respx.get(TOKEN_EXCHANGE_URL).mock(return_value=httpx.Response(200, json=_payload()))
    provider = CopilotTokenProvider(oauth_token="gho_test")
    try:
        results = await asyncio.gather(*(provider.get_token() for _ in range(5)))
        tokens = {t for t, _ in results}
        assert len(tokens) == 1
        assert route.call_count == 1
    finally:
        await provider.close()


@pytest.mark.asyncio
@respx.mock
async def test_uses_default_api_when_endpoints_missing():
    payload = {"token": "abc", "expires_at": int(time.time()) + 1800}
    respx.get(TOKEN_EXCHANGE_URL).mock(return_value=httpx.Response(200, json=payload))
    provider = CopilotTokenProvider(oauth_token="gho_test")
    try:
        _, api = await provider.get_token()
        assert api == "https://api.business.githubcopilot.com"
    finally:
        await provider.close()


@pytest.mark.asyncio
@respx.mock
async def test_raises_on_malformed_response():
    respx.get(TOKEN_EXCHANGE_URL).mock(return_value=httpx.Response(200, json={}))
    provider = CopilotTokenProvider(oauth_token="gho_test")
    try:
        with pytest.raises(RuntimeError):
            await provider.get_token()
    finally:
        await provider.close()
