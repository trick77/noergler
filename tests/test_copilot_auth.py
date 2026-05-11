import asyncio

import pytest

from app.copilot_auth import CopilotTokenProvider, DEFAULT_API_URL


@pytest.mark.asyncio
async def test_passthrough_returns_oauth_token():
    provider = CopilotTokenProvider(oauth_token="gho_test")
    try:
        tok, api = await provider.get_token()
        assert tok == "gho_test"
        assert api == DEFAULT_API_URL
    finally:
        await provider.close()


@pytest.mark.asyncio
async def test_passthrough_is_stable_across_calls():
    provider = CopilotTokenProvider(oauth_token="gho_test")
    try:
        results = await asyncio.gather(*(provider.get_token() for _ in range(5)))
        assert {t for t, _ in results} == {"gho_test"}
        assert {a for _, a in results} == {DEFAULT_API_URL}
    finally:
        await provider.close()


@pytest.mark.asyncio
async def test_endpoints_api_property():
    provider = CopilotTokenProvider(oauth_token="gho_test")
    try:
        assert provider.endpoints_api == DEFAULT_API_URL
    finally:
        await provider.close()
