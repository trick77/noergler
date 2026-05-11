import asyncio

import pytest

from app.copilot_auth import CopilotTokenProvider


@pytest.mark.asyncio
async def test_passthrough_returns_oauth_token():
    provider = CopilotTokenProvider(oauth_token="gho_test")
    try:
        assert await provider.get_token() == "gho_test"
    finally:
        await provider.close()


@pytest.mark.asyncio
async def test_passthrough_is_stable_across_calls():
    provider = CopilotTokenProvider(oauth_token="gho_test")
    try:
        results = await asyncio.gather(*(provider.get_token() for _ in range(5)))
        assert set(results) == {"gho_test"}
    finally:
        await provider.close()
