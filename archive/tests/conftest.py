"""Shared test fixtures.

Tiktoken requires downloading encoding data from the internet.  In
environments where the proxy blocks that request every test touching
``count_tokens`` would fail.  We patch it globally with a cheap
approximation so the full test-suite can run offline.
"""

from unittest.mock import patch

import pytest


def _fake_count_tokens(text: str) -> int:
    """Approximate token count: ~4 chars per token."""
    return max(1, len(text) // 4)


@pytest.fixture(autouse=True)
def _mock_count_tokens():
    with (
        patch("app.llm_client.count_tokens", side_effect=_fake_count_tokens),
        patch("app.reviewer.count_tokens", side_effect=_fake_count_tokens),
    ):
        yield
