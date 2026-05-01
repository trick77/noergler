"""Pricing helpers in app/config.py."""
import json

import httpx
import pytest
import respx

from app.config import (
    LITELLM_PRICING_URL,
    _STATIC_MODEL_PRICING,
    _swap_pricing,
    apply_pricing_overlay,
    estimate_cost_usd,
    fetch_litellm_pricing,
    pricing_for,
)


@pytest.fixture(autouse=True)
def _reset_pricing_table():
    """Each test starts (and ends) with the static defaults installed."""
    _swap_pricing(dict(_STATIC_MODEL_PRICING))
    yield
    _swap_pricing(dict(_STATIC_MODEL_PRICING))


class TestPricingFor:
    def test_known_model_returns_entry(self):
        # Given the static defaults
        # When we look up gpt-5.4
        price = pricing_for("gpt-5.4")
        # Then we get the static entry
        assert price is not None
        assert price.input_per_mtok == 2.50
        assert price.cached_input_per_mtok == 0.25
        assert price.output_per_mtok == 15.00

    def test_unknown_model_returns_none(self):
        assert pricing_for("totally-fictional-model") is None

    def test_dated_suffix_resolves_to_base(self):
        price = pricing_for("claude-sonnet-4-20250514")
        assert price is not None
        assert price.input_per_mtok == 3.00

    def test_dated_suffix_prefers_longest_match(self):
        # Regression: `gpt-5.4-mini-2025-06-01` must resolve to the mini
        # entry, not the (3x more expensive) `gpt-5.4` entry.
        price = pricing_for("gpt-5.4-mini-2025-06-01")
        assert price is not None
        assert price.input_per_mtok == 0.75
        assert price.output_per_mtok == 4.50

    def test_default_llm_model_is_priced(self):
        from app.config import LLMConfig
        default_model = LLMConfig.model_fields["model"].default
        assert pricing_for(default_model) is not None


class TestEstimateCostUsd:
    def test_gpt_5_4_sample(self):
        cost = estimate_cost_usd("gpt-5.4", 100_000, 5_000)
        assert cost is not None
        assert cost == pytest.approx(0.25 + 0.075)

    def test_unknown_model_returns_none(self):
        assert estimate_cost_usd("imaginary-9000", 1000, 1000) is None

    def test_zero_tokens_yields_zero(self):
        assert estimate_cost_usd("gpt-5.4", 0, 0) == 0.0


class TestApplyPricingOverlay:
    def test_overlay_replaces_known_model(self):
        # Given a DB-cached price that differs from the static default
        # When we apply the overlay
        n = apply_pricing_overlay({"gpt-5.4": (9.99, 1.0, 99.0)})
        # Then live pricing reflects the overlay for that model only
        assert n == 1
        assert pricing_for("gpt-5.4").input_per_mtok == 9.99  # type: ignore[union-attr]
        # And other models still come from the static defaults
        assert pricing_for("gpt-4.1").input_per_mtok == 2.00  # type: ignore[union-attr]

    def test_overlay_ignores_unknown_models(self):
        n = apply_pricing_overlay({"unknown-model-x": (1.0, 0.1, 2.0)})
        assert n == 0
        assert pricing_for("unknown-model-x") is None


class TestFetchLitellmPricing:
    @pytest.mark.asyncio
    async def test_overlays_priced_entries(self):
        # Given a LiteLLM payload covering one GPT and one Claude id
        payload = {
            "gpt-5.4": {
                "input_cost_per_token": 0.000004,
                "cache_read_input_token_cost": 0.0000004,
                "output_cost_per_token": 0.00002,
            },
            "openrouter/anthropic/claude-sonnet-4.6": {
                "input_cost_per_token": 0.0000035,
                "output_cost_per_token": 0.0000175,
            },
        }
        with respx.mock:
            respx.get(LITELLM_PRICING_URL).mock(
                return_value=httpx.Response(200, text=json.dumps(payload))
            )
            # When we fetch
            table = await fetch_litellm_pricing()
        # Then GPT-5.4 reflects the per-1M conversion
        assert table is not None
        assert table["gpt-5.4"].input_per_mtok == pytest.approx(4.0)
        assert table["gpt-5.4"].cached_input_per_mtok == pytest.approx(0.4)
        assert table["gpt-5.4"].output_per_mtok == pytest.approx(20.0)
        # And the Claude prefixed key resolves under the bare model id, with
        # cached input falling back to 10% of input when LiteLLM omits it.
        assert table["claude-sonnet-4.6"].input_per_mtok == pytest.approx(3.5)
        assert table["claude-sonnet-4.6"].cached_input_per_mtok == pytest.approx(0.35)
        # And ids missing from LiteLLM keep the static defaults.
        assert table["claude-opus-4.7"] == _STATIC_MODEL_PRICING["claude-opus-4.7"]

    @pytest.mark.asyncio
    async def test_http_failure_returns_none(self):
        with respx.mock:
            respx.get(LITELLM_PRICING_URL).mock(
                return_value=httpx.Response(503, text="boom")
            )
            assert await fetch_litellm_pricing() is None
        # And the live table is untouched (still equals static defaults).
        assert pricing_for("gpt-5.4") == _STATIC_MODEL_PRICING["gpt-5.4"]
