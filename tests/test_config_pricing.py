"""Pricing helpers in app/config.py."""
import pytest

from app.config import estimate_cost_usd, pricing_for


class TestPricingFor:
    def test_known_model_returns_entry(self):
        price = pricing_for("gpt-5.4")
        assert price is not None
        assert price.input_per_mtok == 2.50
        assert price.cached_input_per_mtok == 0.25
        assert price.output_per_mtok == 15.00

    def test_unknown_model_returns_none(self):
        assert pricing_for("totally-fictional-model") is None

    def test_dated_suffix_resolves_to_base(self):
        # Dated/suffixed ids fall back to the base entry, mirroring
        # _context_window_for behaviour.
        price = pricing_for("claude-sonnet-4-20250514")
        assert price is not None
        assert price.input_per_mtok == 3.00


class TestEstimateCostUsd:
    def test_gpt_5_4_sample(self):
        # 100k prompt + 5k completion @ $2.50 / $15.00 per 1M.
        cost = estimate_cost_usd("gpt-5.4", 100_000, 5_000)
        assert cost is not None
        assert cost == pytest.approx(0.25 + 0.075)

    def test_unknown_model_returns_none(self):
        assert estimate_cost_usd("imaginary-9000", 1000, 1000) is None

    def test_zero_tokens_yields_zero(self):
        assert estimate_cost_usd("gpt-5.4", 0, 0) == 0.0
