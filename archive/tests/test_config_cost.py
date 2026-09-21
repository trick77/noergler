"""Cost and context-budget helpers in app/config.py."""
import logging

import pytest

from app.config import TokenUsage, resolve_cost_usd, usable_context_budget


class TestResolveCostUsd:
    """The endpoint's figure is the only source; there is no rate table."""

    def test_reported_cost_is_returned(self):
        assert resolve_cost_usd(TokenUsage(prompt=1_000_000, cost_usd=0.01)) == 0.01

    def test_no_report_yields_no_cost(self):
        # LiteLLM sends the literal "None" for a deployment it can't price;
        # _usd_header turns that into None and the run stays unpriced, with
        # the per-PR cap failing open.
        assert resolve_cost_usd(TokenUsage(prompt=100_000, completion=5_000)) is None

    def test_reported_zero_on_a_real_call_is_kept_and_logged(self, caplog):
        # Trusted as reported: the cap counts $0.00. Logged because it usually
        # means a LiteLLM deployment with its costs explicitly set to 0.
        with caplog.at_level(logging.WARNING, logger="app.config"):
            cost = resolve_cost_usd(TokenUsage(prompt=1_000_000, cost_usd=0.0))
        assert cost == 0.0
        assert any("zero cost" in r.getMessage() for r in caplog.records)

    def test_reported_zero_with_no_tokens_is_silent(self, caplog):
        # Nothing was consumed, so zero is the truthful answer.
        with caplog.at_level(logging.WARNING, logger="app.config"):
            cost = resolve_cost_usd(TokenUsage(cost_usd=0.0))
        assert cost == 0.0
        assert not caplog.records


class TestTokenUsage:
    def test_total_excludes_cached_double_count(self):
        # cached is a subset of prompt, so it must not be added again.
        assert TokenUsage(prompt=100, cached=40, completion=10).total == 110

    def test_cost_defaults_to_none(self):
        # An endpoint that reports no cost leaves the run unpriced rather than
        # estimated — the cap then fails open.
        assert TokenUsage(prompt=100, completion=10).cost_usd is None

    def test_zero_cost_is_preserved_not_treated_as_missing(self):
        # 0.0 is falsy; it must stay a real reported cost, not become None.
        assert TokenUsage(cost_usd=0.0).cost_usd == 0.0


class TestUsableContextBudget:
    def test_below_threshold_uses_flat_headroom(self):
        assert usable_context_budget(128_000) == 112_000

    def test_above_threshold_degrades_by_tail_fraction(self):
        assert usable_context_budget(512_000) == 384_000
        assert usable_context_budget(1_050_000) == 653_000

    def test_at_threshold_is_flat_branch(self):
        assert usable_context_budget(256_000) == 256_000 - 16_000

    def test_tiny_window_clamped_to_floor(self):
        assert usable_context_budget(1_000) == 2_000
