"""Model-catalog resolution and cost helpers in app/config.py."""
import json

import httpx
import pytest
import respx

from app.config import (
    LITELLM_PRICING_URL,
    LLMConfig,
    ModelCatalogEntry,
    ModelCatalogError,
    TokenUsage,
    _swap_active_entry,
    active_entry,
    estimate_cost_usd,
    fetch_model_catalog,
    refresh_active_entry,
    resolve_catalog_entry,
    resolve_or_raise,
    usable_context_budget,
)

# Trimmed from the real LiteLLM catalog — same field names and magnitudes,
# without committing the 1.6MB file as a fixture.
CATALOG = {
    "gpt-5.5": {
        "input_cost_per_token": 5e-06,
        "output_cost_per_token": 3e-05,
        "cache_read_input_token_cost": 5e-07,
        "input_cost_per_token_above_272k_tokens": 1e-05,
        "max_input_tokens": 1050000,
        "litellm_provider": "openai",
        "mode": "chat",
    },
    "gpt-5.4-mini": {
        "input_cost_per_token": 7.5e-07,
        "output_cost_per_token": 4.5e-06,
        "cache_read_input_token_cost": 7.5e-08,
        "max_input_tokens": 400000,
    },
    "gpt-5.4": {
        "input_cost_per_token": 2.5e-06,
        "output_cost_per_token": 1.5e-05,
        "cache_read_input_token_cost": 2.5e-07,
        "max_input_tokens": 1050000,
    },
    # No cache-read rate published.
    "budget-model": {
        "input_cost_per_token": 1e-06,
        "output_cost_per_token": 2e-06,
        "max_input_tokens": 128000,
    },
    "openrouter/anthropic/claude-sonnet-4.6": {
        "input_cost_per_token": 3e-06,
        "output_cost_per_token": 1.5e-05,
        "cache_read_input_token_cost": 3e-07,
        "max_input_tokens": 200000,
    },
    "broken-model": {"input_cost_per_token": 1e-06},  # no output cost / window
    "zero-window": {
        "input_cost_per_token": 1e-06,
        "output_cost_per_token": 2e-06,
        "max_input_tokens": 0,
    },
}


def _mock_catalog(payload: dict[str, object] | None = None, status: int = 200):
    return respx.get(LITELLM_PRICING_URL).mock(
        return_value=httpx.Response(
            status, text=json.dumps(payload if payload is not None else CATALOG)
        )
    )


@pytest.fixture(autouse=True)
def _reset_active_entry():
    """No entry installed before each test — startup is what installs one."""
    import app.config
    app.config._ACTIVE_ENTRY = None
    yield
    app.config._ACTIVE_ENTRY = None


def _entry(**overrides) -> ModelCatalogEntry:
    base = dict(
        model_id="gpt-5.5",
        input_per_mtok=5.00,
        cached_input_per_mtok=0.50,
        output_per_mtok=30.00,
        max_input_tokens=1_050_000,
    )
    base.update(overrides)
    return ModelCatalogEntry(**base)  # pyright: ignore[reportArgumentType]


class TestResolveCatalogEntry:
    def test_exact_match_converts_per_token_to_per_mtok(self):
        entry = resolve_catalog_entry(CATALOG, "gpt-5.5")
        assert entry is not None
        assert entry.model_id == "gpt-5.5"
        assert entry.input_per_mtok == 5.00
        assert entry.cached_input_per_mtok == 0.50
        assert entry.output_per_mtok == 30.00
        assert entry.max_input_tokens == 1_050_000

    def test_provider_prefixed_key_resolves(self):
        # LiteLLM lists some Anthropic models only under a provider prefix.
        entry = resolve_catalog_entry(CATALOG, "claude-sonnet-4.6")
        assert entry is not None
        assert entry.input_per_mtok == 3.00
        assert entry.max_input_tokens == 200_000

    def test_dated_suffix_prefers_longest_base(self):
        # Regression: must resolve to the mini entry, not the 3x pricier base.
        entry = resolve_catalog_entry(CATALOG, "gpt-5.4-mini-2025-06-01")
        assert entry is not None
        assert entry.input_per_mtok == 0.75

    def test_unknown_model_returns_none(self):
        assert resolve_catalog_entry(CATALOG, "totally-fictional-model") is None

    def test_missing_cache_rate_bills_at_full_input_rate(self):
        # No invented discount: an unpublished cache rate means cache hits cost
        # the same as fresh input.
        entry = resolve_catalog_entry(CATALOG, "budget-model")
        assert entry is not None
        assert entry.cached_input_per_mtok == entry.input_per_mtok
        assert entry.supports_caching is False

    def test_malformed_entry_returns_none(self):
        assert resolve_catalog_entry(CATALOG, "broken-model") is None

    def test_non_positive_window_returns_none(self):
        assert resolve_catalog_entry(CATALOG, "zero-window") is None


class TestResolveOrRaise:
    @pytest.mark.asyncio
    async def test_installs_entry_on_success(self):
        with respx.mock:
            _mock_catalog()
            entry = await resolve_or_raise("gpt-5.5")
        assert entry.model_id == "gpt-5.5"
        assert active_entry() == entry

    @pytest.mark.asyncio
    async def test_raises_when_fetch_fails(self):
        # No local fallback — an unreachable catalog must abort startup.
        with respx.mock:
            _mock_catalog(status=500)
            with pytest.raises(ModelCatalogError, match="could not fetch"):
                await resolve_or_raise("gpt-5.5")
        assert active_entry() is None

    @pytest.mark.asyncio
    async def test_raises_when_model_absent(self):
        with respx.mock:
            _mock_catalog()
            with pytest.raises(ModelCatalogError, match="not in the LiteLLM catalog"):
                await resolve_or_raise("ai-gateway-gpt-5.5")
        assert active_entry() is None

    @pytest.mark.asyncio
    async def test_error_names_the_base_model_knob(self):
        with respx.mock:
            _mock_catalog()
            with pytest.raises(ModelCatalogError, match="OPENAI_BASE_MODEL"):
                await resolve_or_raise("ai-gateway-gpt-5.5")


class TestRefreshActiveEntry:
    @pytest.mark.asyncio
    async def test_swaps_in_new_prices(self):
        _swap_active_entry(_entry(input_per_mtok=1.00))
        with respx.mock:
            _mock_catalog()
            assert await refresh_active_entry("gpt-5.5") is True
        current = active_entry()
        assert current is not None
        assert current.input_per_mtok == 5.00

    @pytest.mark.asyncio
    async def test_failed_fetch_keeps_existing_entry(self):
        # The refresh is best-effort: a LiteLLM outage must not degrade or kill
        # a running instance, unlike the startup resolve.
        installed = _entry(input_per_mtok=1.23)
        _swap_active_entry(installed)
        with respx.mock:
            _mock_catalog(status=503)
            assert await refresh_active_entry("gpt-5.5") is False
        assert active_entry() == installed

    @pytest.mark.asyncio
    async def test_model_vanishing_from_catalog_keeps_existing_entry(self):
        installed = _entry()
        _swap_active_entry(installed)
        with respx.mock:
            _mock_catalog(payload={"some-other-model": CATALOG["gpt-5.4"]})
            assert await refresh_active_entry("gpt-5.5") is False
        assert active_entry() == installed


class TestFetchModelCatalog:
    @pytest.mark.asyncio
    async def test_returns_parsed_json(self):
        with respx.mock:
            _mock_catalog()
            data = await fetch_model_catalog()
        assert data is not None
        assert "gpt-5.5" in data

    @pytest.mark.asyncio
    async def test_returns_none_on_http_error(self):
        with respx.mock:
            _mock_catalog(status=404)
            assert await fetch_model_catalog() is None

    @pytest.mark.asyncio
    async def test_returns_none_when_payload_is_not_an_object(self):
        with respx.mock:
            respx.get(LITELLM_PRICING_URL).mock(
                return_value=httpx.Response(200, text="[]")
            )
            assert await fetch_model_catalog() is None


class TestEstimateCostUsd:
    def test_prices_uncached_and_cached_prompt_separately(self):
        entry = _entry()
        usage = TokenUsage(prompt=100_000, cached=80_000, completion=5_000)
        # 20k uncached @ $5 + 80k cached @ $0.50 + 5k out @ $30 per Mtok
        expected = (20_000 * 5.00 + 80_000 * 0.50 + 5_000 * 30.00) / 1_000_000
        assert estimate_cost_usd(usage, entry) == pytest.approx(expected)

    def test_no_cached_tokens_matches_full_input_rate(self):
        # An endpoint that doesn't report cached_tokens degrades to the old
        # upper bound rather than under-billing.
        entry = _entry()
        usage = TokenUsage(prompt=100_000, cached=0, completion=5_000)
        expected = (100_000 * 5.00 + 5_000 * 30.00) / 1_000_000
        assert estimate_cost_usd(usage, entry) == pytest.approx(expected)

    def test_caching_is_cheaper_than_not(self):
        entry = _entry()
        cached = estimate_cost_usd(TokenUsage(prompt=100_000, cached=90_000), entry)
        uncached = estimate_cost_usd(TokenUsage(prompt=100_000), entry)
        assert cached is not None and uncached is not None
        assert cached < uncached

    def test_uses_installed_entry_when_none_passed(self):
        _swap_active_entry(_entry())
        cost = estimate_cost_usd(TokenUsage(prompt=1_000_000))
        assert cost == pytest.approx(5.00)

    def test_returns_none_without_an_entry(self):
        # Fail-open: an unpriced run is recorded without a cost, never blocked.
        assert estimate_cost_usd(TokenUsage(prompt=1000, completion=1000)) is None

    def test_zero_tokens_yields_zero(self):
        assert estimate_cost_usd(TokenUsage(), _entry()) == 0.0


class TestTokenUsage:
    def test_uncached_prompt_is_the_billable_remainder(self):
        assert TokenUsage(prompt=100, cached=30).uncached_prompt == 70

    def test_cached_exceeding_prompt_clamps_to_zero(self):
        # Defensive: a proxy reporting nonsense must not yield a negative bill.
        assert TokenUsage(prompt=10, cached=99).uncached_prompt == 0

    def test_total_excludes_cached_double_count(self):
        # cached is a subset of prompt, so it must not be added again.
        assert TokenUsage(prompt=100, cached=40, completion=10).total == 110

    def test_addition_sums_each_component(self):
        combined = TokenUsage(prompt=10, cached=4, completion=2) + TokenUsage(
            prompt=5, cached=1, completion=3
        )
        assert (combined.prompt, combined.cached, combined.completion) == (15, 5, 5)


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


class TestCatalogModel:
    """`base_model` maps a gateway alias onto a catalog id."""

    def _config(self, **overrides) -> LLMConfig:
        return LLMConfig(
            api_key="k", api_url="https://gw.example.com/v1", **overrides,
        )

    def test_unset_base_model_falls_back_to_model(self):
        cfg = self._config(model="gpt-5.4")
        assert cfg.base_model == ""
        assert cfg.catalog_model == "gpt-5.4"

    def test_base_model_overrides_lookup_key(self):
        cfg = self._config(model="ai-gateway-gpt-5.5", base_model="gpt-5.5")
        assert cfg.catalog_model == "gpt-5.5"
        # The alias itself resolves to nothing — that's the whole problem.
        assert resolve_catalog_entry(CATALOG, cfg.model) is None

    @pytest.mark.asyncio
    async def test_gateway_alias_resolves_via_base_model(self):
        cfg = self._config(model="ai-gateway-gpt-5.5", base_model="gpt-5.5")
        with respx.mock:
            _mock_catalog()
            entry = await resolve_or_raise(cfg.catalog_model)
        # The payoff: prices and a window that clears the 1M floor, with no
        # OPENAI_CONTEXT_WINDOW set by hand.
        assert cfg.context_window == 0
        assert entry.max_input_tokens == 1_050_000
        assert entry.input_per_mtok == 5.00
