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
    fetch_model_catalog,
    refresh_active_entry,
    resolve_catalog_entry,
    resolve_cost_usd,
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
    "broken-model": {"input_cost_per_token": 1e-06},  # no window at all
    # Decoy: the bare key exists with a null window while the provider-prefixed
    # key below carries the real one.
    "claude-decoy": {"input_cost_per_token": 3e-06, "max_input_tokens": None},
    "openrouter/anthropic/claude-decoy": {
        "input_cost_per_token": 3e-06,
        "max_input_tokens": 200000,
    },
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
        matched_key="gpt-5.5",
        max_input_tokens=1_050_000,
    )
    base.update(overrides)
    return ModelCatalogEntry(**base)  # pyright: ignore[reportArgumentType]


class TestResolveCatalogEntry:
    def test_exact_match_yields_the_context_window(self):
        entry = resolve_catalog_entry(CATALOG, "gpt-5.5")
        assert entry is not None
        assert entry.model_id == "gpt-5.5"
        assert entry.max_input_tokens == 1_050_000

    def test_provider_prefixed_key_resolves(self):
        # LiteLLM lists some Anthropic models only under a provider prefix.
        entry = resolve_catalog_entry(CATALOG, "claude-sonnet-4.6")
        assert entry is not None
        assert entry.max_input_tokens == 200_000
        # The key that actually matched is recorded, not just what we asked for.
        assert entry.model_id == "claude-sonnet-4.6"
        assert entry.matched_key == "openrouter/anthropic/claude-sonnet-4.6"

    def test_dated_suffix_prefers_longest_base(self):
        # Regression: must resolve to the mini entry, not the 3x pricier base.
        entry = resolve_catalog_entry(CATALOG, "gpt-5.4-mini-2025-06-01")
        assert entry is not None
        assert entry.max_input_tokens == 400_000
        # A fuzzy match must be traceable — it prices and boots cleanly, so the
        # key it landed on is the only evidence of which rates were applied.
        assert entry.matched_key == "gpt-5.4-mini"
        assert entry.model_id == "gpt-5.4-mini-2025-06-01"

    def test_exact_match_records_itself_as_the_matched_key(self):
        entry = resolve_catalog_entry(CATALOG, "gpt-5.5")
        assert entry is not None
        assert entry.matched_key == entry.model_id == "gpt-5.5"

    def test_unknown_model_returns_none(self):
        assert resolve_catalog_entry(CATALOG, "totally-fictional-model") is None

    def test_entry_without_a_window_returns_none(self):
        assert resolve_catalog_entry(CATALOG, "broken-model") is None

    def test_non_positive_window_returns_none(self):
        assert resolve_catalog_entry(CATALOG, "zero-window") is None

    def test_dated_suffix_resolves_under_a_provider_prefix(self):
        # Regression: the fallback used to compare the requested id against the
        # full catalog key, so `openrouter/anthropic/...` families could never
        # match a dated id — startup aborted on a model that does exist.
        entry = resolve_catalog_entry(CATALOG, "claude-sonnet-4.6-20260101")
        assert entry is not None
        assert entry.matched_key == "openrouter/anthropic/claude-sonnet-4.6"
        assert entry.max_input_tokens == 200_000

    def test_unparseable_first_prefix_does_not_mask_a_later_one(self):
        # Regression: a bare key with a null window used to abort the whole
        # search, hiding a valid provider-prefixed entry.
        entry = resolve_catalog_entry(CATALOG, "claude-decoy")
        assert entry is not None
        assert entry.matched_key == "openrouter/anthropic/claude-decoy"
        assert entry.max_input_tokens == 200_000


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
    async def test_swaps_in_the_new_window(self):
        _swap_active_entry(_entry(max_input_tokens=1))
        with respx.mock:
            _mock_catalog()
            assert await refresh_active_entry("gpt-5.5") is True
        current = active_entry()
        assert current is not None
        assert current.max_input_tokens == 1_050_000

    @pytest.mark.asyncio
    async def test_failed_fetch_keeps_existing_entry(self):
        # The refresh is best-effort: a LiteLLM outage must not degrade or kill
        # a running instance, unlike the startup resolve.
        installed = _entry(max_input_tokens=123_456)
        _swap_active_entry(installed)
        with respx.mock:
            _mock_catalog(status=503)
            assert await refresh_active_entry("gpt-5.5") is False
        assert active_entry() == installed

    @pytest.mark.asyncio
    async def test_shrunken_window_below_floor_is_rejected(self):
        # The catalog is upstream data and can be corrected downward. A refresh
        # must never push a running instance under the window floor that
        # startup enforces — keep the older, valid entry instead.
        installed = _entry(max_input_tokens=1_050_000)
        _swap_active_entry(installed)
        with respx.mock:
            _mock_catalog(payload={"gpt-5.5": {"max_input_tokens": 272_000}})
            assert await refresh_active_entry("gpt-5.5", min_window=1_000_000) is False
        assert active_entry() == installed

    @pytest.mark.asyncio
    async def test_shrunken_window_is_accepted_without_a_floor(self):
        _swap_active_entry(_entry(max_input_tokens=1_050_000))
        with respx.mock:
            _mock_catalog(payload={"gpt-5.5": {"max_input_tokens": 272_000}})
            assert await refresh_active_entry("gpt-5.5") is True
        current = active_entry()
        assert current is not None and current.max_input_tokens == 272_000

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


class TestResolveCostUsd:
    """Endpoint figure first, catalog rates as the fallback."""

    def _priced(self) -> ModelCatalogEntry:
        return ModelCatalogEntry(
            model_id="gpt-5.5", matched_key="gpt-5.5", max_input_tokens=1_050_000,
            input_per_mtok=5.00, cached_input_per_mtok=0.50, output_per_mtok=30.00,
        )

    def test_reported_cost_wins(self):
        cost, reported = resolve_cost_usd(
            TokenUsage(prompt=1_000_000, cost_usd=0.01), self._priced(),
        )
        assert (cost, reported) == (0.01, True)

    def test_falls_back_to_catalog_when_endpoint_reports_nothing(self):
        # Regression: LiteLLM sends the literal "None" for a deployment it
        # can't price, which left the summary with no cost line at all and
        # silently disabled the per-PR cap.
        cost, reported = resolve_cost_usd(
            TokenUsage(prompt=100_000, cached=80_000, completion=5_000),
            self._priced(),
        )
        assert reported is False
        expected = (20_000 * 5.00 + 80_000 * 0.50 + 5_000 * 30.00) / 1_000_000
        assert cost == pytest.approx(expected)

    def test_reported_zero_on_a_real_call_falls_back_to_the_catalog(self):
        # A call that consumed tokens cannot have cost nothing on a priced
        # model — it means the endpoint is misconfigured (e.g. a LiteLLM
        # deployment with input/output cost stored as 0). Trusting it would
        # show $0.00 forever and silently disable the per-PR cap.
        cost, reported = resolve_cost_usd(
            TokenUsage(prompt=1_000_000, cost_usd=0.0), self._priced(),
        )
        assert reported is False
        assert cost == pytest.approx(5.00)

    def test_reported_zero_with_no_tokens_is_kept(self):
        # Nothing was consumed, so zero is the truthful answer.
        cost, reported = resolve_cost_usd(TokenUsage(cost_usd=0.0), self._priced())
        assert (cost, reported) == (0.0, True)

    def test_free_model_still_prices_at_zero(self):
        # The fallback can't invent a cost: a genuinely free model has zero
        # rates in the catalog too.
        free = ModelCatalogEntry(
            model_id="free", matched_key="free", max_input_tokens=1_050_000,
            input_per_mtok=0.0, cached_input_per_mtok=0.0, output_per_mtok=0.0,
        )
        cost, _reported = resolve_cost_usd(
            TokenUsage(prompt=1_000_000, cost_usd=0.0), free,
        )
        assert cost == 0.0

    def test_no_rates_and_no_report_yields_no_cost(self):
        cost, reported = resolve_cost_usd(TokenUsage(prompt=1000), _entry())
        assert (cost, reported) == (None, False)


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
        # The payoff: a window that clears the 1M floor, with no
        # OPENAI_CONTEXT_WINDOW set by hand.
        assert cfg.context_window == 0
        assert entry.max_input_tokens == 1_050_000
