"""Background task that keeps `app.config._MODEL_PRICING` in sync with
the LiteLLM public pricing JSON. Refreshes every 24h, persisting the
result to the `model_pricing` table so a LiteLLM outage at startup falls
back to the last-known-good prices instead of the baked-in static table.
"""
import asyncio
import logging

import asyncpg

from app.config import (
    _MODEL_PRICING,
    _STATIC_MODEL_PRICING,
    _swap_pricing,
    apply_pricing_overlay,
    fetch_litellm_pricing,
)
from app.db.repository import load_model_pricing, upsert_model_pricing

logger = logging.getLogger(__name__)

REFRESH_INTERVAL_SECONDS = 24 * 60 * 60


async def hydrate_from_db(pool: asyncpg.Pool) -> int:
    """Load cached pricing from Postgres into the live table. Best-effort."""
    try:
        cached = await load_model_pricing(pool)
    except Exception as exc:
        logger.warning("model-pricing DB hydrate failed: %s", exc)
        return 0
    if not cached:
        return 0
    overlaid = apply_pricing_overlay(cached)
    logger.info(
        "model-pricing: hydrated %d/%d entries from DB",
        overlaid, len(_STATIC_MODEL_PRICING),
    )
    return overlaid


async def refresh_once(pool: asyncpg.Pool | None) -> bool:
    """One refresh cycle: fetch LiteLLM, swap in-memory, persist to DB.

    Returns True on success, False if the fetch failed (live table left
    untouched in that case).
    """
    table = await fetch_litellm_pricing()
    if table is None:
        return False
    _swap_pricing(table)
    refreshed = sum(
        1 for k in _STATIC_MODEL_PRICING
        if table.get(k) != _STATIC_MODEL_PRICING[k]
    )
    logger.info(
        "model-pricing: refreshed %d/%d entries from LiteLLM",
        refreshed, len(_STATIC_MODEL_PRICING),
    )
    if pool is not None:
        try:
            entries = {
                model_id: (
                    price.input_per_mtok,
                    price.cached_input_per_mtok,
                    price.output_per_mtok,
                )
                for model_id, price in _MODEL_PRICING.items()
            }
            await upsert_model_pricing(pool, entries)
        except Exception as exc:
            logger.warning("model-pricing DB persist failed: %s", exc)
    return True


class PricingRefresher:
    """Background asyncio task that calls `refresh_once` every 24h."""

    def __init__(self, pool: asyncpg.Pool | None) -> None:
        self._pool = pool
        self._task: asyncio.Task | None = None
        self._stop = asyncio.Event()

    def start(self) -> None:
        if self._task is None:
            self._task = asyncio.create_task(self._run(), name="pricing-refresher")

    async def stop(self) -> None:
        self._stop.set()
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except (asyncio.CancelledError, Exception):
                pass
            self._task = None

    async def _run(self) -> None:
        while not self._stop.is_set():
            try:
                await asyncio.wait_for(
                    self._stop.wait(), timeout=REFRESH_INTERVAL_SECONDS
                )
                return  # stop was set
            except asyncio.TimeoutError:
                pass
            await refresh_once(self._pool)
