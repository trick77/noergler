"""Background task that keeps `app.config._MODEL_PRICING` in sync with
the LiteLLM public pricing JSON. Refreshes every 24h, persisting the
result to the `model_pricing` table so a LiteLLM outage at startup falls
back to the last-known-good prices instead of the baked-in static table.
"""
import asyncio
import logging
from typing import final

import asyncpg

from app.config import (
    _MODEL_PRICING,
    _STATIC_MODEL_CONTEXT_WINDOW,
    _STATIC_MODEL_PRICING,
    _swap_context_windows,
    _swap_pricing,
    apply_pricing_overlay,
    fetch_litellm_model_meta,
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

    A single fetch updates both pricing and context-window tables. Returns
    True on success, False if the fetch failed (live tables left untouched).
    """
    meta = await fetch_litellm_model_meta()
    if meta is None:
        return False
    table, ctx_table = meta
    _swap_pricing(table)
    _swap_context_windows(ctx_table)
    refreshed = sum(
        1 for k in _STATIC_MODEL_PRICING
        if table.get(k) != _STATIC_MODEL_PRICING[k]
    )
    ctx_refreshed = sum(
        1 for k in _STATIC_MODEL_CONTEXT_WINDOW
        if ctx_table.get(k) != _STATIC_MODEL_CONTEXT_WINDOW[k]
    )
    logger.info(
        "model-meta: refreshed %d/%d prices, %d/%d context windows from LiteLLM",
        refreshed, len(_STATIC_MODEL_PRICING),
        ctx_refreshed, len(_STATIC_MODEL_CONTEXT_WINDOW),
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
    # NOTE: context windows are deliberately NOT persisted/hydrated like pricing.
    # A stale window is harmless (the 413 handler shrinks the chunk), and the
    # corrected static defaults are already accurate, so a DB cache would be
    # nearly inert. Pricing is persisted because cost accuracy needs the
    # last-known actual prices when LiteLLM is down at cold start.
    return True


@final
class PricingRefresher:
    """Background asyncio task that calls `refresh_once` every 24h."""

    def __init__(self, pool: asyncpg.Pool | None) -> None:
        self._pool = pool
        self._task: asyncio.Task[None] | None = None
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
