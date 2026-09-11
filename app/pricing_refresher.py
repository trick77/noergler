"""Background task that keeps the installed model-catalog entries in sync with
the catalog at `MODEL_CATALOG_URL`. Refreshes every 24h.

Nothing is persisted. The catalog is fetched fresh at startup (per team, see
`LLMClient.check_connectivity`) and re-fetched here on a timer, once per
distinct model across all enabled teams; a failed refresh leaves the entry
loaded at startup in place rather than degrading or exiting.
"""
import asyncio
import logging
from typing import final

from app.config import active_entry, refresh_active_entry
from app.llm_client import _MIN_CONTEXT_WINDOW

logger = logging.getLogger(__name__)

REFRESH_INTERVAL_SECONDS = 24 * 60 * 60


async def refresh_once(model_id: str, catalog_url: str) -> bool:
    """One refresh cycle for one model: re-fetch the catalog and swap the entry in.

    Returns True on success. False means the fetch failed or the model vanished
    from the catalog — in both cases the previously installed entry stays live,
    so a LiteLLM outage or a renamed catalog key can never take a running
    instance down. Only the startup resolve is fatal (for that team).
    """
    before = active_entry(model_id)
    ok = await refresh_active_entry(
        model_id, catalog_url, min_window=_MIN_CONTEXT_WINDOW
    )
    if not ok:
        logger.warning(
            "model-catalog refresh failed for %s — continuing with the entry loaded at "
            "startup (%s)", model_id, before.model_id if before else "none",
        )
        return False
    after = active_entry(model_id)
    if after is not None and before is not None and after != before:
        logger.info(
            "model-catalog: %s updated — context window %d→%d",
            after.model_id, before.max_input_tokens, after.max_input_tokens,
        )
    else:
        logger.info("model-catalog: %s refreshed, no change", model_id)
    return True


@final
class PricingRefresher:
    """Background asyncio task that calls `refresh_once` for every model every 24h."""

    def __init__(self, model_ids: list[str], catalog_url: str) -> None:
        # Distinct, order-preserving: several teams may share a model.
        self._model_ids = list(dict.fromkeys(model_ids))
        self._catalog_url = catalog_url
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
            for model_id in self._model_ids:
                await refresh_once(model_id, self._catalog_url)
