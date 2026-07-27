"""Background task that keeps the installed model-catalog entry in sync with
the LiteLLM public catalog. Refreshes every 24h.

Nothing is persisted. The catalog is fetched fresh at startup (fatally, see
`app.config.resolve_or_raise`) and re-fetched here on a timer; a failed refresh
leaves the entry loaded at startup in place rather than degrading or exiting.
"""
import asyncio
import logging
from typing import final

from app.config import active_entry, refresh_active_entry

logger = logging.getLogger(__name__)

REFRESH_INTERVAL_SECONDS = 24 * 60 * 60


async def refresh_once(model_id: str) -> bool:
    """One refresh cycle: re-fetch the catalog and swap the entry in.

    Returns True on success. False means the fetch failed or the model vanished
    from the catalog — in both cases the previously installed entry stays live,
    so a LiteLLM outage or a renamed catalog key can never take a running
    instance down. Only the startup resolve is fatal.
    """
    before = active_entry()
    ok = await refresh_active_entry(model_id)
    if not ok:
        logger.warning(
            "model-catalog refresh failed — continuing with the entry loaded at "
            "startup (%s)", before.model_id if before else "none",
        )
        return False
    after = active_entry()
    if after is not None and before is not None and after != before:
        logger.info(
            "model-catalog: %s updated — context window %d→%d",
            after.model_id, before.max_input_tokens, after.max_input_tokens,
        )
    else:
        logger.info("model-catalog: refreshed, no change")
    return True


@final
class PricingRefresher:
    """Background asyncio task that calls `refresh_once` every 24h."""

    def __init__(self, model_id: str) -> None:
        self._model_id = model_id
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
            await refresh_once(self._model_id)
