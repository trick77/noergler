import asyncio
import logging
from typing import Callable, Coroutine

logger = logging.getLogger(__name__)

PRKey = tuple[str, str, int]
CoroFactory = Callable[[], Coroutine[None, None, None]]

_RUNNING_ATTR = "_debounce_running"


class PRDebouncer:
    """Coalesce rapid events for the same PR into a single delayed execution.

    On each `schedule(key, ...)` call any pending task still in its debounce
    window is cancelled and replaced. If the previous task is already past
    the debounce window (executing coro_factory), it is left alone — the new
    task will queue behind it on a per-PR lock so the two reviews never run
    concurrently. `asyncio.Task.cancel()` only takes effect at await points
    and cannot cleanly interrupt a mid-flight review, so serialization is
    safer than racing.
    """

    def __init__(self, delay_seconds: float):
        self.delay_seconds = delay_seconds
        self._pending: dict[PRKey, asyncio.Task] = {}
        self._locks: dict[PRKey, asyncio.Lock] = {}

    @property
    def enabled(self) -> bool:
        return self.delay_seconds > 0

    def schedule(self, key: PRKey, coro_factory: CoroFactory) -> None:
        if not self.enabled:
            asyncio.create_task(coro_factory())
            return

        existing = self._pending.get(key)
        if existing and not existing.done():
            if getattr(existing, _RUNNING_ATTR, False):
                logger.info("debounce[%s]: queued behind in-flight review", key)
            else:
                existing.cancel()
                logger.info("debounce[%s]: superseded pending review", key)

        task = asyncio.create_task(self._run(key, coro_factory))
        self._pending[key] = task

    async def _run(self, key: PRKey, coro_factory: CoroFactory) -> None:
        try:
            await asyncio.sleep(self.delay_seconds)
        except asyncio.CancelledError:
            return

        current = asyncio.current_task()
        assert current is not None
        setattr(current, _RUNNING_ATTR, True)

        lock = self._locks.setdefault(key, asyncio.Lock())
        try:
            async with lock:
                await coro_factory()
        except Exception:
            logger.exception("debounce[%s]: task failed", key)
        finally:
            if self._pending.get(key) is current:
                self._pending.pop(key, None)
                self._locks.pop(key, None)

    async def shutdown(self) -> None:
        tasks = list(self._pending.values())
        self._pending.clear()
        for t in tasks:
            t.cancel()
        for t in tasks:
            try:
                await t
            except (asyncio.CancelledError, Exception):
                pass
