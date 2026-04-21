import asyncio
import logging
import time

logger = logging.getLogger(__name__)


class AsyncTokenBucket:
    """Async token bucket rate limiter.

    `rate_per_minute <= 0` disables limiting — `acquire()` returns immediately.
    Otherwise each acquire consumes one token; callers block until the bucket
    has one. Tokens refill continuously at `rate_per_minute / 60` per second,
    capped at `burst`.
    """

    def __init__(self, name: str, rate_per_minute: float, burst: int):
        self.name = name
        self.rate_per_second = max(0.0, rate_per_minute) / 60.0
        self.capacity = max(1, burst)
        self._tokens: float = float(self.capacity)
        self._last_refill: float = time.monotonic()
        self._lock = asyncio.Lock()

    @property
    def enabled(self) -> bool:
        return self.rate_per_second > 0

    def _refill(self) -> None:
        now = time.monotonic()
        elapsed = now - self._last_refill
        if elapsed > 0:
            self._tokens = min(self.capacity, self._tokens + elapsed * self.rate_per_second)
            self._last_refill = now

    async def acquire(self) -> None:
        if not self.enabled:
            return
        while True:
            async with self._lock:
                self._refill()
                if self._tokens >= 1.0:
                    self._tokens -= 1.0
                    return
                needed = 1.0 - self._tokens
                wait_seconds = needed / self.rate_per_second
            log = logger.info if wait_seconds >= 1.0 else logger.debug
            log(
                "rate-limit[%s]: throttling %.2fs (tokens=%.2f cap=%d rate=%.2f/s)",
                self.name, wait_seconds, self._tokens, self.capacity, self.rate_per_second,
            )
            await asyncio.sleep(wait_seconds)
