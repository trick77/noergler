import asyncio
import logging
import time
from dataclasses import dataclass
from typing import Awaitable, Callable

from app.models import WebhookPayload

logger = logging.getLogger(__name__)

PRKey = tuple[str, str, int]
ReviewFn = Callable[[WebhookPayload], Awaitable[None]]

BACKLOG_WARN_THRESHOLD = 10


@dataclass
class _Entry:
    payload: WebhookPayload
    enqueued_at: float


class ReviewQueue:
    """Single-worker PR review queue.

    All reviews run one at a time on a single background worker. Webhooks
    may arrive at any rate; the queue dedupes per-PR: if a PR already has a
    pending entry, the stored payload is replaced with the latest and no
    second slot is enqueued. This collapses rapid bursts (e.g. 50 commits
    pushed in sequence) into at most two reviews (the first one in-flight
    plus the deduped latest state).
    """

    def __init__(self, review_fn: ReviewFn):
        self._review_fn = review_fn
        self._queue: asyncio.Queue[PRKey] = asyncio.Queue()
        self._pending: dict[PRKey, _Entry] = {}
        self._worker: asyncio.Task[None] | None = None

    def _tag(self, key: PRKey) -> str:
        return f"{key[0]}/{key[1]}#{key[2]}"

    def start(self) -> None:
        if self._worker is not None:
            return
        self._worker = asyncio.create_task(self._run(), name="review-queue-worker")
        logger.info("ReviewQueue worker started")

    async def stop(self) -> None:
        if self._worker is None:
            return
        self._worker.cancel()
        try:
            await self._worker
        except (asyncio.CancelledError, Exception):
            pass
        self._worker = None
        logger.info("ReviewQueue worker stopped")

    def submit(self, key: PRKey, payload: WebhookPayload) -> str:
        """Enqueue a review. Returns "queued" for a fresh entry or
        "superseded" when the key was already pending (payload replaced).
        """
        tag = self._tag(key)
        if key in self._pending:
            entry = self._pending[key]
            entry.payload = payload
            entry.enqueued_at = time.monotonic()
            logger.info(
                "queue[%s]: superseded pending payload (depth=%d)",
                tag, self._queue.qsize(),
            )
            return "superseded"
        self._pending[key] = _Entry(payload=payload, enqueued_at=time.monotonic())
        self._queue.put_nowait(key)
        depth = self._queue.qsize()
        logger.info("queue[%s]: enqueued (depth=%d)", tag, depth)
        if depth >= BACKLOG_WARN_THRESHOLD:
            logger.warning("ReviewQueue backlog: %d entries pending", depth)
        return "queued"

    async def _run(self) -> None:
        while True:
            key = await self._queue.get()
            tag = self._tag(key)
            entry = self._pending.pop(key, None)
            if entry is None:
                logger.warning("queue[%s]: dequeued with no payload — skipping", tag)
                continue
            wait = time.monotonic() - entry.enqueued_at
            logger.info(
                "queue[%s]: starting review (waited %.1fs, depth=%d)",
                tag, wait, self._queue.qsize(),
            )
            started = time.monotonic()
            try:
                await self._review_fn(entry.payload)
            except Exception:
                logger.exception("queue[%s]: review failed", tag)
            logger.info(
                "queue[%s]: completed in %.1fs",
                tag, time.monotonic() - started,
            )
