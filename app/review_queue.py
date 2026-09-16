import asyncio
import logging
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from functools import partial
from typing import final

import structlog

from app.models import WebhookPayload

logger = logging.getLogger(__name__)

PRKey = tuple[str, str, int]
# Called with the slug of the team the PR belongs to; the team was
# authenticated by the webhook route, so the worker never has to look it up.
ReviewFn = Callable[[str, WebhookPayload], Awaitable[None]]

# A queued unit of work that is not a PR review: mention Q&A, merge/decline
# rollup. Runs on the same single worker so only one diff/file/prompt set is
# ever resident at a time — the pod's memory limit is sized for one.
JobFn = Callable[[], Awaitable[None]]

BACKLOG_WARN_THRESHOLD = 10


@dataclass
class _Entry:
    team: str
    payload: WebhookPayload
    enqueued_at: float


@dataclass
class _Job:
    tag: str
    team: str
    run: JobFn
    enqueued_at: float


@final
class ReviewQueue:
    """Single-worker PR review queue.

    All reviews run one at a time on a single background worker. Webhooks
    may arrive at any rate; the queue dedupes per-PR: if a PR already has a
    pending entry, the stored payload is replaced with the latest and no
    second slot is enqueued. This collapses rapid bursts (e.g. 50 commits
    pushed in sequence) into at most two reviews (the first one in-flight
    plus the deduped latest state).

    `submit_job` puts other heavy work (mention answers, rollups) on the same
    worker, in arrival order, without dedupe.
    """

    def __init__(self, review_fn: ReviewFn):
        self._review_fn = review_fn
        self._queue: asyncio.Queue[PRKey | _Job] = asyncio.Queue()
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

    def submit(self, key: PRKey, payload: WebhookPayload, team: str) -> str:
        """Enqueue a review. Returns "queued" for a fresh entry or
        "superseded" when the key was already pending (payload replaced).
        """
        tag = self._tag(key)
        if key in self._pending:
            entry = self._pending[key]
            entry.team = team
            entry.payload = payload
            entry.enqueued_at = time.monotonic()
            logger.info(
                "queue[%s]: superseded pending payload (depth=%d)",
                tag, self._queue.qsize(),
            )
            return "superseded"
        self._pending[key] = _Entry(team=team, payload=payload, enqueued_at=time.monotonic())
        self._put(key, tag)
        return "queued"

    def submit_job(self, tag: str, team: str, fn: JobFn) -> str:
        """Enqueue a non-review job. Never deduped; returns "queued"."""
        self._put(_Job(tag=tag, team=team, run=fn, enqueued_at=time.monotonic()), tag)
        return "queued"

    def _put(self, item: PRKey | _Job, tag: str) -> None:
        self._queue.put_nowait(item)
        depth = self._queue.qsize()
        logger.info("queue[%s]: enqueued (depth=%d)", tag, depth)
        if depth >= BACKLOG_WARN_THRESHOLD:
            logger.warning("ReviewQueue backlog: %d entries pending", depth)

    async def _run(self) -> None:
        while True:
            item = await self._queue.get()
            if isinstance(item, _Job):
                tag, team, enqueued_at = item.tag, item.team, item.enqueued_at
                run: JobFn = item.run
            else:
                tag = self._tag(item)
                entry = self._pending.pop(item, None)
                if entry is None:
                    logger.warning("queue[%s]: dequeued with no payload — skipping", tag)
                    continue
                team, enqueued_at = entry.team, entry.enqueued_at
                run = partial(self._review_fn, team, entry.payload)
            wait = time.monotonic() - enqueued_at
            logger.info(
                "queue[%s]: starting %s (waited %.1fs, depth=%d)",
                tag, "job" if isinstance(item, _Job) else "review",
                wait, self._queue.qsize(),
            )
            started = time.monotonic()
            # The worker is one long-lived task, so the team binding must be
            # explicit per job — nothing else would ever clear it.
            structlog.contextvars.bind_contextvars(team=team)
            try:
                await run()
            except Exception:
                logger.exception("queue[%s]: %s failed", tag, "job" if isinstance(item, _Job) else "review")
            logger.info(
                "queue[%s]: completed in %.1fs",
                tag, time.monotonic() - started,
            )
            structlog.contextvars.unbind_contextvars("team")
