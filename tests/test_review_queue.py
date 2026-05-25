import asyncio
from typing import cast

import pytest

from app.models import WebhookPayload
from app.review_queue import ReviewQueue


def _fake_payload(pr_id: int, label: str = "p") -> WebhookPayload:
    class P:
        def __init__(self, i, l):
            self.id = i
            self.label = l
    return cast(WebhookPayload, cast(object, P(pr_id, label)))


@pytest.mark.asyncio
async def test_single_review_runs_and_completes():
    seen: list[int] = []

    async def review(payload):
        seen.append(payload.id)

    queue = ReviewQueue(review)
    queue.start()
    try:
        queue.submit(("P", "r", 1), _fake_payload(1))
        await asyncio.sleep(0.05)
        assert seen == [1]
    finally:
        await queue.stop()


@pytest.mark.asyncio
async def test_dedup_collapses_pending_submits():
    order: list[str] = []

    async def review(payload):
        order.append(f"review:{payload.id}:{payload.label}")
        await asyncio.sleep(0.05)

    queue = ReviewQueue(review)
    queue.start()
    try:
        # submit #1 immediately starts; submits #2 and #3 for the same PR
        # arrive while it runs and must collapse into one queued slot carrying
        # the latest payload.
        queue.submit(("P", "r", 1), _fake_payload(1, "first"))
        await asyncio.sleep(0.01)
        queue.submit(("P", "r", 1), _fake_payload(1, "second"))
        queue.submit(("P", "r", 1), _fake_payload(1, "third"))
        await asyncio.sleep(0.2)
        assert order == ["review:1:first", "review:1:third"]
    finally:
        await queue.stop()


@pytest.mark.asyncio
async def test_reviews_are_serialized_across_prs():
    # Use an event so we can observe whether two reviews run concurrently.
    running = 0
    max_concurrent = 0

    async def review(payload):
        nonlocal running, max_concurrent
        running += 1
        max_concurrent = max(max_concurrent, running)
        await asyncio.sleep(0.05)
        running -= 1

    queue = ReviewQueue(review)
    queue.start()
    try:
        for i in range(5):
            queue.submit(("P", "r", i), _fake_payload(i))
        await asyncio.sleep(0.5)
        assert max_concurrent == 1
    finally:
        await queue.stop()


@pytest.mark.asyncio
async def test_review_exception_does_not_kill_worker():
    calls: list[int] = []

    async def review(payload):
        calls.append(payload.id)
        if payload.id == 1:
            raise RuntimeError("boom")

    queue = ReviewQueue(review)
    queue.start()
    try:
        queue.submit(("P", "r", 1), _fake_payload(1))
        await asyncio.sleep(0.05)
        queue.submit(("P", "r", 2), _fake_payload(2))
        await asyncio.sleep(0.05)
        assert calls == [1, 2]
    finally:
        await queue.stop()


@pytest.mark.asyncio
async def test_submit_returns_outcome():
    async def review(payload):
        await asyncio.sleep(0.1)

    queue = ReviewQueue(review)
    queue.start()
    try:
        # Back-to-back submits with no awaits in between — the worker has
        # no chance to dequeue, so the second submit sees the pending entry
        # and returns "superseded".
        assert queue.submit(("P", "r", 1), _fake_payload(1)) == "queued"
        assert queue.submit(("P", "r", 1), _fake_payload(1)) == "superseded"
        await asyncio.sleep(0.2)
    finally:
        await queue.stop()


@pytest.mark.asyncio
async def test_stop_is_idempotent():
    async def review(payload):
        pass

    queue = ReviewQueue(review)
    queue.start()
    await queue.stop()
    await queue.stop()
