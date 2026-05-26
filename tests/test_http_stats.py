import asyncio
from collections import Counter

import httpx
import pytest

from app.http_stats import (
    enter_http_scope,
    exit_http_scope,
    make_event_hook,
    record_request,
    summarize,
    track_http_requests,
)


def test_record_request_outside_scope_is_noop() -> None:
    record_request("bitbucket", "GET")  # must not raise, must not crash


def test_track_http_requests_counts_within_block() -> None:
    with track_http_requests() as counter:
        record_request("bitbucket", "GET")
        record_request("bitbucket", "POST")
        record_request("jira", "GET")
    assert counter == Counter({"bitbucket:GET": 1, "bitbucket:POST": 1, "jira:GET": 1})


def test_record_request_after_exit_is_noop() -> None:
    with track_http_requests():
        pass
    record_request("bitbucket", "GET")  # no scope active


def test_enter_exit_scope_is_independent_of_with_block() -> None:
    scope = enter_http_scope()
    try:
        record_request("bitbucket", "GET")
    finally:
        exit_http_scope(scope)
    assert scope.counter == Counter({"bitbucket:GET": 1})


def test_summarize_aggregates_per_client() -> None:
    counter = Counter({"bitbucket:GET": 3, "bitbucket:POST": 1, "jira:GET": 2})
    assert summarize(counter) == {"bitbucket": 4, "jira": 2, "total": 6}


def test_summarize_empty() -> None:
    assert summarize(Counter()) == {"bitbucket": 0, "jira": 0, "total": 0}


@pytest.mark.asyncio
async def test_parallel_tasks_have_isolated_counters() -> None:
    async def task(label: str) -> Counter[str]:
        with track_http_requests() as c:
            await asyncio.sleep(0)  # yield so the other task can interleave
            record_request(label, "GET")
            await asyncio.sleep(0)
            record_request(label, "GET")
        return c

    bb, jira = await asyncio.gather(task("bitbucket"), task("jira"))
    assert bb == Counter({"bitbucket:GET": 2})
    assert jira == Counter({"jira:GET": 2})


@pytest.mark.asyncio
async def test_event_hook_increments_counter_on_real_request() -> None:
    hook = make_event_hook("bitbucket")

    async def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"ok": True})

    transport = httpx.MockTransport(handler)
    client = httpx.AsyncClient(
        transport=transport,
        base_url="https://bitbucket.test",
        event_hooks={"request": [hook]},
    )
    try:
        with track_http_requests() as counter:
            await client.get("/rest/api/1.0/application-properties")
            await client.post("/rest/api/1.0/foo", json={})
    finally:
        await client.aclose()
    assert counter == Counter({"bitbucket:GET": 1, "bitbucket:POST": 1})
