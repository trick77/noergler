"""Per-task HTTP request counters.

Used to surface how many round-trips a single review run makes against Bitbucket
and Jira. The counter lives in a `ContextVar` so it propagates automatically to
all coroutines spawned inside `track_http_requests()` — including
`asyncio.gather` fan-outs in the reviewer — without threading state through
every method signature.
"""

from collections import Counter
from collections.abc import Awaitable, Callable, Generator
from contextlib import contextmanager
from contextvars import ContextVar, Token

import httpx

_request_counter: ContextVar[Counter[str] | None] = ContextVar(
    "noergler_http_request_counter", default=None
)


def record_request(client_label: str, method: str) -> None:
    counter = _request_counter.get()
    if counter is None:
        return
    counter[f"{client_label}:{method.upper()}"] += 1


@contextmanager
def track_http_requests() -> Generator[Counter[str], None, None]:
    """Activate a fresh request counter for the duration of the block.

    Nested blocks get their own counter — outer scopes do not see inner counts.
    """
    scope = enter_http_scope()
    try:
        yield scope.counter
    finally:
        exit_http_scope(scope)


class HttpScope:
    """Active request-counter scope (counter + reset token).

    Use `enter_http_scope()` / `exit_http_scope()` when the surrounding code
    already has its own try/finally and adding a `with` block would force a
    full-body reindent.
    """

    __slots__ = ("counter", "_token")

    def __init__(self, counter: Counter[str], token: Token[Counter[str] | None]) -> None:
        self.counter = counter
        self._token = token


def enter_http_scope() -> HttpScope:
    counter: Counter[str] = Counter()
    token = _request_counter.set(counter)
    return HttpScope(counter, token)


def exit_http_scope(scope: HttpScope) -> None:
    _request_counter.reset(scope._token)


def summarize(counter: Counter[str]) -> dict[str, int]:
    """Sum a per-method counter into per-client totals.

    Keys: `bitbucket`, `jira`, `inference` (each int).
    Methods (e.g. `bitbucket:GET`) stay in the source counter for the caller
    to log as breakdown.
    """
    bitbucket = sum(v for k, v in counter.items() if k.startswith("bitbucket:"))
    jira = sum(v for k, v in counter.items() if k.startswith("jira:"))
    inference = sum(v for k, v in counter.items() if k.startswith("inference:"))
    return {"bitbucket": bitbucket, "jira": jira, "inference": inference}


def make_event_hook(client_label: str) -> Callable[[httpx.Request], Awaitable[None]]:
    """Build an httpx request event hook that records under `client_label`."""

    async def _hook(request: httpx.Request) -> None:
        record_request(client_label, request.method)

    return _hook
