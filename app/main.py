import hashlib
import hmac
import logging
import os
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any, cast

import structlog
from fastapi import BackgroundTasks, FastAPI, Header, HTTPException, Request, Response

from app.bitbucket import BitbucketClient
from app.config import AppConfig, load_config, log_config, model_label
from app.logging_config import configure_logging
from app.pricing_refresher import PricingRefresher, hydrate_from_db, refresh_once
from app.copilot_auth import CopilotTokenProvider
from app.db import close_pool, create_pool
from app.llm_client import LLMClient
from app.jira import JiraClient
from app.models import WebhookPayload
from app.review_queue import ReviewQueue
from app.reviewer import Reviewer
from app.riptide_client import RiptideAuthError, RiptideClient

# Logging is configured at import time so uvicorn's own loggers get the JSON
# bridge before the first line is written. We read env vars directly here
# (not via load_config) because config loading itself logs.
configure_logging(
    level=os.environ.get("LOG_LEVEL", "INFO"),
    env=os.environ.get("NOERGLER_ENV", "dev"),
)

_REVIEW_EVENT_KEYS = {"pr:opened", "pr:from_ref_updated"}
_SILENT_PATHS = frozenset({"/health"})

logger = logging.getLogger(__name__)


def _unify_uvicorn_logging() -> None:
    """Force all uvicorn loggers to propagate to the root logger so they go
    through the structlog JSON bridge instead of uvicorn's plain formatter."""
    for name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
        uv_logger = logging.getLogger(name)
        uv_logger.handlers.clear()
        uv_logger.propagate = True

config: AppConfig = cast(AppConfig, None)
reviewer: Reviewer = cast(Reviewer, None)
bitbucket_client: BitbucketClient = cast(BitbucketClient, None)
llm_client: LLMClient = cast(LLMClient, None)
jira_client: JiraClient = cast(JiraClient, None)
copilot_token_provider: CopilotTokenProvider = cast(CopilotTokenProvider, None)
review_queue: ReviewQueue = cast(ReviewQueue, None)
riptide_client: RiptideClient = cast(RiptideClient, None)
pricing_refresher: PricingRefresher = cast(PricingRefresher, None)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    global config, reviewer, bitbucket_client, llm_client, jira_client, copilot_token_provider, review_queue, riptide_client, pricing_refresher

    _unify_uvicorn_logging()
    version = os.environ.get("OPENSHIFT_BUILD_COMMIT") or os.environ.get("NOERGLER_VERSION") or "dev"
    logger.info("noergler version: %s", version)
    config = load_config()
    log_config(config, logger)
    bitbucket_client = BitbucketClient(config.bitbucket)
    riptide_client = RiptideClient.from_env(config.riptide.url, config.riptide.token)

    copilot_token_provider = CopilotTokenProvider(oauth_token=config.llm.oauth_token)

    llm_client = LLMClient(config.llm, config.review, copilot_token_provider)

    jira_client = JiraClient(config.jira)

    checks: dict[str, str | None] = {}

    try:
        db_pool = await create_pool(config.database.url)
        checks["Database"] = None
    except Exception as exc:
        checks["Database"] = str(exc)
        db_pool = None

    try:
        await bitbucket_client.check_connectivity()
        logger.info("Bot username: %s", bitbucket_client.bot_username)
        checks["Bitbucket"] = None
    except Exception as exc:
        checks["Bitbucket"] = str(exc)

    try:
        await jira_client.check_connectivity()
        checks["Jira"] = None
    except Exception as exc:
        checks["Jira"] = str(exc)

    try:
        await llm_client.check_connectivity()
        checks["LLM"] = None
    except Exception as exc:
        checks["LLM"] = str(exc)

    if riptide_client.enabled:
        # Verify riptide reachability + bearer at startup. A wrong token
        # fails fast (RiptideAuthError); transient unreachability only logs.
        try:
            await riptide_client.verify_at_startup()
            checks["Riptide"] = None
        except RiptideAuthError as exc:
            checks["Riptide"] = str(exc)
    else:
        logger.info("riptide_disabled: RIPTIDE_URL/RIPTIDE_TOKEN not set — event forwarding off")

    for name, error in checks.items():
        if error is None:
            logger.info("  ✔ %s: OK", name)
        else:
            logger.error("  ✘ %s: %s", name, error)

    failed = [k for k, v in checks.items() if v is not None]
    if failed:
        if db_pool:
            await close_pool()
        await bitbucket_client.close()
        await llm_client.close()
        await jira_client.close()
        await copilot_token_provider.close()
        await riptide_client.close()
        raise RuntimeError(
            f"Startup aborted — {len(failed)} connection(s) failed: {', '.join(failed)}"
        )

    reviewer = Reviewer(
        bitbucket_client, llm_client, config.review,
        jira=jira_client,
        server_config=config.server,
        db_pool=db_pool,
        riptide=riptide_client,
    )
    review_queue = ReviewQueue(reviewer.review_pull_request)
    review_queue.start()

    # Pricing: hydrate from DB cache (non-fatal if empty), then attempt one
    # live refresh from LiteLLM. The background task takes over from there
    # and refreshes every 24h. None of these are blocking on hard failures —
    # we always fall through to the static defaults baked into app.config.
    if db_pool is not None:
        await hydrate_from_db(db_pool)
    await refresh_once(db_pool)
    pricing_refresher = PricingRefresher(db_pool)
    pricing_refresher.start()

    _app.state.config = config
    _app.state.db_pool = db_pool
    logger.info("Bridge service started, model=%s, api_url=%s", model_label(config.llm.model, config.llm.reasoning_effort), config.llm.api_url)

    yield

    if pricing_refresher is not None:
        await pricing_refresher.stop()
    await review_queue.stop()
    await bitbucket_client.close()
    await llm_client.close()
    await jira_client.close()
    await copilot_token_provider.close()
    await riptide_client.close()
    await close_pool()


app = FastAPI(title="Bitbucket PR Review Bridge", lifespan=lifespan)


@app.middleware("http")
async def access_log(request: Request, call_next: Any) -> Response:
    # Liveness checks fire every few seconds; logging them buries real
    # traffic in Splunk. Pass through unobserved.
    if request.url.path in _SILENT_PATHS:
        return await call_next(request)
    request_id = request.headers.get("x-request-id") or uuid.uuid4().hex
    structlog.contextvars.bind_contextvars(
        request_id=request_id,
        method=request.method,
        path=request.url.path,
    )
    started = time.perf_counter()
    status_code = 500
    try:
        response: Response = await call_next(request)
        status_code = response.status_code
        return response
    finally:
        logger.info(
            "http_request status=%s duration_ms=%s",
            status_code,
            round((time.perf_counter() - started) * 1000, 1),
        )
        structlog.contextvars.clear_contextvars()


def _verify_webhook_signature(body: bytes, signature: str, secret: str) -> bool:
    expected = hmac.new(
        secret.encode(), body, hashlib.sha256
    ).hexdigest()
    # Bitbucket sends "sha256=<hex>" — strip the prefix before comparing
    if signature.startswith("sha256="):
        signature = signature[len("sha256="):]
    return hmac.compare_digest(expected, signature)


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/webhook")
async def webhook(
    request: Request,
    background_tasks: BackgroundTasks,
    x_hub_signature: str | None = Header(None),
    x_event_key: str | None = Header(None),
):
    if x_event_key == "diagnostics:ping":
        return {"status": "ok"}

    body = await request.body()

    if not x_hub_signature:
        # Bitbucket "Test connection" may omit both signature and event key
        if not x_event_key and b"eventKey" not in body:
            logger.info("Test connection received (no signature, no event key)")
            return {"status": "ok"}
        raise HTTPException(status_code=401, detail="Missing signature")
    if not _verify_webhook_signature(
        body, x_hub_signature, config.bitbucket.webhook_secret
    ):
        raise HTTPException(status_code=401, detail="Invalid signature")

    payload_json = await request.json()

    event_key = payload_json.get("eventKey", "")
    if not event_key.startswith("pr:"):
        return {"status": "ignored", "reason": f"not a PR event: {event_key}"}

    try:
        payload = WebhookPayload(**payload_json)
    except Exception as e:
        logger.error("Failed to parse webhook payload: %s", e)
        raise HTTPException(status_code=400, detail="Invalid payload")

    if event_key == "pr:merged":
        background_tasks.add_task(reviewer.handle_pr_merged, payload)
        return {"status": "accepted", "reason": "merged-stats"}

    if event_key == "pr:deleted":
        background_tasks.add_task(reviewer.handle_pr_deleted, payload)
        return {"status": "accepted", "reason": "deleted-purge"}

    if event_key == "pr:comment:added":
        comment_text = payload_json.get("comment", {}).get("text", "")
        comment_id = payload_json.get("comment", {}).get("id")
        parent_id = payload_json.get("commentParentId")
        logger.info("Comment event: id=%s parentId=%s", comment_id, parent_id)
        trigger = f"@{config.bitbucket.username}"
        if trigger.lower() in comment_text.lower():
            background_tasks.add_task(reviewer.handle_mention, payload)
            return {"status": "accepted", "reason": "mention"}
        if parent_id is not None:
            background_tasks.add_task(reviewer.handle_feedback, payload)
            return {"status": "accepted", "reason": "feedback"}
        return {"status": "ignored", "reason": "comment without mention"}

    if event_key not in _REVIEW_EVENT_KEYS:
        logger.warning(
            "Unhandled event %r — check your Bitbucket webhook configuration",
            event_key,
        )
        return {"status": "ignored", "reason": f"unhandled event: {event_key}"}

    pr = payload.pullRequest
    repo = pr.toRef.repository or pr.fromRef.repository
    if repo is None:
        logger.error("Review event for PR %d missing repository info", pr.id)
        return {"status": "ignored", "reason": "missing repository"}
    key = (repo.project.key, repo.slug, pr.id)
    outcome = review_queue.submit(key, payload)
    return {"status": "accepted", "pr_id": pr.id, "queue": outcome}
