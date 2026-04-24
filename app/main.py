import hashlib
import hmac
import logging
from contextlib import asynccontextmanager

from fastapi import BackgroundTasks, FastAPI, Header, HTTPException, Request

from typing import cast

from app.bitbucket import BitbucketClient
from app.config import AppConfig, load_config, log_config
from app.copilot_auth import CopilotTokenProvider
from app.db import close_pool, create_pool
from app.llm_client import LLMClient
from app.jira import JiraClient
from app.metrics import router as metrics_router
from app.models import WebhookPayload
from app.review_queue import ReviewQueue
from app.reviewer import Reviewer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)

_REVIEW_EVENT_KEYS = {"pr:opened", "pr:from_ref_updated"}


class _HealthFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        return "/health" not in record.getMessage()


logging.getLogger("uvicorn.access").addFilter(_HealthFilter())
logger = logging.getLogger(__name__)


def _unify_uvicorn_logging() -> None:
    """Force all uvicorn loggers to propagate to the root logger for consistent formatting."""
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


@asynccontextmanager
async def lifespan(_app: FastAPI):
    global config, reviewer, bitbucket_client, llm_client, jira_client, copilot_token_provider, review_queue

    _unify_uvicorn_logging()
    config = load_config()
    log_config(config, logger)
    bitbucket_client = BitbucketClient(config.bitbucket)

    copilot_token_provider = CopilotTokenProvider(
        oauth_token=config.llm.oauth_token,
        integration_id=config.llm.integration_id,
        editor_version=config.llm.editor_version,
    )
    # Do the first token exchange up front so auth failures surface at startup
    # and so we can honour the endpoints.api returned by the exchange.
    await copilot_token_provider.get_token()
    if copilot_token_provider.endpoints_api != config.llm.api_url:
        logger.info(
            "Overriding api_url with token-exchange endpoints.api: %s -> %s",
            config.llm.api_url, copilot_token_provider.endpoints_api,
        )
        config.llm.api_url = copilot_token_provider.endpoints_api

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
        raise RuntimeError(
            f"Startup aborted — {len(failed)} connection(s) failed: {', '.join(failed)}"
        )

    reviewer = Reviewer(
        bitbucket_client, llm_client, config.review,
        jira=jira_client,
        server_config=config.server,
        db_pool=db_pool,
    )
    review_queue = ReviewQueue(reviewer.review_pull_request)
    review_queue.start()
    _app.state.config = config
    _app.state.db_pool = db_pool
    logger.info("Bridge service started, model=%s, api_url=%s", config.llm.model, config.llm.api_url)

    yield

    await review_queue.stop()
    await bitbucket_client.close()
    await llm_client.close()
    await jira_client.close()
    await copilot_token_provider.close()
    await close_pool()


app = FastAPI(title="Bitbucket PR Review Bridge", lifespan=lifespan)
app.include_router(metrics_router)


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
