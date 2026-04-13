import hashlib
import hmac
import logging
from contextlib import asynccontextmanager

from fastapi import BackgroundTasks, FastAPI, Header, HTTPException, Request

from app.bitbucket import BitbucketClient
from app.config import load_config, log_config
from app.db import close_pool, create_pool
from app.llm_client import LLMClient
from app.jira import JiraClient
from app.models import WebhookPayload
from app.reviewer import Reviewer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logging.getLogger("httpx").setLevel(logging.WARNING)

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

config = None
reviewer = None
bitbucket_client = None
copilot_client = None
jira_client = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global config, reviewer, bitbucket_client, copilot_client, jira_client

    _unify_uvicorn_logging()
    config = load_config()
    log_config(config, logger)
    bitbucket_client = BitbucketClient(config.bitbucket)
    copilot_client = LLMClient(config.copilot, config.review)

    jira_client = None
    if config.jira.enabled:
        jira_client = JiraClient(config.jira)
        logger.info("Jira integration enabled (%s)", config.jira.url)

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

    if jira_client:
        try:
            await jira_client.check_connectivity()
            checks["Jira"] = None
        except Exception as exc:
            checks["Jira"] = str(exc)

    try:
        await copilot_client.check_connectivity()
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
        await copilot_client.close()
        if jira_client:
            await jira_client.close()
        raise RuntimeError(
            f"Startup aborted — {len(failed)} connection(s) failed: {', '.join(failed)}"
        )

    reviewer = Reviewer(
        bitbucket_client, copilot_client, config.review,
        jira=jira_client,
        server_config=config.server,
        db_pool=db_pool,
    )
    logger.info("Bridge service started, model=%s, api_url=%s", config.copilot.model, config.copilot.api_url)

    yield

    await bitbucket_client.close()
    await copilot_client.close()
    if jira_client:
        await jira_client.close()
    await close_pool()


app = FastAPI(title="Bitbucket PR Review Bridge", lifespan=lifespan)


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

    if event_key == "pr:comment:added":
        comment_text = payload_json.get("comment", {}).get("text", "")
        comment_id = payload_json.get("comment", {}).get("id")
        parent_id = payload_json.get("commentParentId")
        logger.info("Comment event: id=%s parentId=%s", comment_id, parent_id)
        trigger = f"@{config.review.mention_trigger}"
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

    background_tasks.add_task(reviewer.review_pull_request, payload)
    return {"status": "accepted", "pr_id": payload.pullRequest.id}
