import hashlib
import hmac
import logging
from contextlib import asynccontextmanager

from fastapi import BackgroundTasks, FastAPI, Header, HTTPException, Request

from app.bitbucket import BitbucketClient
from app.config import load_config
from app.copilot import CopilotClient
from app.models import WebhookPayload
from app.reviewer import Reviewer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logging.getLogger("httpx").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

config = None
reviewer = None
bitbucket_client = None
copilot_client = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global config, reviewer, bitbucket_client, copilot_client

    config = load_config()
    bitbucket_client = BitbucketClient(config.bitbucket)
    copilot_client = CopilotClient(config.copilot, config.review)
    reviewer = Reviewer(
        bitbucket_client,
        copilot_client,
        config.review.allowed_authors,
        max_comments=config.review.max_comments,
        context_lines=config.review.context_lines,
    )

    await copilot_client.validate_model()
    logger.info("Bridge service started, model=%s", config.copilot.model)

    yield

    await bitbucket_client.close()
    await copilot_client.close()


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
    x_hub_signature: str = Header(...),
):
    body = await request.body()

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

    background_tasks.add_task(reviewer.review_pull_request, payload)
    return {"status": "accepted", "pr_id": payload.pullRequest.id}
