import hashlib
import hmac
import logging
import os
import re
import time
import uuid
from collections.abc import Awaitable, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Annotated, cast, final

import structlog
from fastapi import BackgroundTasks, FastAPI, Header, HTTPException, Request, Response
from fastapi.responses import JSONResponse

from app.bitbucket import BitbucketClient
from app.config import AppConfig, TeamConfig, load_config, log_config, model_label
from app.logging_config import configure_logging
from app.pricing_refresher import PricingRefresher
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
# Must match PROBE_EVENT_KEY in scripts/onboard_repo.py.
PROBE_EVENT_KEY = "noergler:probe"
_SILENT_PATHS = frozenset({"/health", "/ready"})
_REQUEST_ID_RE = re.compile(r"^[A-Za-z0-9._-]{1,64}$")

logger = logging.getLogger(__name__)
access_logger = structlog.stdlib.get_logger("app.access")


@final
@dataclass
class TeamRuntime:
    """Everything one enabled team needs at request time.

    The Bitbucket client, the DB pool and the review queue are shared by all
    teams (see AGENTS.md on serialization); the LLM client, the Jira prefixes
    and the riptide emitter are the team's own.
    """

    config: TeamConfig
    llm: LLMClient
    jira: JiraClient
    riptide: RiptideClient
    reviewer: Reviewer


config: AppConfig = cast(AppConfig, cast(object, None))
bitbucket_client: BitbucketClient = cast(BitbucketClient, cast(object, None))
jira_client: JiraClient = cast(JiraClient, cast(object, None))
review_queue: ReviewQueue = cast(ReviewQueue, cast(object, None))
pricing_refresher: PricingRefresher = cast(PricingRefresher, cast(object, None))
# Enabled teams by slug; disabled teams by slug with the reason. A slug lives
# in exactly one of the two. Both are fixed after startup — fixing a team is
# a config change and a redeploy, same as any other setting.
teams: dict[str, TeamRuntime] = {}
disabled_teams: dict[str, str] = {}


async def _review_for_team(slug: str, payload: WebhookPayload) -> None:
    """Queue worker entry: route the job to the team's reviewer."""
    runtime = teams.get(slug)
    if runtime is None:
        # Cannot happen after startup (the set is fixed), but a queued job
        # must never crash the worker.
        logger.error("queued review for unknown team %s dropped", slug)
        return
    await runtime.reviewer.review_pull_request(payload)


async def _start_team(
    team: TeamConfig, db_pool, log: logging.Logger
) -> TeamRuntime | str:
    """Build a team's clients and run its startup checks.

    Returns the runtime, or the reason string when the team must be disabled.
    Every failure here is the team's own (its key, its model, its riptide
    token) and never touches another team.
    """
    llm = LLMClient(team.llm, team.review)
    jira = JiraClient(team.jira)
    riptide = RiptideClient.from_env(
        team.riptide.url if team.riptide else "",
        team.riptide.token if team.riptide else "",
    )

    async def _teardown() -> None:
        await llm.close()
        await jira.close()
        await riptide.close()

    try:
        await llm.check_connectivity()
    except Exception as exc:
        await _teardown()
        return f"LLM check failed: {exc}"

    if riptide.enabled:
        # A wrong token disables the team (RiptideAuthError); anything else
        # (unreachable, malformed ping body) only logs and the team stays
        # enabled — emissions are best-effort anyway.
        try:
            await riptide.verify_at_startup()
        except RiptideAuthError as exc:
            await _teardown()
            return f"riptide check failed: {exc}"
        except Exception as exc:
            log.warning("riptide ping failed, team stays enabled: %s", exc)
    else:
        log.info("riptide disabled for this team — event forwarding off")

    reviewer = Reviewer(
        bitbucket_client, llm, team.review,
        jira=jira,
        server_config=config.server,
        db_pool=db_pool,
        riptide=riptide,
        team_slug=team.slug,
    )
    return TeamRuntime(config=team, llm=llm, jira=jira, riptide=riptide, reviewer=reviewer)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    global config, bitbucket_client, jira_client, review_queue, pricing_refresher

    version = os.environ.get("OPENSHIFT_BUILD_COMMIT") or os.environ.get("NOERGLER_VERSION") or "dev"
    logger.info("noergler version: %s", version)
    config = load_config()
    log_config(config, logger)
    bitbucket_client = BitbucketClient(config.bitbucket)
    jira_client = JiraClient(config.jira)

    # --- Shared layer: any failure here aborts boot. Nothing works without it.
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

    for name, error in checks.items():
        if error is None:
            logger.info("%s: OK", name)
        else:
            logger.error("%s: %s", name, error)

    failed = [k for k, v in checks.items() if v is not None]
    if failed:
        if db_pool:
            await close_pool()
        await bitbucket_client.close()
        await jira_client.close()
        raise RuntimeError(
            f"Startup aborted — {len(failed)} connection(s) failed: {', '.join(failed)}"
        )

    # --- Per-team layer: a failure disables that team only.
    teams.clear()
    disabled_teams.clear()
    disabled_teams.update(config.disabled)
    for slug, team in config.teams.items():
        structlog.contextvars.bind_contextvars(team=slug)
        try:
            try:
                result = await _start_team(team, db_pool, logger)
            except Exception as exc:
                # Nothing a single team does may take the instance down.
                result = f"startup failed: {exc}"
            if isinstance(result, str):
                disabled_teams[slug] = result
                logger.error("team_disabled team=%s reason=%s", slug, result)
            else:
                teams[slug] = result
                logger.info(
                    "team_ready team=%s model=%s riptide=%s",
                    slug, model_label(team.llm.model, team.llm.reasoning_effort),
                    "on" if result.riptide.enabled else "off",
                )
        finally:
            structlog.contextvars.unbind_contextvars("team")

    summary = "teams_ready enabled=%s disabled=%s"
    if disabled_teams:
        logger.warning(summary, sorted(teams), sorted(disabled_teams))
    else:
        logger.info(summary, sorted(teams), sorted(disabled_teams))
    if not teams:
        logger.error("no team is enabled — /ready reports 503 until the config is fixed")

    review_queue = ReviewQueue(_review_for_team)
    review_queue.start()

    # Every enabled team's model was resolved by its LLM connectivity check
    # above; this task only keeps those entries fresh every 24h.
    pricing_refresher = PricingRefresher(
        [rt.config.llm.model for rt in teams.values()], config.llm.catalog_url,
    )
    pricing_refresher.start()

    _app.state.config = config
    _app.state.db_pool = db_pool
    logger.info("Bridge service started, api_url=%s, teams=%d", config.llm.api_url, len(teams))

    yield

    await pricing_refresher.stop()
    await review_queue.stop()
    for runtime in teams.values():
        await runtime.llm.close()
        await runtime.jira.close()
        await runtime.riptide.close()
    await bitbucket_client.close()
    await jira_client.close()
    await close_pool()


app = FastAPI(title="Bitbucket PR Review Bridge", lifespan=lifespan)


@app.middleware("http")
async def access_log(
    request: Request, call_next: Callable[[Request], Awaitable[Response]]
) -> Response:
    # Probes fire every few seconds; logging them buries real traffic in
    # Splunk. Pass through unobserved.
    if request.url.path in _SILENT_PATHS:
        return await call_next(request)
    # Honor caller-supplied X-Request-Id only when it looks like a sane
    # correlation token. Untrusted input must not become an indexed field
    # — an attacker could otherwise inject newlines or huge strings.
    header_id = request.headers.get("x-request-id")
    if header_id and _REQUEST_ID_RE.match(header_id):
        request_id = header_id
    else:
        request_id = uuid.uuid4().hex
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
        # Emit the access record BEFORE clearing contextvars so method/path
        # are still merged onto the log event.
        access_logger.info(
            "http_request",
            status_code=status_code,
            duration_ms=round((time.perf_counter() - started) * 1000, 1),
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


async def _probe(team: TeamConfig, payload: object) -> dict[str, object]:
    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail="probe body must be an object")
    project = payload.get("project")
    repo = payload.get("repo")
    if not isinstance(project, str) or not project.strip():
        raise HTTPException(status_code=400, detail="probe needs a project key")
    if repo is not None and (not isinstance(repo, str) or not repo.strip()):
        raise HTTPException(status_code=400, detail="probe repo must be a slug or null")
    project = project.strip()
    repo = repo.strip() if isinstance(repo, str) else None

    # How the team claims this project in teams.yaml: the whole project, some
    # repos, or not at all. A project probe is owned only by a whole claim; a
    # repo probe by either. The claim kind lets the script tell an admin whose
    # team.json disagrees with teams.yaml exactly what to change.
    scope = next((p for p in team.projects if p.key == project), None)
    claim = "none" if scope is None else ("whole" if scope.repos is None else "repos")
    if repo is None:
        owned = claim == "whole"
    else:
        owned = team.owns(project, repo)

    # Read access as the bot, with its own token. Read only: whether it may
    # also comment is proven by the first review.
    bot_can_read = False
    if owned:
        try:
            if repo is None:
                await bitbucket_client.get_project(project)
            else:
                await bitbucket_client.get_repo(project, repo)
            bot_can_read = True
        except Exception as exc:
            logger.info("probe: bot cannot read %s/%s: %s", project, repo or "*", exc)
    target = project if repo is None else f"{project}/{repo}"
    logger.info("probe target=%s claim=%s owned=%s bot_can_read=%s", target, claim, owned, bot_can_read)
    return {
        "team": team.slug,
        "owned": owned,
        "claim": claim,
        "bot_can_read": bot_can_read,
        "bot_username": config.bitbucket.username,
    }


def _team_status() -> dict[str, object]:
    # Slugs only. The disable reasons carry internal detail (gateway URLs and
    # error bodies, env var names) and belong in the log, not on an
    # unauthenticated probe.
    return {
        "enabled": sorted(teams),
        "disabled": sorted(disabled_teams),
    }


@app.get("/health")
async def health():
    """Liveness: 200 while the process is up. A config fault that leaves
    every team disabled is not fixed by a restart, so it never fails this."""
    return {"status": "ok", "teams": _team_status()}


@app.get("/ready")
async def ready():
    """Readiness: 503 while no team can take traffic."""
    body = {"status": "ok" if teams else "no-teams", "teams": _team_status()}
    return JSONResponse(body, status_code=200 if teams else 503)


@app.post("/webhook/{team_slug}")
async def webhook(
    team_slug: str,
    request: Request,
    background_tasks: BackgroundTasks,
    x_hub_signature: Annotated[str | None, Header()] = None,
    x_event_key: Annotated[str | None, Header()] = None,
):
    # Team identity comes from the path plus the HMAC below, never from the
    # payload: `project.key` in the body is unauthenticated.
    structlog.contextvars.bind_contextvars(team=team_slug)
    runtime = teams.get(team_slug)
    if runtime is None:
        if team_slug in disabled_teams:
            # Reason in the log only; this answer goes out before the HMAC check.
            logger.warning("webhook rejected: team is disabled (%s)", disabled_teams[team_slug])
            raise HTTPException(
                status_code=503,
                detail=f"team {team_slug} is disabled, see the noergler startup log",
            )
        raise HTTPException(status_code=404, detail="unknown team")
    team = runtime.config
    reviewer = runtime.reviewer

    if x_event_key == "diagnostics:ping":
        return {"status": "ok"}

    body = await request.body()

    if not x_hub_signature:
        # Bitbucket "Test connection" may omit both signature and event key
        if not x_event_key and b"eventKey" not in body:
            logger.info("Test connection received (no signature, no event key)")
            return {"status": "ok"}
        raise HTTPException(status_code=401, detail="Missing signature")
    if not _verify_webhook_signature(body, x_hub_signature, team.webhook_secret):
        raise HTTPException(status_code=401, detail="Invalid signature")

    payload_json = await request.json()

    if x_event_key == PROBE_EVENT_KEY:
        # Onboarding probe from scripts/onboard_repo.py: signed with the team
        # secret like any event, so only that team can ask. Answers whether
        # the team owns the target and whether the bot can read it, which
        # proves route, secret, ownership and bot access without a PR.
        return await _probe(team, payload_json)

    event_key = payload_json.get("eventKey", "")
    if not event_key.startswith("pr:"):
        return {"status": "ignored", "reason": f"not a PR event: {event_key}"}

    try:
        payload = WebhookPayload(**payload_json)
    except Exception as e:
        logger.error("Failed to parse webhook payload: %s", e)
        raise HTTPException(status_code=400, detail="Invalid payload")

    # The signature proves the sender holds this team's secret; it does not
    # prove the PR is this team's. Without this check a team could sign a
    # payload naming another team's repo and review it on its own key.
    pr = payload.pullRequest
    repo = pr.toRef.repository or pr.fromRef.repository
    if repo is None:
        logger.error("Event %s for PR %d missing repository info", event_key, pr.id)
        return {"status": "ignored", "reason": "missing repository"}
    if not team.owns(repo.project.key, repo.slug):
        logger.warning(
            "webhook rejected: %s/%s is not owned by team %s",
            repo.project.key, repo.slug, team_slug,
        )
        raise HTTPException(
            status_code=403,
            detail=f"repository {repo.project.key}/{repo.slug} is not owned by team {team_slug}",
        )

    if event_key == "pr:merged":
        background_tasks.add_task(reviewer.handle_pr_merged, payload)
        return {"status": "accepted", "reason": "merged-rollup"}

    if event_key == "pr:declined":
        background_tasks.add_task(reviewer.handle_pr_declined, payload)
        return {"status": "accepted", "reason": "declined-rollup"}

    if event_key == "pr:deleted":
        background_tasks.add_task(reviewer.handle_pr_deleted, payload)
        return {"status": "accepted", "reason": "deleted-purge"}

    if event_key == "pr:comment:deleted":
        background_tasks.add_task(reviewer.handle_comment_deleted, payload)
        return {"status": "accepted", "reason": "comment-deleted"}

    if event_key == "pr:comment:added":
        comment_text = payload_json.get("comment", {}).get("text", "")
        comment_id = payload_json.get("comment", {}).get("id")
        logger.info("Comment event: id=%s", comment_id)
        trigger = f"@{config.bitbucket.username}"
        if trigger.lower() in comment_text.lower():
            background_tasks.add_task(reviewer.handle_mention, payload)
            return {"status": "accepted", "reason": "mention"}
        return {"status": "ignored", "reason": "comment without mention"}

    if event_key not in _REVIEW_EVENT_KEYS:
        logger.warning(
            "Unhandled event %r — check your Bitbucket webhook configuration",
            event_key,
        )
        return {"status": "ignored", "reason": f"unhandled event: {event_key}"}

    key = (repo.project.key, repo.slug, pr.id)
    outcome = review_queue.submit(key, payload, team_slug)
    return {"status": "accepted", "pr_id": pr.id, "queue": outcome}
