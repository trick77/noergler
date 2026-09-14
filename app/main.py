import hashlib
import hmac
import logging
import os
import re
import time
import uuid
from collections.abc import Awaitable, Callable
from contextlib import asynccontextmanager
from dataclasses import asdict, dataclass
from typing import Annotated, cast, final

import httpx
import structlog
from fastapi import BackgroundTasks, FastAPI, Header, HTTPException, Request, Response
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from app import onboarding, team_store
from app.bitbucket import BitbucketClient
from app.config import AppConfig, ProjectScope, TeamConfig, load_config, log_config, model_label
from app.logging_config import configure_logging
from app.db import close_pool, create_pool, get_pool
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

    # The team's own settings change at runtime through the API (one replica,
    # so the in-memory copy is the truth right after the DB write).
    def apply_claims(self, scopes: list[ProjectScope]) -> None:
        self.config.projects = scopes

    def apply_settings(self, settings: team_store.TeamSettings) -> None:
        self.config.review = self.config.review.model_copy(update={
            "auto_review_authors": settings.auto_review_authors,
            "ignore_authors": settings.ignore_authors,
        })
        self.reviewer.auto_review_authors = settings.auto_review_authors
        self.reviewer.ignore_authors = settings.ignore_authors


config: AppConfig = cast(AppConfig, cast(object, None))
bitbucket_client: BitbucketClient = cast(BitbucketClient, cast(object, None))
jira_client: JiraClient = cast(JiraClient, cast(object, None))
review_queue: ReviewQueue = cast(ReviewQueue, cast(object, None))
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


async def _load_team_store(db_pool, configured: dict[str, TeamConfig]) -> dict[str, str]:
    """Claims and author lists come from the DB; `teams.yaml` seeds a slug
    the DB knows nothing about. Returns per-slug reasons for teams whose seed
    conflicts with another team's claims (those are disabled)."""
    errors: dict[str, str] = {}
    claims = await team_store.list_all_claims(db_pool)
    settings = await team_store.get_all_settings(db_pool)
    for slug, team in configured.items():
        if slug in claims:
            team.projects = claims[slug]
        elif team.projects:
            try:
                added = await team_store.add_claims(db_pool, slug, team.projects, claimed_by="teams.yaml")
            except team_store.ClaimConflict as exc:
                errors[slug] = f"teams.yaml seed: {exc}"
                continue
            logger.info("claims seeded from teams.yaml team=%s n=%d", slug, len(added))
        else:
            logger.info("no claims yet team=%s (claim via POST /onboard)", slug)
        if slug in settings:
            team.review = team.review.model_copy(update={
                "auto_review_authors": settings[slug].auto_review_authors,
                "ignore_authors": settings[slug].ignore_authors,
            })
        elif team.review.auto_review_authors or team.review.ignore_authors:
            await team_store.put_settings(
                db_pool, slug,
                team_store.TeamSettings(team.review.auto_review_authors, team.review.ignore_authors),
                updated_by="teams.yaml",
            )
    return errors


@asynccontextmanager
async def lifespan(_app: FastAPI):
    global config, bitbucket_client, jira_client, review_queue

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
    seed_errors = await _load_team_store(db_pool, config.teams)
    for slug, team in config.teams.items():
        structlog.contextvars.bind_contextvars(team=slug)
        try:
            try:
                result = seed_errors.get(slug) or await _start_team(team, db_pool, logger)
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

    _app.state.config = config
    _app.state.db_pool = db_pool
    logger.info("Bridge service started, api_url=%s, teams=%d", config.llm.api_url, len(teams))

    yield

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


def _runtime_for(team_slug: str) -> TeamRuntime:
    """The enabled team behind a `/{route}/{team_slug}` path, or 404 / 503."""
    runtime = teams.get(team_slug)
    if runtime is None:
        if team_slug in disabled_teams:
            # Reason in the log only; this answer goes out before any auth check.
            logger.warning("request rejected: team is disabled (%s)", disabled_teams[team_slug])
            raise HTTPException(
                status_code=503,
                detail=f"team {team_slug} is disabled, see the noergler startup log",
            )
        raise HTTPException(status_code=404, detail="unknown team")
    return runtime


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
    runtime = _runtime_for(team_slug)
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


class OnboardRequest(BaseModel, extra="forbid"):
    action: onboarding.Action = "status"
    # Claim these (and hook them), or with `remove`: unclaim them, drop their
    # hooks and every PR record of the team on them. Same shape as the
    # `projects:` block in teams.yaml.
    projects: list[ProjectScope] | None = None
    # Narrow to these current claims (`KEY` or `KEY/repo`); default all.
    targets: list[str] | None = None
    dry_run: bool = False
    name: str = onboarding.DEFAULT_WEBHOOK_NAME
    prune: bool = True


class ReviewAuthorsRequest(BaseModel, extra="forbid"):
    auto_review_authors: list[str]
    ignore_authors: list[str]


@asynccontextmanager
async def _admin_client(authorization: str | None):
    """A Bitbucket client on the caller's own token, after Bitbucket has
    accepted it. Yields (client, caller username). The token is used for this
    request only and never logged; nothing derived from teams.yaml or the DB
    is answered before this check passes."""
    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(
            status_code=401,
            detail="Authorization: Bearer <your Bitbucket HTTP access token> required",
        )
    token = authorization[len("bearer "):].strip()
    if not token:
        raise HTTPException(status_code=401, detail="empty bearer token")
    async with BitbucketClient(config.bitbucket, token=token) as admin:
        try:
            caller = await admin.whoami()
        except Exception as exc:
            logger.warning("whoami against Bitbucket failed: %s", exc)
            raise HTTPException(status_code=502, detail="Bitbucket did not answer the token check") from exc
        if not caller:
            raise HTTPException(status_code=401, detail="Bitbucket rejected the token")
        yield admin, caller


def _pool():
    pool = get_pool()
    if pool is None:  # only before lifespan finished; never in a served request
        raise HTTPException(status_code=503, detail="database not ready")
    return pool


async def _has_admin(admin: BitbucketClient, target: onboarding.Target) -> bool:
    """Listing webhooks needs admin on the target; Bitbucket answers 401/403
    for anything less. Any other failure is Bitbucket's, not the caller's: 502."""
    try:
        await admin.list_webhooks(target.project, target.repo)
    except httpx.HTTPStatusError as exc:
        if exc.response.status_code in (401, 403):
            return False
        raise HTTPException(
            status_code=502, detail=f"Bitbucket answered HTTP {exc.response.status_code} on {target.key}"
        ) from exc
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=502, detail=f"Bitbucket unreachable on {target.key}: {exc}") from exc
    return True


def _team_view(runtime: TeamRuntime) -> dict[str, object]:
    return {
        "team": runtime.config.slug,
        "projects": [p.model_dump(exclude_none=True) for p in runtime.config.projects],
        "auto_review_authors": runtime.config.review.auto_review_authors,
        "ignore_authors": runtime.config.review.ignore_authors,
    }


@app.post("/onboard/{team_slug}")
async def onboard(
    team_slug: str,
    body: OnboardRequest,
    authorization: Annotated[str | None, Header()] = None,
):
    """Put noergler's webhook on the team's Bitbucket projects/repos, and with
    `projects` in the body claim them for the team first (or, with `remove`,
    give them up along with every PR record on them).

    Authenticated by the caller's own Bitbucket token (`Authorization: Bearer`);
    it is used for this request's Bitbucket calls and dropped. A claim needs
    project admin on the target, proven with that token. Without `projects`
    the targets are the team's current claims, `targets` can only narrow them.
    The webhook secret never leaves the service: it is written into the hook here.
    """
    structlog.contextvars.bind_contextvars(team=team_slug)
    runtime = _runtime_for(team_slug)
    team = runtime.config
    if not config.server.public_url:
        raise HTTPException(
            status_code=503,
            detail="NOERGLER_PUBLIC_URL is not set on this instance; onboarding via API is disabled",
        )
    if body.projects is not None and body.action == "status":
        raise HTTPException(status_code=400, detail="projects only with onboard, grant-bot or remove")
    if body.projects is not None and body.targets is not None:
        raise HTTPException(status_code=400, detail="projects and targets are exclusive")

    webhook_url = f"{config.server.public_url}/webhook/{team_slug}"
    extra: dict[str, object] = {}
    async with _admin_client(authorization) as (admin, caller):
        rows: list[onboarding.StatusRow] | list[onboarding.TargetResult]
        if body.projects is not None and body.action != "remove":
            rows, text, healthy = await _claim_and_onboard(runtime, admin, caller, body, webhook_url, extra)
        elif body.projects is not None:
            rows, text, healthy = await _remove_and_unclaim(runtime, admin, caller, body, webhook_url, extra)
        else:
            try:
                targets = onboarding.targets_for(team, body.targets)
            except onboarding.UnknownTarget as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
            if not targets:
                raise HTTPException(status_code=400, detail="no targets: claim a project first (projects in the body)")
            logger.info(
                "onboard by=%s action=%s targets=%d webhook_url=%s dry_run=%s",
                caller, body.action, len(targets), webhook_url, body.dry_run,
            )
            onboarder = onboarding.Onboarder(
                admin, bitbucket_client, team, webhook_url,
                webhook_name=body.name, dry_run=body.dry_run,
                grant_bot=body.action == "grant-bot", prune=body.prune,
            )
            rows, text, healthy = await onboarding.run(onboarder, body.action, targets)
    logger.info("onboard action=%s done healthy=%s\n%s", body.action, healthy, text)
    return {
        "team": team_slug,
        "action": body.action,
        "webhook_url": webhook_url,
        "healthy": healthy,
        "rows": [asdict(r) for r in rows],
        "text": text,
        **extra,
    }


async def _claim_and_onboard(
    runtime: TeamRuntime, admin: BitbucketClient, caller: str, body: OnboardRequest,
    webhook_url: str, extra: dict[str, object],
) -> tuple[list[onboarding.TargetResult], str, bool]:
    """`projects` with onboard/grant-bot: prove admin per target, claim what
    is proven (all or nothing against other teams), then hook exactly those."""
    team = runtime.config
    scopes = list(body.projects or [])
    proven: list[ProjectScope] = []
    failed: list[onboarding.TargetResult] = []
    for scope in scopes:
        ok_repos: list[str] = []
        for target in onboarding.targets_for(team.model_copy(update={"projects": [scope]})):
            if not await _has_admin(admin, target):
                failed.append(onboarding.TargetResult(
                    target, "failed", detail=f"no project admin on {target.key} with this token; not claimed",
                ))
            elif target.repo is None:
                proven.append(scope)
            else:
                ok_repos.append(target.repo)
        if ok_repos:
            proven.append(ProjectScope(key=scope.key, repos=ok_repos))

    claimed: list[str] = []
    if proven and not body.dry_run:
        try:
            claimed = await team_store.add_claims(_pool(), team.slug, proven, claimed_by=caller)
        except team_store.ClaimConflict as exc:
            raise HTTPException(
                status_code=409,
                detail={"message": str(exc), "conflict": {
                    "target": exc.project if exc.repo is None else f"{exc.project}/{exc.repo}",
                    "team": exc.other_team,
                }},
            ) from exc
        runtime.apply_claims(await team_store.list_claims(_pool(), team.slug))
    extra["claimed"] = claimed
    # For the hook step the proven scopes count as claimed even on a dry run.
    view = team if not body.dry_run else team.model_copy(update={"projects": team.projects + proven})
    targets = onboarding.targets_for(view.model_copy(update={"projects": proven})) if proven else []
    logger.info(
        "onboard by=%s action=%s claimed=%s targets=%d dry_run=%s",
        caller, body.action, claimed, len(targets), body.dry_run,
    )
    onboarder = onboarding.Onboarder(
        admin, bitbucket_client, view, webhook_url,
        webhook_name=body.name, dry_run=body.dry_run,
        grant_bot=body.action == "grant-bot", prune=body.prune,
    )
    rows, _, _ = await onboarding.run(onboarder, body.action, targets) if targets else ([], "", True)
    results = failed + cast(list[onboarding.TargetResult], rows)
    return results, onboarding.render_results(results), onboarding.results_healthy(results)


async def _remove_and_unclaim(
    runtime: TeamRuntime, admin: BitbucketClient, caller: str, body: OnboardRequest,
    webhook_url: str, extra: dict[str, object],
) -> tuple[list[onboarding.TargetResult], str, bool]:
    """`projects` with remove: hooks off, claims gone, every PR record of the
    team on those targets purged (findings cascade). `dry_run` counts only."""
    team = runtime.config
    scopes = list(body.projects or [])
    # A whole-project scope means every claim the team has on that project,
    # whether it holds the project or some of its repos.
    wanted: list[onboarding.Target] = []
    for scope in scopes:
        if scope.repos is None:
            mine = [t for t in onboarding.targets_for(team) if t.project == scope.key]
            if not mine:
                raise HTTPException(status_code=400, detail=f"no claim on {scope.key}")
            wanted.extend(mine)
        else:
            try:
                wanted.extend(onboarding.targets_for(team, [f"{scope.key}/{r}" for r in scope.repos]))
            except onboarding.UnknownTarget as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
    # Giving a target up needs the same proof as taking it: admin on it.
    proven: list[onboarding.Target] = []
    results: list[onboarding.TargetResult] = []
    for target in wanted:
        if await _has_admin(admin, target):
            proven.append(target)
        else:
            results.append(onboarding.TargetResult(
                target, "failed", detail=f"no project admin on {target.key} with this token; not removed",
            ))
    onboarder = onboarding.Onboarder(admin, bitbucket_client, team, webhook_url, webhook_name=body.name, dry_run=body.dry_run)
    rows, _, _ = await onboarding.run(onboarder, "remove", proven) if proven else ([], "", True)
    hook_results = cast(list[onboarding.TargetResult], rows)
    purged = 0
    for result in hook_results:
        t = result.target
        if body.dry_run:
            n = await team_store.count_project_prs(_pool(), team.slug, t.project, t.repo)
            result.detail += f"; dry-run: would purge {n} PR record(s)"
        else:
            n = await team_store.purge_project(_pool(), team.slug, t.project, t.repo)
            result.detail += f"; purged {n} PR record(s)"
        purged += n
    results.extend(hook_results)
    unclaimed: list[str] = []
    if proven and not body.dry_run:
        drop = [ProjectScope(key=t.project) if t.repo is None else ProjectScope(key=t.project, repos=[t.repo]) for t in proven]
        unclaimed = await team_store.remove_claims(_pool(), team.slug, drop)
        runtime.apply_claims(await team_store.list_claims(_pool(), team.slug))
    logger.info("onboard by=%s action=remove unclaimed=%s purged_prs=%d dry_run=%s", caller, unclaimed, purged, body.dry_run)
    extra["unclaimed"] = unclaimed
    extra["purged_prs"] = purged
    return results, onboarding.render_results(results), onboarding.results_healthy(results)


@app.get("/teams/{team_slug}")
async def team_settings(team_slug: str, authorization: Annotated[str | None, Header()] = None):
    """The team's own settings: claims and review author lists."""
    structlog.contextvars.bind_contextvars(team=team_slug)
    runtime = _runtime_for(team_slug)
    async with _admin_client(authorization):
        return _team_view(runtime)


@app.put("/teams/{team_slug}/review-authors")
async def put_review_authors(
    team_slug: str,
    body: ReviewAuthorsRequest,
    authorization: Annotated[str | None, Header()] = None,
):
    """Replace both author lists. Needs project admin on at least one of the
    team's claims (proven with the caller's token). Takes effect immediately."""
    structlog.contextvars.bind_contextvars(team=team_slug)
    runtime = _runtime_for(team_slug)
    async with _admin_client(authorization) as (admin, caller):
        claims = onboarding.targets_for(runtime.config)
        if not claims:
            raise HTTPException(status_code=409, detail="claim a project first (POST /onboard with projects)")
        if not any([await _has_admin(admin, t) for t in claims]):
            raise HTTPException(status_code=403, detail="no project admin on any of the team's claims with this token")
        settings = team_store.TeamSettings(
            [a.strip() for a in body.auto_review_authors if a.strip()],
            [a.strip() for a in body.ignore_authors if a.strip()],
        )
        await team_store.put_settings(_pool(), team_slug, settings, updated_by=caller)
        runtime.apply_settings(settings)
    return _team_view(runtime)
