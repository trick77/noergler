"""Webhook onboarding for a team, driven by `POST /onboard/{team}`.

Onboarding = putting noergler's webhook on each target the team claims in
`teams.yaml` and, on request, granting the bot write access there. Every
Bitbucket write goes through the *team admin's* token (project admin on the
team's projects); the bot's own token is only used to check that the bot can
read a target. The admin token lives in one `BitbucketClient` for the duration
of the request and is never logged or stored.

Targets come from the team block, never from the caller: a whole-project
claim (`key` without `repos`) is one project webhook, a `repos:` list is one
repo webhook per slug. The caller can only narrow that set.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any, Literal

import httpx

from app.bitbucket import BitbucketClient
from app.config import REQUIRED_WEBHOOK_EVENTS, TeamConfig

logger = logging.getLogger(__name__)

DEFAULT_WEBHOOK_NAME = "noergler"

Action = Literal["status", "onboard", "grant-bot", "remove"]


@dataclass(frozen=True)
class Target:
    """A project (repo=None: one project webhook) or a single repo."""

    project: str
    repo: str | None = None

    @property
    def is_project(self) -> bool:
        return self.repo is None

    @property
    def key(self) -> str:
        return self.project if self.repo is None else f"{self.project}/{self.repo}"

    @property
    def label(self) -> str:
        return f"{self.project} (project)" if self.repo is None else self.key

    @property
    def bot_permission(self) -> str:
        return "PROJECT_WRITE" if self.repo is None else "REPO_WRITE"


@dataclass(frozen=True)
class Claim:
    """How `teams.yaml` claims a target for the team, and whether the bot can read it."""

    owned: bool
    kind: str  # "whole", "repos" or "none"
    bot_can_read: bool


@dataclass
class TargetResult:
    target: Target
    status: str  # "ok", "failed", "skipped"
    detail: str = ""
    diff: list[str] = field(default_factory=list)


@dataclass
class StatusRow:
    target: Target
    owned: bool
    bot_can_read: bool
    webhook: str          # "ok", "missing", "stale: ...", "foreign: ...", "HTTP <code>", or why it is not owned
    stray: list[str]      # this instance's repo-level hooks under a project target
    foreign: list[str]    # same-named hooks pointing at another noergler instance


class UnknownTarget(ValueError):
    """A requested target is not in the team's `teams.yaml` block."""


class ForeignHook(Exception):
    """A same-named hook points at another noergler instance; never rewritten."""


def targets_for(team: TeamConfig, subset: list[str] | None = None) -> list[Target]:
    """The team's targets from `teams.yaml`, optionally narrowed to `subset`
    (`KEY` or `KEY/repo`, as they appear in the team block)."""
    targets: list[Target] = []
    for scope in team.projects:
        if scope.repos is None:
            targets.append(Target(scope.key))
        else:
            targets.extend(Target(scope.key, repo) for repo in scope.repos)
    if subset is None:
        return targets
    by_key = {t.key: t for t in targets}
    unknown = [s for s in subset if s not in by_key]
    if unknown:
        raise UnknownTarget(
            f"not in team {team.slug}'s teams.yaml block: {', '.join(unknown)}; "
            f"known: {', '.join(by_key)}"
        )
    return [by_key[s] for s in subset]


def claim_kind(team: TeamConfig, project: str) -> str:
    scope = next((p for p in team.projects if p.key == project), None)
    return "none" if scope is None else ("whole" if scope.repos is None else "repos")


def _not_owned_reason(target: Target, claim: Claim) -> str:
    if target.is_project and claim.kind == "repos":
        return (
            f"teams.yaml lists specific repos of {target.project} for this team; "
            "onboard those, or ask the noergler admin to claim the whole project"
        )
    return "not owned by this team in teams.yaml; ask the noergler admin"


def _whole_claim_reason(target: Target) -> str:
    return (
        f"teams.yaml claims all of {target.project} for this team: one project webhook "
        "covers it, a repo webhook next to it would deliver every event twice"
    )


def _hook_named(hooks: list[dict[str, Any]], name: str) -> dict[str, Any] | None:
    return next((h for h in hooks if h.get("name") == name), None)


def _http_detail(exc: httpx.HTTPStatusError) -> str:
    return f"HTTP {exc.response.status_code}: {exc.response.text[:200]}"


class Onboarder:
    def __init__(
        self,
        admin: BitbucketClient,
        bot: BitbucketClient,
        team: TeamConfig,
        webhook_url: str,
        *,
        webhook_name: str = DEFAULT_WEBHOOK_NAME,
        dry_run: bool = False,
        grant_bot: bool = False,
        prune: bool = True,
    ):
        self.admin = admin
        self.bot = bot
        self.team = team
        self.webhook_url = webhook_url
        self.webhook_name = webhook_name
        self.dry_run = dry_run
        self.grant_bot = grant_bot
        self.prune = prune
        # Hooks are matched by name AND instance: a hook named `noergler` that
        # points at another instance (intg next to prod) is reported, never
        # pruned or rewritten.
        self.instance_url = webhook_url.rsplit("/webhook/", 1)[0]

    # -- claim and access -- #
    async def claim(self, target: Target) -> Claim:
        kind = claim_kind(self.team, target.project)
        if target.repo is None:
            owned = kind == "whole"
        else:
            owned = self.team.owns(target.project, target.repo)
        bot_can_read = False
        if owned:
            try:
                if target.repo is None:
                    await self.bot.get_project(target.project)
                else:
                    await self.bot.get_repo(target.project, target.repo)
                bot_can_read = True
            except Exception as exc:  # noqa: BLE001 — any failure means "cannot read"
                logger.info("bot cannot read %s: %s", target.key, exc)
        return Claim(owned=owned, kind=kind, bot_can_read=bot_can_read)

    def _is_ours(self, hook: dict[str, Any]) -> bool:
        url = hook.get("url")
        return isinstance(url, str) and url.startswith(self.instance_url + "/")

    def _build_webhook_body(self) -> dict[str, Any]:
        return {
            "name": self.webhook_name,
            "url": self.webhook_url,
            "active": True,
            "events": list(REQUIRED_WEBHOOK_EVENTS),
            "configuration": {"secret": self.team.webhook_secret},
            "sslVerificationRequired": True,
        }

    def _diff_webhook(self, existing: dict[str, Any]) -> list[str]:
        diffs: list[str] = []
        if existing.get("url") != self.webhook_url:
            diffs.append(f"url: {existing.get('url')!r} -> {self.webhook_url!r}")
        existing_events = set(existing.get("events") or [])
        required = set(REQUIRED_WEBHOOK_EVENTS)
        if existing_events != required:
            missing = sorted(required - existing_events)
            extra = sorted(existing_events - required)
            diffs.append(f"events: missing={missing} extra={extra}")
        if not existing.get("active", True):
            diffs.append("active: False -> True")
        if not existing.get("sslVerificationRequired", True):
            diffs.append("sslVerificationRequired: False -> True")
        # Bitbucket never returns the stored secret, so only its absence is
        # visible; a secret-only change needs remove + onboard.
        if not (existing.get("configuration") or {}):
            diffs.append("configuration.secret: (unset) -> (set)")
        return diffs

    # -- building blocks -- #
    async def upsert_webhook(self, target: Target) -> tuple[int, list[str]]:
        """Create or update the webhook. Returns (webhook_id, diff)."""
        existing = _hook_named(
            await self.admin.list_webhooks(target.project, target.repo), self.webhook_name
        )
        if existing is not None and not self._is_ours(existing):
            raise ForeignHook(
                f"{self.webhook_name!r} hook points at another noergler ({existing.get('url')}); "
                "gone? remove it with that instance first, else use another name"
            )
        body = self._build_webhook_body()
        if existing is None:
            logger.info("[%s] creating webhook %r", target.key, self.webhook_name)
            if self.dry_run:
                return -1, ["create"]
            created = await self.admin.create_webhook(target.project, target.repo, body)
            return int(created["id"]), ["create"]

        diff = self._diff_webhook(existing)
        if not diff:
            logger.info("[%s] webhook already up to date", target.key)
            return int(existing["id"]), []
        logger.info("[%s] updating webhook id=%s changes=%s", target.key, existing.get("id"), diff)
        if not self.dry_run:
            await self.admin.update_webhook(target.project, target.repo, int(existing["id"]), body)
        return int(existing["id"]), diff

    async def stray_repo_hooks(self, project: str) -> tuple[list[tuple[Target, int]], list[str]]:
        """(this instance's repo-level hooks under a project, foreign same-named
        hooks). With a project webhook in place the former deliver every event
        a second time; the latter belong to another noergler and are left alone."""
        stray: list[tuple[Target, int]] = []
        foreign: list[str] = []
        slugs = [
            repo["slug"] for repo in await self.admin.list_repos(project)
            if isinstance(repo.get("slug"), str)
        ]
        # One listing per repo; a project has dozens, so a few in flight at a
        # time keeps the request short without hammering Bitbucket.
        gate = asyncio.Semaphore(4)

        async def _hooks(slug: str) -> list[dict[str, Any]]:
            async with gate:
                return await self.admin.list_webhooks(project, slug)

        for slug, hooks in zip(slugs, await asyncio.gather(*(_hooks(s) for s in slugs))):
            hook = _hook_named(hooks, self.webhook_name)
            if hook is None:
                continue
            repo_target = Target(project, slug)
            if self._is_ours(hook):
                stray.append((repo_target, int(hook["id"])))
            else:
                foreign.append(f"{repo_target.key} -> {hook.get('url')}")
        return stray, foreign

    async def prune_repo_hooks(self, project: str) -> list[str]:
        pruned: list[str] = []
        stray, foreign = await self.stray_repo_hooks(project)
        for entry in foreign:
            logger.warning("[%s] hook named %r points at another noergler, left alone", entry, self.webhook_name)
        for repo_target, hook_id in stray:
            if not self.dry_run:
                await self.admin.delete_webhook(project, repo_target.repo, hook_id)
                logger.info("[%s] deleted stray repo webhook id=%d", repo_target.key, hook_id)
            pruned.append(repo_target.key)
        return pruned

    async def _refuse_repo_target(self, target: Target, claim: Claim) -> str | None:
        """A repo target is refused when it would sit next to a project webhook
        of this instance: every event would be delivered twice (see prune)."""
        if target.is_project:
            return None
        if claim.kind == "whole":
            return _whole_claim_reason(target)
        try:
            hooks = await self.admin.list_webhooks(target.project, None)
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code in (401, 403):
                # no project admin on a shared project: nothing more to check
                return None
            raise
        project_hook = _hook_named(hooks, self.webhook_name)
        if project_hook is not None and self._is_ours(project_hook):
            return (
                f"{target.project} still has this instance's project webhook (id={project_hook.get('id')}), "
                f"which delivers every event as well; remove it first"
            )
        return None

    # -- orchestrators -- #
    async def status(self, target: Target) -> StatusRow:
        claim = await self.claim(target)
        stray: list[str] = []
        foreign: list[str] = []
        if not claim.owned:
            return StatusRow(target, False, False, _not_owned_reason(target, claim), stray, foreign)
        try:
            refused = await self._refuse_repo_target(target, claim)
        except httpx.HTTPStatusError as exc:
            return StatusRow(
                target, True, claim.bot_can_read,
                f"project hook check HTTP {exc.response.status_code}", stray, foreign,
            )
        if refused is not None:
            return StatusRow(target, True, claim.bot_can_read, f"blocked: {refused}", stray, foreign)
        try:
            existing = _hook_named(
                await self.admin.list_webhooks(target.project, target.repo), self.webhook_name
            )
            if existing is None:
                webhook = "missing"
            elif not self._is_ours(existing):
                webhook = f"foreign: {existing.get('url')}"
            else:
                diff = self._diff_webhook(existing)
                webhook = "ok" if not diff else "stale: " + "; ".join(diff)
        except httpx.HTTPStatusError as exc:
            webhook = f"HTTP {exc.response.status_code}"
        except httpx.HTTPError as exc:
            webhook = f"error: {exc}"
        if target.is_project and not webhook.startswith(("HTTP ", "error: ")):
            # The hook verdict above stands on its own; a failure here only
            # means the repo-level check is unknown, which is not healthy either.
            try:
                found, foreign = await self.stray_repo_hooks(target.project)
                stray = [t.key for t, _ in found]
            except httpx.HTTPStatusError as exc:
                webhook += f" (repo hooks unchecked: HTTP {exc.response.status_code})"
            except httpx.HTTPError as exc:
                webhook += f" (repo hooks unchecked: {exc})"
        return StatusRow(target, claim.owned, claim.bot_can_read, webhook, stray, foreign)

    async def onboard(self, target: Target) -> TargetResult:
        claim = await self.claim(target)
        if not claim.owned:
            return TargetResult(target, "skipped", detail=_not_owned_reason(target, claim))
        try:
            refused = await self._refuse_repo_target(target, claim)
        except httpx.HTTPStatusError as exc:
            return TargetResult(target, "failed", detail=f"project hook check {_http_detail(exc)}")
        if refused is not None:
            return TargetResult(target, "skipped", detail=refused)

        notes: list[str] = []
        bot = self.bot.bot_username
        if not claim.bot_can_read:
            if not self.grant_bot:
                return TargetResult(
                    target, "skipped",
                    detail=f"{bot} cannot read it; grant {bot} {target.bot_permission} "
                           f"in Bitbucket or run grant-bot",
                )
            if not self.dry_run:
                try:
                    await self.admin.grant_user_permission(
                        target.project, target.repo, bot, target.bot_permission
                    )
                except httpx.HTTPStatusError as exc:
                    return TargetResult(
                        target, "failed",
                        detail=f"grant {target.bot_permission} to {bot}: {_http_detail(exc)}",
                    )
                logger.info("[%s] granted %s %s", target.key, bot, target.bot_permission)
            notes.append(f"{bot} granted {target.bot_permission}")

        try:
            webhook_id, diff = await self.upsert_webhook(target)
        except ForeignHook as exc:
            return TargetResult(target, "failed", detail=str(exc))
        except httpx.HTTPStatusError as exc:
            return TargetResult(target, "failed", detail=f"upsert webhook {_http_detail(exc)}")
        except httpx.HTTPError as exc:
            return TargetResult(target, "failed", detail=f"upsert webhook: {exc}")

        if diff == ["create"]:
            notes.insert(0, f"webhook created (id={webhook_id})")
        elif diff:
            notes.insert(0, f"webhook updated (id={webhook_id})")
        else:
            notes.insert(0, "webhook already up to date")

        if target.is_project and self.prune:
            try:
                pruned = await self.prune_repo_hooks(target.project)
            except httpx.HTTPError as exc:
                notes.append(f"prune of repo hooks failed: {exc}")
            else:
                if pruned:
                    notes.append(f"pruned {len(pruned)} repo hook(s): {', '.join(pruned)}")

        prefix = "dry-run: " if self.dry_run else ""
        return TargetResult(target, "ok", detail=prefix + ", ".join(notes), diff=diff)

    async def remove(self, target: Target) -> TargetResult:
        """Delete this instance's webhook from the target. No-op if absent."""
        try:
            hooks = await self.admin.list_webhooks(target.project, target.repo)
        except httpx.HTTPStatusError as exc:
            return TargetResult(target, "failed", detail=f"list webhooks {_http_detail(exc)}")
        except httpx.HTTPError as exc:
            return TargetResult(target, "failed", detail=f"list webhooks: {exc}")

        existing = _hook_named(hooks, self.webhook_name)
        if existing is None:
            return TargetResult(target, "skipped", detail=f"no {self.webhook_name!r} webhook found")
        if not self._is_ours(existing):
            return TargetResult(
                target, "skipped",
                detail=f"{self.webhook_name!r} webhook points at another noergler ({existing.get('url')}), left alone",
            )
        webhook_id = int(existing["id"])
        if self.dry_run:
            return TargetResult(target, "ok", detail=f"dry-run: would remove webhook id={webhook_id}")
        try:
            await self.admin.delete_webhook(target.project, target.repo, webhook_id)
        except httpx.HTTPStatusError as exc:
            return TargetResult(target, "failed", detail=f"delete webhook {_http_detail(exc)}")
        except httpx.HTTPError as exc:
            return TargetResult(target, "failed", detail=f"delete webhook: {exc}")
        logger.info("[%s] removed webhook id=%d", target.key, webhook_id)
        return TargetResult(target, "ok", detail=f"webhook removed: id={webhook_id}")



# -- rendering -- #
def render_status(rows: list[StatusRow]) -> str:
    width = max((len(r.target.label) for r in rows), default=10)
    header = f"{'target'.ljust(width)}  owned  bot   webhook"
    lines = [header, "-" * len(header)]
    for r in rows:
        owned = "yes" if r.owned else "no"
        bot = ("yes" if r.bot_can_read else "no") if r.owned else "-"
        line = f"{r.target.label.ljust(width)}  {owned.ljust(5)}  {bot.ljust(4)}  {r.webhook}"
        if r.stray:
            line += f"  stray repo hooks: {', '.join(r.stray)}"
        if r.foreign:
            line += f"  foreign hooks: {', '.join(r.foreign)}"
        lines.append(line)
    return "\n".join(lines)


def render_results(results: list[TargetResult]) -> str:
    width = max((len(r.target.label) for r in results), default=10)
    header = f"{'target'.ljust(width)}  status   detail"
    lines = [header, "-" * len(header)]
    for r in results:
        lines.append(f"{r.target.label.ljust(width)}  {r.status.ljust(7)}  {r.detail}")
    return "\n".join(lines)


def status_healthy(rows: list[StatusRow]) -> bool:
    return all(r.owned and r.bot_can_read and r.webhook == "ok" and not r.stray for r in rows)


def results_healthy(results: list[TargetResult]) -> bool:
    return all(r.status != "failed" for r in results)


async def run(
    onboarder: Onboarder, action: Action, targets: list[Target]
) -> tuple[list[StatusRow] | list[TargetResult], str, bool]:
    """Run `action` over `targets`, one failure never aborting the rest.
    Returns (rows, table text, healthy)."""
    if action == "status":
        rows: list[StatusRow] = []
        for target in targets:
            try:
                rows.append(await onboarder.status(target))
            except Exception as exc:  # noqa: BLE001 — per-target isolation
                logger.exception("[%s] unexpected error", target.key)
                rows.append(StatusRow(target, False, False, f"error: {exc}", [], []))
        return rows, render_status(rows), status_healthy(rows)

    step = onboarder.remove if action == "remove" else onboarder.onboard
    results: list[TargetResult] = []
    for target in targets:
        try:
            results.append(await step(target))
        except Exception as exc:  # noqa: BLE001 — per-target isolation
            logger.exception("[%s] unexpected error", target.key)
            results.append(TargetResult(target, "failed", detail=f"unexpected error: {exc}"))
    return results, render_results(results), results_healthy(results)
