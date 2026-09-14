"""What a team changes on its own, kept in the DB: project/repo claims and the
review author lists. `teams.yaml` only seeds an empty DB (see `main.lifespan`).

Ownership guarantee: `team_claims` holds a project or a repo for exactly one
team (unique indexes, migration 001). The one overlap the indexes cannot
express, a whole-project claim against another team's repo claims on that
project, is checked here inside a transaction that locks the project's rows.
"""

from __future__ import annotations

import fnmatch
import logging
from dataclasses import dataclass, field
from typing import Any

import asyncpg

from app.config import ProjectScope

logger = logging.getLogger(__name__)


@dataclass
class ClaimConflict(Exception):
    """The target is held by another team; nothing was written."""

    project: str
    repo: str | None
    other_team: str

    def __str__(self) -> str:
        target = self.project if self.repo is None else f"{self.project}/{self.repo}"
        return f"{target} is claimed by team {self.other_team}"


@dataclass(frozen=True)
class TeamSettings:
    auto_review_authors: list[str]
    ignore_authors: list[str]
    exclude_repos: list[str] = field(default_factory=list)

    def excludes(self, repo_slug: str) -> bool:
        """Case-insensitive glob match of a repo slug against `exclude_repos`."""
        return excludes_repo(self.exclude_repos, repo_slug)


def excludes_repo(patterns: list[str], repo_slug: str) -> bool:
    slug = repo_slug.lower()
    return any(fnmatch.fnmatchcase(slug, p.lower()) for p in patterns)


def _scopes_from_rows(rows: list[asyncpg.Record]) -> list[ProjectScope]:
    """Rows (project_key, repo_slug) -> ProjectScope list, whole projects first,
    repo claims grouped per project, both in first-claimed order."""
    whole: list[str] = []
    repos: dict[str, list[str]] = {}
    for row in rows:
        if row["repo_slug"] is None:
            if row["project_key"] not in whole:
                whole.append(row["project_key"])
        else:
            repos.setdefault(row["project_key"], []).append(row["repo_slug"])
    scopes = [ProjectScope(key=k) for k in whole]
    scopes.extend(ProjectScope(key=k, repos=v) for k, v in repos.items() if k not in whole)
    return scopes


async def list_claims(pool: asyncpg.Pool, team_slug: str) -> list[ProjectScope]:
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            "SELECT project_key, repo_slug FROM team_claims WHERE team_slug = $1 ORDER BY id",
            team_slug,
        )
    return _scopes_from_rows(rows)


async def list_all_claims(pool: asyncpg.Pool) -> dict[str, list[ProjectScope]]:
    async with pool.acquire() as conn:
        rows = await conn.fetch("SELECT team_slug, project_key, repo_slug FROM team_claims ORDER BY id")
    by_team: dict[str, list[asyncpg.Record]] = {}
    for row in rows:
        by_team.setdefault(row["team_slug"], []).append(row)
    return {slug: _scopes_from_rows(team_rows) for slug, team_rows in by_team.items()}


async def add_claims(
    pool: asyncpg.Pool, team_slug: str, scopes: list[ProjectScope], claimed_by: str
) -> list[str]:
    """Claim `scopes` for the team. All or nothing: a conflict with another
    team raises `ClaimConflict` and writes nothing. Returns the targets that
    are new (`KEY` or `KEY/repo`); a repo already covered by the team's own
    whole-project claim is not new, a whole-project claim replaces the team's
    own repo claims on that project."""
    added: list[str] = []
    try:
        async with pool.acquire() as conn:
            async with conn.transaction():
                await _add_claims(conn, team_slug, scopes, claimed_by, added)
    except asyncpg.UniqueViolationError as exc:
        # Two first claims on the same project raced past the FOR UPDATE (no
        # rows to lock yet); the index decided, the loser sees a conflict.
        added.clear()
        raise ClaimConflict(scopes[0].key, None, "another team (concurrent claim)") from exc
    if added:
        logger.info("claims added team=%s by=%s targets=%s", team_slug, claimed_by, added)
    return added


async def _add_claims(
    conn: Any, team_slug: str, scopes: list[ProjectScope], claimed_by: str, added: list[str]
) -> None:
    for scope in scopes:
        rows = await conn.fetch(
            "SELECT id, team_slug, repo_slug FROM team_claims WHERE project_key = $1 FOR UPDATE",
            scope.key,
        )
        if scope.repos is None:
            for row in rows:
                if row["team_slug"] != team_slug:
                    raise ClaimConflict(scope.key, row["repo_slug"], row["team_slug"])
            if any(row["repo_slug"] is None for row in rows):
                continue  # already ours as a whole
            # our repo claims on this project are covered by the project claim now
            await conn.execute(
                "DELETE FROM team_claims WHERE project_key = $1 AND team_slug = $2", scope.key, team_slug
            )
            await conn.execute(
                "INSERT INTO team_claims (team_slug, project_key, repo_slug, claimed_by) VALUES ($1, $2, NULL, $3)",
                team_slug, scope.key, claimed_by,
            )
            added.append(scope.key)
            continue
        whole = next((row for row in rows if row["repo_slug"] is None), None)
        if whole is not None and whole["team_slug"] != team_slug:
            raise ClaimConflict(scope.key, None, whole["team_slug"])
        if whole is not None:
            continue  # our whole-project claim already covers every repo
        held = {row["repo_slug"]: row["team_slug"] for row in rows}
        for repo in scope.repos:
            owner = held.get(repo)
            if owner is not None and owner != team_slug:
                raise ClaimConflict(scope.key, repo, owner)
            if owner == team_slug:
                continue
            await conn.execute(
                "INSERT INTO team_claims (team_slug, project_key, repo_slug, claimed_by) VALUES ($1, $2, $3, $4)",
                team_slug, scope.key, repo, claimed_by,
            )
            added.append(f"{scope.key}/{repo}")


async def remove_claims(pool: asyncpg.Pool, team_slug: str, scopes: list[ProjectScope]) -> list[str]:
    """Drop the team's claims named in `scopes`. A whole-project scope drops
    the project claim and any repo claims of the team on it. Returns what
    was dropped."""
    removed: list[str] = []
    async with pool.acquire() as conn:
        async with conn.transaction():
            for scope in scopes:
                if scope.repos is None:
                    rows = await conn.fetch(
                        "DELETE FROM team_claims WHERE team_slug = $1 AND project_key = $2 RETURNING repo_slug",
                        team_slug, scope.key,
                    )
                    removed.extend(
                        scope.key if row["repo_slug"] is None else f"{scope.key}/{row['repo_slug']}" for row in rows
                    )
                    continue
                for repo in scope.repos:
                    rows = await conn.fetch(
                        "DELETE FROM team_claims WHERE team_slug = $1 AND project_key = $2 AND repo_slug = $3 RETURNING id",
                        team_slug, scope.key, repo,
                    )
                    if rows:
                        removed.append(f"{scope.key}/{repo}")
    if removed:
        logger.info("claims removed team=%s targets=%s", team_slug, removed)
    return removed


async def purge_project(pool: asyncpg.Pool, team_slug: str, project_key: str, repo_slug: str | None) -> int:
    """Delete every PR record the team holds on the target (findings cascade).
    The `team_slug` guard keeps a team from purging data another team wrote
    on a project that changed hands. Returns the number of PRs dropped."""
    async with pool.acquire() as conn:
        if repo_slug is None:
            rows = await conn.fetch(
                "DELETE FROM pr_reviews WHERE team_slug = $1 AND project_key = $2 RETURNING id",
                team_slug, project_key,
            )
        else:
            rows = await conn.fetch(
                "DELETE FROM pr_reviews WHERE team_slug = $1 AND project_key = $2 AND repo_slug = $3 RETURNING id",
                team_slug, project_key, repo_slug,
            )
    if rows:
        logger.info("purged %d PR record(s) team=%s target=%s/%s", len(rows), team_slug, project_key, repo_slug or "*")
    return len(rows)


async def count_project_prs(pool: asyncpg.Pool, team_slug: str, project_key: str, repo_slug: str | None) -> int:
    """What `purge_project` would drop (for dry runs)."""
    async with pool.acquire() as conn:
        if repo_slug is None:
            row = await conn.fetchrow(
                "SELECT COUNT(*) AS n FROM pr_reviews WHERE team_slug = $1 AND project_key = $2",
                team_slug, project_key,
            )
        else:
            row = await conn.fetchrow(
                "SELECT COUNT(*) AS n FROM pr_reviews WHERE team_slug = $1 AND project_key = $2 AND repo_slug = $3",
                team_slug, project_key, repo_slug,
            )
    return int(row["n"]) if row else 0


async def get_settings(pool: asyncpg.Pool, team_slug: str) -> TeamSettings | None:
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT auto_review_authors, ignore_authors, exclude_repos FROM team_settings WHERE team_slug = $1",
            team_slug,
        )
    if row is None:
        return None
    return TeamSettings(list(row["auto_review_authors"]), list(row["ignore_authors"]), list(row["exclude_repos"]))


async def get_all_settings(pool: asyncpg.Pool) -> dict[str, TeamSettings]:
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            "SELECT team_slug, auto_review_authors, ignore_authors, exclude_repos FROM team_settings"
        )
    return {
        row["team_slug"]: TeamSettings(
            list(row["auto_review_authors"]), list(row["ignore_authors"]), list(row["exclude_repos"])
        )
        for row in rows
    }


async def put_settings(
    pool: asyncpg.Pool, team_slug: str, settings: TeamSettings, updated_by: str
) -> None:
    async with pool.acquire() as conn:
        await conn.execute(
            """
            INSERT INTO team_settings (team_slug, auto_review_authors, ignore_authors, exclude_repos, updated_by, updated_at)
            VALUES ($1, $2, $3, $4, $5, NOW())
            ON CONFLICT (team_slug) DO UPDATE SET
                auto_review_authors = EXCLUDED.auto_review_authors,
                ignore_authors = EXCLUDED.ignore_authors,
                exclude_repos = EXCLUDED.exclude_repos,
                updated_by = EXCLUDED.updated_by,
                updated_at = NOW()
            """,
            team_slug, settings.auto_review_authors, settings.ignore_authors, settings.exclude_repos, updated_by,
        )
    logger.info(
        "settings updated team=%s by=%s auto_review_authors=%s ignore_authors=%s exclude_repos=%s",
        team_slug, updated_by, settings.auto_review_authors, settings.ignore_authors, settings.exclude_repos,
    )
