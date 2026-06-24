import asyncio
import json
import logging
import re
import time
from collections.abc import Awaitable
from datetime import datetime, timezone
from pathlib import PurePosixPath
from typing import Any, Literal, TypeVar, final

import asyncpg
import httpx
import openai
import structlog

from app.bitbucket import BitbucketClient, IncrementalDiffUnavailable
from app.db import repository
from app.feedback import classify_feedback, disagree_response
from app.markdown_format import wrap_prose
from app.llm_client import (
    INFERENCE_HARD_TIMEOUT_SECONDS,
    LLMClient,
    FileReviewData,
    ReviewSummary,
    count_tokens,
    extract_path,
    format_file_entry,
    is_deleted,
    is_reviewable_diff,
    render_previously_posted_findings,
    split_by_file,
)
from app.config import ReviewConfig, ServerConfig, estimate_cost_usd, model_label
from app.http_stats import HttpScope, enter_http_scope, exit_http_scope, summarize
from app.riptide_client import RiptideClient
from app.context_expansion import expand_all_files
from app.cross_file_context import build_cross_file_context, render_cross_file_context
from app.diff_compression import compress_for_large_pr, is_small_pr, sort_files_by_language_priority
from app.jira import JiraClient, JiraTicket
from app.models import PullRequest, ReviewFinding, WebhookPayload

logger = logging.getLogger(__name__)


# Curated 2026 references surfaced in the "AGENTS.md too large" skip summary.
# Publication dates verified via WebFetch (2026-05); replace if the URLs rot or
# better-dated sources appear.
_AGENTS_MD_FURTHER_READING: tuple[tuple[str, str], ...] = (
    (
        "Upsun — The research is in: your AGENTS.md is probably too long (2026-02-23)",
        "https://developer.upsun.com/posts/ai/agents-md-less-is-more",
    ),
    (
        "Augment Code — A good AGENTS.md is a model upgrade. A bad one is "
        "worse than no docs at all (2026-04-22)",
        "https://www.augmentcode.com/blog/how-to-write-good-agents-dot-md-files",
    ),
    (
        "Caveman — Claude Code skill, \"why use many token when few token do trick\"",
        "https://github.com/juliusbrussee/caveman",
    ),
)

# Markdown link syntax: [Title](URL) — captured as (title, url). The URL group
# is greedy so URLs containing `)` (e.g. Wikipedia-style `Foo_(bar)`) still
# match: anchoring to end-of-string forces the final `)` to be the closer.
_MARKDOWN_LINK_RE = re.compile(r"^\[([^\]]+)\]\((.+)\)$")


def _parse_custom_link(raw: str) -> tuple[str, str] | None:
    """Parse `[Title](URL)` markdown or a bare URL. Returns None if empty."""
    value = raw.strip()
    if not value:
        return None
    m = _MARKDOWN_LINK_RE.match(value)
    if m:
        return (m.group(1).strip(), m.group(2).strip())
    return (value, value)


def _fmt(n: int) -> str:
    return f"{n:,}".replace(",", "'")


def _fmt_k(n: int) -> str:
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M".replace(".0M", "M")
    return f"{round(n / 1000)}k"


def _epoch_ms_to_datetime(epoch_ms: int | None) -> datetime | None:
    if epoch_ms is None:
        return None
    return datetime.fromtimestamp(epoch_ms / 1000, tz=timezone.utc)

SEVERITY_ORDER = {"issue": 0, "suggestion": 1}

# Sentinel marker for the staleness banner we prepend to an existing summary
# comment when a re-review times out. Re-runs detect this and replace the
# previous banner instead of stacking new ones.
_STALE_BANNER_SENTINEL = "<!-- noergler-stale-banner -->"

# Visible prefix of the banner. Used as a defensive fallback in case
# Bitbucket's markdown renderer strips the HTML comment sentinel before
# returning the comment body — without this fallback, banners would stack on
# repeated timeouts. The phrase is stable across the two banner variants
# (with/without prior commit sha).
_STALE_BANNER_VISIBLE_PREFIX = "⚠️ No response from the model within"


def _strip_stale_banner(body: str) -> str:
    """If `body` starts with our staleness banner block, return the body
    without it. Idempotent — safe to call on bodies that don't have one.

    Banner format we write:
        <!-- noergler-stale-banner -->
        ⚠️ No response from the model within ...one-line message...
        <blank line>
        <original body>

    Detection priority: sentinel first; if that's missing (Bitbucket may
    strip HTML comments through its renderer), fall back to the visible
    "⚠️ No response from the model within" prefix on the first content line.
    Strips everything up to and including the first blank line that follows.
    """
    has_sentinel = body.startswith(_STALE_BANNER_SENTINEL)
    if not has_sentinel:
        # Skip leading blank lines, then check the first non-blank.
        lines_iter = iter(body.splitlines())
        first_content = ""
        for line in lines_iter:
            if line.strip():
                first_content = line
                break
        if not first_content.startswith(_STALE_BANNER_VISIBLE_PREFIX):
            return body

    lines = body.splitlines()
    # Find the first non-blank line (the visible banner row, possibly
    # preceded by the sentinel).
    i = 0
    if has_sentinel:
        i = 1
    # Skip the visible banner line itself, then skip until first blank line.
    while i < len(lines) and lines[i].strip() != "":
        i += 1
    while i < len(lines) and lines[i].strip() == "":
        i += 1
    return "\n".join(lines[i:])


# Prepended to the summary comment whenever a PR's accumulated cost is at/over
# the per-PR limit. Like the stale banner, it is sentinel-marked so re-runs
# replace it instead of stacking. The visible prefix is stable across both
# variants (blocked vs. completed-review) so it survives Bitbucket stripping the
# HTML comment through its markdown renderer.
_COST_BANNER_SENTINEL = "<!-- noergler-cost-banner -->"
_COST_BANNER_VISIBLE_PREFIX = "⚠️ **Cost limit exceeded**"


def _strip_cost_banner(body: str) -> str:
    """Remove a leading cost-limit banner block, if present. Idempotent.

    Same detection/stripping strategy as `_strip_stale_banner`: sentinel first,
    visible-prefix fallback, then strip up to and including the first blank line.
    """
    has_sentinel = body.startswith(_COST_BANNER_SENTINEL)
    if not has_sentinel:
        first_content = ""
        for line in body.splitlines():
            if line.strip():
                first_content = line
                break
        if not first_content.startswith(_COST_BANNER_VISIBLE_PREFIX):
            return body

    lines = body.splitlines()
    i = 1 if has_sentinel else 0
    while i < len(lines) and lines[i].strip() != "":
        i += 1
    while i < len(lines) and lines[i].strip() == "":
        i += 1
    return "\n".join(lines[i:])


def _cost_limit_banner(
    cumulative_cost_usd: float,
    limit_usd: float,
    bot_username: str,
    *,
    blocked: bool,
) -> str:
    """Build the sentinel-marked over-limit banner block (no trailing newline).

    `blocked=True` is used when an automatic review was skipped entirely (the
    pushed commit was not reviewed); `blocked=False` is used on a completed run
    (an auto-run that overshot the cap, or a manual @mention review while over).
    Swiss orthography — no ß.
    """
    msg = (
        f"{_COST_BANNER_VISIBLE_PREFIX} — PR total **${cumulative_cost_usd:.2f}**, "
        f"over the **${limit_usd:.2f}** limit. Automatic reviews are paused. "
        f"`@{bot_username}` to review manually, or ask an admin to raise "
        f"`REVIEW_MAX_PR_COST_USD`."
    )
    if blocked:
        msg += " This push was **not** reviewed automatically."
    return f"{_COST_BANNER_SENTINEL}\n{msg}"


# Hard ceiling for the cumulative-diff context, regardless of model. Beyond this
# the LLM tends to drown the focused review in noise.
MAX_CUMULATIVE_CONTEXT_TOKENS = 80_000
MAX_PREVIOUSLY_POSTED_FINDINGS = 50


def _cumulative_diff_budget(max_tokens_per_chunk: int) -> int:
    """Token budget for the cumulative PR diff. ~1/3 of the per-chunk budget so
    the focused review files still fit, capped at the hard ceiling."""
    return min(MAX_CUMULATIVE_CONTEXT_TOKENS, max(2_000, max_tokens_per_chunk // 3))


# Previously-posted findings are bounded both by count and by rendered token size:
# count guards prompt latency, tokens guard against a few very long comments
# inflating the prompt unboundedly.
MAX_PREVIOUSLY_POSTED_FINDINGS_TOKENS = 4_000


def _previously_posted_findings_budget(max_tokens_per_chunk: int) -> int:
    # ~5% of chunk budget: this block is cross-context, not the focus of review;
    # keep it tight so the focused files dominate the prompt.
    return min(MAX_PREVIOUSLY_POSTED_FINDINGS_TOKENS, max(500, max_tokens_per_chunk // 20))
_REVIEW_KEYWORDS = {"review", "review this", "re-review", "rereview"}
_JIRA_TICKET_RE = re.compile(r'\b([A-Z]{2,10}-\d{1,7})\b')
_SECURITY_KEYWORDS = re.compile(
    r"\b(injection|xss|sql[_ -]?injection|authentication|authorization|"
    r"credentials?|secret[s ]?leak|csrf|ssrf|"
    r"path[_ -]?traversal|insecure|vulnerability|sanitiz(?:e|ation)|"
    r"privilege[_ -]?escalation|deserialization|token[_ -]?leak|"
    r"exposed?[_ -]?(?:secret|credential|key|token))\b",
    re.IGNORECASE,
)


_T = TypeVar("_T")


async def _safe_db(coro: Awaitable[_T], fallback: _T | None = None) -> _T | None:
    """Wrap a DB coroutine; on failure log a warning and return the fallback value."""
    try:
        return await coro
    except Exception:
        logger.warning("DB operation failed, using fallback", exc_info=True)
        return fallback


def _is_bot_comment(comment: dict[str, Any], bot_username: str | None) -> bool:
    """Identify a bot comment by author slug."""
    return bool(bot_username) and comment.get("author_slug") == bot_username


def _count_diff_lines(diff: str) -> tuple[int, int]:
    """Count added and removed lines in a unified diff.

    Skips files the reviewer ignores (lock files, generated/minified files,
    binaries, build output, …) so the count reflects real code changes
    rather than reformatted JSON or vendored bundles.
    """
    added = 0
    removed = 0
    for file_diff in split_by_file(diff):
        if not is_reviewable_diff(file_diff):
            continue
        for line in file_diff.splitlines():
            if line.startswith("+") and not line.startswith("+++"):
                added += 1
            elif line.startswith("-") and not line.startswith("---"):
                removed += 1
    return added, removed


def _extract_question(text: str, trigger: str) -> str:
    return re.sub(rf"@{re.escape(trigger)}\b", "", text, flags=re.IGNORECASE).strip()


@final
class Reviewer:
    def __init__(
        self,
        bitbucket: BitbucketClient,
        llm: LLMClient,
        review_config: ReviewConfig,
        jira: JiraClient | None = None,
        server_config: ServerConfig | None = None,
        *,
        db_pool: asyncpg.Pool | None,
        riptide: RiptideClient | None = None,
    ):
        self.bitbucket = bitbucket
        self.llm = llm
        self.review_config = review_config
        self.jira = jira
        self.server_config = server_config or ServerConfig()
        if db_pool is None:
            raise ValueError("db_pool is required")
        self.db_pool = db_pool
        self.auto_review_authors = review_config.auto_review_authors
        self.max_comments = review_config.max_comments
        self.mention_trigger = bitbucket.bot_username
        self.riptide = riptide

    @staticmethod
    def _extract_ticket_id(pr: PullRequest) -> str | None:
        match = _JIRA_TICKET_RE.search(pr.fromRef.displayId)
        if match:
            return match.group(1)
        match = _JIRA_TICKET_RE.search(pr.title)
        if match:
            return match.group(1)
        return None

    @staticmethod
    def _format_ticket_block(ticket: JiraTicket, heading: str) -> list[str]:
        lines = [f"### {heading}: [{ticket.key}]({ticket.url})"]
        lines.append(f"**Title:** {ticket.title}")
        if ticket.issue_type or ticket.status:
            type_status_parts = []
            if ticket.issue_type:
                type_status_parts.append(f"**Type:** {ticket.issue_type}")
            if ticket.status:
                type_status_parts.append(f"**Status:** {ticket.status}")
            lines.append(" · ".join(type_status_parts))
        if ticket.description:
            lines.append(f"**Description:** {ticket.description}")
        if ticket.labels:
            lines.append(f"**Labels:** {', '.join(ticket.labels)}")
        if ticket.acceptance_criteria:
            lines.append(f"**Acceptance criteria:** {ticket.acceptance_criteria}")
        if ticket.subtasks:
            lines.append("**Subtasks:**")
            for st in ticket.subtasks:
                lines.append(f"- {st}")
        return lines

    async def _fetch_ticket_context(
        self, ticket_id: str, parent: JiraTicket | None = None,
    ) -> tuple[str, JiraTicket | None]:
        if not self.jira:
            return "", None
        ticket = await self.jira.fetch_ticket(ticket_id)
        if not ticket:
            return "", None
        blocks: list[str] = []
        if parent:
            blocks.extend(self._format_ticket_block(parent, "Parent ticket"))
            blocks.append("")
            blocks.extend(self._format_ticket_block(ticket, "Sub-task"))
        else:
            blocks.extend(self._format_ticket_block(ticket, "Jira ticket"))
        return "\n".join(blocks), ticket

    def is_auto_review_author(self, author_name: str) -> bool:
        return not self.auto_review_authors or author_name in self.auto_review_authors

    async def _prepare_files(
        self,
        project_key: str,
        repo_slug: str,
        diff: str,
        source_commit: str | None,
        pr_tag: str,
    ) -> tuple[list[FileReviewData], list[str]]:
        all_file_diffs = split_by_file(diff)
        file_diffs = [fd for fd in all_file_diffs if is_reviewable_diff(fd)]
        skipped = len(all_file_diffs) - len(file_diffs)
        logger.info(
            "%s: %d file(s) in diff, %d reviewable, %d skipped (binary/non-reviewable)",
            pr_tag, len(all_file_diffs), len(file_diffs), skipped,
        )
        if not file_diffs:
            return [], []

        content_skipped: list[str] = []
        max_file_lines = self.review_config.max_file_lines

        async def _build_file_data(file_diff: str) -> FileReviewData | None:
            path = extract_path(file_diff)
            if not path:
                first_line = file_diff.split("\n", 1)[0][:200]
                logger.warning("Could not extract path from diff chunk: %r", first_line)
                return None
            deleted = is_deleted(file_diff)
            content = None
            if not deleted and source_commit:
                try:
                    content = await self.bitbucket.fetch_file_content(
                        project_key, repo_slug, source_commit, path
                    )
                except Exception:
                    logger.warning("Failed to fetch content for %s, using diff only", path)
            if content and content.count("\n") + 1 > max_file_lines:
                logger.info("Skipping full content for %s (%d lines > limit %d)", path, content.count("\n") + 1, max_file_lines)
                content = None
                content_skipped.append(path)
            if content:
                logger.info("Including %s with full file content (%d lines)", path, content.count("\n") + 1)
            return FileReviewData(path=path, diff=file_diff, content=content)

        results = await asyncio.gather(*[_build_file_data(fd) for fd in file_diffs])
        files = [f for f in results if f is not None]
        # Stable ordering across re-reviews — keeps unchanged files in the LLM prefix cache.
        files = sort_files_by_language_priority(files)

        total_diff_lines = sum(f.diff.count("\n") + 1 for f in files)
        total_content_lines = sum(f.content.count("\n") + 1 for f in files if f.content)
        files_with_content = sum(1 for f in files if f.content)
        files_diff_only = len(files) - files_with_content
        logger.info(
            "%s: %d file(s) for review — %d diff lines, %d content lines, "
            "%d with full content, %d diff-only",
            pr_tag, len(files), total_diff_lines, total_content_lines,
            files_with_content, files_diff_only,
        )

        return files, content_skipped

    async def review_pull_request(self, payload: WebhookPayload, *, skip_author_check: bool = False) -> None:
        pr_tag = "unknown"
        # Bind pr identity to structlog contextvars so every child log
        # inherits pr_tag/repo and Splunk can filter all activity for one
        # PR with a single query.
        structlog.contextvars.bind_contextvars(pr_tag=pr_tag)
        http_scope: HttpScope | None = None
        try:
            http_scope = enter_http_scope()
            pr = payload.pullRequest
            author_name = pr.author.user.name
            pr_id = pr.id

            project_key, repo_slug = self._extract_project_repo(payload)
            if not project_key or not repo_slug:
                logger.error("Could not extract project/repo from webhook payload")
                return

            pr_tag = f"{project_key}/{repo_slug}#{pr_id}"
            structlog.contextvars.bind_contextvars(
                pr_tag=pr_tag, repo=f"{project_key}/{repo_slug}", pr_id=pr_id
            )
            opened_at = _epoch_ms_to_datetime(pr.createdDate)

            if not skip_author_check and not self.is_auto_review_author(author_name):
                logger.info(
                    "Skipping %s by %s (not in auto-review authors)", pr_tag, author_name
                )
                return

            # If the user removed our summary comment, they want noergler to
            # leave this PR alone. This is the backstop for the pr:comment:deleted
            # webhook (handle_comment_deleted): the webhook marks `ignored_at`
            # immediately, this short-circuits cheaply on it, and the 404 check
            # below catches deletions on repos not yet subscribed to that event.
            # Runs before any path that posts a comment (opt-out / AGENTS.md early
            # returns call _post_or_update_summary). An @mention clears the skip
            # state (see handle_mention), so a reactivated PR falls through here.
            skip_state = await _safe_db(
                repository.get_pr_skip_state(self.db_pool, project_key, repo_slug, pr_id),
                fallback=None,
            )
            if skip_state:
                if skip_state["ignored_at"] is not None:
                    logger.info(
                        "%s: PR ignored (summary comment removed) — skipping", pr_tag
                    )
                    return
                summary_comment_id = skip_state["summary_comment_id"]
                if summary_comment_id is not None:
                    try:
                        await self.bitbucket.fetch_pr_comment(
                            project_key, repo_slug, pr_id, summary_comment_id
                        )
                    except httpx.HTTPStatusError as exc:
                        if exc.response.status_code == 404:
                            await _safe_db(
                                repository.mark_pr_ignored(
                                    self.db_pool, project_key, repo_slug, pr_id
                                )
                            )
                            logger.info(
                                "%s: summary comment %d was deleted — "
                                "ignoring PR from now on",
                                pr_tag, summary_comment_id,
                            )
                            return
                        # Any other HTTP status is not a clear "deleted" signal —
                        # do not silence a live PR; fall through to the review.
                        logger.warning(
                            "%s: could not verify summary comment %d (HTTP %d) — proceeding",
                            pr_tag, summary_comment_id, exc.response.status_code,
                        )
                    except Exception:
                        # Network / timeout / 5xx: never mark ignored on a
                        # transient failure; fall through to the review.
                        logger.warning(
                            "%s: summary-comment check failed — proceeding",
                            pr_tag, exc_info=True,
                        )

            opt_out_keyword = self.review_config.opt_out_branch_keyword
            branch_name = pr.fromRef.displayId
            if opt_out_keyword and opt_out_keyword.lower() in branch_name.lower():
                logger.info(
                    "%s: branch %r contains opt-out keyword %r, skipping review",
                    pr_tag, branch_name, opt_out_keyword,
                )
                pr_review_id = await _safe_db(
                    repository.upsert_pr_review(
                        self.db_pool, project_key, repo_slug, pr_id,
                        last_reviewed_commit=pr.fromRef.latestCommit,
                        author=author_name,
                        pr_title=pr.title,
                        opened_at=opened_at,
                    ),
                    fallback=None,
                )
                await self._post_or_update_summary(
                    project_key, repo_slug, pr_id, pr_review_id,
                    self._build_opt_out_branch_summary(opt_out_keyword, branch_name),
                )
                return

            repo_instructions = await self._fetch_repo_instructions(
                project_key, repo_slug, pr
            )
            if not repo_instructions and self.review_config.require_agents_md:
                logger.info(
                    "%s: no AGENTS.md found on PR or target branch, skipping review "
                    "(set REVIEW_REQUIRE_AGENTS_MD=false to override)",
                    pr_tag,
                )
                pr_review_id = await _safe_db(
                    repository.upsert_pr_review(
                        self.db_pool, project_key, repo_slug, pr_id,
                        last_reviewed_commit=pr.fromRef.latestCommit,
                        author=author_name,
                        pr_title=pr.title,
                        opened_at=opened_at,
                    ),
                    fallback=None,
                )
                await self._post_or_update_summary(
                    project_key, repo_slug, pr_id, pr_review_id,
                    self._build_agents_md_missing_summary(),
                )
                return

            max_tokens = self.review_config.agents_md_max_tokens
            if repo_instructions and max_tokens > 0:
                agents_md_tokens = count_tokens(repo_instructions)
                if agents_md_tokens > max_tokens:
                    logger.info(
                        "%s: AGENTS.md too large (%d > %d tokens), skipping review "
                        "(set REVIEW_AGENTS_MD_MAX_TOKENS=0 to disable the hard limit)",
                        pr_tag, agents_md_tokens, max_tokens,
                    )
                    pr_review_id = await _safe_db(
                        repository.upsert_pr_review(
                            self.db_pool, project_key, repo_slug, pr_id,
                            last_reviewed_commit=pr.fromRef.latestCommit,
                            author=author_name,
                            pr_title=pr.title,
                            opened_at=opened_at,
                        ),
                        fallback=None,
                    )
                    await self._post_or_update_summary(
                        project_key, repo_slug, pr_id, pr_review_id,
                        self._build_agents_md_too_large_summary(agents_md_tokens, max_tokens),
                    )
                    return

            # Per-PR cost cap. Pre-check the cost accumulated *before* this run:
            # once a PR is at/over the limit, stop reviewing it automatically.
            # Enforced on the auto-review path only — a human @mention
            # (skip_author_check=True) is the explicit override and always runs.
            # Fail-open: a None cost (unpriced model, NULL total_cost_usd) never
            # blocks. The run that overshoots the cap completes; only subsequent
            # auto-runs are blocked here.
            if not skip_author_check:
                limit_usd = self.review_config.max_pr_cost_usd
                cumulative_cost = await _safe_db(
                    repository.get_pr_cost(self.db_pool, project_key, repo_slug, pr_id),
                    fallback=None,
                )
                if cumulative_cost is not None and cumulative_cost >= limit_usd:
                    logger.info(
                        "%s: PR cost $%.2f >= limit $%.2f — skipping auto-review "
                        "(@%s to review manually, or raise REVIEW_MAX_PR_COST_USD)",
                        pr_tag, cumulative_cost, limit_usd,
                        self.bitbucket.bot_username,
                    )
                    # Preserve the prior reviewed commit — this push was not
                    # reviewed, so raising the limit later must re-review the
                    # accumulated range rather than skip it.
                    prior_commit = await _safe_db(
                        repository.get_last_reviewed_commit(
                            self.db_pool, project_key, repo_slug, pr_id,
                        ),
                        fallback=None,
                    )
                    pr_review_id = await _safe_db(
                        repository.upsert_pr_review(
                            self.db_pool, project_key, repo_slug, pr_id,
                            last_reviewed_commit=prior_commit,
                            author=author_name,
                            pr_title=pr.title,
                            opened_at=opened_at,
                        ),
                        fallback=None,
                    )
                    await self._post_or_update_cost_limit_notice(
                        project_key, repo_slug, pr_id, pr_review_id,
                        cumulative_cost, limit_usd,
                    )
                    return

            logger.info(
                "Starting review of %s by %s (branch: %s)",
                pr_tag, author_name, pr.fromRef.displayId,
            )
            t0 = time.monotonic()

            # Determine if incremental review is possible
            source_commit = pr.fromRef.latestCommit
            last_reviewed = await _safe_db(
                repository.get_last_reviewed_commit(self.db_pool, project_key, repo_slug, pr_id),
                fallback=None,
            )

            is_incremental = False
            incremental_from: str | None = None
            diff: str = ""
            cumulative_pr_diff: str = ""

            if payload.eventKey == "pr:from_ref_updated" and last_reviewed and source_commit:
                if source_commit == last_reviewed:
                    # Same SHA — nothing actually changed (e.g. retrigger or
                    # no-op force-push). Skip is safe.
                    logger.info(
                        "%s: HEAD unchanged since last review (%s), skipping",
                        pr_tag, source_commit[:10],
                    )
                    return
                try:
                    inc_diff = await self.bitbucket.fetch_commit_diff(
                        project_key, repo_slug, last_reviewed, source_commit,
                    )
                except IncrementalDiffUnavailable as exc:
                    # Expected, well-handled control flow: branch was rebased
                    # / squashed, history rewritten. Log at INFO without a
                    # traceback. Fall through to the full-review path below;
                    # we MUST NOT skip the review just because the
                    # incremental optimization didn't apply.
                    logger.info(
                        "%s: incremental diff unavailable (%s) — running full review",
                        pr_tag, exc,
                    )
                    is_incremental = False
                except Exception:
                    # Genuinely unexpected — keep the warning + traceback so
                    # operators can see the underlying error. Fall through
                    # to full review.
                    logger.warning(
                        "%s: incremental diff failed unexpectedly, falling back to full review",
                        pr_tag, exc_info=True,
                    )
                    is_incremental = False
                else:
                    if inc_diff.strip():
                        diff = inc_diff
                        is_incremental = True
                        incremental_from = last_reviewed
                        logger.info(
                            "%s: incremental review %s..%s",
                            pr_tag, last_reviewed[:10], source_commit[:10],
                        )
                    else:
                        # HEAD moved but compare/diff was empty: rebase to
                        # identical tree, or a Bitbucket edge case. Fall
                        # back to full review — never silently skip when the
                        # SHA actually changed.
                        logger.warning(
                            "%s: HEAD moved %s -> %s but compare/diff is empty — "
                            "falling back to full review to avoid missing changes",
                            pr_tag, last_reviewed[:10], source_commit[:10],
                        )
                        is_incremental = False

            if not is_incremental:
                diff = await self.bitbucket.fetch_pr_diff(
                    project_key, repo_slug, pr_id, context_lines=0
                )
                if not diff.strip():
                    logger.info("%s has empty diff, skipping", pr_tag)
                    return
            else:
                # Cumulative PR diff used as cross-file context so the LLM can verify
                # invariants split across multiple commits (e.g. entity rename in commit 1,
                # repository method rename in commit 2). Best-effort: a failure here must
                # not block the incremental review.
                try:
                    cumulative_pr_diff = await self.bitbucket.fetch_pr_diff(
                        project_key, repo_slug, pr_id, context_lines=0
                    )
                except Exception:
                    logger.warning(
                        "%s: failed to fetch cumulative PR diff for context",
                        pr_tag, exc_info=True,
                    )
                    cumulative_pr_diff = ""
                if cumulative_pr_diff:
                    cum_budget = _cumulative_diff_budget(self.llm.max_tokens_per_chunk)
                    cum_tokens = count_tokens(cumulative_pr_diff)
                    if cum_tokens > cum_budget:
                        logger.warning(
                            "%s: cumulative PR diff %d tokens exceeds budget %d "
                            "(model context %s), dropping",
                            pr_tag, cum_tokens, cum_budget,
                            self.llm.context_window,
                        )
                        cumulative_pr_diff = ""
            files, content_skipped = await self._prepare_files(
                project_key, repo_slug, diff, source_commit, pr_tag
            )

            if not files:
                logger.info("%s has no reviewable files after content fetch, skipping", pr_tag)
                return

            # Count diff stats before compression (total PR scope).
            diff_added, diff_removed = _count_diff_lines(diff)
            total_files = len(files)

            other_modified: list[str] = []
            deleted_paths: list[str] = []
            renamed_paths: list[str] = []

            template = self.llm.prompt_template
            max_tokens = self.llm.max_tokens_per_chunk

            rc = self.review_config
            if is_small_pr(files, max_tokens, template, count_tokens, format_file_entry):
                logger.info(
                    "%s: small PR (%d files) — expanding context (before=%d, after=%d, dynamic=%s)",
                    pr_tag, len(files),
                    rc.diff_extra_lines_before, rc.diff_extra_lines_after,
                    rc.diff_allow_dynamic_context,
                )
                files = expand_all_files(
                    files,
                    before=rc.diff_extra_lines_before,
                    after=rc.diff_extra_lines_after,
                    max_dynamic_before=rc.diff_max_extra_lines_dynamic_context,
                    dynamic_context=rc.diff_allow_dynamic_context,
                )
            else:
                compression = compress_for_large_pr(
                    files, max_tokens, template, count_tokens, format_file_entry,
                )
                files = compression.included_files
                other_modified = compression.other_modified_paths
                deleted_paths = compression.deleted_file_paths
                renamed_paths = compression.renamed_file_paths
                logger.info(
                    "%s: large PR — compression applied: %d included, %d other_modified, %d deleted, %d renamed",
                    pr_tag, len(files), len(other_modified), len(deleted_paths), len(renamed_paths),
                )
                files = expand_all_files(
                    files,
                    before=rc.diff_extra_lines_before,
                    after=rc.diff_extra_lines_after,
                    max_dynamic_before=rc.diff_max_extra_lines_dynamic_context,
                    dynamic_context=rc.diff_allow_dynamic_context,
                )

            if not files:
                logger.info("%s has no reviewable files after compression, skipping", pr_tag)
                return

            # Build cross-file relationship map
            cross_file_rels = build_cross_file_context(files)
            cross_file_ctx = render_cross_file_context(cross_file_rels)
            if cross_file_ctx:
                logger.info(
                    "%s: cross-file context: %d relationship(s)",
                    pr_tag, len(cross_file_rels),
                )

            agents_md_found = bool(repo_instructions)

            ticket_id = self._extract_ticket_id(pr)
            ticket_context = ""
            ticket: JiraTicket | None = None
            parent_ticket: JiraTicket | None = None
            if ticket_id and self.jira:
                ticket, parent_ticket = await self.jira.fetch_ticket_with_parent(ticket_id)
                if ticket:
                    ticket_context, ticket = await self._fetch_ticket_context(
                        ticket_id, parent=parent_ticket,
                    )
                    logger.info("%s: linked Jira ticket %s", pr_tag, ticket_id)

            # Load previously posted findings once: feed them to the LLM as
            # "do not re-raise" context AND reuse for the post-hoc dedup below.
            existing_findings = await _safe_db(
                repository.get_existing_findings_for_prompt(
                    self.db_pool, project_key, repo_slug, pr_id,
                ),
                fallback=[],
            ) or []
            # Tail = most recent (DB returns ORDER BY id ASC). On a long-lived PR
            # with >MAX prior findings the oldest are dropped from the prompt but
            # remain in `existing_keys` below, so post-hoc dedup is unaffected.
            findings_for_prompt = existing_findings[-MAX_PREVIOUSLY_POSTED_FINDINGS:]
            findings_budget = _previously_posted_findings_budget(self.llm.max_tokens_per_chunk)
            # Drop oldest in chunks until rendered block fits the budget.
            # Step is 25% of remaining count to avoid an O(n^2) re-render per item.
            while findings_for_prompt and count_tokens(
                render_previously_posted_findings(findings_for_prompt)
            ) > findings_budget:
                drop = max(1, len(findings_for_prompt) // 4)
                findings_for_prompt = findings_for_prompt[drop:]

            llm_result = await self.llm.review_diff(
                files, repo_instructions,
                other_modified_paths=other_modified,
                deleted_file_paths=deleted_paths,
                renamed_file_paths=renamed_paths,
                ticket_context=ticket_context,
                ticket_compliance_check=self.review_config.ticket_compliance_check,
                cross_file_context=cross_file_ctx,
                cumulative_pr_diff=cumulative_pr_diff,
                previously_posted_findings=findings_for_prompt,
            )

            # Deadline-exceeded: never post a normal summary. The model didn't
            # respond within our wall-clock cap; partial findings (if any) are
            # discarded. We post a "Review skipped" notice or prepend a
            # staleness banner to the existing summary so the human reader
            # knows the latest commit was not reviewed. Stats are skipped to
            # avoid polluting metrics with a 0/0 row.
            if llm_result.timed_out:
                short_new = (source_commit or "")[:8] or "unknown"
                logger.error(
                    "Review of %s aborted — no response within %.0fs (commit %s)",
                    pr_tag, INFERENCE_HARD_TIMEOUT_SECONDS, short_new,
                )
                # Read the prior successful commit BEFORE the upsert (which
                # would otherwise overwrite it with this failed commit).
                prior_commit = await _safe_db(
                    repository.get_last_reviewed_commit(
                        self.db_pool, project_key, repo_slug, pr_id,
                    ),
                    fallback=None,
                )
                # Preserve the prior commit — the timed-out run did not
                # actually review the new one, so advancing the pointer would
                # cause the next incremental review to skip the un-reviewed
                # range.
                pr_review_id = await _safe_db(
                    repository.upsert_pr_review(
                        self.db_pool, project_key, repo_slug, pr_id,
                        last_reviewed_commit=prior_commit,
                        author=author_name,
                        pr_title=pr.title,
                        opened_at=opened_at,
                    ),
                    fallback=None,
                )
                await self._post_or_update_timeout_notice(
                    project_key, repo_slug, pr_id, pr_review_id,
                    new_commit=source_commit,
                    prior_commit=prior_commit,
                )
                return

            # Deduplicate against existing findings in DB
            existing_keys = {
                (f["file_path"], f["line_number"], f["severity"])
                for f in existing_findings
            }
            findings = [
                f for f in llm_result.findings
                if (f.file, f.line, f.severity) not in existing_keys
            ]

            findings, truncated = _sort_and_limit(findings, self.max_comments)

            # Upsert PR review record to get pr_review_id for linking findings
            pr_review_id = await _safe_db(
                repository.upsert_pr_review(
                    self.db_pool, project_key, repo_slug, pr_id,
                    last_reviewed_commit=source_commit,
                    author=author_name,
                    pr_title=pr.title,
                    opened_at=opened_at,
                ),
                fallback=None,
            )

            posted = 0
            failed = 0
            for finding in findings:
                try:
                    comment_id = await self.bitbucket.post_inline_comment(
                        project_key, repo_slug, pr_id, finding
                    )
                    posted += 1
                    if pr_review_id:
                        await _safe_db(
                            repository.insert_finding(
                                self.db_pool, pr_review_id,
                                file_path=finding.file,
                                line_number=finding.line,
                                severity=finding.severity,
                                comment_text=finding.comment,
                                suggestion=finding.suggestion,
                                bitbucket_comment_id=comment_id,
                                commit_sha=source_commit,
                                is_incremental=is_incremental,
                            )
                        )
                except Exception:
                    failed += 1
                    logger.error(
                        "Failed to post inline comment on %s:%d",
                        finding.file,
                        finding.line,
                        exc_info=True,
                    )

            elapsed = time.monotonic() - t0

            # Per-run + cumulative USD cost. Both are None when the model
            # has no entry in _MODEL_PRICING (e.g. an unmapped Copilot id).
            run_cost_usd = estimate_cost_usd(
                self.llm.config.model,
                llm_result.prompt_tokens,
                llm_result.completion_tokens,
            )
            cumulative_cost_usd: float | None = None
            if run_cost_usd is not None:
                cumulative_cost_usd = await _safe_db(
                    repository.add_pr_cost(
                        self.db_pool, project_key, repo_slug, pr_id, run_cost_usd,
                    )
                )

            summary = self._build_summary(
                findings,
                truncated,
                agents_md_found=agents_md_found,
                skipped_files=llm_result.skipped_files,
                content_skipped_files=content_skipped,
                token_usage=(llm_result.prompt_tokens, llm_result.completion_tokens),
                run_cost_usd=run_cost_usd,
                cumulative_cost_usd=cumulative_cost_usd,
                prompt_breakdown=llm_result.prompt_breakdown,
                ticket=ticket,
                parent_ticket=parent_ticket,
                compliance_requirements=llm_result.compliance_requirements,
                compliance_extraction_failed=llm_result.compliance_extraction_failed,
                elapsed=elapsed,
                jira_enabled=self.jira is not None,
                ticket_compliance_check=self.review_config.ticket_compliance_check,
                summary=llm_result.summary,
                reviewed_commit=source_commit,
                incremental_from=incremental_from,
                files_reviewed=len(files),
                total_files=total_files + len(deleted_paths) + len(renamed_paths),
                diff_added=diff_added,
                diff_removed=diff_removed,
                cross_file_symbols=[r.symbol for r in cross_file_rels] if cross_file_rels else None,
                chunk_count=llm_result.chunk_count,
                chunk_budget=self.llm.max_tokens_per_chunk,
                context_window=self.llm.context_window,
            )

            # Completed run that is at/over the cost limit: keep the over-limit
            # banner on top of the summary so the over-budget state stays loud.
            # Covers the auto-run that overshot the cap and any manual @mention
            # review while already over. No strip needed — `summary` is freshly
            # built each run, so the banner is applied exactly once.
            if (
                cumulative_cost_usd is not None
                and cumulative_cost_usd >= self.review_config.max_pr_cost_usd
            ):
                summary = (
                    _cost_limit_banner(
                        cumulative_cost_usd,
                        self.review_config.max_pr_cost_usd,
                        self.bitbucket.bot_username,
                        blocked=False,
                    )
                    + "\n\n"
                    + summary
                )

            await self._post_or_update_summary(
                project_key, repo_slug, pr_id, pr_review_id, summary
            )

            review_model_name = model_label(self.llm.config.model, self.llm.config.reasoning_effort)

            # Fold this run into the per-PR rollup that ships to riptide at
            # PR close. We persist regardless of whether riptide is wired —
            # the accumulator is also useful for local introspection, and
            # toggling riptide on later still sees a complete history.
            await _safe_db(
                repository.record_review_run_stats(
                    self.db_pool,
                    project_key=project_key,
                    repo_slug=repo_slug,
                    pr_id=pr_id,
                    model=review_model_name,
                    prompt_tokens=llm_result.prompt_tokens,
                    completion_tokens=llm_result.completion_tokens,
                    elapsed_ms=int(elapsed * 1000),
                    findings_count=len(findings),
                    source_commit_sha=source_commit,
                    lines_added=diff_added,
                    lines_removed=diff_removed,
                    files_changed=total_files + len(deleted_paths) + len(renamed_paths),
                )
            )

            parts = [
                f"Review of {pr_tag} completed in {elapsed:.1f}s",
                f"{len(findings)} issue{'s' if len(findings) != 1 else ''}",
                f"{posted} comment{'s' if posted != 1 else ''} posted",
            ]
            if failed:
                parts.append(f"{failed} failed")
            logger.info(" — ".join(parts))

        except Exception:
            logger.error("Review of %s failed", pr_tag, exc_info=True)
        finally:
            if http_scope is not None:
                totals = summarize(http_scope.counter)
                logger.info(
                    "Review HTTP totals - bitbucket=%d jira=%d inference=%d (%s)",
                    totals.get("bitbucket", 0),
                    totals.get("jira", 0),
                    totals.get("inference", 0),
                    dict(http_scope.counter),
                )
                exit_http_scope(http_scope)
            structlog.contextvars.unbind_contextvars("pr_tag", "repo", "pr_id")

    async def handle_comment_deleted(self, payload: WebhookPayload) -> None:
        """Primary signal for the opt-out feature: if the user deleted our
        summary comment, mark the PR ignored immediately.

        Bitbucket's `pr:comment:deleted` payload carries the deleted comment id
        directly, so this is robust where the pull-time guard's 404 check is
        fragile (a summary with replies may be soft-deleted, returning 200 with
        a tombstone instead of 404). The pull-time guard in review_pull_request
        stays as a backstop for repos not yet subscribed to this event.
        """
        comment = payload.comment
        if not comment:
            return

        project_key, repo_slug = self._extract_project_repo(payload)
        if not project_key or not repo_slug:
            return
        pr_id = payload.pullRequest.id
        pr_tag = f"{project_key}/{repo_slug}#{pr_id}"
        structlog.contextvars.bind_contextvars(
            pr_tag=pr_tag, repo=f"{project_key}/{repo_slug}", pr_id=pr_id
        )

        skip_state = await _safe_db(
            repository.get_pr_skip_state(self.db_pool, project_key, repo_slug, pr_id),
            fallback=None,
        )
        if not skip_state or skip_state["summary_comment_id"] != comment.id:
            # A different comment was deleted — none of our business.
            return

        await _safe_db(
            repository.mark_pr_ignored(self.db_pool, project_key, repo_slug, pr_id)
        )
        logger.info(
            "%s: summary comment %d deleted (webhook) — ignoring PR from now on",
            pr_tag, comment.id,
        )

    async def handle_mention(self, payload: WebhookPayload) -> None:
        comment = payload.comment
        if not comment:
            return

        # Self-loop prevention: ignore bot's own comments
        if comment.author.name == self.bitbucket.bot_username:
            logger.debug("Ignoring own comment (bot)")
            return

        pr = payload.pullRequest
        if pr.state and pr.state != "OPEN":
            logger.info("Ignoring mention on non-open PR (state=%s)", pr.state)
            return

        project_key, repo_slug = self._extract_project_repo(payload)
        if not project_key or not repo_slug:
            logger.error("Could not extract project/repo from webhook payload")
            return

        pr_tag = f"{project_key}/{repo_slug}#{pr.id}"

        # Any @mention reactivates a PR that was ignored after its summary
        # comment was removed. Clearing the skip state (and the stale
        # summary_comment_id, done in reactivate_pr) lets the next review post
        # a fresh summary and stops the review guard from re-ignoring the PR.
        skip_state = await _safe_db(
            repository.get_pr_skip_state(self.db_pool, project_key, repo_slug, pr.id),
            fallback=None,
        )
        if skip_state and skip_state["ignored_at"] is not None:
            await _safe_db(
                repository.reactivate_pr(self.db_pool, project_key, repo_slug, pr.id)
            )
            logger.info("%s: reactivating ignored PR via @mention", pr_tag)

        question = _extract_question(comment.text, self.mention_trigger)

        if not question or question.lower() in _REVIEW_KEYWORDS:
            logger.info("Mention triggers full review (question=%r)", question)
            await self.review_pull_request(payload, skip_author_check=True)
            return
        logger.info("Handling mention Q&A on %s: %r", pr_tag, question)

        try:
            diff = await self.bitbucket.fetch_pr_diff(project_key, repo_slug, pr.id, context_lines=0)
            source_commit = pr.fromRef.latestCommit
            files, _ = await self._prepare_files(
                project_key, repo_slug, diff, source_commit, pr_tag
            )
            if not files:
                await self.bitbucket.reply_to_comment(
                    project_key, repo_slug, pr.id, comment.id,
                    "No reviewable files in this PR.",
                )
                return
            repo_instructions = await self._fetch_repo_instructions(
                project_key, repo_slug, pr
            )

            ticket_context = ""
            ticket_id = self._extract_ticket_id(pr)
            if ticket_id and self.jira:
                ticket, parent_ticket = await self.jira.fetch_ticket_with_parent(ticket_id)
                if ticket:
                    ticket_context, _ = await self._fetch_ticket_context(
                        ticket_id, parent=parent_ticket,
                    )

            answer = await self.llm.answer_question(
                question, files, repo_instructions,
                ticket_context=ticket_context,
            )
            await self.bitbucket.reply_to_comment(
                project_key, repo_slug, pr.id, comment.id, answer,
            )
            logger.info("Posted Q&A reply on %s", pr_tag)
        except openai.APITimeoutError:
            timeout_minutes = int(INFERENCE_HARD_TIMEOUT_SECONDS / 60)
            logger.error(
                "Mention Q&A on %s aborted — no response within %.0fs",
                pr_tag, INFERENCE_HARD_TIMEOUT_SECONDS,
            )
            try:
                await self.bitbucket.reply_to_comment(
                    project_key, repo_slug, pr.id, comment.id,
                    f"⚠️ No response from the model within {timeout_minutes} "
                    f"minutes. Please try again, or simplify the question if "
                    f"it requires a lot of context.",
                )
            except Exception:
                logger.error(
                    "Failed to post mention timeout reply on %s", pr_tag,
                    exc_info=True,
                )
        except Exception:
            logger.error("Mention Q&A on %s failed", pr_tag, exc_info=True)

    async def handle_pr_merged(self, payload: WebhookPayload) -> None:
        pr = payload.pullRequest
        project_key, repo_slug = self._extract_project_repo(payload)
        if not project_key or not repo_slug:
            return

        pr_tag = f"{project_key}/{repo_slug}#{pr.id}"

        try:
            # Mark PR as merged; data retained for metrics
            await _safe_db(
                repository.mark_pr_merged(self.db_pool, project_key, repo_slug, pr.id)
            )
            frozen_cost = await _safe_db(
                repository.freeze_pr_cost(self.db_pool, project_key, repo_slug, pr.id)
            )
            if frozen_cost is not None:
                logger.info(
                    "%s merged — frozen LLM cost $%.4f (upper bound)",
                    pr_tag, frozen_cost,
                )

            # Forward the per-PR rollup (cost + diff size + tokens + findings)
            # to riptide. Best-effort: a failure here must not block the
            # rest of the merge handler.
            try:
                merge_commit_sha = self._extract_merge_commit_sha(payload)
                await self._emit_pr_rollup_to_riptide(
                    project_key=project_key,
                    repo_slug=repo_slug,
                    pr_id=pr.id,
                    pr_tag=pr_tag,
                    outcome="merged",
                    merge_commit_sha=merge_commit_sha,
                )
            except Exception:
                logger.warning("riptide emit (pr_completed merged) failed", exc_info=True)

            comments = await self.bitbucket.fetch_pr_comments(project_key, repo_slug, pr.id)
            bot_username = self.bitbucket.bot_username

            noergler_inline = {
                c["id"]: c for c in comments
                if _is_bot_comment(c, bot_username)
                and c.get("path") is not None
                and c.get("id") is not None
            }

            if not noergler_inline:
                logger.info("%s merged — no review comments", pr_tag)
                return

            disagreed_parent_ids: set[int] = set()
            for c in comments:
                parent_id = c.get("parent_id")
                if parent_id is None or parent_id not in noergler_inline:
                    continue
                if _is_bot_comment(c, bot_username):
                    continue
                if classify_feedback(c.get("text", "")) == "negative":
                    disagreed_parent_ids.add(parent_id)

            disagreed = len(disagreed_parent_ids)
            total = len(noergler_inline)
            useful_pct = (total - disagreed) / total * 100
            logger.info(
                "%s merged — %d comments, %d disagreed (%.0f%% useful)",
                pr_tag, total, disagreed, useful_pct,
            )
        except Exception:
            logger.error("Merged stats for %s failed", pr_tag, exc_info=True)

    async def handle_pr_declined(self, payload: WebhookPayload) -> None:
        pr = payload.pullRequest
        project_key, repo_slug = self._extract_project_repo(payload)
        if not project_key or not repo_slug:
            return

        pr_tag = f"{project_key}/{repo_slug}#{pr.id}"

        await _safe_db(
            repository.mark_pr_declined(self.db_pool, project_key, repo_slug, pr.id)
        )
        logger.info("%s declined — marked, data retained", pr_tag)

        try:
            await self._emit_pr_rollup_to_riptide(
                project_key=project_key,
                repo_slug=repo_slug,
                pr_id=pr.id,
                pr_tag=pr_tag,
                outcome="declined",
                merge_commit_sha=None,
            )
        except Exception:
            logger.warning("riptide emit (pr_completed declined) failed", exc_info=True)

    async def handle_pr_deleted(self, payload: WebhookPayload) -> None:
        pr = payload.pullRequest
        project_key, repo_slug = self._extract_project_repo(payload)
        if not project_key or not repo_slug:
            return

        pr_tag = f"{project_key}/{repo_slug}#{pr.id}"

        await _safe_db(
            repository.mark_pr_deleted(self.db_pool, project_key, repo_slug, pr.id)
        )
        logger.info("%s deleted — marked, data retained", pr_tag)

        try:
            await self._emit_pr_rollup_to_riptide(
                project_key=project_key,
                repo_slug=repo_slug,
                pr_id=pr.id,
                pr_tag=pr_tag,
                outcome="deleted",
                merge_commit_sha=None,
            )
        except Exception:
            logger.warning("riptide emit (pr_completed deleted) failed", exc_info=True)

    def _extract_merge_commit_sha(self, payload: WebhookPayload) -> str | None:
        """Pull the merge commit SHA from a pr:merged payload.

        Bitbucket places it on pullRequest.properties.mergeCommit.id. Riptide's
        pr_completed validator rejects merged events with merge_commit_sha=None
        (intentional, to catch sender bugs), so a missing SHA here causes the
        emit to fail loudly downstream.
        """
        props = payload.pullRequest.properties
        if props is None or props.mergeCommit is None:
            return None
        sha = props.mergeCommit.id
        # Treat empty string the same as missing — Bitbucket has been seen
        # to populate the field as "" on some deployments.
        return sha if sha else None

    async def _emit_pr_rollup_to_riptide(
        self,
        *,
        project_key: str,
        repo_slug: str,
        pr_id: int,
        pr_tag: str,
        outcome: Literal["merged", "declined", "deleted"],
        merge_commit_sha: str | None,
    ) -> None:
        """Aggregate per-run review stats and POST a pr_completed event.

        No-op when riptide is not configured, when no review ever ran on
        the PR, or when the rollup has already been emitted (idempotency
        via riptide_emitted_at).
        """
        if self.riptide is None or not self.riptide.enabled:
            return

        # Refresh the final cumulative PR diff from Bitbucket at close time
        # — the per-run accumulators may have only seen incremental diffs.
        # For deleted PRs the diff is gone, so skip the fetch entirely and
        # fall back to the per-run trail rather than always log-and-except.
        final_lines_added: int | None = None
        final_lines_removed: int | None = None
        final_files_changed: int | None = None
        if outcome != "deleted":
            try:
                full_diff = await self.bitbucket.fetch_pr_diff(
                    project_key, repo_slug, pr_id, context_lines=0
                )
                if full_diff.strip():
                    final_lines_added, final_lines_removed = _count_diff_lines(full_diff)
                    # files_changed = number of distinct paths in the diff
                    # ('diff --git a/... b/...' header). Cheaper than parsing.
                    final_files_changed = sum(
                        1 for line in full_diff.splitlines() if line.startswith("diff --git ")
                    )
            except Exception:
                logger.info(
                    "%s: final PR diff unavailable at close — using last-known per-run stats",
                    pr_tag,
                )

        # Use the source-branch HEAD as recorded by the latest review run.
        snapshot = await _safe_db(
            repository.claim_rollup_for_emit(
                self.db_pool,
                project_key=project_key,
                repo_slug=repo_slug,
                pr_id=pr_id,
                final_source_commit_sha=None,  # already set by record_review_run_stats
                final_merge_commit_sha=merge_commit_sha,
                final_lines_added=final_lines_added,
                final_lines_removed=final_lines_removed,
                final_files_changed=final_files_changed,
            )
        )
        if snapshot is None:
            logger.debug(
                "%s: rollup not emitted (already emitted or no review runs)",
                pr_tag,
            )
            return

        source_sha = snapshot.get("final_source_commit_sha")
        if not source_sha:
            logger.warning(
                "%s: rollup has no source_commit_sha — skipping emit "
                "(should not happen if record_review_run_stats ran)",
                pr_tag,
            )
            return

        first_review_at = snapshot.get("first_review_at") or datetime.now(timezone.utc)
        total_cost_usd = snapshot.get("total_cost_usd")
        models_used = list(snapshot.get("models_used") or [])

        await self.riptide.emit_pr_completed(
            outcome=outcome,
            pr_key=pr_tag,
            repo=f"{project_key}/{repo_slug}",
            source_commit_sha=source_sha,
            merge_commit_sha=snapshot.get("final_merge_commit_sha"),
            lines_added=snapshot.get("final_lines_added") or 0,
            lines_removed=snapshot.get("final_lines_removed") or 0,
            files_changed=snapshot.get("final_files_changed") or 0,
            total_runs=snapshot["total_runs"],
            total_prompt_tokens=snapshot["total_prompt_tokens"],
            total_completion_tokens=snapshot["total_completion_tokens"],
            total_elapsed_ms=snapshot["total_elapsed_ms"],
            total_findings_count=snapshot["total_findings_count"],
            total_cost_usd=total_cost_usd,
            models_used=models_used,
            first_review_at=first_review_at,
            closed_at=datetime.now(timezone.utc),
        )

    async def handle_feedback(self, payload: WebhookPayload) -> None:
        comment = payload.comment
        if not comment or payload.commentParentId is None:
            logger.debug("Feedback skipped: no comment or no commentParentId")
            return

        # Self-loop prevention: ignore bot's own replies
        if comment.author.name == self.bitbucket.bot_username:
            logger.debug("Feedback skipped: reply is from bot")
            return

        pr = payload.pullRequest
        project_key, repo_slug = self._extract_project_repo(payload)
        if not project_key or not repo_slug:
            logger.debug("Feedback skipped: could not extract project/repo")
            return

        pr_tag = f"{project_key}/{repo_slug}#{pr.id}"
        parent_id = payload.commentParentId

        try:
            # Look up the parent finding by Bitbucket comment ID
            finding = await _safe_db(
                repository.get_finding_by_comment_id(self.db_pool, parent_id),
                fallback=None,
            )
            if finding is None:
                logger.info(
                    "Feedback skipped on %s: parent %d not found in DB",
                    pr_tag, parent_id,
                )
                return

            parent_file = finding["file_path"]
            parent_line = finding["line_number"]
            parent_severity = finding["severity"]
            parent_commit_sha = finding.get("commit_sha")
            suggestion_text = ""

            classification = classify_feedback(comment.text)
            if classification != "negative":
                logger.debug("Feedback skipped on %s: classified as %s", pr_tag, classification)
                return

            already_disagreed = await _safe_db(
                repository.has_negative_feedback(
                    self.db_pool, project_key, repo_slug, pr.id, parent_id,
                ),
                fallback=False,
            )
            if already_disagreed:
                logger.info(
                    "Feedback skipped on %s: parent %d already has a negative feedback",
                    pr_tag, parent_id,
                )
                return

            logger.info(
                "Disagree feedback: %s",
                json.dumps({
                    "event": "disagree",
                    "pr": pr_tag,
                    "comment_id": parent_id,
                    "file": parent_file or "",
                    "line": parent_line,
                    "suggestion": suggestion_text,
                    "author": comment.author.name,
                }),
            )

            reacted = await self.bitbucket.add_comment_reaction(
                project_key, repo_slug, pr.id, comment.id
            )
            if not reacted:
                await self.bitbucket.reply_to_comment(
                    project_key, repo_slug, pr.id, comment.id,
                    disagree_response(),
                )

            # Persist feedback event
            await _safe_db(
                repository.insert_feedback(
                    self.db_pool,
                    project_key=project_key,
                    repo_slug=repo_slug,
                    pr_id=pr.id,
                    bitbucket_comment_id=parent_id,
                    feedback_author=comment.author.name,
                    classification=classification,
                    file_path=parent_file,
                    severity=parent_severity,
                )
            )
            if self.riptide is not None and self.riptide.enabled:
                # `disagreed` mirrors noergler's "negative" classification.
                # `acknowledged` is reserved for a future positive-feedback path.
                try:
                    await self.riptide.emit_feedback(
                        pr_key=pr_tag,
                        finding_id=str(parent_id),
                        verdict="disagreed",
                        actor=comment.author.name,
                        repo=f"{project_key}/{repo_slug}",
                        commit_sha=parent_commit_sha,
                        occurred_at=datetime.now(timezone.utc),
                    )
                except Exception:
                    logger.warning("riptide emit (feedback) failed", exc_info=True)
        except Exception:
            logger.error("Feedback handling on %s failed", pr_tag, exc_info=True)

    def _extract_project_repo(
        self, payload: WebhookPayload
    ) -> tuple[str, str]:
        pr = payload.pullRequest
        # Bitbucket Server webhooks nest repository under fromRef/toRef
        ref = pr.toRef
        if ref.repository:
            return ref.repository.project.key, ref.repository.slug
        ref = pr.fromRef
        if ref.repository:
            return ref.repository.project.key, ref.repository.slug
        return ("", "")

    @staticmethod
    def _build_opt_out_branch_summary(keyword: str, branch: str) -> str:
        return (
            "### Review skipped — opt-out keyword in branch name 🛑\n\n"
            f"The source branch `{branch}` contains the opt-out keyword "
            f"`{keyword}`, so the auto-review was skipped. No LLM or Jira "
            "calls were made for this PR.\n\n"
            "**What to do**\n"
            "- Rename the branch to remove the keyword and push again; the "
            "next webhook event will trigger a full review.\n"
            "- Or @mention the bot explicitly in a comment if you want a "
            "one-off review on this branch — mentions are not muted.\n\n"
            "**Configure**\n"
            "- The keyword is set via `REVIEW_OPT_OUT_BRANCH_KEYWORD` on the "
            "noergler service (empty string disables the feature).\n"
        )

    @staticmethod
    def _build_agents_md_missing_summary() -> str:
        return (
            "### Review skipped — no `AGENTS.md` found 🛑\n\n"
            "This repository has no `AGENTS.md` on the PR branch or the target branch. "
            "Project-specific review guidelines are **vital** for producing targeted, "
            "high-signal feedback — without them the reviewer falls back to generic nits, "
            "so the review was not run.\n\n"
            "**What to do**\n"
            "- Add an `AGENTS.md` file to the repository root describing project "
            "conventions, forbidden patterns, and areas the reviewer should focus on.\n"
            "- Push the file to the PR branch (or merge it into the target branch) and "
            "the next webhook event on this PR will trigger a full review.\n\n"
            "**Opt-out**\n"
            "- To review PRs without an `AGENTS.md`, set "
            "`REVIEW_REQUIRE_AGENTS_MD=false` on the noergler service and restart.\n"
        )

    def _build_agents_md_too_large_summary(self, tokens: int, limit: int) -> str:
        links: list[tuple[str, str]] = []
        custom = _parse_custom_link(self.review_config.agents_md_custom_link)
        if custom is not None:
            links.append(custom)
        links.extend(_AGENTS_MD_FURTHER_READING)
        further_reading = "\n".join(f"- [{title}]({url})" for title, url in links)
        return (
            "### Review skipped — `AGENTS.md` too large 🛑\n\n"
            f"`AGENTS.md` weighs in at ~{_fmt(tokens)} tokens, exceeding the configured "
            f"hard limit of {_fmt(limit)} tokens. Oversized agent instructions crowd out "
            "the actual diff, degrade review quality (context rot), and inflate "
            "inference cost — so the review was not run.\n\n"
            "**What to do**\n"
            "- Trim `AGENTS.md`: drop sections that are there for humans to skim, "
            "remove duplicated README content, replace prose with terse rules.\n"
            "- Move detailed reference material into separate files and link to "
            "them; keep the main file lean.\n"
            "- Push the slimmer file and the next webhook event on this PR will "
            "trigger a full review.\n\n"
            "**Configure**\n"
            "- Raise the limit via `REVIEW_AGENTS_MD_MAX_TOKENS` on the noergler "
            "service, or set it to `0` to disable the hard cut-off entirely "
            "(the soft warning via `REVIEW_AGENTS_MD_WARN_TOKENS` remains).\n\n"
            "**Further reading**\n"
            f"{further_reading}\n"
        )

    async def _post_or_update_summary(
        self,
        project_key: str,
        repo_slug: str,
        pr_id: int,
        pr_review_id: int | None,
        summary: str,
    ) -> None:
        """Post a new summary PR comment or update the existing one, tracking it in DB."""
        existing_summary = None
        if pr_review_id:
            db_summary_info = await _safe_db(
                repository.get_summary_comment_info(self.db_pool, pr_review_id),
                fallback=None,
            )
            if db_summary_info:
                existing_summary = {
                    "id": db_summary_info["summary_comment_id"],
                    "version": db_summary_info["summary_comment_version"],
                }

        summary_comment_id = None
        summary_comment_version = None
        try:
            updated_version = None
            if existing_summary:
                updated_version = await self.bitbucket.update_pr_comment(
                    project_key, repo_slug, pr_id,
                    existing_summary["id"], existing_summary["version"], summary,
                )

            if updated_version is not None and existing_summary:
                summary_comment_id = existing_summary["id"]
                summary_comment_version = updated_version
            else:
                post_result = await self.bitbucket.post_pr_comment(
                    project_key, repo_slug, pr_id, summary
                )
                if post_result:
                    summary_comment_id, summary_comment_version = post_result

            if pr_review_id and summary_comment_id is not None:
                await _safe_db(
                    repository.update_summary_comment(
                        self.db_pool, pr_review_id,
                        summary_comment_id, summary_comment_version or 0,
                    )
                )
        except Exception:
            logger.error("Failed to post summary comment", exc_info=True)

    async def _post_or_update_timeout_notice(
        self,
        project_key: str,
        repo_slug: str,
        pr_id: int,
        pr_review_id: int | None,
        new_commit: str | None,
        prior_commit: str | None,
    ) -> None:
        """Post a deadline-exceeded notice or prepend a staleness banner.

        - **No prior summary tracked** → post a fresh "Review skipped"
          comment that becomes the noergler comment for this PR. The next
          successful review will update it in place via
          `_post_or_update_summary`.
        - **Prior summary tracked** → fetch its body, strip any previous
          staleness banner (idempotency on repeated timeouts), prepend a
          fresh banner naming the failed and prior commits.

        We always preserve the original body. The next successful review
        replaces the entire comment, banner and all.
        """
        timeout_minutes = int(INFERENCE_HARD_TIMEOUT_SECONDS / 60)
        short_new = (new_commit or "")[:8] or "unknown"

        existing_summary = None
        if pr_review_id:
            existing_summary = await _safe_db(
                repository.get_summary_comment_info(self.db_pool, pr_review_id),
                fallback=None,
            )

        if not existing_summary:
            body = (
                f"⚠️ **Review skipped** — no response from the model within "
                f"{timeout_minutes} minutes for commit `{short_new}`. "
                f"The review did not complete. Push a new commit or "
                f"`@{self.bitbucket.bot_username}` to retry."
            )
            try:
                post_result = await self.bitbucket.post_pr_comment(
                    project_key, repo_slug, pr_id, body,
                )
            except Exception:
                logger.error("Failed to post timeout notice", exc_info=True)
                return
            if post_result and pr_review_id:
                comment_id, version = post_result
                await _safe_db(
                    repository.update_summary_comment(
                        self.db_pool, pr_review_id,
                        comment_id, version or 0,
                    )
                )
            return

        comment_id = existing_summary["summary_comment_id"]
        version = existing_summary["summary_comment_version"]
        try:
            existing_payload = await self.bitbucket.fetch_pr_comment(
                project_key, repo_slug, pr_id, comment_id,
            )
        except Exception:
            logger.error(
                "Failed to fetch existing summary for staleness banner",
                exc_info=True,
            )
            return
        existing_body = existing_payload.get("text", "") or ""
        # Bitbucket's optimistic locking: prefer the live version over the
        # one cached in our DB to avoid a 409.
        live_version = existing_payload.get("version", version)

        body_without_banner = _strip_stale_banner(existing_body)
        short_old = (prior_commit or "")[:8] if prior_commit else None
        if short_old:
            banner = (
                f"{_STALE_BANNER_SENTINEL}\n"
                f"⚠️ No response from the model within {timeout_minutes} "
                f"minutes on commit `{short_new}` — findings below reflect "
                f"the earlier commit `{short_old}`."
            )
        else:
            banner = (
                f"{_STALE_BANNER_SENTINEL}\n"
                f"⚠️ No response from the model within {timeout_minutes} "
                f"minutes on commit `{short_new}` — findings below reflect "
                f"an earlier commit."
            )
        new_body = banner + "\n\n" + body_without_banner

        try:
            new_version = await self.bitbucket.update_pr_comment(
                project_key, repo_slug, pr_id,
                comment_id, live_version, new_body,
            )
        except Exception:
            logger.error(
                "Failed to update existing summary with staleness banner",
                exc_info=True,
            )
            return
        if new_version is not None and pr_review_id:
            await _safe_db(
                repository.update_summary_comment(
                    self.db_pool, pr_review_id,
                    comment_id, new_version,
                )
            )

    async def _post_or_update_cost_limit_notice(
        self,
        project_key: str,
        repo_slug: str,
        pr_id: int,
        pr_review_id: int | None,
        cumulative_cost_usd: float,
        limit_usd: float,
    ) -> None:
        """Surface a blocked auto-review on the Bitbucket PR summary comment.

        - **No prior summary tracked** → post a fresh banner-only comment that
          becomes the noergler comment for this PR.
        - **Prior summary tracked** → fetch its body, strip any previous cost
          banner (idempotency on repeated blocked pushes), prepend a fresh
          `blocked=True` banner. The body underneath reflects the last review
          (older than the un-reviewed push). A later successful review replaces
          the whole comment.

        Nothing is written to Jira. Mirrors `_post_or_update_timeout_notice`.
        """
        banner = _cost_limit_banner(
            cumulative_cost_usd, limit_usd, self.bitbucket.bot_username,
            blocked=True,
        )

        existing_summary = None
        if pr_review_id:
            existing_summary = await _safe_db(
                repository.get_summary_comment_info(self.db_pool, pr_review_id),
                fallback=None,
            )

        if not existing_summary:
            try:
                post_result = await self.bitbucket.post_pr_comment(
                    project_key, repo_slug, pr_id, banner,
                )
            except Exception:
                logger.error("Failed to post cost-limit notice", exc_info=True)
                return
            if post_result and pr_review_id:
                comment_id, version = post_result
                await _safe_db(
                    repository.update_summary_comment(
                        self.db_pool, pr_review_id,
                        comment_id, version or 0,
                    )
                )
            return

        comment_id = existing_summary["summary_comment_id"]
        version = existing_summary["summary_comment_version"]
        try:
            existing_payload = await self.bitbucket.fetch_pr_comment(
                project_key, repo_slug, pr_id, comment_id,
            )
        except Exception:
            logger.error(
                "Failed to fetch existing summary for cost-limit banner",
                exc_info=True,
            )
            return
        existing_body = existing_payload.get("text", "") or ""
        # Prefer the live version over our cached one to avoid a 409.
        live_version = existing_payload.get("version", version)

        new_body = banner + "\n\n" + _strip_cost_banner(existing_body)

        try:
            new_version = await self.bitbucket.update_pr_comment(
                project_key, repo_slug, pr_id,
                comment_id, live_version, new_body,
            )
        except Exception:
            logger.error(
                "Failed to update existing summary with cost-limit banner",
                exc_info=True,
            )
            return
        if new_version is not None and pr_review_id:
            await _safe_db(
                repository.update_summary_comment(
                    self.db_pool, pr_review_id,
                    comment_id, new_version,
                )
            )

    async def _fetch_repo_instructions(
        self, project: str, repo: str, pr: PullRequest
    ) -> str:
        """Fetch AGENTS.md from the PR branch, falling back to the target branch."""
        for ref in [pr.fromRef.latestCommit, pr.toRef.displayId]:
            if not ref:
                continue
            try:
                content = await self.bitbucket.fetch_file_content(
                    project, repo, ref, "AGENTS.md"
                )
                if content.strip():
                    logger.info("Loaded AGENTS.md from ref %s", ref)
                    return content
            except Exception:
                logger.debug("AGENTS.md not found at ref %s", ref)
        return ""

    @staticmethod
    def _plural(n: int, word: str) -> str:
        return f"{n} {word}" if n == 1 else f"{n} {word}s"

    def _format_agents_md_scope_line(self, agents_md_tokens: int | None) -> str:
        base = "Using project-specific review guidelines from `AGENTS.md`"
        warn_threshold = self.review_config.agents_md_warn_tokens
        if not agents_md_tokens or warn_threshold <= 0:
            return f"{base} ✅"
        pct = round(agents_md_tokens / warn_threshold * 100)
        stats = f"(~{agents_md_tokens} / {warn_threshold} tokens, {pct}%)"
        if agents_md_tokens > warn_threshold:
            return f"{base} {stats} — risk of context bloat, consider trimming ⚠️"
        return f"{base} {stats} ✅"

    _VERDICT_LABEL = {
        "approve": "Approve ✅",
        "approve_with_followups": "Approve with follow-ups ⚠️",
        "request_changes": "Request changes 🛑",
    }

    def _build_summary(
        self,
        findings: list[ReviewFinding],
        truncated: bool = False,
        agents_md_found: bool = False,
        skipped_files: list[str] | None = None,
        content_skipped_files: list[str] | None = None,
        token_usage: tuple[int, int] | None = None,
        prompt_breakdown: dict[str, int] | None = None,
        ticket: JiraTicket | None = None,
        parent_ticket: JiraTicket | None = None,
        compliance_requirements: list[dict[str, Any]] | None = None,
        compliance_extraction_failed: bool = False,
        elapsed: float | None = None,
        jira_enabled: bool = False,
        ticket_compliance_check: bool = True,
        summary: ReviewSummary | None = None,
        reviewed_commit: str | None = None,
        incremental_from: str | None = None,
        files_reviewed: int | None = None,
        total_files: int | None = None,
        diff_added: int | None = None,
        diff_removed: int | None = None,
        cross_file_symbols: list[str] | None = None,
        chunk_count: int | None = None,
        chunk_budget: int | None = None,
        context_window: int | None = None,
        run_cost_usd: float | None = None,
        cumulative_cost_usd: float | None = None,
    ) -> str:
        summary = summary or ReviewSummary()
        sections: list[str] = []

        # --- Header (incremental update banner) — always shown when applicable.
        if incremental_from and reviewed_commit:
            sections.append(
                f"### Review summary (incremental update)\n"
                f"- Changes reviewed: `{incremental_from[:10]}` .. `{reviewed_commit[:10]}`"
            )

        # --- 1. Overview (always rendered)
        # Italic "_Not provided._" / "_Not assessed._" fallbacks below are
        # distinct from the LLM-emitted "None." / "None notable." sentinels:
        # the italic form means "the model did not return this field at all"
        # (a defect worth seeing) while the plain sentinel means "the model
        # looked and had nothing to report" (the intended clean-case output).
        overview_body = wrap_prose(summary.overview.strip()) if summary.overview else "_Not provided._"
        sections.append("### Overview\n" + overview_body)

        # --- 2. Strengths (always rendered; `None.` when empty)
        if summary.strengths:
            strengths_body = "\n".join(f"- {s}" for s in summary.strengths)
        else:
            strengths_body = "None."
        sections.append("### Strengths\n" + strengths_body)

        # --- 3. Issues / Suggestions (always rendered; numbered headlines only)
        issues_lines: list[str] = ["### Issues / Suggestions"]
        if not findings:
            issues_lines.append("None.")
        else:
            security_findings = [f for f in findings if _SECURITY_KEYWORDS.search(f.comment)]
            if security_findings:
                issues_lines.append(
                    f"- {self._plural(len(security_findings), 'potential security issue')} 🔒 "
                    "— review carefully"
                )
            if truncated:
                issues_lines.append(
                    f"- Showing top {len(findings)} findings by severity. "
                    "Additional findings were omitted."
                )
            if security_findings or truncated:
                issues_lines.append("")
            for idx, f in enumerate(findings, start=1):
                headline = (f.headline or "").strip()
                if not headline:
                    headline = f.comment.splitlines()[0].strip() if f.comment else "(no description)"
                issues_lines.append(f"{idx}. {headline}")
        sections.append("\n".join(issues_lines))

        # --- 4. Security / Performance (always rendered; `None notable.` when empty)
        sec_body = wrap_prose(summary.security_performance.strip()) or "None notable."
        sections.append("### Security / Performance\n" + sec_body)

        # --- 5. Test Coverage (always rendered)
        tc_body = wrap_prose(summary.test_coverage.strip()) or "_Not assessed._"
        sections.append("### Test Coverage\n" + tc_body)

        # --- 6. Ticket / Requirement Compliance (conditional on ticket presence)
        if ticket:
            reqs = compliance_requirements if ticket_compliance_check else None
            has_compliance = bool(reqs)
            if has_compliance:
                met_count = sum(1 for r in reqs if r.get("met"))
                total_count = len(reqs)
                if met_count == total_count:
                    compliance_label, compliance_emoji = "Fully compliant", "✅"
                elif met_count > 0:
                    compliance_label, compliance_emoji = "Partially compliant", "⚠️"
                else:
                    compliance_label, compliance_emoji = "Not compliant", "❌"
                verdict_suffix = f" · **{compliance_label}** {compliance_emoji}"
                heading = "### Requirement Compliance"
            else:
                verdict_suffix = ""
                heading = "### Ticket"

            ticket_lines = [heading]
            if parent_ticket:
                ticket_lines.append(
                    f"**[{parent_ticket.key}]({parent_ticket.url})** — {parent_ticket.title}"
                )
                ticket_lines.append(
                    f"**↳ [{ticket.key}]({ticket.url})** — {ticket.title}{verdict_suffix}"
                )
            else:
                ticket_lines.append(
                    f"**[{ticket.key}]({ticket.url})** — {ticket.title}{verdict_suffix}"
                )
            if has_compliance:
                for r in reqs:
                    mark = "✅" if r.get("met") else "❌"
                    ticket_lines.append(f"- {r.get('requirement', '???')} {mark}")
            else:
                # Make the reason for missing compliance data explicit so
                # reviewers can tell config/ticket/LLM problems apart.
                # Order matters: a missing AC means there was nothing to
                # extract in the first place, so "extraction failed" would be
                # misleading. Check ticket-side conditions before LLM-side.
                if not ticket_compliance_check:
                    reason = "_Compliance check disabled in config_"
                elif not ticket.acceptance_criteria:
                    reason = "_No acceptance criteria found in ticket_"
                elif compliance_extraction_failed:
                    reason = "_Compliance extraction failed during LLM review — check logs_"
                else:
                    reason = "_No acceptance criteria are verifiable from code changes_"
                ticket_lines.append(reason)
            sections.append("\n".join(ticket_lines))

        # --- 7. Verdict (always rendered)
        verdict_label = self._VERDICT_LABEL.get(
            summary.verdict_decision, self._VERDICT_LABEL["approve"],
        )
        rationale = wrap_prose(summary.verdict_rationale.strip()) or "_No rationale provided._"
        sections.append(f"### Recommendation\n**{verdict_label}** — {rationale}")

        # --- Scope
        scope: list[str] = []
        if agents_md_found:
            scope.append(self._format_agents_md_scope_line(
                (prompt_breakdown or {}).get("repo_instructions"),
            ))
        else:
            scope.append(
                "Tip: Add an `AGENTS.md` to your repository root with project-specific "
                "review guidelines for more targeted feedback. 💡"
            )
        if not ticket:
            if jira_enabled:
                scope.append("No ticket found in branch name or PR title ℹ️")
            else:
                scope.append("Jira is not enabled ℹ️")

        if files_reviewed is not None and total_files is not None:
            if files_reviewed == total_files:
                files_str = f"Reviewed {files_reviewed} files"
            else:
                skipped_count = total_files - files_reviewed
                files_str = (
                    f"Reviewed {files_reviewed} of {total_files} files "
                    f"({skipped_count} skipped: lock files, binaries, config)"
                )
            diff_parts = []
            if diff_added:
                diff_parts.append(f"+{diff_added}")
            if diff_removed:
                diff_parts.append(f"-{diff_removed}")
            if diff_parts:
                files_str += f", {' / '.join(diff_parts)} lines"
            scope.append(files_str)
        elif diff_added is not None or diff_removed is not None:
            parts = []
            if diff_added:
                parts.append(f"+{diff_added}")
            if diff_removed:
                parts.append(f"-{diff_removed}")
            if parts:
                scope.append(f"Diff: {' / '.join(parts)} lines")

        if cross_file_symbols:
            symbol_list = ", ".join(f"`{s}`" for s in cross_file_symbols[:5])
            suffix = f" and {len(cross_file_symbols) - 5} more" if len(cross_file_symbols) > 5 else ""
            dep_word = "dependency" if len(cross_file_symbols) == 1 else "dependencies"
            scope.append(
                f"{len(cross_file_symbols)} cross-file {dep_word} analyzed "
                f"({symbol_list}{suffix})"
            )

        if skipped_files:
            file_list = ", ".join(f"`{PurePosixPath(f).name}`" for f in skipped_files)
            scope.append(f"Not reviewed (too large): {file_list} ⚠️")
        if content_skipped_files:
            file_list = ", ".join(f"`{PurePosixPath(f).name}`" for f in content_skipped_files)
            scope.append(f"Reviewed without full file context (too large): {file_list} ⚠️")

        # --- Cost / telemetry
        telemetry: list[str] = []
        if chunk_count is not None and chunk_budget and token_usage:
            prompt_t = token_usage[0]
            used_k = _fmt_k(prompt_t)
            budget_k = _fmt_k(chunk_budget)
            if chunk_count == 1:
                pct = round(prompt_t / chunk_budget * 100) if chunk_budget else 0
                window_suffix = f", model max {_fmt_k(context_window)}" if context_window else ""
                telemetry.append(
                    f"Tokens used: {used_k} of {budget_k} available "
                    f"({pct}% used{window_suffix}) · 1 pass"
                )
            else:
                avg_pct = round(prompt_t / chunk_count / chunk_budget * 100) if chunk_budget else 0
                cap_clause = (
                    f"cap {budget_k}/pass, model max {_fmt_k(context_window)}"
                    if context_window
                    else f"cap {budget_k}/pass"
                )
                telemetry.append(
                    f"Tokens used: {used_k} total across {chunk_count} passes "
                    f"(avg {avg_pct}% used/pass, {cap_clause})"
                )

        if token_usage:
            if prompt_breakdown:
                t = prompt_breakdown["template"]
                r = prompt_breakdown["repo_instructions"]
                f = prompt_breakdown["files"]
                telemetry.append(
                    f"Input tokens: ~{_fmt(t)} review prompt · "
                    f"~{_fmt(r)} AGENTS.md · ~{_fmt(f)} file content"
                )
            prompt_t, completion_t = token_usage
            model = model_label(self.llm.config.model, self.llm.config.reasoning_effort)
            total = prompt_t + completion_t
            stats = (
                f"Model: `{model}` · ↑ {_fmt(prompt_t)} · ↓ {_fmt(completion_t)} "
                f"({_fmt(total)} total)"
            )
            if elapsed is not None:
                stats += f" · ⏱️ {elapsed:.1f}s"
            telemetry.append(stats)

        # Omitted entirely when the model has no pricing entry — better than
        # printing a misleading "$0.00".
        if run_cost_usd is not None:
            cost_line = f"Estimated cost: ${run_cost_usd:.2f} this run"
            if cumulative_cost_usd is not None:
                cost_line += f", ${cumulative_cost_usd:.2f} PR total"
                # Show the per-PR budget alongside the running total so the
                # limit is transparent on every summary, not only at the cap.
                cost_line += f" / ${self.review_config.max_pr_cost_usd:.2f} limit"
            telemetry.append(cost_line)

        footnote = [*scope]
        footnote.extend(telemetry)
        if footnote:
            sections.append("---\n" + "\n".join(f"- _{m}_" for m in footnote))

        return "\n\n".join(sections)



def _sort_and_limit(
    findings: list[ReviewFinding], max_comments: int
) -> tuple[list[ReviewFinding], bool]:
    sorted_findings = sorted(findings, key=lambda f: SEVERITY_ORDER.get(f.severity, 99))
    truncated = len(sorted_findings) > max_comments
    return sorted_findings[:max_comments], truncated
