import asyncio
import json
import logging
import re
import time
from pathlib import PurePosixPath

from app.bitbucket import NOERGLER_MARKER, BitbucketClient
from app.db import repository
from app.feedback import classify_feedback, random_response
from app.llm_client import (
    LLMClient,
    FileReviewData,
    count_tokens,
    extract_path,
    format_file_entry,
    is_deleted,
    is_reviewable_diff,
    split_by_file,
)
from app.config import ReviewConfig, ServerConfig
from app.context_expansion import expand_all_files
from app.cross_file_context import build_cross_file_context, render_cross_file_context
from app.diff_compression import compress_for_large_pr, is_small_pr
from app.jira import JiraClient, JiraTicket
from app.models import PullRequest, ReviewFinding, WebhookPayload

logger = logging.getLogger(__name__)


def _fmt(n: int) -> str:
    return f"{n:,}".replace(",", "'")

SEVERITY_ORDER = {"critical": 0, "warning": 1}
_EFFORT_LABELS = {
    1: "Trivial: typo, comment, config tweak",
    2: "Small: single-function change, clear intent",
    3: "Medium: multiple files, some logic changes",
    4: "Large: significant logic changes, needs careful review",
    5: "Very large: architectural changes, complex interactions",
}
_SEVERITY_RE = re.compile(r"\*\*Severity Level:\*\*\s*(\w+)")
_REVIEW_KEYWORDS = {"review", "review this", "re-review", "rereview"}
_LAST_REVIEWED_RE = re.compile(r"<!--\s*noergler:last_reviewed_commit=([0-9a-f]+)\s*-->")
_JIRA_TICKET_RE = re.compile(r'\b([A-Z]{2,10}-\d{1,7})\b')
_SECURITY_KEYWORDS = re.compile(
    r"\b(injection|xss|sql[_ -]?injection|authentication|authorization|"
    r"credentials?|secret[s ]?leak|csrf|ssrf|"
    r"path[_ -]?traversal|insecure|vulnerability|sanitiz(?:e|ation)|"
    r"privilege[_ -]?escalation|deserialization|token[_ -]?leak|"
    r"exposed?[_ -]?(?:secret|credential|key|token))\b",
    re.IGNORECASE,
)


async def _safe_db(coro, fallback=None):
    """Wrap a DB coroutine; on failure log a warning and return the fallback value."""
    try:
        return await coro
    except Exception:
        logger.warning("DB operation failed, using fallback", exc_info=True)
        return fallback


def _is_bot_comment(comment: dict, bot_username: str | None) -> bool:
    """Identify a bot comment by author slug (preferred) or legacy marker (backward compat)."""
    if bot_username and comment.get("author_slug") == bot_username:
        return True
    return NOERGLER_MARKER in comment.get("text", "")


def _count_diff_lines(diff: str) -> tuple[int, int]:
    """Count added and removed lines in a unified diff."""
    added = 0
    removed = 0
    for line in diff.splitlines():
        if line.startswith("+") and not line.startswith("+++"):
            added += 1
        elif line.startswith("-") and not line.startswith("---"):
            removed += 1
    return added, removed


def _extract_question(text: str, trigger: str) -> str:
    return re.sub(rf"@{re.escape(trigger)}\b", "", text, flags=re.IGNORECASE).strip()


class Reviewer:
    def __init__(
        self,
        bitbucket: BitbucketClient,
        copilot: LLMClient,
        review_config: ReviewConfig,
        jira: JiraClient | None = None,
        server_config: ServerConfig | None = None,
        db_pool=None,
    ):
        self.bitbucket = bitbucket
        self.copilot = copilot
        self.review_config = review_config
        self.jira = jira
        self.server_config = server_config or ServerConfig()
        self.db_pool = db_pool
        self.auto_review_authors = review_config.auto_review_authors
        self.max_comments = review_config.max_comments
        self.mention_trigger = review_config.mention_trigger
        self.ramsay_authors = review_config.ramsay_authors

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

    def _tone_for_author(self, author: str) -> str:
        return "ramsay" if author in self.ramsay_authors else "default"

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
        try:
            pr = payload.pullRequest
            author_name = pr.author.user.name
            pr_id = pr.id

            project_key, repo_slug = self._extract_project_repo(payload)
            if not project_key or not repo_slug:
                logger.error("Could not extract project/repo from webhook payload")
                return

            pr_tag = f"{project_key}/{repo_slug}#{pr_id}"

            if not skip_author_check and not self.is_auto_review_author(author_name):
                logger.info(
                    "Skipping %s by %s (not in auto-review authors)", pr_tag, author_name
                )
                return

            logger.info(
                "Starting review of %s by %s (branch: %s)",
                pr_tag, author_name, pr.fromRef.displayId,
            )
            t0 = time.monotonic()

            # Fetch existing comments early (needed for fallback dedup and summary detection)
            existing = await self._fetch_existing_comments(project_key, repo_slug, pr_id)

            # Determine if incremental review is possible
            source_commit = pr.fromRef.latestCommit
            last_reviewed = None

            # Get last reviewed commit: DB first, then fallback to comment parsing
            if self.db_pool:
                last_reviewed = await _safe_db(
                    repository.get_last_reviewed_commit(self.db_pool, project_key, repo_slug, pr_id),
                    fallback=None,
                )
            if last_reviewed is None:
                last_reviewed = _extract_last_reviewed_commit(existing)

            is_incremental = False
            incremental_from: str | None = None

            if payload.eventKey == "pr:from_ref_updated" and last_reviewed and source_commit:
                try:
                    inc_diff = await self.bitbucket.fetch_commit_diff(
                        project_key, repo_slug, last_reviewed, source_commit,
                    )
                    if inc_diff.strip():
                        diff = inc_diff
                        is_incremental = True
                        incremental_from = last_reviewed
                        logger.info(
                            "%s: incremental review %s..%s",
                            pr_tag, last_reviewed[:10], source_commit[:10],
                        )
                    else:
                        logger.info("%s: no changes since last review, skipping", pr_tag)
                        return
                except Exception:
                    logger.warning(
                        "%s: incremental diff failed (rebase?), falling back to full review",
                        pr_tag, exc_info=True,
                    )
                    is_incremental = False

            if not is_incremental:
                diff = await self.bitbucket.fetch_pr_diff(
                    project_key, repo_slug, pr_id, context_lines=0
                )
                if not diff.strip():
                    logger.info("%s has empty diff, skipping", pr_tag)
                    return
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

            template = self.copilot.prompt_template
            max_tokens = self.copilot.config.max_tokens_per_chunk

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

            repo_instructions = await self._fetch_repo_instructions(
                project_key, repo_slug, pr
            )
            agents_md_found = bool(repo_instructions) or any(
                f.path == "AGENTS.md" for f in files
            )

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

            tone = self._tone_for_author(author_name)
            llm_result = await self.copilot.review_diff(
                files, repo_instructions, tone=tone,
                other_modified_paths=other_modified,
                deleted_file_paths=deleted_paths,
                renamed_file_paths=renamed_paths,
                ticket_context=ticket_context,
                ticket_compliance_check=self.review_config.ticket_compliance_check,
                cross_file_context=cross_file_ctx,
            )

            # Deduplicate: DB-backed (preferred) or comment-based fallback
            original_count = len(llm_result.findings)
            existing_keys = None
            if self.db_pool:
                existing_keys = await _safe_db(
                    repository.get_existing_finding_keys(self.db_pool, project_key, repo_slug, pr_id),
                    fallback=None,
                )

            if existing_keys is not None:
                findings = [
                    f for f in llm_result.findings
                    if (f.file, f.line, f.severity) not in existing_keys
                ]
            else:
                findings = _deduplicate(
                    llm_result.findings, existing,
                    bot_username=self.bitbucket.bot_username,
                )

            deduplicated_count = original_count - len(findings)
            findings, truncated = _sort_and_limit(findings, self.max_comments)

            # Upsert PR review record to get pr_review_id for linking findings
            pr_review_id = None
            if self.db_pool:
                pr_review_id = await _safe_db(
                    repository.upsert_pr_review(
                        self.db_pool, project_key, repo_slug, pr_id,
                        last_reviewed_commit=source_commit,
                        author=author_name,
                        pr_title=pr.title,
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
                    if self.db_pool and pr_review_id:
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
            summary = self._build_summary(
                findings,
                truncated,
                agents_md_found=agents_md_found,
                skipped_files=llm_result.skipped_files,
                content_skipped_files=content_skipped,
                token_usage=(llm_result.prompt_tokens, llm_result.completion_tokens),
                prompt_breakdown=llm_result.prompt_breakdown,
                review_effort=llm_result.review_effort,
                ticket=ticket,
                parent_ticket=parent_ticket,
                compliance_requirements=llm_result.compliance_requirements,
                elapsed=elapsed,
                jira_enabled=self.jira is not None,
                ticket_compliance_check=self.review_config.ticket_compliance_check,
                change_summary=llm_result.change_summary,
                reviewed_commit=source_commit,
                incremental_from=incremental_from,
                files_reviewed=len(files),
                total_files=total_files + len(deleted_paths) + len(renamed_paths),
                diff_added=diff_added,
                diff_removed=diff_removed,
                cross_file_symbols=[r.symbol for r in cross_file_rels] if cross_file_rels else None,
            )

            # Find existing summary: DB first, then fallback to comment scanning
            existing_summary = None
            if self.db_pool and pr_review_id:
                db_summary_info = await _safe_db(
                    repository.get_summary_comment_info(self.db_pool, pr_review_id),
                    fallback=None,
                )
                if db_summary_info:
                    existing_summary = {
                        "id": db_summary_info["summary_comment_id"],
                        "version": db_summary_info["summary_comment_version"],
                    }

            if existing_summary is None:
                existing_summary = next(
                    (c for c in existing
                     if _is_bot_comment(c, self.bitbucket.bot_username)
                     and "Review summary" in c.get("text", "")
                     and c.get("path") is None
                     and c.get("id")),
                    None,
                )

            summary_comment_id = None
            summary_comment_version = None
            try:
                updated_version = None
                if existing_summary:
                    updated_version = await self.bitbucket.update_pr_comment(
                        project_key, repo_slug, pr_id,
                        existing_summary["id"], existing_summary["version"], summary,
                    )

                if updated_version is not None:
                    summary_comment_id = existing_summary["id"]
                    summary_comment_version = updated_version
                else:
                    post_result = await self.bitbucket.post_pr_comment(
                        project_key, repo_slug, pr_id, summary
                    )
                    if post_result:
                        summary_comment_id, summary_comment_version = post_result

                if self.db_pool and pr_review_id and summary_comment_id is not None:
                    await _safe_db(
                        repository.update_summary_comment(
                            self.db_pool, pr_review_id,
                            summary_comment_id, summary_comment_version or 0,
                        )
                    )
            except Exception:
                logger.error("Failed to post summary comment", exc_info=True)

            # Persist review statistics
            if self.db_pool:
                counts = {"critical": 0, "warning": 0}
                for f in findings:
                    counts[f.severity] = counts.get(f.severity, 0) + 1
                security_count = len([f for f in findings if _SECURITY_KEYWORDS.search(f.comment)])
                await _safe_db(
                    repository.insert_review_stats(
                        self.db_pool,
                        project_key=project_key,
                        repo_slug=repo_slug,
                        pr_id=pr_id,
                        author=author_name,
                        is_incremental=is_incremental,
                        reviewed_commit=source_commit,
                        diff_added=diff_added,
                        diff_removed=diff_removed,
                        files_reviewed=len(files),
                        total_files=total_files + len(deleted_paths) + len(renamed_paths),
                        critical_count=counts.get("critical", 0),
                        warning_count=counts.get("warning", 0),
                        security_count=security_count,
                        review_effort=llm_result.review_effort,
                        prompt_tokens=llm_result.prompt_tokens,
                        completion_tokens=llm_result.completion_tokens,
                        model_name=self.copilot.config.model,
                        elapsed_seconds=elapsed,
                        cross_file_deps=len(cross_file_rels) if cross_file_rels else 0,
                        skipped_files=len(llm_result.skipped_files),
                        content_skipped=len(content_skipped),
                        findings_posted=posted,
                        findings_deduplicated=deduplicated_count,
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

    async def handle_mention(self, payload: WebhookPayload) -> None:
        comment = payload.comment
        if not comment:
            return

        # Self-loop prevention: ignore bot's own comments
        if comment.author.name == self.bitbucket.bot_username or NOERGLER_MARKER in comment.text:
            logger.debug("Ignoring own comment (bot)")
            return

        pr = payload.pullRequest
        if pr.state and pr.state != "OPEN":
            logger.info("Ignoring mention on non-open PR (state=%s)", pr.state)
            return

        question = _extract_question(comment.text, self.mention_trigger)

        if not question or question.lower() in _REVIEW_KEYWORDS:
            logger.info("Mention triggers full review (question=%r)", question)
            await self.review_pull_request(payload, skip_author_check=True)
            return

        project_key, repo_slug = self._extract_project_repo(payload)
        if not project_key or not repo_slug:
            logger.error("Could not extract project/repo from webhook payload")
            return

        pr_tag = f"{project_key}/{repo_slug}#{pr.id}"
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

            tone = self._tone_for_author(pr.author.user.name)
            answer = await self.copilot.answer_question(
                question, files, repo_instructions, tone=tone,
                ticket_context=ticket_context,
            )
            await self.bitbucket.reply_to_comment(
                project_key, repo_slug, pr.id, comment.id, answer,
            )
            logger.info("Posted Q&A reply on %s", pr_tag)
        except Exception:
            logger.error("Mention Q&A on %s failed", pr_tag, exc_info=True)

    async def handle_pr_merged(self, payload: WebhookPayload) -> None:
        pr = payload.pullRequest
        project_key, repo_slug = self._extract_project_repo(payload)
        if not project_key or not repo_slug:
            return

        pr_tag = f"{project_key}/{repo_slug}#{pr.id}"

        try:
            # DB: delete PR review record (cascades to findings; stats/feedback retained)
            if self.db_pool:
                await _safe_db(
                    repository.delete_pr_review(self.db_pool, project_key, repo_slug, pr.id)
                )

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

            disagreed = 0
            for c in comments:
                parent_id = c.get("parent_id")
                if parent_id in noergler_inline and not _is_bot_comment(c, bot_username):
                    if classify_feedback(c.get("text", "")) == "negative":
                        disagreed += 1

            total = len(noergler_inline)
            useful_pct = (total - disagreed) / total * 100
            logger.info(
                "%s merged — %d comments, %d disagreed (%.0f%% useful)",
                pr_tag, total, disagreed, useful_pct,
            )
        except Exception:
            logger.error("Merged stats for %s failed", pr_tag, exc_info=True)

    async def handle_feedback(self, payload: WebhookPayload) -> None:
        comment = payload.comment
        if not comment or payload.commentParentId is None:
            logger.debug("Feedback skipped: no comment or no commentParentId")
            return

        # Self-loop prevention: ignore bot's own replies
        if comment.author.name == self.bitbucket.bot_username or NOERGLER_MARKER in comment.text:
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
            # Try DB path first: look up the parent finding by Bitbucket comment ID
            parent_file: str | None = None
            parent_line: int | None = None
            parent_severity: str | None = None
            suggestion_text = ""
            using_db_for_parent = False
            existing: list[dict] | None = None

            if self.db_pool:
                finding = await _safe_db(
                    repository.get_finding_by_comment_id(self.db_pool, parent_id),
                    fallback=None,
                )
                if finding is not None:
                    using_db_for_parent = True
                    parent_file = finding["file_path"]
                    parent_line = finding["line_number"]
                    parent_severity = finding["severity"]

            if not using_db_for_parent:
                # Fallback: fetch existing comments and validate parent
                existing = await self.bitbucket.fetch_pr_comments(project_key, repo_slug, pr.id)
                parent_comment = next(
                    (c for c in existing if c.get("id") == parent_id), None
                )

                if not parent_comment or not _is_bot_comment(parent_comment, self.bitbucket.bot_username):
                    logger.debug(
                        "Feedback skipped on %s: parent %d not found or not a bot comment",
                        pr_tag, parent_id,
                    )
                    return

                if not parent_comment.get("path"):
                    logger.debug(
                        "Feedback skipped on %s: parent %d is not an inline comment",
                        pr_tag, parent_id,
                    )
                    return

                parent_file = parent_comment.get("path")
                parent_line = parent_comment.get("line")

                parent_text = parent_comment.get("text", "")
                if "**Suggestion:** " in parent_text:
                    suggestion_text = parent_text.split("**Suggestion:** ", 1)[1].split("\n")[0]

            classification = classify_feedback(comment.text)
            if classification != "negative":
                logger.debug("Feedback skipped on %s: classified as %s", pr_tag, classification)
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
                    random_response(),
                )

            # Persist feedback event in DB
            if self.db_pool:
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

            # Aggregate stats log (fallback path only — DB path has persistent stats)
            if not using_db_for_parent and existing is not None:
                bot_username = self.bitbucket.bot_username
                noergler_comment_ids = {
                    c["id"] for c in existing
                    if _is_bot_comment(c, bot_username) and c.get("id") is not None
                }
                negative = 0
                for c in existing:
                    if c.get("parent_id") in noergler_comment_ids and c.get("id") != comment.id:
                        if classify_feedback(c.get("text", "")) == "negative":
                            negative += 1
                # Include the current reply (it may not be in the fetched list yet)
                if classification == "negative":
                    negative += 1

                logger.info(
                    "Feedback on %s: %d disagreed / %d review comments",
                    pr_tag, negative, len(noergler_comment_ids),
                )
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

    async def _fetch_existing_comments(
        self, project: str, repo: str, pr_id: int
    ) -> list[dict]:
        try:
            return await self.bitbucket.fetch_pr_comments(project, repo, pr_id)
        except Exception:
            logger.warning(
                "Failed to fetch existing comments for dedup, skipping",
                exc_info=True,
            )
            return []

    @staticmethod
    def _plural(n: int, word: str) -> str:
        return f"{n} {word}" if n == 1 else f"{n} {word}s"

    def _build_summary(
        self,
        findings: list[ReviewFinding],
        truncated: bool = False,
        agents_md_found: bool = False,
        skipped_files: list[str] | None = None,
        content_skipped_files: list[str] | None = None,
        token_usage: tuple[int, int] | None = None,
        prompt_breakdown: dict[str, int] | None = None,
        review_effort: int | None = None,
        ticket: JiraTicket | None = None,
        parent_ticket: JiraTicket | None = None,
        compliance_requirements: list[dict] | None = None,
        elapsed: float | None = None,
        jira_enabled: bool = False,
        ticket_compliance_check: bool = True,
        change_summary: list[str] | None = None,
        reviewed_commit: str | None = None,
        incremental_from: str | None = None,
        files_reviewed: int | None = None,
        total_files: int | None = None,
        diff_added: int | None = None,
        diff_removed: int | None = None,
        cross_file_symbols: list[str] | None = None,
    ) -> str:
        if incremental_from and reviewed_commit:
            summary = "### Review summary (incremental update)\n"
            summary += f"- Changes reviewed: `{incremental_from[:10]}` .. `{reviewed_commit[:10]}`\n"
        else:
            summary = "### Review summary (initial review)\n"

        if not findings:
            summary += "- No issues found ✅"
        else:
            counts = {"critical": 0, "warning": 0}
            for f in findings:
                counts[f.severity] = counts.get(f.severity, 0) + 1

            if counts["critical"]:
                summary += f"- {self._plural(counts['critical'], 'critical')} ❌\n"
            if counts["warning"]:
                summary += f"- {self._plural(counts['warning'], 'warning')} ⚠️\n"

            security_findings = [f for f in findings if _SECURITY_KEYWORDS.search(f.comment)]
            if security_findings:
                summary += f"- {self._plural(len(security_findings), 'potential security issue')} — review carefully 🔒\n"

            if truncated:
                summary += f"- Showing top {len(findings)} findings by severity. Additional findings were omitted.\n"

            summary = summary.rstrip("\n")

        if change_summary:
            summary += "\n\n### What changed\n"
            summary += "\n".join(f"- {item}" for item in change_summary)

        ticket_section = ""
        if ticket:
            ticket_lines = ["### Ticket"]
            if parent_ticket:
                ticket_lines.append(f"**[{parent_ticket.key}]({parent_ticket.url})** — {parent_ticket.title}")
                ticket_lines.append(f"**↳ [{ticket.key}]({ticket.url})** — {ticket.title}")
            else:
                ticket_lines.append(f"**[{ticket.key}]({ticket.url})**")
            if ticket_compliance_check and compliance_requirements:
                met_count = sum(1 for r in compliance_requirements if r.get("met"))
                total_count = len(compliance_requirements)
                if met_count == total_count:
                    compliance_level = "Fully compliant"
                    compliance_emoji = "✅"
                elif met_count > 0:
                    compliance_level = "Partially compliant"
                    compliance_emoji = "⚠️"
                else:
                    compliance_level = "Not compliant"
                    compliance_emoji = "❌"
                ticket_lines.append(f"- Compliance: **{compliance_level}** {compliance_emoji}")
                req_lines = [f"    - {r.get('requirement', '???')} {'✅' if r.get('met') else '❌'}" for r in compliance_requirements]
                ticket_lines.append(f"  - Requirements:\n" + "\n".join(req_lines))
            elif not ticket_compliance_check:
                ticket_lines.append("- Ticket compliance check is disabled ℹ️")
            ticket_section = "\n\n" + "\n".join(ticket_lines)
        meta = []
        if not ticket:
            if jira_enabled:
                meta.append("No ticket found in branch name or PR title ℹ️")
            else:
                meta.append("Jira is not enabled ℹ️")
        if review_effort is not None and findings:
            label = _EFFORT_LABELS.get(review_effort, "")
            meta.append(f"Estimated review effort: **{review_effort}/5** — {label} 📊")

        if files_reviewed is not None and total_files is not None:
            if files_reviewed == total_files:
                meta.append(f"Reviewed {files_reviewed} files 📂")
            else:
                skipped_count = total_files - files_reviewed
                meta.append(
                    f"Reviewed {files_reviewed} of {total_files} files "
                    f"({skipped_count} skipped: lock files, binaries, config) 📂"
                )

        if diff_added is not None or diff_removed is not None:
            parts = []
            if diff_added:
                parts.append(f"+{diff_added}")
            if diff_removed:
                parts.append(f"-{diff_removed}")
            if parts:
                meta.append(f"Diff: {' / '.join(parts)} lines")

        if cross_file_symbols:
            symbol_list = ", ".join(f"`{s}`" for s in cross_file_symbols[:5])
            suffix = f" and {len(cross_file_symbols) - 5} more" if len(cross_file_symbols) > 5 else ""
            meta.append(f"{len(cross_file_symbols)} cross-file dependencies analyzed ({symbol_list}{suffix}) 🔗")

        if skipped_files:
            file_list = ", ".join(f"`{PurePosixPath(f).name}`" for f in skipped_files)
            meta.append(f"Not reviewed (too large): {file_list} ⚠️")
        if content_skipped_files:
            file_list = ", ".join(f"`{PurePosixPath(f).name}`" for f in content_skipped_files)
            meta.append(f"Reviewed without full file context (too large): {file_list} ⚠️")

        if agents_md_found:
            meta.append("Using project-specific review guidelines from `AGENTS.md` ✅")
        else:
            meta.append("Tip: Add an `AGENTS.md` to your repository root with project-specific review guidelines for more targeted feedback. 💡")

        if token_usage:
            prompt_t, completion_t = token_usage
            model = self.copilot.config.model
            total = prompt_t + completion_t
            stats = f"Model: `{model}` · {_fmt(prompt_t)}↑ {_fmt(completion_t)}↓ ({_fmt(total)} total)"
            if elapsed is not None:
                stats += f" · ⏱️ {elapsed:.1f}s"
            meta.append(stats)
            if prompt_breakdown:
                t = prompt_breakdown['template']
                r = prompt_breakdown['repo_instructions']
                f = prompt_breakdown['files']
                meta.append(f"Tokens: ~{_fmt(t)} template · ~{_fmt(r)} repo · ~{_fmt(f)} file content")

        if ticket_section:
            summary += ticket_section

        if meta:
            summary += "\n\n### Info\n" + "\n".join(f"- {m}" for m in meta)

        if reviewed_commit:
            summary += f"\n\n<!-- noergler:last_reviewed_commit={reviewed_commit} -->"

        return summary


def _extract_last_reviewed_commit(existing_comments: list[dict]) -> str | None:
    """Find the last-reviewed commit hash from the summary comment metadata (fallback)."""
    for comment in existing_comments:
        text = comment.get("text", "")
        if _is_bot_comment(comment, None) and "Review summary" in text:
            match = _LAST_REVIEWED_RE.search(text)
            if match:
                return match.group(1)
    return None


def _deduplicate(
    findings: list[ReviewFinding],
    existing: list[dict],
    bot_username: str | None = None,
) -> list[ReviewFinding]:
    existing_keys: set[tuple[str | None, int | None, str]] = set()
    for comment in existing:
        if not _is_bot_comment(comment, bot_username):
            continue
        match = _SEVERITY_RE.search(comment.get("text", ""))
        if match:
            severity = match.group(1).lower()
            existing_keys.add((comment.get("path"), comment.get("line"), severity))
    return [
        f for f in findings
        if (f.file, f.line, f.severity) not in existing_keys
    ]


def _sort_and_limit(
    findings: list[ReviewFinding], max_comments: int
) -> tuple[list[ReviewFinding], bool]:
    sorted_findings = sorted(findings, key=lambda f: SEVERITY_ORDER.get(f.severity, 99))
    truncated = len(sorted_findings) > max_comments
    return sorted_findings[:max_comments], truncated
