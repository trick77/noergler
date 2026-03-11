import asyncio
import logging
import re
import time

from app.bitbucket import NOERGLER_MARKER, BitbucketClient
from app.copilot import (
    CopilotClient,
    FileReviewData,
    extract_path,
    is_deleted,
    is_reviewable_diff,
    split_by_file,
)
from app.models import PullRequest, ReviewFinding, WebhookPayload

logger = logging.getLogger(__name__)

SEVERITY_ORDER = {"critical": 0, "warning": 1}
_SEVERITY_RE = re.compile(r"\*\*(\w+):\*\*")
_REVIEW_KEYWORDS = {"review", "review this", "re-review", "rereview"}


def _extract_question(text: str, trigger: str) -> str:
    return re.sub(rf"@{re.escape(trigger)}\b", "", text, flags=re.IGNORECASE).strip()


class Reviewer:
    def __init__(
        self,
        bitbucket: BitbucketClient,
        copilot: CopilotClient,
        auto_review_authors: list[str],
        max_comments: int = 25,
        max_lines_per_file: int = 1000,
        context_lines: int = 0,
        mention_trigger: str = "noergler",
        ramsay_authors: list[str] | None = None,
    ):
        self.bitbucket = bitbucket
        self.copilot = copilot
        self.auto_review_authors = auto_review_authors
        self.max_comments = max_comments
        self.max_lines_per_file = max_lines_per_file
        self.context_lines = context_lines
        self.mention_trigger = mention_trigger
        self.ramsay_authors = ramsay_authors or []

    def _tone_for_author(self, author: str) -> str:
        return "ramsay" if author in self.ramsay_authors else "default"

    def is_auto_review_author(self, author_name: str) -> bool:
        return not self.auto_review_authors or author_name in self.auto_review_authors

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

            diff = await self.bitbucket.fetch_pr_diff(
                project_key, repo_slug, pr_id, context_lines=self.context_lines
            )
            if not diff.strip():
                logger.info("%s has empty diff, skipping", pr_tag)
                return

            # Phase 1: split by file, filter by extension/binary
            all_file_diffs = split_by_file(diff)
            file_diffs = [fd for fd in all_file_diffs if is_reviewable_diff(fd)]
            skipped = len(all_file_diffs) - len(file_diffs)
            logger.info(
                "%s: %d file(s) in diff, %d reviewable, %d skipped (binary/non-reviewable)",
                pr_tag, len(all_file_diffs), len(file_diffs), skipped,
            )
            if not file_diffs:
                logger.info("%s has no reviewable files, skipping", pr_tag)
                return

            source_commit = pr.fromRef.latestCommit

            # Fetch full file content in parallel for non-deleted files
            skipped_large: list[str] = []

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
                # Phase 2: skip if content exceeds max_lines_per_file
                line_count = content.count("\n") + 1 if content else 0
                if content and line_count > self.max_lines_per_file:
                    logger.info(
                        "Skipping %s: %d lines exceeds limit of %d",
                        path, line_count, self.max_lines_per_file,
                    )
                    skipped_large.append(path)
                    return None
                # Drop full file content when diff is larger (context overlap makes it redundant)
                diff_lines = file_diff.count("\n") + 1
                if content and diff_lines >= line_count:
                    logger.info(
                        "Dropping full file content for %s: diff (%d lines) >= file (%d lines)",
                        path, diff_lines, line_count,
                    )
                    content = None
                logger.info("Including %s for review (%d lines, deleted=%s)", path, line_count, deleted)
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

            if not files:
                logger.info("%s has no reviewable files after content fetch, skipping", pr_tag)
                if skipped_large:
                    summary = self._build_summary([], skipped_files=skipped_large)
                    try:
                        await self.bitbucket.post_pr_comment(
                            project_key, repo_slug, pr_id, summary
                        )
                    except Exception:
                        logger.error("Failed to post summary comment", exc_info=True)
                return

            repo_instructions = await self._fetch_repo_instructions(
                project_key, repo_slug, pr
            )
            agents_md_found = bool(repo_instructions) or any(
                f.path == "AGENTS.md" for f in files
            )

            tone = self._tone_for_author(author_name)
            result = await self.copilot.review_diff(files, repo_instructions, tone=tone)

            existing = await self._fetch_existing_comments(project_key, repo_slug, pr_id)
            findings = _deduplicate(result.findings, existing)
            findings, truncated = _sort_and_limit(findings, self.max_comments)

            posted = 0
            failed = 0
            for finding in findings:
                try:
                    await self.bitbucket.post_inline_comment(
                        project_key, repo_slug, pr_id, finding
                    )
                    posted += 1
                except Exception:
                    failed += 1
                    logger.error(
                        "Failed to post inline comment on %s:%d",
                        finding.file,
                        finding.line,
                        exc_info=True,
                    )

            summary = self._build_summary(
                findings,
                truncated,
                agents_md_found=agents_md_found,
                skipped_files=skipped_large + result.skipped_files,
                token_usage=(result.prompt_tokens, result.completion_tokens),
            )
            try:
                await self.bitbucket.post_pr_comment(
                    project_key, repo_slug, pr_id, summary
                )
            except Exception:
                logger.error("Failed to post summary comment", exc_info=True)

            elapsed = time.monotonic() - t0
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

        if NOERGLER_MARKER in comment.text:
            logger.debug("Ignoring own comment (marker found)")
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
            diff = await self.bitbucket.fetch_pr_diff(project_key, repo_slug, pr.id, context_lines=self.context_lines)
            repo_instructions = await self._fetch_repo_instructions(
                project_key, repo_slug, pr
            )
            tone = self._tone_for_author(pr.author.user.name)
            answer = await self.copilot.answer_question(question, diff, repo_instructions, tone=tone)
            await self.bitbucket.reply_to_comment(
                project_key, repo_slug, pr.id, comment.id, answer,
            )
            logger.info("Posted Q&A reply on %s", pr_tag)
        except Exception:
            logger.error("Mention Q&A on %s failed", pr_tag, exc_info=True)

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
        token_usage: tuple[int, int] | None = None,
    ) -> str:
        if not findings:
            summary = "**Noergler review summary:** No issues found. ✅"
        else:
            counts = {"critical": 0, "warning": 0}
            for f in findings:
                counts[f.severity] = counts.get(f.severity, 0) + 1

            rows = []
            if counts["critical"]:
                rows.append(f"| ❌ Critical | {counts['critical']:>5} |")
            if counts["warning"]:
                rows.append(f"| ⚠️ Warning  | {counts['warning']:>5} |")

            table = "\n".join([
                "| Severity | Count |",
                "|----------|------:|",
                *rows,
            ])

            summary = f"**Noergler review summary:** {self._plural(len(findings), 'issue')} found\n\n{table}"
            if truncated:
                summary += f"\n\n_Showing top {len(findings)} findings by severity. Additional findings were omitted._"

        if skipped_files:
            file_list = ", ".join(f"`{f}`" for f in skipped_files)
            summary += f"\n\n⚠️ _Not reviewed (too large): {file_list}_"

        if agents_md_found:
            summary += "\n\n✅ _Using project-specific review guidelines from `AGENTS.md`._"
        else:
            summary += "\n\n💡 _Tip: Add an `AGENTS.md` to your repository root with project-specific review guidelines for more targeted feedback._"

        if token_usage:
            prompt_t, completion_t = token_usage
            model = self.copilot.config.model
            summary += f"\n\n_Model: `{model}` — tokens used: {prompt_t + completion_t:,} ({prompt_t:,} prompt + {completion_t:,} completion)_"

        return summary


def _deduplicate(
    findings: list[ReviewFinding], existing: list[dict]
) -> list[ReviewFinding]:
    existing_keys: set[tuple[str | None, int | None, str]] = set()
    for comment in existing:
        if NOERGLER_MARKER not in comment.get("text", ""):
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
