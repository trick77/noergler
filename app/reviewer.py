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

SEVERITY_ORDER = {"error": 0, "warning": 1, "info": 2}
_SEVERITY_RE = re.compile(r"\*\*\[(\w+)\]\*\*")


class Reviewer:
    def __init__(
        self,
        bitbucket: BitbucketClient,
        copilot: CopilotClient,
        allowed_authors: list[str],
        max_comments: int = 25,
        max_lines_per_file: int = 1000,
    ):
        self.bitbucket = bitbucket
        self.copilot = copilot
        self.allowed_authors = allowed_authors
        self.max_comments = max_comments
        self.max_lines_per_file = max_lines_per_file

    def is_author_allowed(self, author_name: str) -> bool:
        return not self.allowed_authors or author_name in self.allowed_authors

    async def review_pull_request(self, payload: WebhookPayload) -> None:
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

            if not self.is_author_allowed(author_name):
                logger.info(
                    "Skipping %s by %s (not in allowed authors)", pr_tag, author_name
                )
                return

            logger.info(
                "Starting review of %s by %s (branch: %s)",
                pr_tag, author_name, pr.fromRef.displayId,
            )
            t0 = time.monotonic()

            diff = await self.bitbucket.fetch_pr_diff(
                project_key, repo_slug, pr_id
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
                    return None
                logger.info("Including %s for review (%d lines, deleted=%s)", path, line_count, deleted)
                return FileReviewData(path=path, diff=file_diff, content=content)

            results = await asyncio.gather(*[_build_file_data(fd) for fd in file_diffs])
            files = [f for f in results if f is not None]

            if not files:
                logger.info("%s has no reviewable files after content fetch, skipping", pr_tag)
                return

            repo_instructions = await self._fetch_repo_instructions(
                project_key, repo_slug, pr
            )

            findings = await self.copilot.review_diff(files, repo_instructions)

            existing = await self._fetch_existing_comments(project_key, repo_slug, pr_id)
            findings = _deduplicate(findings, existing)
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

            summary = self._build_summary(findings, truncated)
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
        if word == "info":
            return f"{n} {word}"
        return f"{n} {word}" if n == 1 else f"{n} {word}s"

    def _build_summary(
        self, findings: list[ReviewFinding], truncated: bool = False
    ) -> str:
        if not findings:
            return "**Noergler review summary:** No issues found."

        counts = {"error": 0, "warning": 0, "info": 0}
        for f in findings:
            counts[f.severity] = counts.get(f.severity, 0) + 1

        parts = []
        if counts["error"]:
            parts.append(f"🔴 {self._plural(counts['error'], 'error')}")
        if counts["warning"]:
            parts.append(f"🟠 {self._plural(counts['warning'], 'warning')}")
        if counts["info"]:
            parts.append(f"🔵 {self._plural(counts['info'], 'info')}")

        summary = f"**Noergler review summary:** {self._plural(len(findings), 'issue')} found — {', '.join(parts)}"
        if truncated:
            summary += f"\n\n_Showing top {len(findings)} findings by severity. Additional findings were omitted._"
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
