import logging
import re
import time

from app.bitbucket import NITPICK_MARKER, BitbucketClient
from app.copilot import CopilotClient
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
    ):
        self.bitbucket = bitbucket
        self.copilot = copilot
        self.allowed_authors = allowed_authors
        self.max_comments = max_comments

    def is_author_allowed(self, author_name: str) -> bool:
        return author_name in self.allowed_authors

    async def review_pull_request(self, payload: WebhookPayload) -> None:
        pr = payload.pullRequest
        author_name = pr.author.user.name
        pr_id = pr.id

        project_key, repo_slug = self._extract_project_repo(payload)
        if not project_key or not repo_slug:
            logger.error("Could not extract project/repo from webhook payload")
            return

        if not self.is_author_allowed(author_name):
            logger.info(
                "Skipping PR %d by %s (not in allowed authors)", pr_id, author_name
            )
            return

        logger.info("Starting review of PR %d by %s", pr_id, author_name)
        t0 = time.monotonic()

        diff = await self.bitbucket.fetch_pr_diff(project_key, repo_slug, pr_id)
        if not diff.strip():
            logger.info("PR %d has empty diff, skipping", pr_id)
            return

        repo_instructions = await self._fetch_repo_instructions(
            project_key, repo_slug, pr
        )

        findings = await self.copilot.review_diff(diff, repo_instructions)

        existing = await self._fetch_existing_comments(project_key, repo_slug, pr_id)
        findings = _deduplicate(findings, existing)
        findings, truncated = _sort_and_limit(findings, self.max_comments)

        to_commit = pr.fromRef.latestCommit or ""

        posted = 0
        failed = 0
        for finding in findings:
            try:
                await self.bitbucket.post_inline_comment(
                    project_key, repo_slug, pr_id, finding, to_commit
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
            f"Review of PR {pr_id} completed in {elapsed:.1f}s",
            f"{len(findings)} issue{'s' if len(findings) != 1 else ''}",
            f"{posted} comment{'s' if posted != 1 else ''} posted",
        ]
        if failed:
            parts.append(f"{failed} failed")
        logger.info(" — ".join(parts))

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

    def _build_summary(
        self, findings: list[ReviewFinding], truncated: bool = False
    ) -> str:
        if not findings:
            return "**Nitpick Review:** No issues found."

        counts = {"error": 0, "warning": 0, "info": 0}
        for f in findings:
            counts[f.severity] = counts.get(f.severity, 0) + 1

        parts = []
        if counts["error"]:
            parts.append(f"{counts['error']} error(s)")
        if counts["warning"]:
            parts.append(f"{counts['warning']} warning(s)")
        if counts["info"]:
            parts.append(f"{counts['info']} info")

        summary = f"**Nitpick Review:** {len(findings)} issue(s) found — {', '.join(parts)}"
        if truncated:
            summary += f"\n\n_Showing top {len(findings)} findings by severity. Additional findings were omitted._"
        return summary


def _deduplicate(
    findings: list[ReviewFinding], existing: list[dict]
) -> list[ReviewFinding]:
    existing_keys: set[tuple[str | None, int | None, str]] = set()
    for comment in existing:
        if NITPICK_MARKER not in comment.get("text", ""):
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
