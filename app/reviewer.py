import asyncio
import logging
import re
import time

from app.bitbucket import NOERGLER_MARKER, BitbucketClient
from app.feedback import classify_feedback, random_response
from app.copilot import (
    CopilotClient,
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
_JIRA_TICKET_RE = re.compile(r'\b([A-Z]{2,10}-\d{1,7})\b')
_SECURITY_KEYWORDS = re.compile(
    r"\b(injection|xss|sql[_ -]?injection|authentication|authorization|"
    r"credentials?|secret[s ]?leak|csrf|ssrf|"
    r"path[_ -]?traversal|insecure|vulnerability|sanitiz(?:e|ation)|"
    r"privilege[_ -]?escalation|deserialization|token[_ -]?leak|"
    r"exposed?[_ -]?(?:secret|credential|key|token))\b",
    re.IGNORECASE,
)


def _extract_question(text: str, trigger: str) -> str:
    return re.sub(rf"@{re.escape(trigger)}\b", "", text, flags=re.IGNORECASE).strip()


class Reviewer:
    def __init__(
        self,
        bitbucket: BitbucketClient,
        copilot: CopilotClient,
        review_config: ReviewConfig,
        jira: JiraClient | None = None,
        server_config: ServerConfig | None = None,
    ):
        self.bitbucket = bitbucket
        self.copilot = copilot
        self.review_config = review_config
        self.jira = jira
        self.server_config = server_config or ServerConfig()
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

    async def _fetch_ticket_context(self, ticket_id: str) -> tuple[str, JiraTicket | None]:
        if not self.jira:
            return "", None
        ticket = await self.jira.fetch_ticket(ticket_id)
        if not ticket:
            return "", None
        lines = [f"### Jira ticket: [{ticket.key}]({ticket.url})"]
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
        return "\n".join(lines), ticket

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

            # Always fetch minimal diff first for size estimation
            diff = await self.bitbucket.fetch_pr_diff(
                project_key, repo_slug, pr_id, context_lines=0
            )
            if not diff.strip():
                logger.info("%s has empty diff, skipping", pr_tag)
                return

            source_commit = pr.fromRef.latestCommit
            files, content_skipped = await self._prepare_files(
                project_key, repo_slug, diff, source_commit, pr_tag
            )

            if not files:
                logger.info("%s has no reviewable files after content fetch, skipping", pr_tag)
                return

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

            repo_instructions = await self._fetch_repo_instructions(
                project_key, repo_slug, pr
            )
            agents_md_found = bool(repo_instructions) or any(
                f.path == "AGENTS.md" for f in files
            )

            ticket_id = self._extract_ticket_id(pr)
            ticket_context = ""
            ticket: JiraTicket | None = None
            if ticket_id and self.jira:
                ticket_context, ticket = await self._fetch_ticket_context(ticket_id)
                if ticket:
                    logger.info("%s: linked Jira ticket %s", pr_tag, ticket_id)

            tone = self._tone_for_author(author_name)
            result = await self.copilot.review_diff(
                files, repo_instructions, tone=tone,
                other_modified_paths=other_modified,
                deleted_file_paths=deleted_paths,
                renamed_file_paths=renamed_paths,
                ticket_context=ticket_context,
                ticket_compliance_check=self.review_config.ticket_compliance_check,
            )

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

            elapsed = time.monotonic() - t0
            summary = self._build_summary(
                findings,
                truncated,
                agents_md_found=agents_md_found,
                skipped_files=result.skipped_files,
                content_skipped_files=content_skipped,
                token_usage=(result.prompt_tokens, result.completion_tokens),
                prompt_breakdown=result.prompt_breakdown,
                review_effort=result.review_effort,
                ticket=ticket,
                compliance_requirements=result.compliance_requirements,
                elapsed=elapsed,
                jira_enabled=self.jira is not None,
                ticket_compliance_check=self.review_config.ticket_compliance_check,
            )
            try:
                existing_summary = next(
                    (c for c in existing if NOERGLER_MARKER in c.get("text", "")
                     and "Review summary" in c.get("text", "")
                     and c.get("path") is None and c.get("id")),
                    None,
                )
                updated = False
                if existing_summary:
                    updated = await self.bitbucket.update_pr_comment(
                        project_key, repo_slug, pr_id,
                        existing_summary["id"], existing_summary["version"], summary,
                    )
                if not updated:
                    await self.bitbucket.post_pr_comment(
                        project_key, repo_slug, pr_id, summary
                    )
            except Exception:
                logger.error("Failed to post summary comment", exc_info=True)

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
                ticket_context, _ = await self._fetch_ticket_context(ticket_id)

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
            comments = await self.bitbucket.fetch_pr_comments(project_key, repo_slug, pr.id)

            noergler_inline = {
                c["id"]: c for c in comments
                if NOERGLER_MARKER in c.get("text", "")
                and c.get("path") is not None
                and c.get("id") is not None
            }

            if not noergler_inline:
                logger.info("%s merged — no review comments", pr_tag)
                return

            not_helpful = 0
            for c in comments:
                parent_id = c.get("parent_id")
                if parent_id in noergler_inline and NOERGLER_MARKER not in c.get("text", ""):
                    if classify_feedback(c.get("text", "")) == "negative":
                        not_helpful += 1

            total = len(noergler_inline)
            useful_pct = (total - not_helpful) / total * 100
            logger.info(
                "%s merged — %d comments, %d not helpful (%.0f%% useful)",
                pr_tag, total, not_helpful, useful_pct,
            )
        except Exception:
            logger.error("Merged stats for %s failed", pr_tag, exc_info=True)

    async def handle_feedback(self, payload: WebhookPayload) -> None:
        comment = payload.comment
        if not comment or not comment.parent:
            return

        if NOERGLER_MARKER in comment.text:
            return

        pr = payload.pullRequest
        project_key, repo_slug = self._extract_project_repo(payload)
        if not project_key or not repo_slug:
            return

        pr_tag = f"{project_key}/{repo_slug}#{pr.id}"
        parent_id = comment.parent.id

        try:
            existing = await self.bitbucket.fetch_pr_comments(project_key, repo_slug, pr.id)
            parent_comment = next(
                (c for c in existing if c.get("id") == parent_id), None
            )

            if not parent_comment or NOERGLER_MARKER not in parent_comment.get("text", ""):
                return

            classification = classify_feedback(comment.text)
            if classification != "negative":
                return

            logger.info(
                "%s: %s feedback on comment %d from %s",
                pr_tag, classification, parent_id, comment.author.name,
            )

            reacted = await self.bitbucket.add_comment_reaction(
                project_key, repo_slug, pr.id, comment.id
            )
            if not reacted:
                await self.bitbucket.reply_to_comment(
                    project_key, repo_slug, pr.id, comment.id,
                    random_response(),
                    include_marker=False,
                )

            # Aggregate stats
            noergler_comment_ids = {
                c["id"] for c in existing
                if NOERGLER_MARKER in c.get("text", "") and c.get("id") is not None
            }
            positive = 0
            negative = 0
            feedback_count = 0
            counted_ids: set[int] = set()
            for c in existing:
                if c.get("parent_id") in noergler_comment_ids and c.get("id") != comment.id:
                    fb = classify_feedback(c.get("text", ""))
                    feedback_count += 1
                    counted_ids.add(c["id"])
                    if fb == "positive":
                        positive += 1
                    else:
                        negative += 1
            # Include the current reply (it may not be in the fetched list yet)
            feedback_count += 1
            if classification == "positive":
                positive += 1
            else:
                negative += 1

            logger.info(
                "Feedback on %s: %d/%d comments — %d positive, %d negative",
                pr_tag, feedback_count, len(noergler_comment_ids), positive, negative,
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
        compliance_requirements: list[dict] | None = None,
        elapsed: float | None = None,
        jira_enabled: bool = False,
        ticket_compliance_check: bool = True,
    ) -> str:
        if not findings:
            summary = "🤖 **Review summary:** No issues found. ✅"
        else:
            counts = {"critical": 0, "warning": 0}
            for f in findings:
                counts[f.severity] = counts.get(f.severity, 0) + 1

            parts = []
            if counts["critical"]:
                parts.append(f"❌ {self._plural(counts['critical'], 'critical')}")
            if counts["warning"]:
                parts.append(f"⚠️ {self._plural(counts['warning'], 'warning')}")

            summary = f"🤖 **Review summary:** {self._plural(len(findings), 'issue')} found — {', '.join(parts)}"

            security_findings = [f for f in findings if _SECURITY_KEYWORDS.search(f.comment)]

            if truncated:
                summary += f"\n\nShowing top {len(findings)} findings by severity. Additional findings were omitted."

        ticket_section = ""
        if ticket:
            ticket_lines = [f"**🎫 Ticket: [{ticket.key}]({ticket.url})**"]
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
                ticket_lines.append(f"- {compliance_emoji} Compliance: **{compliance_level}**")
                req_lines = [f"    - {'✅' if r.get('met') else '❌'} {r.get('requirement', '???')}" for r in compliance_requirements]
                ticket_lines.append(f"  - 📋 Requirements:\n" + "\n".join(req_lines))
            elif not ticket_compliance_check:
                ticket_lines.append("- ℹ️ Ticket compliance check is disabled")
            ticket_section = "\n\n" + "\n".join(ticket_lines)
        elif jira_enabled:
            ticket_section = "\n\n" + "ℹ️ No Jira ticket found in branch name or PR title"

        meta = []
        if findings and security_findings:
            meta.append(f"🔒 **Security:** {self._plural(len(security_findings), 'potential security issue')} detected — review these findings carefully.")
        if review_effort is not None:
            label = _EFFORT_LABELS.get(review_effort, "")
            meta.append(f"📊 Estimated review effort: **{review_effort}/5** — {label}")

        if skipped_files:
            file_list = ", ".join(f"`{f}`" for f in skipped_files)
            meta.append(f"⚠️ Not reviewed (too large): {file_list}")
        if content_skipped_files:
            file_list = ", ".join(f"`{f}`" for f in content_skipped_files)
            meta.append(f"⚠️ Reviewed without full file context (too large): {file_list}")

        if agents_md_found:
            meta.append("✅ Using project-specific review guidelines from `AGENTS.md`")
        else:
            meta.append("💡 Tip: Add an `AGENTS.md` to your repository root with project-specific review guidelines for more targeted feedback.")

        if ticket_section:
            summary += ticket_section

        if meta:
            summary += "\n\n" + "\n".join(f"- {m}" for m in meta)

        if token_usage:
            prompt_t, completion_t = token_usage
            model = self.copilot.config.model
            total = prompt_t + completion_t
            stats_line = f"_Model: `{model}` · {_fmt(prompt_t)}↑ {_fmt(completion_t)}↓ ({_fmt(total)} total)"
            if elapsed is not None:
                stats_line += f" · ⏱️ {elapsed:.1f}s"
            stats_line += "_"
            summary += f"\n\n{stats_line}"
            if prompt_breakdown:
                t = prompt_breakdown['template']
                r = prompt_breakdown['repo_instructions']
                f = prompt_breakdown['files']
                summary += (
                    f"\n_↳ ~{_fmt(t)} template · ~{_fmt(r)} repo · ~{_fmt(f)} files_"
                )

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
