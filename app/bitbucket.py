import logging
import re
import ssl

import httpx

from app.config import BitbucketConfig
from app.models import ReviewFinding

logger = logging.getLogger(__name__)

# Kept for backward-compatible identification of comments posted before bot_username was tracked
NOERGLER_MARKER = "— _noergler_"
SEVERITY_EMOJI = {"critical": "❌", "warning": "⚠️"}


class BitbucketClient:
    def __init__(self, config: BitbucketConfig):
        self.config = config
        self.bot_username: str | None = None
        self.client = httpx.AsyncClient(
            base_url=config.base_url,
            headers={
                "Authorization": f"Bearer {config.token}",
                "Accept": "application/json",
            },
            timeout=30.0,
            verify=ssl.create_default_context(),
        )

    async def close(self):
        await self.client.aclose()

    async def check_connectivity(self) -> None:
        try:
            response = await self.client.get("/rest/api/1.0/application-properties")
            response.raise_for_status()
            data = response.json()
            logger.info(
                "Bitbucket connectivity OK — %s v%s",
                data.get("displayName", "?"),
                data.get("version", "?"),
            )
        except Exception as exc:
            logger.warning("Bitbucket connectivity check failed: %r", exc)

    async def fetch_authenticated_user(self) -> None:
        """Discover and store the bot's Bitbucket username via the whoami endpoint."""
        try:
            response = await self.client.get(
                "/plugins/servlet/applinks/whoami",
                headers={"Accept": "text/plain"},
            )
            if response.status_code == 200:
                username = response.text.strip()
                if username:
                    self.bot_username = username
                    logger.info("Bot username: %s", self.bot_username)
                    return
        except Exception:
            pass
        logger.warning(
            "Could not determine bot username; falling back to marker-based comment identification"
        )

    async def fetch_pr_diff(
        self, project: str, repo: str, pr_id: int, context_lines: int = 0
    ) -> str:
        url = f"/rest/api/1.0/projects/{project}/repos/{repo}/pull-requests/{pr_id}/diff"
        params = {"contextLines": context_lines} if context_lines > 0 else {}
        response = await self.client.get(
            url, headers={"Accept": "text/plain"}, params=params
        )
        response.raise_for_status()
        return response.text

    async def fetch_commit_diff(
        self, project: str, repo: str, from_commit: str, to_commit: str,
    ) -> str:
        """Fetch diff between two commits (for incremental review)."""
        url = f"/rest/api/1.0/projects/{project}/repos/{repo}/compare/diff"
        params = {"from": from_commit, "to": to_commit}
        response = await self.client.get(
            url, headers={"Accept": "text/plain"}, params=params
        )
        response.raise_for_status()
        return response.text

    async def fetch_file_content(
        self, project: str, repo: str, commit: str, path: str
    ) -> str:
        url = f"/rest/api/1.0/projects/{project}/repos/{repo}/raw/{path}"
        response = await self.client.get(
            url,
            params={"at": commit},
        )
        response.raise_for_status()
        return response.text

    async def post_inline_comment(
        self,
        project: str,
        repo: str,
        pr_id: int,
        finding: ReviewFinding,
    ) -> int | None:
        url = f"/rest/api/1.0/projects/{project}/repos/{repo}/pull-requests/{pr_id}/comments"
        path = re.sub(r"^[ab]/", "", finding.file)
        parts = [
            f"**Suggestion:** {finding.comment}",
            f"**Severity Level:** {finding.severity.capitalize()} {SEVERITY_EMOJI.get(finding.severity, '❓')}",
        ]
        if finding.suggestion:
            parts.append(f"**Suggested change:**\n```\n{finding.suggestion}\n```")
        parts.append("_Wrong finding? Reply \"disagree\" if this comment is incorrect or hallucinated._")
        text = "\n\n".join(parts)
        payload = {
            "text": text,
            "anchor": {
                "path": path,
                "line": finding.line,
                "fileType": "TO",
            },
        }
        # Try ADDED first (new lines), fall back to CONTEXT (unchanged lines)
        for line_type in ("ADDED", "CONTEXT"):
            payload["anchor"]["lineType"] = line_type
            response = await self.client.post(url, json=payload)
            if response.status_code != 400:
                break
        if response.status_code == 400:
            logger.warning(
                "Inline comment rejected for %s:%d — %s",
                finding.file, finding.line, response.text,
            )
        response.raise_for_status()
        logger.info("Posted inline comment on %s:%d", finding.file, finding.line)
        return response.json().get("id")

    async def post_pr_comment(
        self, project: str, repo: str, pr_id: int, text: str
    ) -> tuple[int, int] | None:
        url = f"/rest/api/1.0/projects/{project}/repos/{repo}/pull-requests/{pr_id}/comments"
        response = await self.client.post(url, json={"text": text})
        response.raise_for_status()
        data = response.json()
        logger.info("Posted summary comment on PR %d", pr_id)
        return data.get("id"), data.get("version", 0)

    async def reply_to_comment(
        self, project: str, repo: str, pr_id: int,
        parent_comment_id: int, text: str,
    ) -> None:
        url = f"/rest/api/1.0/projects/{project}/repos/{repo}/pull-requests/{pr_id}/comments"
        payload = {
            "text": text,
            "parent": {"id": parent_comment_id},
        }
        response = await self.client.post(url, json=payload)
        response.raise_for_status()
        logger.info("Replied to comment %d on PR %d", parent_comment_id, pr_id)

    async def update_pr_comment(
        self, project: str, repo: str, pr_id: int,
        comment_id: int, version: int, text: str,
    ) -> int | None:
        url = f"/rest/api/1.0/projects/{project}/repos/{repo}/pull-requests/{pr_id}/comments/{comment_id}"
        payload = {"text": text, "version": version}
        response = await self.client.put(url, json=payload)
        if response.status_code == 409:
            logger.warning(
                "Version conflict updating comment %d on PR %d, falling back to new comment",
                comment_id, pr_id,
            )
            return None
        response.raise_for_status()
        logger.info("Updated summary comment %d on PR %d", comment_id, pr_id)
        return response.json().get("version")

    async def add_comment_reaction(
        self, project: str, repo: str, pr_id: int, comment_id: int, emoticon: str = "eyes"
    ) -> bool:
        """Try to add an emoji reaction. Returns True on success, False on failure."""
        url = (
            f"/rest/comment-likes/latest/projects/{project}/repos/{repo}"
            f"/pull-requests/{pr_id}/comments/{comment_id}/reactions"
        )
        try:
            response = await self.client.put(url, json={"emoticon": emoticon})
            if response.status_code < 300:
                return True
            logger.debug("Reaction API returned %d for comment %d", response.status_code, comment_id)
            return False
        except Exception:
            logger.debug("Reaction API failed for comment %d", comment_id, exc_info=False)
            return False

    async def fetch_pr_comments(
        self, project: str, repo: str, pr_id: int
    ) -> list[dict]:
        """Fetch all PR activity comments and return dicts with text, path, line."""
        comments: list[dict] = []
        start = 0
        while True:
            url = f"/rest/api/1.0/projects/{project}/repos/{repo}/pull-requests/{pr_id}/activities"
            response = await self.client.get(url, params={"start": start, "limit": 1000})
            response.raise_for_status()
            data = response.json()
            for activity in data.get("values", []):
                if activity.get("action") != "COMMENTED":
                    continue
                comment = activity.get("comment", {})
                text = comment.get("text", "")
                anchor = activity.get("commentAnchor") or comment.get("anchor") or {}
                author = comment.get("author", {})
                comments.append({
                    "id": comment.get("id"),
                    "version": comment.get("version"),
                    "text": text,
                    "path": anchor.get("path"),
                    "line": anchor.get("line"),
                    "parent_id": comment.get("parent", {}).get("id"),
                    "author_slug": author.get("slug") or author.get("name"),
                })
            if data.get("isLastPage", True):
                break
            start = data.get("nextPageStart", start + 1000)
        return comments
