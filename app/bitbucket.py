import logging
import ssl

import httpx

from app.config import BitbucketConfig
from app.models import ReviewFinding

logger = logging.getLogger(__name__)

NITPICK_MARKER = "<!-- nitpick -->"


class BitbucketClient:
    def __init__(self, config: BitbucketConfig):
        self.config = config
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

    async def fetch_pr_diff(self, project: str, repo: str, pr_id: int) -> str:
        url = f"/rest/api/1.0/projects/{project}/repos/{repo}/pull-requests/{pr_id}/diff"
        response = await self.client.get(
            url, headers={"Accept": "text/plain"}
        )
        response.raise_for_status()
        return response.text

    async def fetch_file_content(
        self, project: str, repo: str, commit: str, path: str
    ) -> str:
        url = f"/rest/api/1.0/projects/{project}/repos/{repo}/browse/{path}"
        response = await self.client.get(
            url,
            params={"at": commit},
            headers={"Accept": "text/plain"},
        )
        response.raise_for_status()
        return response.text

    async def post_inline_comment(
        self,
        project: str,
        repo: str,
        pr_id: int,
        finding: ReviewFinding,
        to_commit: str,
    ) -> None:
        url = f"/rest/api/1.0/projects/{project}/repos/{repo}/pull-requests/{pr_id}/comments"
        payload = {
            "text": f"{NITPICK_MARKER}\n**[{finding.severity.upper()}]** {finding.comment}",
            "anchor": {
                "path": finding.file,
                "line": finding.line,
                "lineType": "ADDED",
                "fileType": "TO",
            },
        }
        response = await self.client.post(url, json=payload)
        response.raise_for_status()
        logger.debug("Posted inline comment on %s:%d", finding.file, finding.line)

    async def post_pr_comment(
        self, project: str, repo: str, pr_id: int, text: str
    ) -> None:
        url = f"/rest/api/1.0/projects/{project}/repos/{repo}/pull-requests/{pr_id}/comments"
        response = await self.client.post(url, json={"text": f"{NITPICK_MARKER}\n{text}"})
        response.raise_for_status()
        logger.debug("Posted summary comment on PR %d", pr_id)

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
                anchor = comment.get("anchor") or activity.get("commentAnchor") or {}
                comments.append({
                    "text": text,
                    "path": anchor.get("path"),
                    "line": anchor.get("line"),
                })
            if data.get("isLastPage", True):
                break
            start = data.get("nextPageStart", start + 1000)
        return comments
