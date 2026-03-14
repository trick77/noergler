import logging
import re
from dataclasses import dataclass, field

import httpx

from app.config import JiraConfig

logger = logging.getLogger(__name__)


@dataclass
class JiraTicket:
    key: str
    title: str
    description: str | None
    labels: list[str]
    acceptance_criteria: str | None
    subtasks: list[str] = field(default_factory=list)
    url: str = ""
    issue_type: str | None = None
    status: str | None = None

    MAX_DESCRIPTION_LENGTH = 5000


_HEADING_RE = re.compile(r"^h[1-6]\.\s*", re.MULTILINE)
_BLOCK_TAG_RE = re.compile(r"\{(noformat|code(?::[^}]*)?|quote|panel(?::[^}]*)?)}", re.IGNORECASE)
_COLOR_RE = re.compile(r"\{color(?::[^}]*)?\}(.*?)\{color}", re.DOTALL)
_IMAGE_RE = re.compile(r"!(?:[^|!\n]+\|)?[^!\n]+!")
_LINK_RE = re.compile(r"\[([^|]+)\|([^\]]+)]")
_TABLE_HEADER_RE = re.compile(r"\|\|")
_BOLD_RE = re.compile(r"(?<!\w)\*(.+?)\*(?!\w)")
_ITALIC_RE = re.compile(r"(?<!\w)_(.+?)_(?!\w)")
_STRIKE_RE = re.compile(r"(?<!\w)-(.+?)-(?!\w)")
_BLANK_LINES_RE = re.compile(r"\n{3,}")


def _strip_jira_markup(text: str) -> str:
    text = _HEADING_RE.sub("", text)
    text = _BLOCK_TAG_RE.sub("", text)
    text = _COLOR_RE.sub(r"\1", text)
    text = _IMAGE_RE.sub("", text)
    text = _LINK_RE.sub(r"\1 (\2)", text)
    text = _TABLE_HEADER_RE.sub(" | ", text)
    text = _BOLD_RE.sub(r"\1", text)
    text = _ITALIC_RE.sub(r"\1", text)
    text = _STRIKE_RE.sub(r"\1", text)
    text = _BLANK_LINES_RE.sub("\n\n", text)
    return text.strip()


def _extract_acceptance_criteria(description: str, prefixes: list[str]) -> str | None:
    if not prefixes:
        return None
    lines: list[str] = []
    for prefix in prefixes:
        pattern = re.compile(
            rf"^\s*{re.escape(prefix)}(?:[- ]?\d+)?\s*[:.]?\s*.*$",
            re.IGNORECASE | re.MULTILINE,
        )
        for match in pattern.finditer(description):
            line = match.group(0).strip()
            if line and line not in lines:
                lines.append(line)
    return "\n".join(lines) if lines else None


class JiraClient:
    def __init__(self, config: JiraConfig):
        self.config = config
        self.client = httpx.AsyncClient(
            headers={
                "Authorization": f"Bearer {config.token}",
                "Accept": "application/json",
            },
            timeout=30.0,
        )

    async def fetch_ticket(self, ticket_id: str) -> JiraTicket | None:
        url = (
            f"{self.config.url.rstrip('/')}/rest/api/2/issue/{ticket_id}"
            f"?fields=summary,description,labels,subtasks,issuetype,status"
        )
        try:
            response = await self.client.get(url)
            if response.status_code == 404:
                logger.info("Jira ticket %s not found", ticket_id)
                return None
            response.raise_for_status()
        except httpx.ConnectError:
            logger.warning("Failed to connect to Jira for ticket %s", ticket_id)
            return None
        except httpx.HTTPStatusError:
            logger.warning("Jira API error for ticket %s", ticket_id, exc_info=True)
            return None

        data = response.json()
        fields = data.get("fields", {})

        description = fields.get("description")
        if description and len(description) > JiraTicket.MAX_DESCRIPTION_LENGTH:
            description = description[:JiraTicket.MAX_DESCRIPTION_LENGTH] + "..."

        if description:
            description = _strip_jira_markup(description)

        subtasks_raw = fields.get("subtasks", [])
        subtasks = [
            f"{st['key']}: {st.get('fields', {}).get('summary', '')}"
            for st in subtasks_raw
            if isinstance(st, dict) and "key" in st
        ]

        acceptance_criteria = _extract_acceptance_criteria(
            description, self.config.acceptance_criteria_prefixes
        ) if description else None

        issue_type = fields.get("issuetype", {}).get("name") if isinstance(fields.get("issuetype"), dict) else None
        status = fields.get("status", {}).get("name") if isinstance(fields.get("status"), dict) else None

        browse_url = f"{self.config.url.rstrip('/')}/browse/{data.get('key', ticket_id)}"

        return JiraTicket(
            key=data.get("key", ticket_id),
            title=fields.get("summary", ""),
            description=description,
            labels=fields.get("labels", []),
            acceptance_criteria=acceptance_criteria,
            subtasks=subtasks,
            url=browse_url,
            issue_type=issue_type,
            status=status,
        )

    async def check_connectivity(self) -> bool:
        try:
            url = f"{self.config.url.rstrip('/')}/rest/api/2/myself"
            response = await self.client.get(url)
            response.raise_for_status()
            data = response.json()
            logger.info(
                "Jira connectivity OK — authenticated as %s",
                data.get("displayName", data.get("name", "?")),
            )
            return True
        except Exception as exc:
            logger.warning("Jira connectivity check failed: %s", exc)
            return False

    async def close(self):
        await self.client.aclose()
