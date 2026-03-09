import json
import logging
import ssl
from pathlib import Path

import httpx
import tiktoken

from app.config import CopilotConfig, ReviewConfig
from app.models import ReviewFinding

logger = logging.getLogger(__name__)


def _count_tokens(text: str) -> int:
    try:
        enc = tiktoken.encoding_for_model("gpt-4")
    except KeyError:
        enc = tiktoken.get_encoding("cl100k_base")
    return len(enc.encode(text))


def _load_prompt_template(template_path: str) -> str:
    path = Path(template_path)
    if not path.exists():
        raise FileNotFoundError(f"Prompt template not found: {template_path}")
    return path.read_text()


def split_diff_into_chunks(
    diff_text: str, max_tokens: int, prompt_template: str
) -> list[str]:
    prompt_overhead = _count_tokens(prompt_template.replace("{diff}", ""))
    available_tokens = max_tokens - prompt_overhead

    file_diffs = _split_by_file(diff_text)

    chunks: list[str] = []
    current_chunk_parts: list[str] = []
    current_tokens = 0

    for file_diff in file_diffs:
        file_tokens = _count_tokens(file_diff)

        if file_tokens > available_tokens:
            if current_chunk_parts:
                chunks.append("\n".join(current_chunk_parts))
                current_chunk_parts = []
                current_tokens = 0
            chunks.append(file_diff)
            continue

        if current_tokens + file_tokens > available_tokens:
            chunks.append("\n".join(current_chunk_parts))
            current_chunk_parts = []
            current_tokens = 0

        current_chunk_parts.append(file_diff)
        current_tokens += file_tokens

    if current_chunk_parts:
        chunks.append("\n".join(current_chunk_parts))

    return chunks if chunks else [diff_text]


def _split_by_file(diff_text: str) -> list[str]:
    parts: list[str] = []
    current_lines: list[str] = []

    for line in diff_text.splitlines(keepends=True):
        if line.startswith("diff --git ") and current_lines:
            parts.append("".join(current_lines))
            current_lines = []
        current_lines.append(line)

    if current_lines:
        parts.append("".join(current_lines))

    return parts


def _parse_review_response(content: str) -> list[ReviewFinding]:
    content = content.strip()
    if content.startswith("```"):
        lines = content.splitlines()
        lines = lines[1:]  # remove opening fence
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        content = "\n".join(lines)

    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        logger.error("Failed to parse review response as JSON: %s", content[:200])
        return []

    if not isinstance(data, list):
        logger.error("Review response is not a JSON array")
        return []

    findings = []
    for item in data:
        try:
            findings.append(ReviewFinding(**item))
        except Exception:
            logger.warning("Skipping malformed finding: %s", item)

    return findings


class CopilotClient:
    def __init__(self, config: CopilotConfig, review_config: ReviewConfig):
        self.config = config
        self.review_config = review_config
        self.client = httpx.AsyncClient(
            headers={
                "Authorization": f"Bearer {config.github_token}",
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
            timeout=120.0,
            verify=ssl.create_default_context(),
        )
        self.prompt_template = _load_prompt_template(
            review_config.review_prompt_template
        )

    async def close(self):
        await self.client.aclose()

    async def validate_model(self) -> dict | None:
        base_url = self.config.api_url.split("/inference")[0]
        models_url = base_url + "/catalog/models"
        try:
            response = await self.client.get(models_url)
            response.raise_for_status()
            models_data = response.json()
            model_list = models_data.get("data", models_data) if isinstance(models_data, dict) else models_data
            for model in model_list:
                if isinstance(model, dict) and model.get("id") == self.config.model:
                    logger.info(
                        "Model %s validated. max_prompt_tokens=%s",
                        self.config.model,
                        model.get("max_prompt_tokens", "unknown"),
                    )
                    return model
            logger.warning(
                "Model %s not found in available models", self.config.model
            )
        except Exception:
            logger.warning("Could not validate model against models API", exc_info=True)
        return None

    async def review_diff(self, diff_text: str) -> list[ReviewFinding]:
        chunks = split_diff_into_chunks(
            diff_text,
            self.config.max_tokens_per_chunk,
            self.prompt_template,
        )

        all_findings: list[ReviewFinding] = []
        for i, chunk in enumerate(chunks):
            logger.info("Reviewing chunk %d/%d", i + 1, len(chunks))
            prompt = self.prompt_template.replace("{diff}", chunk)
            findings = await self._call_api(prompt)
            all_findings.extend(findings)

        return all_findings

    async def _call_api(self, prompt: str) -> list[ReviewFinding]:
        payload = {
            "model": self.config.model,
            "messages": [
                {"role": "system", "content": "You are a code review assistant. Always respond with valid JSON."},
                {"role": "user", "content": prompt},
            ],
        }

        response = await self.client.post(self.config.api_url, json=payload)
        response.raise_for_status()

        data = response.json()
        content = data["choices"][0]["message"]["content"]
        return _parse_review_response(content)
