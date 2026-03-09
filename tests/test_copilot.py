import json

import httpx
import pytest
import respx

from app.config import CopilotConfig, ReviewConfig
from app.copilot import (
    CopilotClient,
    _parse_review_response,
    split_diff_into_chunks,
)


@pytest.fixture
def copilot_config():
    return CopilotConfig(
        model="openai/gpt-4.1",
        github_token="test-token",
        api_url="https://models.github.ai/inference/chat/completions",
        max_tokens_per_chunk=80000,
    )


@pytest.fixture
def review_config():
    return ReviewConfig(
        allowed_authors=["jan.username"],
        review_prompt_template="prompts/review.txt",
    )


class TestParseReviewResponse:
    def test_valid_json_array(self):
        content = json.dumps([
            {"file": "src/main.py", "line": 10, "severity": "error", "comment": "Bug here"}
        ])
        findings = _parse_review_response(content)
        assert len(findings) == 1
        assert findings[0].file == "src/main.py"
        assert findings[0].line == 10
        assert findings[0].severity == "error"

    def test_empty_array(self):
        findings = _parse_review_response("[]")
        assert findings == []

    def test_wrapped_in_code_fence(self):
        content = "```json\n[{\"file\": \"a.py\", \"line\": 1, \"severity\": \"warning\", \"comment\": \"test\"}]\n```"
        findings = _parse_review_response(content)
        assert len(findings) == 1

    def test_invalid_json(self):
        findings = _parse_review_response("not json at all")
        assert findings == []

    def test_not_an_array(self):
        findings = _parse_review_response('{"file": "a.py"}')
        assert findings == []

    def test_malformed_item_skipped(self):
        content = json.dumps([
            {"file": "a.py", "line": 1, "severity": "error", "comment": "good"},
            {"bad": "item"},
        ])
        findings = _parse_review_response(content)
        assert len(findings) == 1


class TestSplitDiffIntoChunks:
    def test_single_file_fits(self):
        diff = "diff --git a/file.py b/file.py\n+hello\n"
        template = "Review:\n{diff}"
        chunks = split_diff_into_chunks(diff, 80000, template)
        assert len(chunks) == 1
        assert "file.py" in chunks[0]

    def test_multiple_files_split(self):
        # Create a diff with two files, each large enough to exceed the limit
        file_a = "diff --git a/a.py b/a.py\n" + "+added line\n" * 50
        file_b = "diff --git a/b.py b/b.py\n" + "+another line\n" * 50
        diff = file_a + file_b
        template = "Review:\n{diff}"
        # Token limit just big enough for one file but not both
        chunks = split_diff_into_chunks(diff, 200, template)
        assert len(chunks) >= 2


class TestCopilotClient:
    @pytest.mark.asyncio
    @respx.mock
    async def test_review_diff(self, copilot_config, review_config):
        review_response = {
            "choices": [
                {
                    "message": {
                        "content": json.dumps([
                            {
                                "file": "src/main.py",
                                "line": 5,
                                "severity": "warning",
                                "comment": "Unused variable",
                            }
                        ])
                    }
                }
            ]
        }

        respx.post("https://models.github.ai/inference/chat/completions").mock(
            return_value=httpx.Response(200, json=review_response)
        )

        client = CopilotClient(copilot_config, review_config)
        try:
            findings = await client.review_diff("diff --git a/src/main.py b/src/main.py\n+x = 1\n")
            assert len(findings) == 1
            assert findings[0].severity == "warning"
        finally:
            await client.close()

    @pytest.mark.asyncio
    @respx.mock
    async def test_validate_model_found(self, copilot_config, review_config):
        models_response = {
            "data": [
                {"id": "openai/gpt-4.1", "max_prompt_tokens": 128000},
                {"id": "openai/gpt-5-mini", "max_prompt_tokens": 64000},
            ]
        }

        respx.get("https://models.github.ai/catalog/models").mock(
            return_value=httpx.Response(200, json=models_response)
        )

        client = CopilotClient(copilot_config, review_config)
        try:
            result = await client.validate_model()
            assert result is not None
            assert result["id"] == "openai/gpt-4.1"
        finally:
            await client.close()

    @pytest.mark.asyncio
    @respx.mock
    async def test_validate_model_not_found(self, copilot_config, review_config):
        models_response = {"data": [{"id": "other-model"}]}

        respx.get("https://models.github.ai/catalog/models").mock(
            return_value=httpx.Response(200, json=models_response)
        )

        client = CopilotClient(copilot_config, review_config)
        try:
            result = await client.validate_model()
            assert result is None
        finally:
            await client.close()
