import json

import httpx
import pytest
import respx

from app.config import CopilotConfig, ReviewConfig
from app.copilot import (
    CopilotClient,
    _is_reviewable,
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


class TestIsReviewable:
    def test_source_files_are_reviewable(self):
        for ext in [".py", ".java", ".ts", ".tsx", ".js", ".go", ".rs", ".rb", ".c", ".cpp", ".h"]:
            diff = f"diff --git a/src/main{ext} b/src/main{ext}\n+code\n"
            assert _is_reviewable(diff), f"{ext} should be reviewable"

    def test_binary_extensions_skipped(self):
        for ext in [".png", ".jpg", ".pdf", ".zip", ".exe", ".jar", ".pyc"]:
            diff = f"diff --git a/assets/file{ext} b/assets/file{ext}\n+something\n"
            assert not _is_reviewable(diff), f"{ext} should be skipped"

    def test_json_and_lock_files_skipped(self):
        for name in ["package.json", "yarn.lock", "Pipfile.lock"]:
            diff = f"diff --git a/{name} b/{name}\n+content\n"
            assert not _is_reviewable(diff), f"{name} should be skipped"

    def test_minified_files_skipped(self):
        for name in ["bundle.min.js", "styles.min.css"]:
            diff = f"diff --git a/dist/{name} b/dist/{name}\n+minified\n"
            assert not _is_reviewable(diff), f"{name} should be skipped"

    def test_binary_files_differ_marker_skipped(self):
        diff = "diff --git a/image.dat b/image.dat\nBinary files /dev/null and b/image.dat differ\n"
        assert not _is_reviewable(diff)

    def test_config_files_are_reviewable(self):
        for name in ["Dockerfile", "Makefile", ".gitignore", "config.yaml", "deploy.yml", "setup.cfg"]:
            diff = f"diff --git a/{name} b/{name}\n+content\n"
            assert _is_reviewable(diff), f"{name} should be reviewable"

    def test_large_file_skipped(self):
        lines = "+line\n" * 1500
        diff = f"diff --git a/big.py b/big.py\n{lines}"
        assert not _is_reviewable(diff, max_lines=1000)

    def test_file_within_limit_passes(self):
        lines = "+line\n" * 500
        diff = f"diff --git a/small.py b/small.py\n{lines}"
        assert _is_reviewable(diff, max_lines=1000)

    def test_custom_max_lines(self):
        lines = "+line\n" * 60
        diff = f"diff --git a/file.py b/file.py\n{lines}"
        assert not _is_reviewable(diff, max_lines=50)
        assert _is_reviewable(diff, max_lines=100)

    def test_split_diff_filters_large_files(self):
        small_diff = "diff --git a/small.py b/small.py\n+ok\n"
        big_diff = "diff --git a/big.py b/big.py\n" + "+line\n" * 1500
        combined = small_diff + big_diff
        template = "Review:\n{diff}"
        chunks = split_diff_into_chunks(combined, 80000, template, max_lines_per_file=1000)
        full = "\n".join(chunks)
        assert "small.py" in full
        assert "big.py" not in full

    def test_split_diff_filters_non_reviewable(self):
        source_diff = "diff --git a/main.py b/main.py\n+code\n"
        image_diff = "diff --git a/logo.png b/logo.png\n+binary\n"
        json_diff = "diff --git a/package.json b/package.json\n+deps\n"
        combined = source_diff + image_diff + json_diff
        template = "Review:\n{diff}"
        chunks = split_diff_into_chunks(combined, 80000, template)
        full = "\n".join(chunks)
        assert "main.py" in full
        assert "logo.png" not in full
        assert "package.json" not in full


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
            ],
            "usage": {"prompt_tokens": 100, "completion_tokens": 50},
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
