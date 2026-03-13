from unittest.mock import AsyncMock

import pytest

from app.bitbucket import NOERGLER_MARKER
from app.config import ReviewConfig
from app.models import WebhookPayload
from app.reviewer import Reviewer, _extract_question


def _make_mention_payload(
    comment_text: str = "@noergler explain this",
    comment_id: int = 100,
    author: str = "someone",
    pr_state: str = "OPEN",
) -> WebhookPayload:
    return WebhookPayload(**{
        "eventKey": "pr:comment:added",
        "comment": {
            "id": comment_id,
            "text": comment_text,
            "author": {"name": author},
        },
        "pullRequest": {
            "id": 42,
            "title": "Test PR",
            "state": pr_state,
            "fromRef": {
                "id": "refs/heads/feature",
                "displayId": "feature",
                "latestCommit": "abc123",
                "repository": {"slug": "my-repo", "project": {"key": "PROJ"}},
            },
            "toRef": {
                "id": "refs/heads/main",
                "displayId": "main",
                "latestCommit": "def456",
                "repository": {"slug": "my-repo", "project": {"key": "PROJ"}},
            },
            "author": {"user": {"name": "pr-author"}},
        },
    })


@pytest.fixture
def mock_bitbucket():
    client = AsyncMock()
    client.fetch_pr_diff = AsyncMock(return_value="diff --git a/file.py b/file.py\n+hello\n")
    client.fetch_file_content = AsyncMock(return_value="hello\n")
    client.fetch_pr_comments = AsyncMock(return_value=[])
    client.post_inline_comment = AsyncMock()
    client.post_pr_comment = AsyncMock()
    client.reply_to_comment = AsyncMock()
    return client


@pytest.fixture
def mock_copilot():
    client = AsyncMock()
    client.config.model = "openai/gpt-5"
    client.config.max_tokens_per_chunk = 80000
    client.prompt_template = "Review these files:\n{files}\n{tone}\n{repo_instructions}"
    client.review_diff = AsyncMock(return_value=[])
    client.answer_question = AsyncMock(return_value="Here is the answer.")
    return client


@pytest.fixture
def reviewer(mock_bitbucket, mock_copilot):
    return Reviewer(
        mock_bitbucket, mock_copilot,
        ReviewConfig(auto_review_authors=["pr-author"]),
    )


class TestHandleMention:
    @pytest.mark.asyncio
    async def test_qa_path(self, reviewer, mock_bitbucket, mock_copilot):
        payload = _make_mention_payload("@noergler explain this")
        await reviewer.handle_mention(payload)

        mock_copilot.answer_question.assert_called_once()
        call_args = mock_copilot.answer_question.call_args
        assert call_args[0][0] == "explain this"
        mock_bitbucket.reply_to_comment.assert_called_once()
        assert mock_bitbucket.reply_to_comment.call_args[0][3] == 100

    @pytest.mark.asyncio
    async def test_empty_mention_triggers_review(self, reviewer, mock_copilot):
        payload = _make_mention_payload("@noergler")
        await reviewer.handle_mention(payload)

        mock_copilot.answer_question.assert_not_called()
        mock_copilot.review_diff.assert_called_once()

    @pytest.mark.asyncio
    async def test_review_keyword_triggers_review(self, reviewer, mock_copilot):
        payload = _make_mention_payload("@noergler review this")
        await reviewer.handle_mention(payload)

        mock_copilot.answer_question.assert_not_called()
        mock_copilot.review_diff.assert_called_once()

    @pytest.mark.asyncio
    async def test_self_loop_ignored(self, reviewer, mock_bitbucket, mock_copilot):
        payload = _make_mention_payload(f"@noergler some text\n\n{NOERGLER_MARKER}")
        await reviewer.handle_mention(payload)

        mock_copilot.answer_question.assert_not_called()
        mock_copilot.review_diff.assert_not_called()
        mock_bitbucket.reply_to_comment.assert_not_called()

    @pytest.mark.asyncio
    async def test_closed_pr_ignored(self, reviewer, mock_bitbucket, mock_copilot):
        payload = _make_mention_payload("@noergler explain this", pr_state="MERGED")
        await reviewer.handle_mention(payload)

        mock_copilot.answer_question.assert_not_called()
        mock_bitbucket.reply_to_comment.assert_not_called()

    @pytest.mark.asyncio
    async def test_no_comment_returns_without_error(self, reviewer, mock_copilot):
        payload = WebhookPayload(**{
            "eventKey": "pr:comment:added",
            "pullRequest": {
                "id": 42,
                "title": "Test PR",
                "fromRef": {
                    "id": "refs/heads/feature",
                    "displayId": "feature",
                    "latestCommit": "abc123",
                    "repository": {"slug": "my-repo", "project": {"key": "PROJ"}},
                },
                "toRef": {
                    "id": "refs/heads/main",
                    "displayId": "main",
                    "latestCommit": "def456",
                    "repository": {"slug": "my-repo", "project": {"key": "PROJ"}},
                },
                "author": {"user": {"name": "pr-author"}},
            },
        })
        await reviewer.handle_mention(payload)
        mock_copilot.answer_question.assert_not_called()


class TestExtractQuestion:
    def test_basic(self):
        assert _extract_question("@noergler explain this", "noergler") == "explain this"

    def test_case_insensitive(self):
        assert _extract_question("@NOERGLER explain this", "noergler") == "explain this"

    def test_multiple_mentions(self):
        result = _extract_question("@noergler hey @noergler", "noergler")
        assert result == "hey"

    def test_only_trigger(self):
        assert _extract_question("@noergler", "noergler") == ""

    def test_custom_trigger(self):
        assert _extract_question("@mybot what is this?", "mybot") == "what is this?"
