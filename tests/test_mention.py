from unittest.mock import AsyncMock

import pytest

from app.bitbucket import NOERGLER_MARKER
from app.config import ReviewConfig
from app.jira import JiraTicket
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
    client.bot_username = "noergler"
    client.fetch_pr_diff = AsyncMock(return_value="diff --git a/file.py b/file.py\n+hello\n")
    client.fetch_file_content = AsyncMock(return_value="hello\n")
    client.fetch_pr_comments = AsyncMock(return_value=[])
    client.post_inline_comment = AsyncMock(return_value=None)
    client.post_pr_comment = AsyncMock(return_value=(123, 0))
    client.reply_to_comment = AsyncMock()
    return client


@pytest.fixture
def mock_llm():
    client = AsyncMock()
    client.config.model = "openai/gpt-4.1"
    client.max_tokens_per_chunk = 80000
    client.context_window = 1_000_000
    client.prompt_template = "Review these files:\n{files}\n{repo_instructions}"
    client.review_diff = AsyncMock(return_value=[])
    client.answer_question = AsyncMock(return_value="Here is the answer.")
    return client


@pytest.fixture
def reviewer(mock_bitbucket, mock_llm):
    return Reviewer(
        mock_bitbucket, mock_llm,
        ReviewConfig(auto_review_authors=["pr-author"]),
        db_pool=AsyncMock(),
    )


class TestHandleMention:
    @pytest.mark.asyncio
    async def test_qa_path(self, reviewer, mock_bitbucket, mock_llm):
        payload = _make_mention_payload("@noergler explain this")
        await reviewer.handle_mention(payload)

        mock_llm.answer_question.assert_called_once()
        call_args = mock_llm.answer_question.call_args
        assert call_args[0][0] == "explain this"
        mock_bitbucket.reply_to_comment.assert_called_once()
        assert mock_bitbucket.reply_to_comment.call_args[0][3] == 100

    @pytest.mark.asyncio
    async def test_empty_mention_triggers_review(self, reviewer, mock_llm):
        payload = _make_mention_payload("@noergler")
        await reviewer.handle_mention(payload)

        mock_llm.answer_question.assert_not_called()
        mock_llm.review_diff.assert_called_once()

    @pytest.mark.asyncio
    async def test_review_keyword_triggers_review(self, reviewer, mock_llm):
        payload = _make_mention_payload("@noergler review this")
        await reviewer.handle_mention(payload)

        mock_llm.answer_question.assert_not_called()
        mock_llm.review_diff.assert_called_once()

    @pytest.mark.asyncio
    async def test_self_loop_ignored(self, reviewer, mock_bitbucket, mock_llm):
        payload = _make_mention_payload(f"@noergler some text\n\n{NOERGLER_MARKER}")
        await reviewer.handle_mention(payload)

        mock_llm.answer_question.assert_not_called()
        mock_llm.review_diff.assert_not_called()
        mock_bitbucket.reply_to_comment.assert_not_called()

    @pytest.mark.asyncio
    async def test_closed_pr_ignored(self, reviewer, mock_bitbucket, mock_llm):
        payload = _make_mention_payload("@noergler explain this", pr_state="MERGED")
        await reviewer.handle_mention(payload)

        mock_llm.answer_question.assert_not_called()
        mock_bitbucket.reply_to_comment.assert_not_called()

    @pytest.mark.asyncio
    async def test_qa_with_ticket_context(self, mock_bitbucket, mock_llm):
        mock_jira = AsyncMock()
        ticket = JiraTicket(
            key="SEP-123",
            title="Add login",
            description="Implement login page",
            labels=["frontend"],
            acceptance_criteria=None,
            url="https://jira.example.com/browse/SEP-123",
            issue_type="Story",
            status="In Progress",
        )
        mock_jira.fetch_ticket_with_parent = AsyncMock(return_value=(ticket, None))
        mock_jira.fetch_ticket = AsyncMock(return_value=ticket)

        reviewer = Reviewer(
            mock_bitbucket, mock_llm,
            ReviewConfig(auto_review_authors=["pr-author"]),
            jira=mock_jira,
            db_pool=AsyncMock(),
        )
        payload = _make_mention_payload("@noergler explain this")
        # Set branch to include ticket ID
        payload.pullRequest.fromRef.displayId = "feature/SEP-123-login"
        await reviewer.handle_mention(payload)

        mock_llm.answer_question.assert_called_once()
        call_kwargs = mock_llm.answer_question.call_args[1]
        assert "SEP-123" in call_kwargs["ticket_context"]
        assert "Add login" in call_kwargs["ticket_context"]

    @pytest.mark.asyncio
    async def test_qa_without_jira(self, reviewer, mock_llm):
        payload = _make_mention_payload("@noergler explain this")
        await reviewer.handle_mention(payload)

        mock_llm.answer_question.assert_called_once()
        call_kwargs = mock_llm.answer_question.call_args[1]
        assert call_kwargs["ticket_context"] == ""

    @pytest.mark.asyncio
    async def test_no_comment_returns_without_error(self, reviewer, mock_llm):
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
        mock_llm.answer_question.assert_not_called()


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
