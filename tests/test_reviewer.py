from unittest.mock import AsyncMock

import pytest

from app.bitbucket import NOERGLER_MARKER
from app.config import ReviewConfig
from app.copilot import CopilotClient, FileReviewData
from app.jira import JiraClient, JiraTicket
from app.models import ReviewFinding, WebhookPayload
from app.reviewer import Reviewer, _deduplicate, _sort_and_limit


def _review_config(**overrides) -> ReviewConfig:
    defaults = dict(auto_review_authors=["username"])
    defaults.update(overrides)
    return ReviewConfig(**defaults)


def _make_payload(
    author: str = "username",
    branch: str = "feature",
    title: str = "Test PR",
) -> WebhookPayload:
    return WebhookPayload(**{
        "eventKey": "pr:opened",
        "pullRequest": {
            "id": 42,
            "title": title,
            "fromRef": {
                "id": f"refs/heads/{branch}",
                "displayId": branch,
                "latestCommit": "abc123",
                "repository": {"slug": "my-repo", "project": {"key": "PROJ"}},
            },
            "toRef": {
                "id": "refs/heads/main",
                "displayId": "main",
                "latestCommit": "def456",
                "repository": {"slug": "my-repo", "project": {"key": "PROJ"}},
            },
            "author": {"user": {"name": author}},
        },
    })


def _make_mention_payload(
    mention_text: str = "@noergler what does this do?",
    author: str = "username",
) -> WebhookPayload:
    return WebhookPayload(**{
        "eventKey": "pr:comment:added",
        "pullRequest": {
            "id": 42,
            "title": "Test PR",
            "state": "OPEN",
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
            "author": {"user": {"name": author}},
        },
        "comment": {"id": 99, "text": mention_text, "author": {"name": author}},
    })


def _make_real_payload() -> WebhookPayload:
    """Build a WebhookPayload from a real-world Bitbucket Server webhook (anonymized)."""
    return WebhookPayload(**{
        "eventKey": "pr:opened",
        "date": "2025-06-17T06:55:51+0000",
        "actor": {
            "name": "username",
            "emailAddress": "user@example.com",
            "active": True,
            "displayName": "Nit Pick",
            "id": 12345,
            "slug": "username",
            "type": "NORMAL",
            "links": {
                "self": [{"href": "https://bitbucket.example.com/users/username"}]
            },
        },
        "pullRequest": {
            "id": 1,
            "version": 0,
            "title": "test nitpick",
            "state": "OPEN",
            "open": True,
            "closed": False,
            "createdDate": 1750143351522,
            "updatedDate": 1750143351522,
            "fromRef": {
                "id": "refs/heads/test-branch",
                "displayId": "test-branch",
                "latestCommit": "d596f83abcdef1234567890abcdef1234567890ab",
                "repository": {
                    "slug": "test",
                    "id": 999,
                    "name": "test",
                    "description": "test repo for nitpick",
                    "hierarchyId": "abc123def456",
                    "scmId": "git",
                    "state": "AVAILABLE",
                    "statusMessage": "Available",
                    "forkable": True,
                    "project": {
                        "key": "~USERNAME",
                        "id": 100,
                        "name": "Nit Pick",
                        "description": "Personal project of Nit Pick",
                        "public": False,
                        "type": "PERSONAL",
                        "owner": {
                            "name": "username",
                            "emailAddress": "user@example.com",
                            "active": True,
                            "displayName": "Nit Pick",
                            "id": 12345,
                            "slug": "username",
                            "type": "NORMAL",
                            "links": {
                                "self": [{"href": "https://bitbucket.example.com/users/username"}]
                            },
                        },
                        "links": {
                            "self": [{"href": "https://bitbucket.example.com/users/username"}]
                        },
                    },
                    "public": False,
                    "archived": False,
                    "links": {
                        "clone": [
                            {"href": "https://bitbucket.example.com/scm/~username/test.git", "name": "http"},
                            {"href": "ssh://git@bitbucket.example.com:7999/~username/test.git", "name": "ssh"},
                        ],
                        "self": [{"href": "https://bitbucket.example.com/users/username/repos/test/browse"}],
                    },
                },
            },
            "toRef": {
                "id": "refs/heads/master",
                "displayId": "master",
                "latestCommit": "a1b2c3d4e5f67890a1b2c3d4e5f67890a1b2c3d4",
                "repository": {
                    "slug": "test",
                    "id": 999,
                    "name": "test",
                    "description": "test repo for nitpick",
                    "hierarchyId": "abc123def456",
                    "scmId": "git",
                    "state": "AVAILABLE",
                    "statusMessage": "Available",
                    "forkable": True,
                    "project": {
                        "key": "~USERNAME",
                        "id": 100,
                        "name": "Nit Pick",
                        "description": "Personal project of Nit Pick",
                        "public": False,
                        "type": "PERSONAL",
                        "owner": {
                            "name": "username",
                            "emailAddress": "user@example.com",
                            "active": True,
                            "displayName": "Nit Pick",
                            "id": 12345,
                            "slug": "username",
                            "type": "NORMAL",
                            "links": {
                                "self": [{"href": "https://bitbucket.example.com/users/username"}]
                            },
                        },
                        "links": {
                            "self": [{"href": "https://bitbucket.example.com/users/username"}]
                        },
                    },
                    "public": False,
                    "archived": False,
                    "links": {
                        "clone": [
                            {"href": "https://bitbucket.example.com/scm/~username/test.git", "name": "http"},
                            {"href": "ssh://git@bitbucket.example.com:7999/~username/test.git", "name": "ssh"},
                        ],
                        "self": [{"href": "https://bitbucket.example.com/users/username/repos/test/browse"}],
                    },
                },
            },
            "locked": False,
            "author": {
                "user": {
                    "name": "username",
                    "emailAddress": "user@example.com",
                    "active": True,
                    "displayName": "Nit Pick",
                    "id": 12345,
                    "slug": "username",
                    "type": "NORMAL",
                    "links": {
                        "self": [{"href": "https://bitbucket.example.com/users/username"}]
                    },
                },
                "role": "AUTHOR",
                "approved": False,
                "status": "UNAPPROVED",
            },
            "reviewers": [],
            "participants": [],
            "links": {
                "self": [{"href": "https://bitbucket.example.com/users/username/repos/test/pull-requests/1"}]
            },
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
    client.update_pr_comment = AsyncMock(return_value=True)
    return client


def _make_review_result(findings=None, skipped_files=None, review_effort=1):
    return CopilotClient.ReviewResult(
        findings=findings or [],
        skipped_files=skipped_files or [],
        prompt_tokens=100,
        completion_tokens=50,
        review_effort=review_effort,
    )


@pytest.fixture
def mock_copilot():
    client = AsyncMock()
    client.config.model = "openai/gpt-4.1"
    client.config.max_tokens_per_chunk = 80000
    client.prompt_template = "Review these files:\n{files}\n{tone}\n{repo_instructions}"
    client.review_diff = AsyncMock(return_value=_make_review_result([
        ReviewFinding(file="file.py", line=1, severity="warning", comment="Test issue"),
    ]))
    return client


@pytest.fixture
def reviewer(mock_bitbucket, mock_copilot):
    return Reviewer(mock_bitbucket, mock_copilot, _review_config())


class TestReviewer:
    @pytest.mark.asyncio
    async def test_review_allowed_author(self, reviewer, mock_bitbucket, mock_copilot):
        payload = _make_payload("username")
        await reviewer.review_pull_request(payload)

        # Single diff fetch with context_lines=0, context expanded locally
        mock_bitbucket.fetch_pr_diff.assert_called_once_with("PROJ", "my-repo", 42, context_lines=0)

        mock_copilot.review_diff.assert_called_once()
        call_args = mock_copilot.review_diff.call_args
        files = call_args[0][0]
        assert len(files) == 1
        assert isinstance(files[0], FileReviewData)
        assert files[0].path == "file.py"

        mock_bitbucket.post_inline_comment.assert_called_once()
        mock_bitbucket.post_pr_comment.assert_called_once()

        summary_text = mock_bitbucket.post_pr_comment.call_args[0][3]
        assert "1 issue found" in summary_text

    @pytest.mark.asyncio
    async def test_skip_disallowed_author(self, reviewer, mock_bitbucket, mock_copilot):
        payload = _make_payload("other.user")
        await reviewer.review_pull_request(payload)

        mock_bitbucket.fetch_pr_diff.assert_not_called()
        mock_copilot.review_diff.assert_not_called()

    @pytest.mark.asyncio
    async def test_skip_author_check_allows_non_listed_author(self, reviewer, mock_bitbucket, mock_copilot):
        payload = _make_payload("other.user")
        await reviewer.review_pull_request(payload, skip_author_check=True)

        # Single diff fetch, context expanded locally
        mock_bitbucket.fetch_pr_diff.assert_called_once()
        mock_copilot.review_diff.assert_called_once()

    @pytest.mark.asyncio
    async def test_skip_empty_diff(self, reviewer, mock_bitbucket, mock_copilot):
        mock_bitbucket.fetch_pr_diff.return_value = "   \n"
        payload = _make_payload("username")
        await reviewer.review_pull_request(payload)

        mock_copilot.review_diff.assert_not_called()

    @pytest.mark.asyncio
    async def test_no_findings(self, reviewer, mock_bitbucket, mock_copilot):
        mock_copilot.review_diff.return_value = _make_review_result([])
        payload = _make_payload("username")
        await reviewer.review_pull_request(payload)

        mock_bitbucket.post_inline_comment.assert_not_called()
        summary_text = mock_bitbucket.post_pr_comment.call_args[0][3]
        assert "No issues found" in summary_text

    @pytest.mark.asyncio
    async def test_content_fetch_failure_falls_back_to_diff_only(self, mock_bitbucket, mock_copilot):
        mock_bitbucket.fetch_file_content = AsyncMock(side_effect=Exception("not found"))
        rev = Reviewer(mock_bitbucket, mock_copilot, _review_config())
        payload = _make_payload("username")
        await rev.review_pull_request(payload)

        # Should still review, just with content=None
        mock_copilot.review_diff.assert_called_once()
        files = mock_copilot.review_diff.call_args[0][0]
        assert files[0].content is None

    @pytest.mark.asyncio
    async def test_content_preserved_for_small_pr(self, mock_bitbucket, mock_copilot):
        mock_bitbucket.fetch_file_content = AsyncMock(return_value="hello\n")
        rev = Reviewer(mock_bitbucket, mock_copilot, _review_config())
        payload = _make_payload("username")
        await rev.review_pull_request(payload)

        mock_copilot.review_diff.assert_called_once()
        files = mock_copilot.review_diff.call_args[0][0]
        assert len(files) == 1
        assert files[0].content == "hello\n"

    @pytest.mark.asyncio
    async def test_context_expanded_for_large_pr(self, mock_bitbucket, mock_copilot):
        file_content = "\n".join(f"line {i}" for i in range(50))
        # Return a diff with a hunk and enough content that file_content provides context
        mock_bitbucket.fetch_pr_diff = AsyncMock(
            return_value="diff --git a/file.py b/file.py\n@@ -10,1 +10,1 @@\n-old\n+new\n"
        )
        mock_bitbucket.fetch_file_content = AsyncMock(return_value=file_content)
        # Use a tiny max_tokens to force the large PR path
        mock_copilot.config.max_tokens_per_chunk = 1
        mock_copilot.prompt_template = "{files}\n{tone}\n{repo_instructions}"
        mock_copilot.review_diff = AsyncMock(return_value=_make_review_result())

        rev = Reviewer(mock_bitbucket, mock_copilot, _review_config())
        payload = _make_payload("username")
        await rev.review_pull_request(payload)

        # Single diff fetch — no second call for expanded context
        mock_bitbucket.fetch_pr_diff.assert_called_once()
        # review_diff may or may not be called depending on compression,
        # but the key assertion is no second fetch_pr_diff call

    def test_is_auto_review_author(self, reviewer):
        assert reviewer.is_auto_review_author("username") is True
        assert reviewer.is_auto_review_author("other.user") is False

    def test_is_auto_review_author_empty_list(self, mock_bitbucket, mock_copilot):
        rev = Reviewer(mock_bitbucket, mock_copilot, _review_config(auto_review_authors=[]))
        assert rev.is_auto_review_author("anyone") is True

    def test_build_summary_mixed(self, reviewer):
        findings = [
            ReviewFinding(file="a.py", line=1, severity="critical", comment="err"),
            ReviewFinding(file="b.py", line=2, severity="warning", comment="warn"),
        ]
        summary = reviewer._build_summary(findings)
        assert "2 issues found" in summary
        assert "critical" in summary
        assert "warning" in summary

    def test_build_summary_empty(self, reviewer):
        assert "No issues found. ✅" in reviewer._build_summary([])

    @pytest.mark.asyncio
    async def test_review_real_webhook_payload(self, mock_bitbucket, mock_copilot):
        payload = _make_real_payload()
        reviewer = Reviewer(mock_bitbucket, mock_copilot, _review_config(auto_review_authors=["username"]))
        await reviewer.review_pull_request(payload)

        # Single diff fetch, context expanded locally
        mock_bitbucket.fetch_pr_diff.assert_called_once_with("~USERNAME", "test", 1, context_lines=0)
        mock_copilot.review_diff.assert_called_once()
        mock_bitbucket.post_inline_comment.assert_called_once()
        mock_bitbucket.post_pr_comment.assert_called_once()


class TestDedupAndLimit:
    def test_findings_limited_to_max_comments(self):
        findings = [
            ReviewFinding(file=f"f{i}.py", line=i, severity="warning", comment=f"issue {i}")
            for i in range(30)
        ]
        limited, truncated = _sort_and_limit(findings, max_comments=5)
        assert len(limited) == 5
        assert truncated is True

    def test_no_truncation_when_under_limit(self):
        findings = [
            ReviewFinding(file="a.py", line=1, severity="warning", comment="issue"),
        ]
        limited, truncated = _sort_and_limit(findings, max_comments=25)
        assert len(limited) == 1
        assert truncated is False

    def test_findings_sorted_by_severity(self):
        findings = [
            ReviewFinding(file="a.py", line=1, severity="warning", comment="warn"),
            ReviewFinding(file="b.py", line=2, severity="critical", comment="err"),
        ]
        sorted_findings, _ = _sort_and_limit(findings, max_comments=25)
        assert [f.severity for f in sorted_findings] == ["critical", "warning"]

    def test_dedup_skips_existing_nitpick_comments(self):
        findings = [
            ReviewFinding(file="a.py", line=10, severity="critical", comment="bug"),
            ReviewFinding(file="b.py", line=20, severity="warning", comment="style"),
        ]
        existing = [
            {"text": f"**Suggestion:** bug\n\n**Severity Level:** Critical ❌\n\n{NOERGLER_MARKER}", "path": "a.py", "line": 10},
        ]
        result = _deduplicate(findings, existing)
        assert len(result) == 1
        assert result[0].file == "b.py"

    def test_dedup_ignores_human_comments(self):
        findings = [
            ReviewFinding(file="a.py", line=10, severity="critical", comment="bug"),
        ]
        existing = [
            {"text": "**Severity Level:** Critical ❌", "path": "a.py", "line": 10},
        ]
        result = _deduplicate(findings, existing)
        assert len(result) == 1

    def test_dedup_ignores_different_severity(self):
        findings = [
            ReviewFinding(file="a.py", line=10, severity="warning", comment="style"),
        ]
        existing = [
            {"text": f"**Suggestion:** bug\n\n**Severity Level:** Critical ❌\n\n{NOERGLER_MARKER}", "path": "a.py", "line": 10},
        ]
        result = _deduplicate(findings, existing)
        assert len(result) == 1

    @pytest.fixture
    def reviewer(self, mock_bitbucket, mock_copilot):
        return Reviewer(mock_bitbucket, mock_copilot, _review_config())

    def test_build_summary_truncated(self, reviewer):
        findings = [
            ReviewFinding(file="a.py", line=1, severity="critical", comment="err"),
            ReviewFinding(file="b.py", line=2, severity="warning", comment="warn"),
        ]
        summary = reviewer._build_summary(findings, truncated=True)
        assert "2 issues found" in summary
        assert "Additional findings were omitted" in summary

    def test_build_summary_agents_md_not_found(self, reviewer):
        findings = [
            ReviewFinding(file="a.py", line=1, severity="warning", comment="warn"),
        ]
        summary = reviewer._build_summary(findings, agents_md_found=False)
        assert "- 💡" in summary
        assert "AGENTS.md" in summary
        assert "Tip:" in summary

    def test_build_summary_agents_md_found(self, reviewer):
        findings = [
            ReviewFinding(file="a.py", line=1, severity="warning", comment="warn"),
        ]
        summary = reviewer._build_summary(findings, agents_md_found=True)
        assert "- ✅" in summary
        assert "Using project-specific review guidelines" in summary
        assert "Tip:" not in summary

    def test_build_summary_no_findings_agents_md_not_found(self, reviewer):
        summary = reviewer._build_summary([], agents_md_found=False)
        assert "No issues found" in summary
        assert "- 💡" in summary
        assert "Tip:" in summary

    def test_build_summary_no_findings_agents_md_found(self, reviewer):
        summary = reviewer._build_summary([], agents_md_found=True)
        assert "No issues found" in summary
        assert "- ✅" in summary
        assert "Using project-specific review guidelines" in summary
        assert "Tip:" not in summary

    def test_build_summary_skipped_files(self, reviewer):
        findings = [
            ReviewFinding(file="a.py", line=1, severity="warning", comment="warn"),
        ]
        summary = reviewer._build_summary(findings, skipped_files=["huge.py", "big.js"])
        assert "- ⚠️ Not reviewed (too large)" in summary
        assert "`huge.py`" in summary
        assert "`big.js`" in summary

    def test_build_summary_no_skipped_files(self, reviewer):
        summary = reviewer._build_summary([], skipped_files=[])
        assert "Not reviewed" not in summary

    def test_build_summary_token_usage(self, reviewer):
        findings = [
            ReviewFinding(file="a.py", line=1, severity="warning", comment="warn"),
        ]
        summary = reviewer._build_summary(findings, token_usage=(1000, 500))
        assert "Model: `openai/gpt-4.1`" in summary
        assert "1,000↑" in summary
        assert "500↓" in summary
        assert "1,500 total" in summary

    def test_build_summary_prompt_breakdown(self, reviewer):
        findings = [
            ReviewFinding(file="a.py", line=1, severity="warning", comment="warn"),
        ]
        breakdown = {"template": 500, "repo_instructions": 200, "files": 7258}
        summary = reviewer._build_summary(
            findings,
            token_usage=(7958, 6764),
            prompt_breakdown=breakdown,
        )
        assert "~500 template" in summary
        assert "~200 repo" in summary
        assert "~7,258 files" in summary

    def test_build_summary_prompt_breakdown_without_token_usage(self, reviewer):
        findings = [
            ReviewFinding(file="a.py", line=1, severity="warning", comment="warn"),
        ]
        breakdown = {"template": 500, "repo_instructions": 0, "files": 7258}
        summary = reviewer._build_summary(findings, prompt_breakdown=breakdown)
        assert "template" not in summary

    @pytest.mark.asyncio
    async def test_findings_limited_in_review(self, mock_bitbucket, mock_copilot):
        mock_copilot.review_diff.return_value = _make_review_result([
            ReviewFinding(file=f"f{i}.py", line=i, severity="warning", comment=f"issue {i}")
            for i in range(30)
        ])
        rev = Reviewer(mock_bitbucket, mock_copilot, _review_config(max_comments=5))
        payload = _make_payload("username")
        await rev.review_pull_request(payload)

        assert mock_bitbucket.post_inline_comment.call_count == 5
        summary_text = mock_bitbucket.post_pr_comment.call_args[0][3]
        assert "Additional findings were omitted" in summary_text

    @pytest.mark.asyncio
    async def test_dedup_graceful_on_fetch_failure(self, mock_bitbucket, mock_copilot):
        mock_bitbucket.fetch_pr_comments.side_effect = Exception("API error")
        rev = Reviewer(mock_bitbucket, mock_copilot, _review_config())
        payload = _make_payload("username")
        await rev.review_pull_request(payload)

        mock_bitbucket.post_inline_comment.assert_called_once()
        mock_bitbucket.post_pr_comment.assert_called_once()

    @pytest.mark.asyncio
    async def test_persistent_comment_updates_existing(self, mock_bitbucket, mock_copilot):
        mock_bitbucket.update_pr_comment = AsyncMock(return_value=True)
        mock_bitbucket.fetch_pr_comments.return_value = [
            {
                "id": 77,
                "version": 2,
                "text": f"🤖 **Review summary:** old\n\n{NOERGLER_MARKER}",
                "path": None,
                "line": None,
            },
        ]
        rev = Reviewer(mock_bitbucket, mock_copilot, _review_config())
        payload = _make_payload("username")
        await rev.review_pull_request(payload)

        mock_bitbucket.update_pr_comment.assert_called_once()
        call_args = mock_bitbucket.update_pr_comment.call_args[0]
        assert call_args[3] == 77  # comment_id
        assert call_args[4] == 2   # version
        mock_bitbucket.post_pr_comment.assert_not_called()

    @pytest.mark.asyncio
    async def test_persistent_comment_creates_new_when_none_exists(self, mock_bitbucket, mock_copilot):
        mock_bitbucket.fetch_pr_comments.return_value = []
        rev = Reviewer(mock_bitbucket, mock_copilot, _review_config())
        payload = _make_payload("username")
        await rev.review_pull_request(payload)

        mock_bitbucket.post_pr_comment.assert_called_once()
        mock_bitbucket.update_pr_comment.assert_not_called()

    @pytest.mark.asyncio
    async def test_persistent_comment_ignores_inline_markers(self, mock_bitbucket, mock_copilot):
        mock_bitbucket.fetch_pr_comments.return_value = [
            {
                "id": 88,
                "version": 1,
                "text": f"**Suggestion:** bug\n\n{NOERGLER_MARKER}",
                "path": "a.py",
                "line": 10,
            },
        ]
        rev = Reviewer(mock_bitbucket, mock_copilot, _review_config())
        payload = _make_payload("username")
        await rev.review_pull_request(payload)

        mock_bitbucket.post_pr_comment.assert_called_once()
        mock_bitbucket.update_pr_comment.assert_not_called()

    def test_build_summary_effort_score(self, reviewer):
        findings = [
            ReviewFinding(file="a.py", line=1, severity="warning", comment="warn"),
        ]
        summary = reviewer._build_summary(findings, review_effort=3)
        assert "- 📊" in summary
        assert "**3/5**" in summary
        assert "Medium" in summary

    def test_build_summary_no_effort_score(self, reviewer):
        summary = reviewer._build_summary([], review_effort=None)
        assert "📊" not in summary

    def test_build_summary_security_section(self, reviewer):
        findings = [
            ReviewFinding(file="a.py", line=1, severity="critical", comment="SQL injection vulnerability found"),
            ReviewFinding(file="b.py", line=2, severity="warning", comment="unused import"),
        ]
        summary = reviewer._build_summary(findings)
        assert "- 🔒" in summary
        assert "1 potential security issue" in summary

    def test_build_summary_no_security_section(self, reviewer):
        findings = [
            ReviewFinding(file="a.py", line=1, severity="warning", comment="unused variable"),
        ]
        summary = reviewer._build_summary(findings)
        assert "🔒" not in summary

    def test_build_summary_multiple_security_findings(self, reviewer):
        findings = [
            ReviewFinding(file="a.py", line=1, severity="critical", comment="SQL injection risk in query"),
            ReviewFinding(file="b.py", line=5, severity="warning", comment="XSS vulnerability in template"),
        ]
        summary = reviewer._build_summary(findings)
        assert "🔒" in summary
        assert "2 potential security issues" in summary


class TestHandleMention:
    @pytest.mark.asyncio
    async def test_mention_uses_file_level_preparation(self, mock_bitbucket, mock_copilot):
        mock_copilot.answer_question = AsyncMock(return_value="Here's the answer.")
        mock_bitbucket.reply_to_comment = AsyncMock()

        rev = Reviewer(mock_bitbucket, mock_copilot, _review_config())
        payload = _make_mention_payload("@noergler what does this do?")
        await rev.handle_mention(payload)

        # answer_question should receive FileReviewData list, not raw string
        mock_copilot.answer_question.assert_called_once()
        call_args = mock_copilot.answer_question.call_args
        question = call_args[0][0]
        files = call_args[0][1]
        assert question == "what does this do?"
        assert isinstance(files, list)
        assert len(files) == 1
        assert isinstance(files[0], FileReviewData)
        assert files[0].path == "file.py"

        mock_bitbucket.reply_to_comment.assert_called_once()
        assert mock_bitbucket.reply_to_comment.call_args[0][4] == "Here's the answer."

    @pytest.mark.asyncio
    async def test_mention_no_reviewable_files(self, mock_bitbucket, mock_copilot):
        mock_bitbucket.fetch_pr_diff.return_value = (
            "diff --git a/image.png b/image.png\nBinary files differ\n"
        )
        mock_bitbucket.reply_to_comment = AsyncMock()

        rev = Reviewer(mock_bitbucket, mock_copilot, _review_config())
        payload = _make_mention_payload("@noergler what does this do?")
        await rev.handle_mention(payload)

        mock_copilot.answer_question.assert_not_called()
        mock_bitbucket.reply_to_comment.assert_called_once()
        reply_text = mock_bitbucket.reply_to_comment.call_args[0][4]
        assert "No reviewable files" in reply_text

    @pytest.mark.asyncio
    async def test_mention_triggers_full_review(self, mock_bitbucket, mock_copilot):
        mock_bitbucket.reply_to_comment = AsyncMock()
        rev = Reviewer(mock_bitbucket, mock_copilot, _review_config())
        payload = _make_mention_payload("@noergler review")
        await rev.handle_mention(payload)

        # Should trigger review_diff, not answer_question
        mock_copilot.review_diff.assert_called_once()
        mock_copilot.answer_question.assert_not_called()


class TestTicketExtraction:
    def test_extract_ticket_id_from_branch(self, mock_bitbucket, mock_copilot):
        rev = Reviewer(mock_bitbucket, mock_copilot, _review_config())
        payload = _make_payload(branch="feature/SEP-22888-description")
        assert rev._extract_ticket_id(payload.pullRequest) == "SEP-22888"

    def test_extract_ticket_id_from_branch_slash(self, mock_bitbucket, mock_copilot):
        rev = Reviewer(mock_bitbucket, mock_copilot, _review_config())
        payload = _make_payload(branch="SEP-22888/description")
        assert rev._extract_ticket_id(payload.pullRequest) == "SEP-22888"

    def test_extract_ticket_id_from_title(self, mock_bitbucket, mock_copilot):
        rev = Reviewer(mock_bitbucket, mock_copilot, _review_config())
        payload = _make_payload(branch="feature/no-ticket", title="SEP-22888 Fix login")
        assert rev._extract_ticket_id(payload.pullRequest) == "SEP-22888"

    def test_extract_ticket_id_none(self, mock_bitbucket, mock_copilot):
        rev = Reviewer(mock_bitbucket, mock_copilot, _review_config())
        payload = _make_payload(branch="feature/no-ticket", title="Fix login bug")
        assert rev._extract_ticket_id(payload.pullRequest) is None

    def test_extract_ticket_id_branch_priority(self, mock_bitbucket, mock_copilot):
        rev = Reviewer(mock_bitbucket, mock_copilot, _review_config())
        payload = _make_payload(branch="feature/SEP-111-fix", title="SEP-222 other fix")
        assert rev._extract_ticket_id(payload.pullRequest) == "SEP-111"


class TestBuildSummaryWithTicket:
    @pytest.fixture
    def reviewer(self, mock_bitbucket, mock_copilot):
        return Reviewer(mock_bitbucket, mock_copilot, _review_config())

    def test_build_summary_with_ticket(self, reviewer):
        ticket = JiraTicket(
            key="SEP-22888",
            title="Config security",
            description=None,
            labels=[],
            acceptance_criteria=None,
            url="https://jira.example.com/browse/SEP-22888",
        )
        summary = reviewer._build_summary([], ticket=ticket)
        assert "**🎫 Ticket: [SEP-22888](https://jira.example.com/browse/SEP-22888)**" in summary

    def test_build_summary_compliance_all_met(self, reviewer):
        ticket = JiraTicket(
            key="SEP-100", title="Test", description=None,
            labels=[], acceptance_criteria=None,
            url="https://jira.example.com/browse/SEP-100",
        )
        requirements = [
            {"requirement": "Implement auth filter", "met": True},
            {"requirement": "Add config endpoint", "met": True},
        ]
        summary = reviewer._build_summary(
            [], ticket=ticket, compliance_requirements=requirements
        )
        assert "✅ Compliance: **Fully compliant**" in summary
        assert "    - ✅ Implement auth filter" in summary
        assert "    - ✅ Add config endpoint" in summary
        assert "📋 Requirements:" in summary

    def test_build_summary_compliance_partial(self, reviewer):
        ticket = JiraTicket(
            key="SEP-100", title="Test", description=None,
            labels=[], acceptance_criteria=None,
            url="https://jira.example.com/browse/SEP-100",
        )
        requirements = [
            {"requirement": "Implement auth filter", "met": True},
            {"requirement": "Write integration tests", "met": False},
        ]
        summary = reviewer._build_summary(
            [], ticket=ticket, compliance_requirements=requirements
        )
        assert "⚠️ Compliance: **Partially compliant**" in summary
        assert "    - ✅ Implement auth filter" in summary
        assert "    - ❌ Write integration tests" in summary

    def test_build_summary_compliance_none_met(self, reviewer):
        ticket = JiraTicket(
            key="SEP-100", title="Test", description=None,
            labels=[], acceptance_criteria=None,
            url="https://jira.example.com/browse/SEP-100",
        )
        requirements = [
            {"requirement": "Implement auth filter", "met": False},
            {"requirement": "Write tests", "met": False},
        ]
        summary = reviewer._build_summary(
            [], ticket=ticket, compliance_requirements=requirements
        )
        assert "❌ Compliance: **Not compliant**" in summary
        assert "    - ❌ Implement auth filter" in summary
        assert "    - ❌ Write tests" in summary

    def test_build_summary_no_requirements(self, reviewer):
        ticket = JiraTicket(
            key="SEP-100", title="Test", description=None,
            labels=[], acceptance_criteria=None,
            url="https://jira.example.com/browse/SEP-100",
        )
        summary = reviewer._build_summary([], ticket=ticket, compliance_requirements=[])
        assert "📋" not in summary
        assert "compliance" not in summary.lower()

    def test_build_summary_no_ticket(self, reviewer):
        summary = reviewer._build_summary([])
        assert "🎫" not in summary
        assert "compliance" not in summary.lower()

    def test_build_summary_ticket_compliance_disabled(self, reviewer):
        ticket = JiraTicket(
            key="SEP-100", title="Test", description=None,
            labels=[], acceptance_criteria=None,
            url="https://jira.example.com/browse/SEP-100",
        )
        requirements = [{"requirement": "Add login", "met": True}]
        summary = reviewer._build_summary(
            [], ticket=ticket, compliance_requirements=requirements,
            ticket_compliance_check=False,
        )
        assert "SEP-100" in summary
        assert "ℹ️ Ticket compliance check is disabled" in summary
        assert "📋" not in summary

    def test_build_summary_jira_enabled_no_ticket(self, reviewer):
        summary = reviewer._build_summary([], jira_enabled=True)
        assert "ℹ️ No Jira ticket found in branch name or PR title" in summary

    @pytest.mark.asyncio
    async def test_fetch_ticket_context_includes_type_and_status(self, mock_bitbucket, mock_copilot):
        ticket = JiraTicket(
            key="SEP-100", title="Fix bug", description=None,
            labels=[], acceptance_criteria=None,
            url="https://jira.example.com/browse/SEP-100",
            issue_type="Bug", status="In Progress",
        )
        mock_jira = AsyncMock()
        mock_jira.fetch_ticket = AsyncMock(return_value=ticket)
        reviewer = Reviewer(mock_bitbucket, mock_copilot, _review_config(), jira=mock_jira)
        context, returned_ticket = await reviewer._fetch_ticket_context("SEP-100")
        assert returned_ticket is ticket
        assert "**Type:** Bug" in context
        assert "**Status:** In Progress" in context

    @pytest.mark.asyncio
    async def test_fetch_ticket_context_omits_type_when_absent(self, mock_bitbucket, mock_copilot):
        ticket = JiraTicket(
            key="SEP-100", title="Fix bug", description=None,
            labels=[], acceptance_criteria=None,
            url="https://jira.example.com/browse/SEP-100",
        )
        mock_jira = AsyncMock()
        mock_jira.fetch_ticket = AsyncMock(return_value=ticket)
        reviewer = Reviewer(mock_bitbucket, mock_copilot, _review_config(), jira=mock_jira)
        context, _ = await reviewer._fetch_ticket_context("SEP-100")
        assert "**Type:**" not in context
        assert "**Status:**" not in context

    def test_build_summary_jira_not_configured_no_ticket(self, reviewer):
        summary = reviewer._build_summary([], jira_enabled=False)
        assert "No Jira ticket found" not in summary

    def test_build_summary_with_elapsed(self, reviewer):
        summary = reviewer._build_summary(
            [], token_usage=(1000, 500), elapsed=12.3
        )
        assert "⏱️ 12.3s" in summary


class TestReviewWithJira:
    @pytest.mark.asyncio
    async def test_review_with_jira_context(self, mock_bitbucket, mock_copilot):
        mock_jira = AsyncMock()
        mock_jira.fetch_ticket = AsyncMock(return_value=JiraTicket(
            key="SEP-123",
            title="Add login",
            description="Implement login page",
            labels=["frontend"],
            acceptance_criteria=None,
            url="https://jira.example.com/browse/SEP-123",
        ))
        result = _make_review_result(
            findings=[
                ReviewFinding(file="file.py", line=1, severity="warning", comment="Test issue"),
            ],
        )
        result.compliance_requirements = [
            {"requirement": "Implement login page", "met": True},
            {"requirement": "Add tests", "met": False},
        ]
        mock_copilot.review_diff.return_value = result

        rev = Reviewer(mock_bitbucket, mock_copilot, _review_config(), jira=mock_jira)
        payload = _make_payload(branch="feature/SEP-123-login")
        await rev.review_pull_request(payload)

        mock_jira.fetch_ticket.assert_called_once_with("SEP-123")
        call_kwargs = mock_copilot.review_diff.call_args[1]
        assert "SEP-123" in call_kwargs["ticket_context"]
        assert "Add login" in call_kwargs["ticket_context"]

        summary_text = mock_bitbucket.update_pr_comment.call_args[0][5] if mock_bitbucket.update_pr_comment.called else mock_bitbucket.post_pr_comment.call_args[0][3]
        assert "SEP-123" in summary_text
        assert "⚠️ Compliance: **Partially compliant**" in summary_text
        assert "    - ✅ Implement login page" in summary_text
        assert "    - ❌ Add tests" in summary_text

    @pytest.mark.asyncio
    async def test_review_jira_disabled(self, mock_bitbucket, mock_copilot):
        rev = Reviewer(mock_bitbucket, mock_copilot, _review_config())
        payload = _make_payload(branch="feature/SEP-123-login")
        await rev.review_pull_request(payload)

        mock_copilot.review_diff.assert_called_once()
        call_kwargs = mock_copilot.review_diff.call_args[1]
        assert call_kwargs["ticket_context"] == ""

    @pytest.mark.asyncio
    async def test_review_passes_ticket_compliance_check_flag(self, mock_bitbucket, mock_copilot):
        mock_jira = AsyncMock()
        mock_jira.fetch_ticket = AsyncMock(return_value=JiraTicket(
            key="SEP-123", title="Add login", description=None,
            labels=[], acceptance_criteria=None,
            url="https://jira.example.com/browse/SEP-123",
        ))
        mock_copilot.review_diff.return_value = _make_review_result()

        rev = Reviewer(
            mock_bitbucket, mock_copilot,
            _review_config(ticket_compliance_check=False),
            jira=mock_jira,
        )
        payload = _make_payload(branch="feature/SEP-123-login")
        await rev.review_pull_request(payload)

        call_kwargs = mock_copilot.review_diff.call_args[1]
        assert call_kwargs["ticket_compliance_check"] is False
        assert "SEP-123" in call_kwargs["ticket_context"]
