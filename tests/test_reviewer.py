import logging
from unittest.mock import AsyncMock, patch

import pytest

from app.config import ReviewConfig
from app.llm_client import LLMClient, FileReviewData
from app.jira import JiraTicket
from app.models import ReviewFinding, WebhookPayload
from app.reviewer import Reviewer, _count_diff_lines, _sort_and_limit


def _review_config(**overrides) -> ReviewConfig:
    kwargs: dict = {"auto_review_authors": ["username"]}
    kwargs.update(overrides)
    return ReviewConfig(**kwargs)


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
    client.bot_username = "noergler"
    client.fetch_pr_diff = AsyncMock(return_value="diff --git a/file.py b/file.py\n+hello\n")
    client.fetch_file_content = AsyncMock(return_value="hello\n")
    client.fetch_pr_comments = AsyncMock(return_value=[])
    client.post_inline_comment = AsyncMock(return_value=None)
    client.post_pr_comment = AsyncMock(return_value=(123, 0))
    client.update_pr_comment = AsyncMock(return_value=True)
    return client


def _make_review_result(findings=None, skipped_files=None, review_effort=1):
    return LLMClient.ReviewResult(
        findings=findings or [],
        skipped_files=skipped_files or [],
        prompt_tokens=100,
        completion_tokens=50,
        review_effort=review_effort,
    )


@pytest.fixture
def mock_llm():
    client = AsyncMock()
    client.config.model = "gpt-5.3-codex"
    client.max_tokens_per_chunk = 80000
    client.context_window = 1_000_000
    client.prompt_template = "Review these files:\n{files}\n{repo_instructions}"
    client.review_diff = AsyncMock(return_value=_make_review_result([
        ReviewFinding(file="file.py", line=1, severity="warning", comment="Test issue"),
    ]))
    return client


@pytest.fixture
def reviewer(mock_bitbucket, mock_llm):
    return Reviewer(mock_bitbucket, mock_llm, _review_config(), db_pool=AsyncMock())


class TestReviewer:
    @pytest.mark.asyncio
    async def test_review_allowed_author(self, reviewer, mock_bitbucket, mock_llm):
        payload = _make_payload("username")
        await reviewer.review_pull_request(payload)

        # Single diff fetch with context_lines=0, context expanded locally
        mock_bitbucket.fetch_pr_diff.assert_called_once_with("PROJ", "my-repo", 42, context_lines=0)

        mock_llm.review_diff.assert_called_once()
        call_args = mock_llm.review_diff.call_args
        files = call_args[0][0]
        assert len(files) == 1
        assert isinstance(files[0], FileReviewData)
        assert files[0].path == "file.py"

        mock_bitbucket.post_inline_comment.assert_called_once()
        mock_bitbucket.post_pr_comment.assert_called_once()

        summary_text = mock_bitbucket.post_pr_comment.call_args[0][3]
        assert "### Review summary" in summary_text
        assert "1 warning" in summary_text

    @pytest.mark.asyncio
    async def test_skip_disallowed_author(self, reviewer, mock_bitbucket, mock_llm):
        payload = _make_payload("other.user")
        await reviewer.review_pull_request(payload)

        mock_bitbucket.fetch_pr_diff.assert_not_called()
        mock_llm.review_diff.assert_not_called()

    @pytest.mark.asyncio
    async def test_skip_author_check_allows_non_listed_author(self, reviewer, mock_bitbucket, mock_llm):
        payload = _make_payload("other.user")
        await reviewer.review_pull_request(payload, skip_author_check=True)

        # Single diff fetch, context expanded locally
        mock_bitbucket.fetch_pr_diff.assert_called_once()
        mock_llm.review_diff.assert_called_once()

    @pytest.mark.asyncio
    async def test_skip_empty_diff(self, reviewer, mock_bitbucket, mock_llm):
        mock_bitbucket.fetch_pr_diff.return_value = "   \n"
        payload = _make_payload("username")
        await reviewer.review_pull_request(payload)

        mock_llm.review_diff.assert_not_called()

    @pytest.mark.asyncio
    async def test_no_findings(self, reviewer, mock_bitbucket, mock_llm):
        mock_llm.review_diff.return_value = _make_review_result([])
        payload = _make_payload("username")
        await reviewer.review_pull_request(payload)

        mock_bitbucket.post_inline_comment.assert_not_called()
        summary_text = mock_bitbucket.post_pr_comment.call_args[0][3]
        assert "No issues found" in summary_text

    @pytest.mark.asyncio
    async def test_review_skipped_when_agents_md_missing_and_required(self, mock_bitbucket, mock_llm):
        mock_bitbucket.fetch_file_content = AsyncMock(return_value="")
        rev = Reviewer(mock_bitbucket, mock_llm, _review_config(), db_pool=AsyncMock())
        payload = _make_payload("username")
        await rev.review_pull_request(payload)

        mock_llm.review_diff.assert_not_called()
        mock_bitbucket.fetch_pr_diff.assert_not_called()
        mock_bitbucket.post_inline_comment.assert_not_called()
        mock_bitbucket.post_pr_comment.assert_called_once()
        summary_text = mock_bitbucket.post_pr_comment.call_args[0][3]
        assert "AGENTS.md" in summary_text
        assert "REVIEW_REQUIRE_AGENTS_MD" in summary_text

    @pytest.mark.asyncio
    async def test_review_proceeds_when_agents_md_missing_and_not_required(self, mock_bitbucket, mock_llm):
        mock_bitbucket.fetch_file_content = AsyncMock(return_value="")
        rev = Reviewer(
            mock_bitbucket, mock_llm, _review_config(require_agents_md=False),
            db_pool=AsyncMock(),
        )
        payload = _make_payload("username")
        await rev.review_pull_request(payload)

        mock_llm.review_diff.assert_called_once()
        mock_bitbucket.post_pr_comment.assert_called_once()
        summary_text = mock_bitbucket.post_pr_comment.call_args[0][3]
        assert "Tip:" in summary_text

    def test_build_agents_md_missing_summary_mentions_setting(self):
        summary = Reviewer._build_agents_md_missing_summary()
        assert "AGENTS.md" in summary
        assert "REVIEW_REQUIRE_AGENTS_MD" in summary

    @pytest.mark.asyncio
    async def test_review_skipped_when_branch_contains_opt_out_keyword(self, mock_bitbucket, mock_llm):
        rev = Reviewer(mock_bitbucket, mock_llm, _review_config(), db_pool=AsyncMock())
        payload = _make_payload("username", branch="feature/x-noergloff")
        await rev.review_pull_request(payload)

        mock_llm.review_diff.assert_not_called()
        mock_bitbucket.fetch_pr_diff.assert_not_called()
        mock_bitbucket.fetch_file_content.assert_not_called()
        mock_bitbucket.post_inline_comment.assert_not_called()
        mock_bitbucket.post_pr_comment.assert_called_once()
        summary_text = mock_bitbucket.post_pr_comment.call_args[0][3]
        assert "noergloff" in summary_text
        assert "feature/x-noergloff" in summary_text
        assert "REVIEW_OPT_OUT_BRANCH_KEYWORD" in summary_text

    @pytest.mark.asyncio
    async def test_opt_out_keyword_match_is_case_insensitive(self, mock_bitbucket, mock_llm):
        rev = Reviewer(mock_bitbucket, mock_llm, _review_config(), db_pool=AsyncMock())
        payload = _make_payload("username", branch="FEATURE/NOERGLOFF-123")
        await rev.review_pull_request(payload)

        mock_llm.review_diff.assert_not_called()
        mock_bitbucket.post_pr_comment.assert_called_once()

    @pytest.mark.asyncio
    async def test_opt_out_disabled_when_keyword_empty(self, mock_bitbucket, mock_llm):
        rev = Reviewer(
            mock_bitbucket, mock_llm,
            _review_config(opt_out_branch_keyword="", require_agents_md=False),
            db_pool=AsyncMock(),
        )
        payload = _make_payload("username", branch="feature/x-noergloff")
        await rev.review_pull_request(payload)

        mock_llm.review_diff.assert_called_once()

    @pytest.mark.asyncio
    async def test_opt_out_does_not_affect_non_matching_branch(self, reviewer, mock_llm):
        payload = _make_payload("username", branch="feature/normal-work")
        await reviewer.review_pull_request(payload)

        mock_llm.review_diff.assert_called_once()

    def test_build_opt_out_branch_summary_mentions_keyword_and_branch(self):
        summary = Reviewer._build_opt_out_branch_summary("noergloff", "feature/x-noergloff")
        assert "noergloff" in summary
        assert "feature/x-noergloff" in summary
        assert "REVIEW_OPT_OUT_BRANCH_KEYWORD" in summary

    @pytest.mark.asyncio
    async def test_content_fetch_failure_falls_back_to_diff_only(self, mock_bitbucket, mock_llm):
        mock_bitbucket.fetch_file_content = AsyncMock(side_effect=Exception("not found"))
        rev = Reviewer(
            mock_bitbucket, mock_llm, _review_config(require_agents_md=False),
            db_pool=AsyncMock(),
        )
        payload = _make_payload("username")
        await rev.review_pull_request(payload)

        # Should still review, just with content=None
        mock_llm.review_diff.assert_called_once()
        files = mock_llm.review_diff.call_args[0][0]
        assert files[0].content is None

    @pytest.mark.asyncio
    async def test_content_preserved_for_small_pr(self, mock_bitbucket, mock_llm):
        mock_bitbucket.fetch_file_content = AsyncMock(return_value="hello\n")
        rev = Reviewer(mock_bitbucket, mock_llm, _review_config(), db_pool=AsyncMock())
        payload = _make_payload("username")
        await rev.review_pull_request(payload)

        mock_llm.review_diff.assert_called_once()
        files = mock_llm.review_diff.call_args[0][0]
        assert len(files) == 1
        assert files[0].content == "hello\n"

    @pytest.mark.asyncio
    async def test_context_expanded_for_large_pr(self, mock_bitbucket, mock_llm):
        file_content = "\n".join(f"line {i}" for i in range(50))
        # Return a diff with a hunk and enough content that file_content provides context
        mock_bitbucket.fetch_pr_diff = AsyncMock(
            return_value="diff --git a/file.py b/file.py\n@@ -10,1 +10,1 @@\n-old\n+new\n"
        )
        mock_bitbucket.fetch_file_content = AsyncMock(return_value=file_content)
        # Use a tiny max_tokens to force the large PR path
        mock_llm.max_tokens_per_chunk = 1
        mock_llm.prompt_template = "{files}\n{repo_instructions}"
        mock_llm.review_diff = AsyncMock(return_value=_make_review_result())

        rev = Reviewer(mock_bitbucket, mock_llm, _review_config(), db_pool=AsyncMock())
        payload = _make_payload("username")
        await rev.review_pull_request(payload)

        # Single diff fetch — no second call for expanded context
        mock_bitbucket.fetch_pr_diff.assert_called_once()
        # review_diff may or may not be called depending on compression,
        # but the key assertion is no second fetch_pr_diff call

    def test_is_auto_review_author(self, reviewer):
        assert reviewer.is_auto_review_author("username") is True
        assert reviewer.is_auto_review_author("other.user") is False

    def test_is_auto_review_author_empty_list(self, mock_bitbucket, mock_llm):
        rev = Reviewer(mock_bitbucket, mock_llm, _review_config(auto_review_authors=[]), db_pool=AsyncMock())
        assert rev.is_auto_review_author("anyone") is True

    def test_build_summary_mixed(self, reviewer):
        findings = [
            ReviewFinding(file="a.py", line=1, severity="critical", comment="err"),
            ReviewFinding(file="b.py", line=2, severity="warning", comment="warn"),
        ]
        summary = reviewer._build_summary(findings)
        assert "### Review summary" in summary
        assert "1 critical ❌" in summary
        assert "1 warning ⚠️" in summary

    def test_build_summary_empty(self, reviewer):
        summary = reviewer._build_summary([])
        assert "### Review summary" in summary
        assert "- No issues found ✅" in summary

    @pytest.mark.asyncio
    async def test_review_real_webhook_payload(self, mock_bitbucket, mock_llm):
        payload = _make_real_payload()
        reviewer = Reviewer(mock_bitbucket, mock_llm, _review_config(auto_review_authors=["username"]), db_pool=AsyncMock())
        await reviewer.review_pull_request(payload)

        # Single diff fetch, context expanded locally
        mock_bitbucket.fetch_pr_diff.assert_called_once_with("~USERNAME", "test", 1, context_lines=0)
        mock_llm.review_diff.assert_called_once()
        mock_bitbucket.post_inline_comment.assert_called_once()
        mock_bitbucket.post_pr_comment.assert_called_once()


class TestSortAndLimit:
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

    @pytest.fixture
    def reviewer(self, mock_bitbucket, mock_llm):
        return Reviewer(mock_bitbucket, mock_llm, _review_config(), db_pool=AsyncMock())

    def test_build_summary_truncated(self, reviewer):
        findings = [
            ReviewFinding(file="a.py", line=1, severity="critical", comment="err"),
            ReviewFinding(file="b.py", line=2, severity="warning", comment="warn"),
        ]
        summary = reviewer._build_summary(findings, truncated=True)
        assert "1 critical ❌" in summary
        assert "Additional findings were omitted" in summary

    def test_build_summary_top_findings_under_limit(self, reviewer):
        findings = [
            ReviewFinding(file="a.py", line=10, severity="critical", comment="Bad thing"),
            ReviewFinding(file="b.py", line=20, severity="warning", comment="Mild thing"),
        ]
        summary = reviewer._build_summary(findings)
        assert "**Top findings:**" in summary
        assert "- ❌ Bad thing" in summary
        assert "- ⚠️ Mild thing" in summary
        # File path / line number must not leak into the summary
        assert "a.py" not in summary
        assert "b.py" not in summary
        assert "…and" not in summary

    def test_build_summary_top_findings_over_limit(self, reviewer):
        findings = [
            ReviewFinding(file=f"f{i}.py", line=i, severity="critical", comment=f"issue {i}")
            for i in range(8)
        ]
        summary = reviewer._build_summary(findings)
        assert "**Top findings:**" in summary
        # Only first 5 rendered as one-liners
        assert "- ❌ issue 0" in summary
        assert "- ❌ issue 4" in summary
        assert "issue 5" not in summary
        assert "- …and 3 more" in summary

    def test_build_summary_top_findings_keeps_long_comment(self, reviewer):
        long_comment = "x" * 200
        findings = [ReviewFinding(file="a.py", line=1, severity="critical", comment=long_comment)]
        summary = reviewer._build_summary(findings)
        # Full single-line comment must be rendered verbatim — no char truncation.
        assert long_comment in summary

    def test_build_summary_top_findings_first_line_only(self, reviewer):
        findings = [
            ReviewFinding(file="a.py", line=1, severity="critical",
                          comment="First line.\nSecond line should be dropped."),
        ]
        summary = reviewer._build_summary(findings)
        assert "First line." in summary
        assert "Second line should be dropped" not in summary

    def test_build_summary_no_top_findings_when_empty(self, reviewer):
        summary = reviewer._build_summary([])
        assert "**Top findings:**" not in summary

    def test_build_summary_agents_md_not_found(self, reviewer):
        findings = [
            ReviewFinding(file="a.py", line=1, severity="warning", comment="warn"),
        ]
        summary = reviewer._build_summary(findings, agents_md_found=False)
        assert "💡" in summary
        assert "AGENTS.md" in summary
        assert "Tip:" in summary

    def test_build_summary_agents_md_found(self, reviewer):
        findings = [
            ReviewFinding(file="a.py", line=1, severity="warning", comment="warn"),
        ]
        summary = reviewer._build_summary(findings, agents_md_found=True)
        assert "✅" in summary
        assert "Using project-specific review guidelines" in summary
        assert "Tip:" not in summary

    def test_build_summary_no_findings_agents_md_not_found(self, reviewer):
        summary = reviewer._build_summary([], agents_md_found=False)
        assert "- No issues found ✅" in summary
        assert "💡" in summary
        assert "Tip:" in summary

    def test_build_summary_no_findings_agents_md_found(self, reviewer):
        summary = reviewer._build_summary([], agents_md_found=True)
        assert "- No issues found ✅" in summary
        assert "Using project-specific review guidelines" in summary
        assert "Tip:" not in summary

    def test_build_summary_agents_md_within_token_limit(self, reviewer):
        findings = [
            ReviewFinding(file="a.py", line=1, severity="warning", comment="warn"),
        ]
        summary = reviewer._build_summary(
            findings,
            agents_md_found=True,
            prompt_breakdown={"template": 100, "repo_instructions": 1240, "files": 500},
        )
        assert "Using project-specific review guidelines from `AGENTS.md`" in summary
        assert "~1240 / 4000 tokens, 31%" in summary
        assert "✅" in summary
        assert "context bloat" not in summary

    def test_build_summary_agents_md_exceeds_token_limit(self, reviewer):
        findings = [
            ReviewFinding(file="a.py", line=1, severity="warning", comment="warn"),
        ]
        summary = reviewer._build_summary(
            findings,
            agents_md_found=True,
            prompt_breakdown={"template": 100, "repo_instructions": 9000, "files": 500},
        )
        assert "Using project-specific review guidelines from `AGENTS.md`" in summary
        assert "~9000 / 4000 tokens, 225%" in summary
        assert "context bloat" in summary
        assert "⚠️" in summary

    def test_build_summary_agents_md_no_token_breakdown(self, reviewer):
        findings = [
            ReviewFinding(file="a.py", line=1, severity="warning", comment="warn"),
        ]
        summary = reviewer._build_summary(
            findings,
            agents_md_found=True,
            prompt_breakdown=None,
        )
        assert "Using project-specific review guidelines from `AGENTS.md` ✅" in summary
        assert "tokens," not in summary
        assert "context bloat" not in summary

    def test_build_summary_agents_md_warn_disabled(self, mock_bitbucket, mock_llm):
        """agents_md_warn_tokens=0 disables the token-count annotation entirely."""
        from app.reviewer import Reviewer
        reviewer = Reviewer(
            mock_bitbucket, mock_llm,
            _review_config(agents_md_warn_tokens=0),
            db_pool=AsyncMock(),
        )
        findings = [
            ReviewFinding(file="a.py", line=1, severity="warning", comment="warn"),
        ]
        summary = reviewer._build_summary(
            findings,
            agents_md_found=True,
            prompt_breakdown={"template": 100, "repo_instructions": 9999, "files": 500},
        )
        assert "Using project-specific review guidelines from `AGENTS.md` ✅" in summary
        assert "tokens," not in summary
        assert "context bloat" not in summary

    def test_build_summary_skipped_files(self, reviewer):
        findings = [
            ReviewFinding(file="a.py", line=1, severity="warning", comment="warn"),
        ]
        summary = reviewer._build_summary(findings, skipped_files=["huge.py", "big.js"])
        assert "- Not reviewed (too large)" in summary
        assert "⚠️" in summary
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
        assert "Model: `gpt-5.3-codex`" in summary
        assert "1'000↑" in summary
        assert "500↓" in summary
        assert "1'500 total" in summary

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
        assert "~500 review prompt" in summary
        assert "~200 AGENTS.md" in summary
        assert "~7'258 file content" in summary

    def test_build_summary_prompt_breakdown_without_token_usage(self, reviewer):
        findings = [
            ReviewFinding(file="a.py", line=1, severity="warning", comment="warn"),
        ]
        breakdown = {"template": 500, "repo_instructions": 0, "files": 7258}
        summary = reviewer._build_summary(findings, prompt_breakdown=breakdown)
        assert "template" not in summary

    def test_build_summary_chunk_count_single_pass(self, reviewer):
        summary = reviewer._build_summary(
            [], chunk_count=1, chunk_budget=80000, token_usage=(60000, 1000)
        )
        assert "Tokens used: 60k of 80k available (75% used) · 1 pass" in summary

    def test_build_summary_chunk_count_multi(self, reviewer):
        summary = reviewer._build_summary(
            [], chunk_count=3, chunk_budget=80000, token_usage=(240000, 3000)
        )
        assert "Tokens used: 240k total across 3 passes (avg 100% used/pass, cap 80k/pass)" in summary

    def test_build_summary_chunk_count_absent_when_none(self, reviewer):
        summary = reviewer._build_summary([])
        assert "Tokens used:" not in summary
        assert "Context:" not in summary
        assert "chunk budget" not in summary

    def test_build_summary_chunk_budget_with_context_window(self, reviewer):
        summary = reviewer._build_summary(
            [],
            chunk_count=1,
            chunk_budget=256_000,
            context_window=272_000,
            token_usage=(245_000, 2_000),
        )
        assert "Tokens used: 245k of 256k available (96% used, model max 272k) · 1 pass" in summary

    @pytest.mark.asyncio
    async def test_findings_limited_in_review(self, mock_bitbucket, mock_llm):
        mock_llm.review_diff.return_value = _make_review_result([
            ReviewFinding(file=f"f{i}.py", line=i, severity="warning", comment=f"issue {i}")
            for i in range(30)
        ])
        rev = Reviewer(mock_bitbucket, mock_llm, _review_config(max_comments=5), db_pool=AsyncMock())
        payload = _make_payload("username")
        await rev.review_pull_request(payload)

        assert mock_bitbucket.post_inline_comment.call_count == 5
        summary_text = mock_bitbucket.post_pr_comment.call_args[0][3]
        assert "Additional findings were omitted" in summary_text

    @pytest.mark.asyncio
    async def test_dedup_graceful_on_fetch_failure(self, mock_bitbucket, mock_llm):
        mock_bitbucket.fetch_pr_comments.side_effect = Exception("API error")
        rev = Reviewer(mock_bitbucket, mock_llm, _review_config(), db_pool=AsyncMock())
        payload = _make_payload("username")
        await rev.review_pull_request(payload)

        mock_bitbucket.post_inline_comment.assert_called_once()
        mock_bitbucket.post_pr_comment.assert_called_once()

    @pytest.mark.asyncio
    async def test_persistent_comment_updates_existing(self, mock_bitbucket, mock_llm, monkeypatch):
        mock_bitbucket.update_pr_comment = AsyncMock(return_value=True)
        monkeypatch.setattr(
            "app.reviewer.repository.upsert_pr_review",
            AsyncMock(return_value=99),
        )
        monkeypatch.setattr(
            "app.reviewer.repository.get_summary_comment_info",
            AsyncMock(return_value={"summary_comment_id": 77, "summary_comment_version": 2}),
        )
        rev = Reviewer(mock_bitbucket, mock_llm, _review_config(), db_pool=AsyncMock())
        payload = _make_payload("username")
        await rev.review_pull_request(payload)

        mock_bitbucket.update_pr_comment.assert_called_once()
        call_args = mock_bitbucket.update_pr_comment.call_args[0]
        assert call_args[3] == 77  # comment_id
        assert call_args[4] == 2   # version
        mock_bitbucket.post_pr_comment.assert_not_called()

    @pytest.mark.asyncio
    async def test_persistent_comment_creates_new_when_none_exists(self, mock_bitbucket, mock_llm):
        mock_bitbucket.fetch_pr_comments.return_value = []
        rev = Reviewer(mock_bitbucket, mock_llm, _review_config(), db_pool=AsyncMock())
        payload = _make_payload("username")
        await rev.review_pull_request(payload)

        mock_bitbucket.post_pr_comment.assert_called_once()
        mock_bitbucket.update_pr_comment.assert_not_called()

    @pytest.mark.asyncio
    async def test_persistent_comment_ignores_inline_markers(self, mock_bitbucket, mock_llm):
        mock_bitbucket.fetch_pr_comments.return_value = [
            {
                "id": 88,
                "version": 1,
                "text": "**Suggestion:** bug",
                "path": "a.py",
                "line": 10,
                "author_slug": "noergler",
            },
        ]
        rev = Reviewer(mock_bitbucket, mock_llm, _review_config(), db_pool=AsyncMock())
        payload = _make_payload("username")
        await rev.review_pull_request(payload)

        mock_bitbucket.post_pr_comment.assert_called_once()
        mock_bitbucket.update_pr_comment.assert_not_called()

    def test_build_summary_effort_score_not_rendered(self, reviewer):
        findings = [
            ReviewFinding(file="a.py", line=1, severity="warning", comment="warn"),
        ]
        summary = reviewer._build_summary(findings)
        assert "📊" not in summary
        assert "review effort" not in summary.lower()

    def test_build_summary_security_section(self, reviewer):
        findings = [
            ReviewFinding(file="a.py", line=1, severity="critical", comment="SQL injection vulnerability found"),
            ReviewFinding(file="b.py", line=2, severity="warning", comment="unused import"),
        ]
        summary = reviewer._build_summary(findings)
        assert "- 1 potential security issue" in summary
        assert "🔒" in summary

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
        assert "- 2 potential security issues" in summary
        assert "🔒" in summary

    def test_build_summary_with_change_summary(self, reviewer):
        summary = reviewer._build_summary(
            [], change_summary=["Added retry logic", "Replaced sync with async I/O"]
        )
        assert "**What changed:**" in summary
        assert "- Added retry logic" in summary
        assert "- Replaced sync with async I/O" in summary

    def test_build_summary_without_change_summary(self, reviewer):
        summary = reviewer._build_summary([])
        assert "What changed" not in summary

    def test_build_summary_what_changed_above_top_findings(self, reviewer):
        findings = [
            ReviewFinding(file="a.py", line=1, severity="critical", comment="bug A"),
            ReviewFinding(file="b.py", line=2, severity="warning", comment="issue B"),
        ]
        summary = reviewer._build_summary(
            findings, change_summary=["Added retry logic"]
        )
        what_idx = summary.index("**What changed:**")
        top_idx = summary.index("**Top findings:**")
        assert what_idx < top_idx

    def test_build_summary_what_changed_when_no_findings(self, reviewer):
        summary = reviewer._build_summary(
            [], change_summary=["Refactors X without behavior change"]
        )
        assert "No issues found" in summary
        assert "**What changed:**" in summary
        assert "- Refactors X without behavior change" in summary

    def test_build_summary_no_divider_before_meta(self, reviewer):
        findings = [
            ReviewFinding(file="a.py", line=1, severity="warning", comment="warn"),
        ]
        summary = reviewer._build_summary(findings, agents_md_found=True)
        assert "\n\n---\n" not in summary

    def test_build_summary_initial_review_header(self, reviewer):
        summary = reviewer._build_summary([])
        assert "### Review summary\n" in summary
        assert "(initial review)" not in summary

    def test_build_summary_files_reviewed_all(self, reviewer):
        summary = reviewer._build_summary([], files_reviewed=5, total_files=5)
        assert "Reviewed 5 files" in summary
        assert "📂" not in summary

    def test_build_summary_files_reviewed_partial(self, reviewer):
        summary = reviewer._build_summary([], files_reviewed=8, total_files=12)
        assert "Reviewed 8 of 12 files" in summary
        assert "4 skipped" in summary

    def test_build_summary_diff_size(self, reviewer):
        summary = reviewer._build_summary([], diff_added=142, diff_removed=38)
        assert "+142 / -38 lines" in summary

    def test_build_summary_diff_size_additions_only(self, reviewer):
        summary = reviewer._build_summary([], diff_added=50, diff_removed=0)
        assert "+50 lines" in summary
        assert "-" not in summary.split("Diff:")[1].split("\n")[0]

    def test_build_summary_diff_size_deletions_only(self, reviewer):
        summary = reviewer._build_summary([], diff_added=0, diff_removed=30)
        assert "-30 lines" in summary

    def test_build_summary_cross_file_symbols(self, reviewer):
        summary = reviewer._build_summary(
            [], cross_file_symbols=["get_user", "UserCache", "process_order"]
        )
        assert "3 cross-file dependencies analyzed" in summary
        assert "`get_user`" in summary
        assert "`UserCache`" in summary
        assert "🔗" not in summary

    def test_build_summary_cross_file_symbols_truncated(self, reviewer):
        symbols = [f"func_{i}" for i in range(8)]
        summary = reviewer._build_summary([], cross_file_symbols=symbols)
        assert "8 cross-file dependencies analyzed" in summary
        assert "and 3 more" in summary

    def test_build_summary_no_cross_file_symbols(self, reviewer):
        summary = reviewer._build_summary([], cross_file_symbols=None)
        assert "cross-file" not in summary


class TestCountDiffLines:
    def test_counts_added_and_removed(self):
        diff = (
            "--- a/file.py\n"
            "+++ b/file.py\n"
            "@@ -1,3 +1,4 @@\n"
            "-old_line\n"
            "+new_line\n"
            "+another_new\n"
            " context\n"
        )
        added, removed = _count_diff_lines(diff)
        assert added == 2
        assert removed == 1

    def test_ignores_diff_headers(self):
        diff = (
            "--- a/file.py\n"
            "+++ b/file.py\n"
            "+real addition\n"
        )
        added, removed = _count_diff_lines(diff)
        assert added == 1
        assert removed == 0

    def test_empty_diff(self):
        added, removed = _count_diff_lines("")
        assert added == 0
        assert removed == 0


class TestHandleMention:
    @pytest.mark.asyncio
    async def test_mention_uses_file_level_preparation(self, mock_bitbucket, mock_llm):
        mock_llm.answer_question = AsyncMock(return_value="Here's the answer.")
        mock_bitbucket.reply_to_comment = AsyncMock()

        rev = Reviewer(mock_bitbucket, mock_llm, _review_config(), db_pool=AsyncMock())
        payload = _make_mention_payload("@noergler what does this do?")
        await rev.handle_mention(payload)

        # answer_question should receive FileReviewData list, not raw string
        mock_llm.answer_question.assert_called_once()
        call_args = mock_llm.answer_question.call_args
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
    async def test_mention_no_reviewable_files(self, mock_bitbucket, mock_llm):
        mock_bitbucket.fetch_pr_diff.return_value = (
            "diff --git a/image.png b/image.png\nBinary files differ\n"
        )
        mock_bitbucket.reply_to_comment = AsyncMock()

        rev = Reviewer(mock_bitbucket, mock_llm, _review_config(), db_pool=AsyncMock())
        payload = _make_mention_payload("@noergler what does this do?")
        await rev.handle_mention(payload)

        mock_llm.answer_question.assert_not_called()
        mock_bitbucket.reply_to_comment.assert_called_once()
        reply_text = mock_bitbucket.reply_to_comment.call_args[0][4]
        assert "No reviewable files" in reply_text

    @pytest.mark.asyncio
    async def test_mention_triggers_full_review(self, mock_bitbucket, mock_llm):
        mock_bitbucket.reply_to_comment = AsyncMock()
        rev = Reviewer(mock_bitbucket, mock_llm, _review_config(), db_pool=AsyncMock())
        payload = _make_mention_payload("@noergler review")
        await rev.handle_mention(payload)

        # Should trigger review_diff, not answer_question
        mock_llm.review_diff.assert_called_once()
        mock_llm.answer_question.assert_not_called()


class TestMaxFileLines:
    @pytest.mark.asyncio
    async def test_file_under_limit_keeps_content(self, mock_bitbucket, mock_llm):
        content = "\n".join(f"line {i}" for i in range(50))  # 50 lines
        mock_bitbucket.fetch_file_content = AsyncMock(return_value=content)
        mock_llm.review_diff = AsyncMock(return_value=_make_review_result())

        rev = Reviewer(mock_bitbucket, mock_llm, _review_config(max_file_lines=100), db_pool=AsyncMock())
        payload = _make_payload("username")
        await rev.review_pull_request(payload)

        mock_llm.review_diff.assert_called_once()
        files = mock_llm.review_diff.call_args[0][0]
        assert len(files) == 1
        assert files[0].content == content

    @pytest.mark.asyncio
    async def test_file_over_limit_falls_back_to_diff_only(self, mock_bitbucket, mock_llm):
        content = "\n".join(f"line {i}" for i in range(200))  # 200 lines
        mock_bitbucket.fetch_file_content = AsyncMock(return_value=content)
        mock_llm.review_diff = AsyncMock(return_value=_make_review_result())

        rev = Reviewer(mock_bitbucket, mock_llm, _review_config(max_file_lines=100), db_pool=AsyncMock())
        payload = _make_payload("username")
        await rev.review_pull_request(payload)

        mock_llm.review_diff.assert_called_once()
        files = mock_llm.review_diff.call_args[0][0]
        assert len(files) == 1
        assert files[0].content is None

    @pytest.mark.asyncio
    async def test_content_skipped_appears_in_summary(self, mock_bitbucket, mock_llm):
        content = "\n".join(f"line {i}" for i in range(200))  # 200 lines
        mock_bitbucket.fetch_file_content = AsyncMock(return_value=content)
        mock_llm.review_diff = AsyncMock(return_value=_make_review_result())

        rev = Reviewer(mock_bitbucket, mock_llm, _review_config(max_file_lines=100), db_pool=AsyncMock())
        payload = _make_payload("username")
        await rev.review_pull_request(payload)

        summary_text = mock_bitbucket.post_pr_comment.call_args[0][3]
        assert "Reviewed without full file context (too large)" in summary_text
        assert "`file.py`" in summary_text

    def test_build_summary_content_skipped_files(self, mock_bitbucket, mock_llm):
        rev = Reviewer(mock_bitbucket, mock_llm, _review_config(), db_pool=AsyncMock())
        summary = rev._build_summary([], content_skipped_files=["large.py"])
        assert "Reviewed without full file context (too large)" in summary
        assert "`large.py`" in summary

    def test_build_summary_no_content_skipped_files(self, mock_bitbucket, mock_llm):
        rev = Reviewer(mock_bitbucket, mock_llm, _review_config(), db_pool=AsyncMock())
        summary = rev._build_summary([], content_skipped_files=[])
        assert "Reviewed without full file context" not in summary


class TestTicketExtraction:
    def test_extract_ticket_id_from_branch(self, mock_bitbucket, mock_llm):
        rev = Reviewer(mock_bitbucket, mock_llm, _review_config(), db_pool=AsyncMock())
        payload = _make_payload(branch="feature/SEP-22888-description")
        assert rev._extract_ticket_id(payload.pullRequest) == "SEP-22888"

    def test_extract_ticket_id_from_branch_slash(self, mock_bitbucket, mock_llm):
        rev = Reviewer(mock_bitbucket, mock_llm, _review_config(), db_pool=AsyncMock())
        payload = _make_payload(branch="SEP-22888/description")
        assert rev._extract_ticket_id(payload.pullRequest) == "SEP-22888"

    def test_extract_ticket_id_from_title(self, mock_bitbucket, mock_llm):
        rev = Reviewer(mock_bitbucket, mock_llm, _review_config(), db_pool=AsyncMock())
        payload = _make_payload(branch="feature/no-ticket", title="SEP-22888 Fix login")
        assert rev._extract_ticket_id(payload.pullRequest) == "SEP-22888"

    def test_extract_ticket_id_none(self, mock_bitbucket, mock_llm):
        rev = Reviewer(mock_bitbucket, mock_llm, _review_config(), db_pool=AsyncMock())
        payload = _make_payload(branch="feature/no-ticket", title="Fix login bug")
        assert rev._extract_ticket_id(payload.pullRequest) is None

    def test_extract_ticket_id_branch_priority(self, mock_bitbucket, mock_llm):
        rev = Reviewer(mock_bitbucket, mock_llm, _review_config(), db_pool=AsyncMock())
        payload = _make_payload(branch="feature/SEP-111-fix", title="SEP-222 other fix")
        assert rev._extract_ticket_id(payload.pullRequest) == "SEP-111"


class TestBuildSummaryWithTicket:
    @pytest.fixture
    def reviewer(self, mock_bitbucket, mock_llm):
        return Reviewer(mock_bitbucket, mock_llm, _review_config(), db_pool=AsyncMock())

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
        # Ticket alone (no compliance data) → Ticket section with key + title
        assert "### Ticket" in summary
        assert "### Requirement compliance" not in summary
        assert "**[SEP-22888](https://jira.example.com/browse/SEP-22888)**" in summary
        assert "Config security" in summary

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
        assert "### Requirement compliance" in summary
        assert "**Fully compliant** ✅" in summary
        assert "- Implement auth filter ✅" in summary
        assert "- Add config endpoint ✅" in summary

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
        assert "**Partially compliant** ⚠️" in summary
        assert "- Implement auth filter ✅" in summary
        assert "- Write integration tests ❌" in summary

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
        assert "**Not compliant** ❌" in summary
        assert "- Implement auth filter ❌" in summary
        assert "- Write tests ❌" in summary

    def test_build_summary_no_requirements(self, reviewer):
        ticket = JiraTicket(
            key="SEP-100", title="Test", description=None,
            labels=[], acceptance_criteria=None,
            url="https://jira.example.com/browse/SEP-100",
        )
        summary = reviewer._build_summary([], ticket=ticket, compliance_requirements=[])
        assert "### Requirement compliance" not in summary
        assert "### Ticket" in summary
        assert "SEP-100" in summary

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
        # Compliance check off → Ticket section (not Requirement compliance)
        assert "### Requirement compliance" not in summary
        assert "### Ticket" in summary
        assert "SEP-100" in summary
        assert "📋" not in summary

    def test_build_summary_jira_enabled_no_ticket(self, reviewer):
        summary = reviewer._build_summary([], jira_enabled=True)
        assert "No ticket found in branch name or PR title ℹ️" in summary
        assert "### Ticket" not in summary

    @pytest.mark.asyncio
    async def test_fetch_ticket_context_includes_type_and_status(self, mock_bitbucket, mock_llm):
        ticket = JiraTicket(
            key="SEP-100", title="Fix bug", description=None,
            labels=[], acceptance_criteria=None,
            url="https://jira.example.com/browse/SEP-100",
            issue_type="Bug", status="In Progress",
        )
        mock_jira = AsyncMock()
        mock_jira.fetch_ticket = AsyncMock(return_value=ticket)
        reviewer = Reviewer(mock_bitbucket, mock_llm, _review_config(), jira=mock_jira, db_pool=AsyncMock())
        context, returned_ticket = await reviewer._fetch_ticket_context("SEP-100")
        assert returned_ticket is ticket
        assert "**Type:** Bug" in context
        assert "**Status:** In Progress" in context

    @pytest.mark.asyncio
    async def test_fetch_ticket_context_omits_type_when_absent(self, mock_bitbucket, mock_llm):
        ticket = JiraTicket(
            key="SEP-100", title="Fix bug", description=None,
            labels=[], acceptance_criteria=None,
            url="https://jira.example.com/browse/SEP-100",
        )
        mock_jira = AsyncMock()
        mock_jira.fetch_ticket = AsyncMock(return_value=ticket)
        reviewer = Reviewer(mock_bitbucket, mock_llm, _review_config(), jira=mock_jira, db_pool=AsyncMock())
        context, _ = await reviewer._fetch_ticket_context("SEP-100")
        assert "**Type:**" not in context
        assert "**Status:**" not in context

    @pytest.mark.asyncio
    async def test_fetch_ticket_context_with_parent(self, mock_bitbucket, mock_llm):
        ticket = JiraTicket(
            key="SEP-101", title="Implement subtask", description=None,
            labels=[], acceptance_criteria=None,
            url="https://jira.example.com/browse/SEP-101",
            issue_type="Sub-task", parent_key="SEP-100",
        )
        parent = JiraTicket(
            key="SEP-100", title="Parent story", description="Full desc",
            labels=["backend"], acceptance_criteria="AC-1: Must work",
            url="https://jira.example.com/browse/SEP-100",
            issue_type="Story", status="In Progress",
        )
        mock_jira = AsyncMock()
        mock_jira.fetch_ticket = AsyncMock(return_value=ticket)
        reviewer = Reviewer(mock_bitbucket, mock_llm, _review_config(), jira=mock_jira, db_pool=AsyncMock())
        context, returned_ticket = await reviewer._fetch_ticket_context("SEP-101", parent=parent)
        assert returned_ticket is ticket
        assert "### Parent ticket: [SEP-100]" in context
        assert "### Sub-task: [SEP-101]" in context
        assert "Parent story" in context
        assert "Implement subtask" in context
        assert "Full desc" in context
        assert "AC-1: Must work" in context

    def test_build_summary_jira_not_configured_no_ticket(self, reviewer):
        summary = reviewer._build_summary([], jira_enabled=False)
        assert "Jira is not enabled ℹ️" in summary
        assert "No ticket found" not in summary

    def test_build_summary_with_elapsed(self, reviewer):
        summary = reviewer._build_summary(
            [], token_usage=(1000, 500), elapsed=12.3
        )
        assert "⏱️ 12.3s" in summary


class TestReviewWithJira:
    @pytest.mark.asyncio
    async def test_review_with_jira_context(self, mock_bitbucket, mock_llm):
        ticket = JiraTicket(
            key="SEP-123",
            title="Add login",
            description="Implement login page",
            labels=["frontend"],
            acceptance_criteria=None,
            url="https://jira.example.com/browse/SEP-123",
        )
        mock_jira = AsyncMock()
        mock_jira.fetch_ticket_with_parent = AsyncMock(return_value=(ticket, None))
        mock_jira.fetch_ticket = AsyncMock(return_value=ticket)
        result = _make_review_result(
            findings=[
                ReviewFinding(file="file.py", line=1, severity="warning", comment="Test issue"),
            ],
        )
        result.compliance_requirements = [
            {"requirement": "Implement login page", "met": True},
            {"requirement": "Add tests", "met": False},
        ]
        mock_llm.review_diff.return_value = result

        rev = Reviewer(mock_bitbucket, mock_llm, _review_config(), jira=mock_jira, db_pool=AsyncMock())
        payload = _make_payload(branch="feature/SEP-123-login")
        await rev.review_pull_request(payload)

        mock_jira.fetch_ticket_with_parent.assert_called_once_with("SEP-123")
        call_kwargs = mock_llm.review_diff.call_args[1]
        assert "SEP-123" in call_kwargs["ticket_context"]
        assert "Add login" in call_kwargs["ticket_context"]

        summary_text = mock_bitbucket.update_pr_comment.call_args[0][5] if mock_bitbucket.update_pr_comment.called else mock_bitbucket.post_pr_comment.call_args[0][3]
        assert "SEP-123" in summary_text
        assert "**Partially compliant** ⚠️" in summary_text
        assert "- Implement login page ✅" in summary_text
        assert "- Add tests ❌" in summary_text

    @pytest.mark.asyncio
    async def test_review_with_parent_ticket(self, mock_bitbucket, mock_llm):
        subtask = JiraTicket(
            key="SEP-124", title="Implement subtask", description=None,
            labels=[], acceptance_criteria=None,
            url="https://jira.example.com/browse/SEP-124",
            issue_type="Sub-task", parent_key="SEP-123",
        )
        parent = JiraTicket(
            key="SEP-123", title="Parent story", description="Full description",
            labels=["backend"], acceptance_criteria=None,
            url="https://jira.example.com/browse/SEP-123",
            issue_type="Story",
        )
        mock_jira = AsyncMock()
        mock_jira.fetch_ticket_with_parent = AsyncMock(return_value=(subtask, parent))
        mock_jira.fetch_ticket = AsyncMock(return_value=subtask)
        result = _make_review_result()
        result.compliance_requirements = [{"requirement": "Wire endpoint", "met": True}]
        mock_llm.review_diff.return_value = result

        rev = Reviewer(mock_bitbucket, mock_llm, _review_config(), jira=mock_jira, db_pool=AsyncMock())
        payload = _make_payload(branch="feature/SEP-124-subtask")
        await rev.review_pull_request(payload)

        call_kwargs = mock_llm.review_diff.call_args[1]
        assert "Parent ticket" in call_kwargs["ticket_context"]
        assert "SEP-123" in call_kwargs["ticket_context"]
        assert "Sub-task" in call_kwargs["ticket_context"]
        assert "SEP-124" in call_kwargs["ticket_context"]

        summary_text = mock_bitbucket.update_pr_comment.call_args[0][5] if mock_bitbucket.update_pr_comment.called else mock_bitbucket.post_pr_comment.call_args[0][3]
        assert "### Requirement compliance" in summary_text
        assert "[SEP-123]" in summary_text
        assert "↳" in summary_text
        assert "SEP-124" in summary_text

    @pytest.mark.asyncio
    async def test_review_jira_disabled(self, mock_bitbucket, mock_llm):
        rev = Reviewer(mock_bitbucket, mock_llm, _review_config(), db_pool=AsyncMock())
        payload = _make_payload(branch="feature/SEP-123-login")
        await rev.review_pull_request(payload)

        mock_llm.review_diff.assert_called_once()
        call_kwargs = mock_llm.review_diff.call_args[1]
        assert call_kwargs["ticket_context"] == ""

    @pytest.mark.asyncio
    async def test_review_passes_ticket_compliance_check_flag(self, mock_bitbucket, mock_llm):
        ticket = JiraTicket(
            key="SEP-123", title="Add login", description=None,
            labels=[], acceptance_criteria=None,
            url="https://jira.example.com/browse/SEP-123",
        )
        mock_jira = AsyncMock()
        mock_jira.fetch_ticket_with_parent = AsyncMock(return_value=(ticket, None))
        mock_jira.fetch_ticket = AsyncMock(return_value=ticket)
        mock_llm.review_diff.return_value = _make_review_result()

        rev = Reviewer(
            mock_bitbucket, mock_llm,
            _review_config(ticket_compliance_check=False),
            jira=mock_jira,
            db_pool=AsyncMock(),
        )
        payload = _make_payload(branch="feature/SEP-123-login")
        await rev.review_pull_request(payload)

        call_kwargs = mock_llm.review_diff.call_args[1]
        assert call_kwargs["ticket_compliance_check"] is False
        assert "SEP-123" in call_kwargs["ticket_context"]


def _make_feedback_payload(
    reply_text: str = "\U0001f44d",
    parent_id: int = 10,
    author: str = "dev",
) -> WebhookPayload:
    return WebhookPayload(**{
        "eventKey": "pr:comment:added",
        "commentParentId": parent_id,
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
            "author": {"user": {"name": "username"}},
        },
        "comment": {
            "id": 200,
            "text": reply_text,
            "author": {"name": author},
        },
    })


class TestHandleFeedback:
    @pytest.mark.asyncio
    async def test_positive_feedback_ignored(self, mock_bitbucket, mock_llm):
        mock_bitbucket.fetch_pr_comments.return_value = [
            {"id": 10, "text": "Bug here", "path": "a.py", "line": 5, "parent_id": None, "author_slug": "noergler"},
        ]
        mock_bitbucket.add_comment_reaction = AsyncMock(return_value=True)

        rev = Reviewer(mock_bitbucket, mock_llm, _review_config(), db_pool=AsyncMock())
        await rev.handle_feedback(_make_feedback_payload("\U0001f44d", parent_id=10))

        mock_bitbucket.add_comment_reaction.assert_not_called()
        mock_bitbucket.reply_to_comment.assert_not_called()

    @pytest.mark.asyncio
    async def test_non_disagree_negative_text_ignored(self, mock_bitbucket, mock_llm):
        mock_bitbucket.fetch_pr_comments.return_value = [
            {"id": 10, "text": "Bug here", "path": "a.py", "line": 5, "parent_id": None, "author_slug": "noergler"},
        ]
        mock_bitbucket.add_comment_reaction = AsyncMock(return_value=True)

        rev = Reviewer(mock_bitbucket, mock_llm, _review_config(), db_pool=AsyncMock())
        await rev.handle_feedback(_make_feedback_payload("wrong", parent_id=10))

        mock_bitbucket.add_comment_reaction.assert_not_called()
        mock_bitbucket.reply_to_comment.assert_not_called()

    @pytest.mark.asyncio
    async def test_negative_feedback_acknowledged(self, mock_bitbucket, mock_llm, caplog, monkeypatch):
        import json as _json
        monkeypatch.setattr(
            "app.reviewer.repository.get_finding_by_comment_id",
            AsyncMock(return_value={"file_path": "a.py", "line_number": 5, "severity": "critical"}),
        )
        mock_bitbucket.add_comment_reaction = AsyncMock(return_value=True)

        rev = Reviewer(mock_bitbucket, mock_llm, _review_config(), db_pool=AsyncMock())
        import logging
        with caplog.at_level(logging.INFO, logger="app.reviewer"):
            await rev.handle_feedback(_make_feedback_payload("disagree", parent_id=10))

        mock_bitbucket.add_comment_reaction.assert_called_once()
        disagree_logs = [r for r in caplog.records if "Disagree feedback" in r.message]
        assert len(disagree_logs) == 1
        payload = _json.loads(disagree_logs[0].message.split("Disagree feedback: ", 1)[1])
        assert payload["event"] == "disagree"
        assert payload["comment_id"] == 10
        assert payload["file"] == "a.py"
        assert payload["line"] == 5

    @pytest.mark.asyncio
    async def test_reaction_fallback_to_reply(self, mock_bitbucket, mock_llm, monkeypatch):
        from app.feedback import _FUN_RESPONSES

        monkeypatch.setattr(
            "app.reviewer.repository.get_finding_by_comment_id",
            AsyncMock(return_value={"file_path": "a.py", "line_number": 5, "severity": "warning"}),
        )
        mock_bitbucket.add_comment_reaction = AsyncMock(return_value=False)
        mock_bitbucket.reply_to_comment = AsyncMock()

        rev = Reviewer(mock_bitbucket, mock_llm, _review_config(), db_pool=AsyncMock())
        await rev.handle_feedback(_make_feedback_payload("disagree", parent_id=10))

        mock_bitbucket.reply_to_comment.assert_called_once()
        assert mock_bitbucket.reply_to_comment.call_args[0][4] in _FUN_RESPONSES

    @pytest.mark.asyncio
    async def test_ignores_reply_to_non_noergler_comment(self, mock_bitbucket, mock_llm):
        mock_bitbucket.fetch_pr_comments.return_value = [
            {"id": 10, "text": "Human comment", "path": "a.py", "line": 5, "parent_id": None},
        ]
        mock_bitbucket.add_comment_reaction = AsyncMock()

        rev = Reviewer(mock_bitbucket, mock_llm, _review_config(), db_pool=AsyncMock())
        await rev.handle_feedback(_make_feedback_payload("+1", parent_id=10))

        mock_bitbucket.add_comment_reaction.assert_not_called()

    @pytest.mark.asyncio
    async def test_logs_disagree_feedback(self, mock_bitbucket, mock_llm, caplog, monkeypatch):
        monkeypatch.setattr(
            "app.reviewer.repository.get_finding_by_comment_id",
            AsyncMock(return_value={"file_path": "b.py", "line_number": 3, "severity": "warning"}),
        )
        mock_bitbucket.add_comment_reaction = AsyncMock(return_value=True)

        rev = Reviewer(mock_bitbucket, mock_llm, _review_config(), db_pool=AsyncMock())
        import logging
        with caplog.at_level(logging.INFO):
            await rev.handle_feedback(_make_feedback_payload("disagree", parent_id=11))

        assert any("Disagree feedback" in r.message for r in caplog.records)

    @pytest.mark.asyncio
    async def test_ignores_bot_own_reply(self, mock_bitbucket, mock_llm):
        rev = Reviewer(mock_bitbucket, mock_llm, _review_config(), db_pool=AsyncMock())
        payload = _make_feedback_payload(
            "👀 Feedback noted, thanks!", parent_id=10, author="noergler",
        )
        await rev.handle_feedback(payload)

        mock_bitbucket.fetch_pr_comments.assert_not_called()
        mock_bitbucket.add_comment_reaction.assert_not_called()

    @pytest.mark.asyncio
    async def test_ignores_reply_to_summary_comment(self, mock_bitbucket, mock_llm):
        mock_bitbucket.fetch_pr_comments.return_value = [
            {"id": 10, "text": "Summary", "path": None, "line": None, "parent_id": None, "author_slug": "noergler"},
        ]
        mock_bitbucket.add_comment_reaction = AsyncMock()

        rev = Reviewer(mock_bitbucket, mock_llm, _review_config(), db_pool=AsyncMock())
        await rev.handle_feedback(_make_feedback_payload("disagree", parent_id=10))

        mock_bitbucket.add_comment_reaction.assert_not_called()
        mock_bitbucket.reply_to_comment.assert_not_called()



class TestHandlePrMerged:
    @pytest.mark.asyncio
    async def test_mixed_feedback(self, mock_bitbucket, mock_llm, caplog):
        mock_bitbucket.fetch_pr_comments.return_value = [
            {"id": 10, "text": "Bug", "path": "a.py", "line": 5, "parent_id": None, "author_slug": "noergler"},
            {"id": 11, "text": "Warning", "path": "b.py", "line": 3, "parent_id": None, "author_slug": "noergler"},
            {"id": 12, "text": "Issue", "path": "c.py", "line": 1, "parent_id": None, "author_slug": "noergler"},
            {"id": 20, "text": "disagree", "path": None, "line": None, "parent_id": 10, "author_slug": "dev"},
            {"id": 21, "text": "I disagree with this", "path": None, "line": None, "parent_id": 11, "author_slug": "dev"},
            {"id": 22, "text": "+1", "path": None, "line": None, "parent_id": 12, "author_slug": "dev"},
        ]

        rev = Reviewer(mock_bitbucket, mock_llm, _review_config(), db_pool=AsyncMock())
        import logging
        with caplog.at_level(logging.INFO):
            await rev.handle_pr_merged(_make_payload())

        stat_record = next(r for r in caplog.records if "merged" in r.message)
        assert "3 comments" in stat_record.message
        assert "2 disagreed" in stat_record.message
        assert "33% useful" in stat_record.message

    @pytest.mark.asyncio
    async def test_no_noergler_comments(self, mock_bitbucket, mock_llm, caplog):
        mock_bitbucket.fetch_pr_comments.return_value = [
            {"id": 10, "text": "Human comment", "path": "a.py", "line": 5, "parent_id": None},
        ]

        rev = Reviewer(mock_bitbucket, mock_llm, _review_config(), db_pool=AsyncMock())
        import logging
        with caplog.at_level(logging.INFO):
            await rev.handle_pr_merged(_make_payload())

        stat_record = next(r for r in caplog.records if "merged" in r.message)
        assert "no review comments" in stat_record.message

    @pytest.mark.asyncio
    async def test_all_comments_useful(self, mock_bitbucket, mock_llm, caplog):
        mock_bitbucket.fetch_pr_comments.return_value = [
            {"id": 10, "text": "Bug", "path": "a.py", "line": 5, "parent_id": None, "author_slug": "noergler"},
            {"id": 11, "text": "Warning", "path": "b.py", "line": 3, "parent_id": None, "author_slug": "noergler"},
            {"id": 20, "text": "+1 great catch", "path": None, "line": None, "parent_id": 10, "author_slug": "dev"},
        ]

        rev = Reviewer(mock_bitbucket, mock_llm, _review_config(), db_pool=AsyncMock())
        import logging
        with caplog.at_level(logging.INFO):
            await rev.handle_pr_merged(_make_payload())

        stat_record = next(r for r in caplog.records if "merged" in r.message)
        assert "2 comments" in stat_record.message
        assert "0 disagreed" in stat_record.message
        assert "100% useful" in stat_record.message



class TestHandlePrDeleted:
    @pytest.mark.asyncio
    async def test_purges_data_and_logs(self, mock_bitbucket, mock_llm, caplog):
        mock_pool = AsyncMock()
        mock_pool.acquire = AsyncMock()

        rev = Reviewer(mock_bitbucket, mock_llm, _review_config(), db_pool=mock_pool)

        with patch("app.reviewer.repository.purge_pr_data", new_callable=AsyncMock) as mock_purge:
            mock_purge.return_value = {
                "review_findings": 5, "pr_reviews": 1,
                "feedback_events": 1,
            }
            with caplog.at_level(logging.INFO):
                await rev.handle_pr_deleted(_make_payload())

            mock_purge.assert_awaited_once_with(mock_pool, "PROJ", "my-repo", 42)
        assert "purged 7 row(s)" in caplog.text

    @pytest.mark.asyncio
    async def test_no_data_to_purge(self, mock_bitbucket, mock_llm, caplog):
        mock_pool = AsyncMock()

        rev = Reviewer(mock_bitbucket, mock_llm, _review_config(), db_pool=mock_pool)

        with patch("app.reviewer.repository.purge_pr_data", new_callable=AsyncMock) as mock_purge:
            mock_purge.return_value = {
                "review_findings": 0, "pr_reviews": 0,
                "feedback_events": 0,
            }
            with caplog.at_level(logging.INFO):
                await rev.handle_pr_deleted(_make_payload())

        assert "no data to purge" in caplog.text

    @pytest.mark.asyncio
    async def test_db_error_logged(self, mock_bitbucket, mock_llm, caplog):
        mock_pool = AsyncMock()

        rev = Reviewer(mock_bitbucket, mock_llm, _review_config(), db_pool=mock_pool)

        with patch("app.reviewer.repository.purge_pr_data", new_callable=AsyncMock) as mock_purge:
            mock_purge.side_effect = RuntimeError("connection lost")
            with caplog.at_level(logging.WARNING):
                await rev.handle_pr_deleted(_make_payload())

        # _safe_db logs the warning, then handle_pr_deleted logs the error
        assert "DB operation failed" in caplog.text
        assert "Purge for deleted" in caplog.text


class TestIncrementalReview:
    @pytest.fixture
    def mock_bitbucket(self):
        client = AsyncMock()
        client.bot_username = "noergler"
        client.fetch_pr_diff = AsyncMock(return_value="diff --git a/file.py b/file.py\n+hello\n")
        client.fetch_commit_diff = AsyncMock(return_value="diff --git a/new.py b/new.py\n+world\n")
        client.fetch_file_content = AsyncMock(return_value="world\n")
        client.fetch_pr_comments = AsyncMock(return_value=[])
        client.post_inline_comment = AsyncMock(return_value=None)
        client.post_pr_comment = AsyncMock(return_value=(123, 0))
        client.update_pr_comment = AsyncMock(return_value=True)
        return client

    @pytest.fixture
    def mock_llm(self):
        client = AsyncMock()
        client.config.model = "gpt-5.3-codex"
        client.max_tokens_per_chunk = 80000
        client.context_window = 1_000_000
        client.prompt_template = "Review these files:\n{files}\n{repo_instructions}"
        client.review_diff = AsyncMock(return_value=_make_review_result([
            ReviewFinding(file="new.py", line=1, severity="warning", comment="Test issue"),
        ]))
        return client

    @pytest.mark.asyncio
    async def test_incremental_review_uses_commit_diff(self, mock_bitbucket, mock_llm, monkeypatch):
        """When DB has last_reviewed_commit and event is from_ref_updated, use commit diff."""
        monkeypatch.setattr(
            "app.reviewer.repository.get_last_reviewed_commit",
            AsyncMock(return_value="aabbccdd1234"),
        )

        rev = Reviewer(mock_bitbucket, mock_llm, _review_config(), db_pool=AsyncMock())
        payload = _make_payload()
        payload.eventKey = "pr:from_ref_updated"

        await rev.review_pull_request(payload)

        mock_bitbucket.fetch_commit_diff.assert_called_once_with(
            "PROJ", "my-repo", "aabbccdd1234", "abc123"
        )
        # Should NOT have called fetch_pr_diff since incremental succeeded
        mock_bitbucket.fetch_pr_diff.assert_not_called()
        mock_llm.review_diff.assert_called_once()

    @pytest.mark.asyncio
    async def test_full_review_when_no_last_reviewed_commit(self, mock_bitbucket, mock_llm, monkeypatch):
        """When DB has no last_reviewed_commit, fall back to full review."""
        monkeypatch.setattr(
            "app.reviewer.repository.get_last_reviewed_commit",
            AsyncMock(return_value=None),
        )

        rev = Reviewer(mock_bitbucket, mock_llm, _review_config(), db_pool=AsyncMock())
        payload = _make_payload()
        payload.eventKey = "pr:from_ref_updated"

        await rev.review_pull_request(payload)

        mock_bitbucket.fetch_commit_diff.assert_not_called()
        mock_bitbucket.fetch_pr_diff.assert_called_once()

    @pytest.mark.asyncio
    async def test_fallback_on_commit_diff_error(self, mock_bitbucket, mock_llm, monkeypatch):
        """When incremental diff fails (e.g. rebase), fall back to full review."""
        monkeypatch.setattr(
            "app.reviewer.repository.get_last_reviewed_commit",
            AsyncMock(return_value="aabbccdd1234"),
        )
        mock_bitbucket.fetch_commit_diff.side_effect = Exception("404 Not Found")

        rev = Reviewer(mock_bitbucket, mock_llm, _review_config(), db_pool=AsyncMock())
        payload = _make_payload()
        payload.eventKey = "pr:from_ref_updated"

        await rev.review_pull_request(payload)

        mock_bitbucket.fetch_commit_diff.assert_called_once()
        mock_bitbucket.fetch_pr_diff.assert_called_once()
        mock_llm.review_diff.assert_called_once()

    @pytest.mark.asyncio
    async def test_skip_when_incremental_diff_empty(self, mock_bitbucket, mock_llm, monkeypatch):
        """When incremental diff is empty, skip review entirely."""
        monkeypatch.setattr(
            "app.reviewer.repository.get_last_reviewed_commit",
            AsyncMock(return_value="aabbccdd1234"),
        )
        mock_bitbucket.fetch_commit_diff.return_value = "   \n"

        rev = Reviewer(mock_bitbucket, mock_llm, _review_config(), db_pool=AsyncMock())
        payload = _make_payload()
        payload.eventKey = "pr:from_ref_updated"

        await rev.review_pull_request(payload)

        mock_llm.review_diff.assert_not_called()

    @pytest.mark.asyncio
    async def test_pr_opened_always_does_full_review(self, mock_bitbucket, mock_llm, monkeypatch):
        """pr:opened always does full review even if DB has last_reviewed_commit."""
        monkeypatch.setattr(
            "app.reviewer.repository.get_last_reviewed_commit",
            AsyncMock(return_value="aabbccdd1234"),
        )

        rev = Reviewer(mock_bitbucket, mock_llm, _review_config(), db_pool=AsyncMock())
        payload = _make_payload()
        payload.eventKey = "pr:opened"

        await rev.review_pull_request(payload)

        mock_bitbucket.fetch_commit_diff.assert_not_called()
        mock_bitbucket.fetch_pr_diff.assert_called_once()

    @pytest.mark.asyncio
    async def test_summary_no_commit_metadata_in_comment(self, mock_bitbucket, mock_llm):
        """Summary comment should not contain commit metadata HTML comments."""
        rev = Reviewer(mock_bitbucket, mock_llm, _review_config(), db_pool=AsyncMock())
        payload = _make_payload()
        await rev.review_pull_request(payload)

        summary_text = mock_bitbucket.post_pr_comment.call_args[0][3]
        assert "<!-- noergler:" not in summary_text

    @pytest.mark.asyncio
    async def test_incremental_summary_header(self, mock_bitbucket, mock_llm, monkeypatch):
        """Incremental review summary should indicate it's incremental."""
        monkeypatch.setattr(
            "app.reviewer.repository.get_last_reviewed_commit",
            AsyncMock(return_value="aabbccdd1234"),
        )
        monkeypatch.setattr(
            "app.reviewer.repository.get_summary_comment_info",
            AsyncMock(return_value={"summary_comment_id": 1, "summary_comment_version": 2}),
        )
        monkeypatch.setattr(
            "app.reviewer.repository.upsert_pr_review",
            AsyncMock(return_value=99),
        )

        rev = Reviewer(mock_bitbucket, mock_llm, _review_config(), db_pool=AsyncMock())
        payload = _make_payload()
        payload.eventKey = "pr:from_ref_updated"

        await rev.review_pull_request(payload)

        # Summary should be updated (existing summary found)
        update_call = mock_bitbucket.update_pr_comment.call_args
        summary_text = update_call[0][5]  # text is the 6th positional arg
        assert "incremental update" in summary_text
        assert "aabbccdd12" in summary_text
        assert "abc123" in summary_text
