import httpx
import pytest
import respx

from app.bitbucket import NOERGLER_MARKER, BitbucketClient
from app.config import BitbucketConfig
from app.models import ReviewFinding

BASE_URL = "https://bitbucket.company.com"


@pytest.fixture
def bb_config():
    return BitbucketConfig(
        base_url=BASE_URL,
        token="test-token",
        webhook_secret="test-secret",
    )


@pytest.fixture
def client(bb_config):
    return BitbucketClient(bb_config)


class TestBitbucketClient:
    @pytest.mark.asyncio
    @respx.mock
    async def test_fetch_pr_diff(self, client):
        diff_text = "diff --git a/file.py b/file.py\n+hello\n"
        respx.get(
            f"{BASE_URL}/rest/api/1.0/projects/PROJ/repos/my-repo/pull-requests/1/diff"
        ).mock(return_value=httpx.Response(200, text=diff_text))

        result = await client.fetch_pr_diff("PROJ", "my-repo", 1)
        assert result == diff_text
        await client.close()

    @pytest.mark.asyncio
    @respx.mock
    async def test_fetch_pr_diff_with_context_lines(self, client):
        diff_text = "diff --git a/file.py b/file.py\n+hello\n"
        route = respx.get(
            f"{BASE_URL}/rest/api/1.0/projects/PROJ/repos/my-repo/pull-requests/1/diff"
        ).mock(return_value=httpx.Response(200, text=diff_text))

        result = await client.fetch_pr_diff("PROJ", "my-repo", 1, context_lines=20)
        assert result == diff_text
        assert route.calls[0].request.url.params["contextLines"] == "20"
        await client.close()

    @pytest.mark.asyncio
    @respx.mock
    async def test_fetch_pr_diff_no_context_lines_param_by_default(self, client):
        diff_text = "diff --git a/file.py b/file.py\n+hello\n"
        route = respx.get(
            f"{BASE_URL}/rest/api/1.0/projects/PROJ/repos/my-repo/pull-requests/1/diff"
        ).mock(return_value=httpx.Response(200, text=diff_text))

        result = await client.fetch_pr_diff("PROJ", "my-repo", 1)
        assert result == diff_text
        assert "contextLines" not in route.calls[0].request.url.params
        await client.close()

    @pytest.mark.asyncio
    @respx.mock
    async def test_fetch_file_content(self, client):
        content = "print('hello')\n"
        respx.get(
            f"{BASE_URL}/rest/api/1.0/projects/PROJ/repos/my-repo/raw/src/main.py"
        ).mock(return_value=httpx.Response(200, text=content))

        result = await client.fetch_file_content("PROJ", "my-repo", "abc123", "src/main.py")
        assert result == content
        await client.close()

    @pytest.mark.asyncio
    @respx.mock
    async def test_post_inline_comment(self, client):
        respx.post(
            f"{BASE_URL}/rest/api/1.0/projects/PROJ/repos/my-repo/pull-requests/1/comments"
        ).mock(return_value=httpx.Response(201, json={"id": 1}))

        finding = ReviewFinding(
            file="src/main.py", line=10, severity="critical", comment="Bug here"
        )
        await client.post_inline_comment("PROJ", "my-repo", 1, finding)
        await client.close()

    @pytest.mark.asyncio
    @respx.mock
    async def test_post_pr_comment(self, client):
        respx.post(
            f"{BASE_URL}/rest/api/1.0/projects/PROJ/repos/my-repo/pull-requests/1/comments"
        ).mock(return_value=httpx.Response(201, json={"id": 2}))

        await client.post_pr_comment("PROJ", "my-repo", 1, "Review summary")
        await client.close()

    @pytest.mark.asyncio
    @respx.mock
    async def test_post_inline_comment_falls_back_to_context_line_type(self, client):
        route = respx.post(
            f"{BASE_URL}/rest/api/1.0/projects/PROJ/repos/my-repo/pull-requests/1/comments"
        ).mock(side_effect=[
            httpx.Response(400, json={"errors": [{"message": "invalid line"}]}),
            httpx.Response(201, json={"id": 1}),
        ])

        finding = ReviewFinding(
            file="src/main.py", line=5, severity="suggestion", comment="Consider refactoring"
        )
        await client.post_inline_comment("PROJ", "my-repo", 1, finding)

        assert route.call_count == 2
        import json
        first_body = json.loads(route.calls[0].request.content)
        second_body = json.loads(route.calls[1].request.content)
        assert first_body["anchor"]["lineType"] == "ADDED"
        assert second_body["anchor"]["lineType"] == "CONTEXT"
        await client.close()

    @pytest.mark.asyncio
    @respx.mock
    async def test_post_inline_comment_includes_marker(self, client):
        route = respx.post(
            f"{BASE_URL}/rest/api/1.0/projects/PROJ/repos/my-repo/pull-requests/1/comments"
        ).mock(return_value=httpx.Response(201, json={"id": 1}))

        finding = ReviewFinding(
            file="src/main.py", line=10, severity="critical", comment="Bug here"
        )
        await client.post_inline_comment("PROJ", "my-repo", 1, finding)

        body = route.calls[0].request.content.decode()
        assert NOERGLER_MARKER in body
        await client.close()

    @pytest.mark.asyncio
    @respx.mock
    async def test_reply_to_comment(self, client):
        import json as _json

        route = respx.post(
            f"{BASE_URL}/rest/api/1.0/projects/PROJ/repos/my-repo/pull-requests/1/comments"
        ).mock(return_value=httpx.Response(201, json={"id": 99}))

        await client.reply_to_comment("PROJ", "my-repo", 1, 42, "Here is the answer")

        assert route.call_count == 1
        body = _json.loads(route.calls[0].request.content)
        assert body["parent"]["id"] == 42
        assert NOERGLER_MARKER in body["text"]
        assert "Here is the answer" in body["text"]
        await client.close()

    @pytest.mark.asyncio
    @respx.mock
    async def test_fetch_pr_comments(self, client):
        activities_response = {
            "values": [
                {
                    "action": "COMMENTED",
                    "comment": {
                        "text": f"❌ **Critical:** bug\n\n{NOERGLER_MARKER}",
                        "anchor": {"path": "a.py", "line": 10},
                    },
                },
                {
                    "action": "APPROVED",
                },
                {
                    "action": "COMMENTED",
                    "comment": {
                        "text": "Human comment",
                    },
                },
            ],
            "isLastPage": True,
        }
        respx.get(
            f"{BASE_URL}/rest/api/1.0/projects/PROJ/repos/my-repo/pull-requests/1/activities"
        ).mock(return_value=httpx.Response(200, json=activities_response))

        comments = await client.fetch_pr_comments("PROJ", "my-repo", 1)
        assert len(comments) == 2
        assert comments[0]["path"] == "a.py"
        assert comments[0]["line"] == 10
        assert NOERGLER_MARKER in comments[0]["text"]
        assert comments[1]["text"] == "Human comment"
        await client.close()

    @pytest.mark.asyncio
    @respx.mock
    async def test_fetch_pr_comments_pagination(self, client):
        page1 = {
            "values": [
                {
                    "action": "COMMENTED",
                    "comment": {"text": "comment1", "anchor": {"path": "a.py", "line": 1}},
                },
            ],
            "isLastPage": False,
            "nextPageStart": 1000,
        }
        page2 = {
            "values": [
                {
                    "action": "COMMENTED",
                    "comment": {"text": "comment2", "anchor": {"path": "b.py", "line": 2}},
                },
            ],
            "isLastPage": True,
        }
        route = respx.get(
            f"{BASE_URL}/rest/api/1.0/projects/PROJ/repos/my-repo/pull-requests/1/activities"
        ).mock(side_effect=[
            httpx.Response(200, json=page1),
            httpx.Response(200, json=page2),
        ])

        comments = await client.fetch_pr_comments("PROJ", "my-repo", 1)
        assert len(comments) == 2
        assert comments[0]["path"] == "a.py"
        assert comments[1]["path"] == "b.py"
        assert route.call_count == 2
        await client.close()
