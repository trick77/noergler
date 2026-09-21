"""Tests for cross-file context analysis."""

import pytest

from app.llm_client import FileReviewData
from app.cross_file_context import (
    CrossFileRelationship,
    SymbolReference,
    _extract_changed_symbols,
    _find_references,
    build_cross_file_context,
    render_cross_file_context,
)


# ---------------------------------------------------------------------------
# Symbol extraction
# ---------------------------------------------------------------------------

class TestExtractChangedSymbols:
    def test_python_function(self):
        f = FileReviewData(
            path="service.py",
            diff=(
                "diff --git a/service.py b/service.py\n"
                "--- a/service.py\n"
                "+++ b/service.py\n"
                "@@ -1,3 +1,4 @@\n"
                "+def get_user(user_id, include_deleted=False):\n"
                "+    pass\n"
            ),
        )
        assert _extract_changed_symbols(f) == ["get_user"]

    def test_python_async_def(self):
        f = FileReviewData(
            path="service.py",
            diff=(
                "diff --git a/service.py b/service.py\n"
                "+++ b/service.py\n"
                "@@ -1,3 +1,4 @@\n"
                "+async def fetch_data(url):\n"
            ),
        )
        assert _extract_changed_symbols(f) == ["fetch_data"]

    def test_python_class(self):
        f = FileReviewData(
            path="models.py",
            diff=(
                "diff --git a/models.py b/models.py\n"
                "+++ b/models.py\n"
                "@@ -1,3 +1,4 @@\n"
                "+class UserFilter:\n"
            ),
        )
        assert _extract_changed_symbols(f) == ["UserFilter"]

    def test_typescript_function(self):
        f = FileReviewData(
            path="service.ts",
            diff=(
                "diff --git a/service.ts b/service.ts\n"
                "+++ b/service.ts\n"
                "@@ -1,3 +1,4 @@\n"
                "+export function getUser(id: string): User {\n"
            ),
        )
        assert _extract_changed_symbols(f) == ["getUser"]

    def test_typescript_class(self):
        f = FileReviewData(
            path="component.ts",
            diff=(
                "diff --git a/component.ts b/component.ts\n"
                "+++ b/component.ts\n"
                "@@ -1,3 +1,4 @@\n"
                "+export class UserService {\n"
            ),
        )
        assert _extract_changed_symbols(f) == ["UserService"]

    def test_java_method(self):
        f = FileReviewData(
            path="Service.java",
            diff=(
                "diff --git a/Service.java b/Service.java\n"
                "+++ b/Service.java\n"
                "@@ -1,3 +1,4 @@\n"
                "+    public void processOrder(Order order) {\n"
            ),
        )
        # The jvm scope pattern matches, but the name extractor looks for
        # fun/class/interface/enum keywords.  A plain "public void" method
        # won't match the JVM _SYMBOL_NAME_PATTERNS (which is intentional —
        # we focus on top-level declarations).
        # This test documents the current behaviour.
        symbols = _extract_changed_symbols(f)
        # Java methods without class/fun keyword are not extracted
        assert symbols == []

    def test_java_class(self):
        f = FileReviewData(
            path="Order.java",
            diff=(
                "diff --git a/Order.java b/Order.java\n"
                "+++ b/Order.java\n"
                "@@ -1,3 +1,4 @@\n"
                "+public class OrderService {\n"
            ),
        )
        assert _extract_changed_symbols(f) == ["OrderService"]

    def test_kotlin_fun(self):
        f = FileReviewData(
            path="Service.kt",
            diff=(
                "diff --git a/Service.kt b/Service.kt\n"
                "+++ b/Service.kt\n"
                "@@ -1,3 +1,4 @@\n"
                "+    fun processOrder(order: Order) {\n"
            ),
        )
        assert _extract_changed_symbols(f) == ["processOrder"]

    def test_ignores_short_names(self):
        f = FileReviewData(
            path="utils.py",
            diff=(
                "diff --git a/utils.py b/utils.py\n"
                "+++ b/utils.py\n"
                "@@ -1,3 +1,4 @@\n"
                "+def fn(x):\n"
            ),
        )
        # "fn" is only 2 chars, below _MIN_SYMBOL_LEN=3
        assert _extract_changed_symbols(f) == []

    def test_ignores_context_lines(self):
        f = FileReviewData(
            path="service.py",
            diff=(
                "diff --git a/service.py b/service.py\n"
                "+++ b/service.py\n"
                "@@ -1,3 +1,4 @@\n"
                " def existing_function():\n"
                "+    return 42\n"
            ),
        )
        # "existing_function" is on a context line, not an added line
        assert _extract_changed_symbols(f) == []

    def test_multiple_symbols(self):
        f = FileReviewData(
            path="service.py",
            diff=(
                "diff --git a/service.py b/service.py\n"
                "+++ b/service.py\n"
                "@@ -1,5 +1,8 @@\n"
                "+def fetch_user(user_id):\n"
                "+    pass\n"
                "+\n"
                "+class UserCache:\n"
                "+    pass\n"
            ),
        )
        assert _extract_changed_symbols(f) == ["fetch_user", "UserCache"]

    def test_no_duplicates(self):
        f = FileReviewData(
            path="service.py",
            diff=(
                "diff --git a/service.py b/service.py\n"
                "+++ b/service.py\n"
                "@@ -1,3 +1,4 @@\n"
                "+def process(data):\n"
                "@@ -10,3 +11,4 @@\n"
                "+def process(other_data):\n"
            ),
        )
        assert _extract_changed_symbols(f) == ["process"]


# ---------------------------------------------------------------------------
# Reference finding
# ---------------------------------------------------------------------------

class TestFindReferences:
    def test_finds_reference_in_content(self):
        target = FileReviewData(
            path="controller.py",
            diff="",
            content=(
                "from service import get_user\n"
                "\n"
                "def handler():\n"
                "    user = get_user(request.user_id)\n"
                "    return user\n"
            ),
        )
        refs = _find_references("get_user", target)
        assert len(refs) == 2
        assert refs[0].line_number == 1
        assert refs[1].line_number == 4

    def test_no_partial_match(self):
        target = FileReviewData(
            path="controller.py",
            diff="",
            content="def get_user_name():\n    pass\n",
        )
        refs = _find_references("get_user", target)
        assert len(refs) == 0

    def test_falls_back_to_diff(self):
        target = FileReviewData(
            path="controller.py",
            diff=(
                "+    result = get_user(1)\n"
            ),
            content=None,
        )
        refs = _find_references("get_user", target)
        assert len(refs) == 1

    def test_skips_comments(self):
        target = FileReviewData(
            path="controller.py",
            diff="",
            content="# get_user is deprecated\ndef handler():\n    get_user(1)\n",
        )
        refs = _find_references("get_user", target)
        assert len(refs) == 1
        assert refs[0].line_number == 3

    def test_max_refs_limit(self):
        lines = [f"    get_user({i})" for i in range(20)]
        target = FileReviewData(
            path="bulk.py",
            diff="",
            content="\n".join(lines),
        )
        refs = _find_references("get_user", target)
        assert len(refs) == 5  # _MAX_REFS_PER_SYMBOL


# ---------------------------------------------------------------------------
# build_cross_file_context
# ---------------------------------------------------------------------------

class TestBuildCrossFileContext:
    def test_single_file_returns_empty(self):
        f = FileReviewData(
            path="service.py",
            diff="+def get_user(id):\n",
            content="def get_user(id):\n    pass\n",
        )
        assert build_cross_file_context([f]) == []

    def test_finds_cross_file_relationship(self):
        service = FileReviewData(
            path="service.py",
            diff=(
                "diff --git a/service.py b/service.py\n"
                "+++ b/service.py\n"
                "@@ -1,3 +1,4 @@\n"
                "+def get_user(user_id, include_deleted=False):\n"
                "+    pass\n"
            ),
            content="def get_user(user_id, include_deleted=False):\n    pass\n",
        )
        controller = FileReviewData(
            path="controller.py",
            diff=(
                "diff --git a/controller.py b/controller.py\n"
                "+++ b/controller.py\n"
                "@@ -1,3 +1,3 @@\n"
                " def handler():\n"
                "-    user = get_user(request.id)\n"
                "+    user = get_user(request.id, False)\n"
            ),
            content=(
                "from service import get_user\n"
                "\n"
                "def handler():\n"
                "    user = get_user(request.id, False)\n"
            ),
        )
        rels = build_cross_file_context([service, controller])
        assert len(rels) == 1
        assert rels[0].symbol == "get_user"
        assert rels[0].defined_in == "service.py"
        assert any(r.file == "controller.py" for r in rels[0].references)

    def test_no_relationships_when_symbols_not_referenced(self):
        a = FileReviewData(
            path="a.py",
            diff=(
                "diff --git a/a.py b/a.py\n"
                "+++ b/a.py\n"
                "@@ -1,3 +1,4 @@\n"
                "+def isolated_function():\n"
            ),
            content="def isolated_function():\n    pass\n",
        )
        b = FileReviewData(
            path="b.py",
            diff="",
            content="def other():\n    return 1\n",
        )
        rels = build_cross_file_context([a, b])
        assert len(rels) == 0

    def test_no_symbols_extracted(self):
        a = FileReviewData(
            path="data.py",
            diff=(
                "diff --git a/data.py b/data.py\n"
                "+++ b/data.py\n"
                "@@ -1,3 +1,4 @@\n"
                "+    x = 42\n"
            ),
            content="x = 42\n",
        )
        b = FileReviewData(
            path="other.py",
            diff="",
            content="y = 1\n",
        )
        assert build_cross_file_context([a, b]) == []


# ---------------------------------------------------------------------------
# render_cross_file_context
# ---------------------------------------------------------------------------

class TestRenderCrossFileContext:
    def test_empty_relationships(self):
        assert render_cross_file_context([]) == ""

    def test_renders_relationships(self):
        rels = [
            CrossFileRelationship(
                symbol="get_user",
                defined_in="service.py",
                references=[
                    SymbolReference(file="controller.py", line_number=4, line_text="user = get_user(request.id)"),
                ],
            ),
        ]
        output = render_cross_file_context(rels)
        assert "## Cross-file relationships" in output
        assert "`get_user`" in output
        assert "`service.py`" in output
        assert "`controller.py:4`" in output

    def test_truncation(self):
        refs = [
            SymbolReference(file=f"file{i}.py", line_number=i, line_text=f"call_{i}()")
            for i in range(50)
        ]
        rels = [
            CrossFileRelationship(symbol="big_func", defined_in="source.py", references=refs),
        ]
        output = render_cross_file_context(rels)
        assert "truncated" in output
