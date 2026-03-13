import pytest

from app.context_expansion import (
    expand_all_files,
    expand_context,
    find_enclosing_scope_line,
    parse_hunks,
)
from app.copilot import FileReviewData


class TestParseHunks:
    def test_single_hunk(self):
        diff = (
            "diff --git a/file.py b/file.py\n"
            "--- a/file.py\n"
            "+++ b/file.py\n"
            "@@ -10,3 +10,4 @@\n"
            "-old line\n"
            "+new line\n"
            "+added line"
        )
        headers, hunks = parse_hunks(diff)
        assert len(headers) == 3
        assert len(hunks) == 1
        assert hunks[0].old_start == 10
        assert hunks[0].old_count == 3
        assert hunks[0].new_start == 10
        assert hunks[0].new_count == 4

    def test_multiple_hunks(self):
        diff = (
            "diff --git a/file.py b/file.py\n"
            "@@ -1,2 +1,3 @@\n"
            "+added\n"
            "@@ -20,1 +21,1 @@\n"
            "-old\n"
            "+new"
        )
        headers, hunks = parse_hunks(diff)
        assert len(hunks) == 2
        assert hunks[0].new_start == 1
        assert hunks[1].new_start == 21

    def test_no_hunks(self):
        diff = "diff --git a/file.py b/file.py\n--- a/file.py\n+++ b/file.py"
        headers, hunks = parse_hunks(diff)
        assert len(hunks) == 0
        assert len(headers) == 3


class TestFindEnclosingScope:
    def test_python_def(self):
        lines = [
            "import os",
            "",
            "def my_function():",
            "    x = 1",
            "    y = 2",
            "    return x + y",
        ]
        # Searching backwards from line 5 (1-based), hunk_new_start=5
        result = find_enclosing_scope_line(lines, 5, 8, "app/utils.py")
        assert result == 3  # def my_function() is on line 3

    def test_python_class(self):
        lines = [
            "import os",
            "",
            "class MyClass:",
            "    def method(self):",
            "        pass",
        ]
        result = find_enclosing_scope_line(lines, 5, 8, "app/models.py")
        assert result == 4  # finds def method first

    def test_python_async_def(self):
        lines = [
            "",
            "async def handler():",
            "    await something()",
            "    return result",
        ]
        result = find_enclosing_scope_line(lines, 4, 8, "app/main.py")
        assert result == 2

    def test_java_method(self):
        lines = [
            "package com.example;",
            "",
            "public class Foo {",
            "    private int bar() {",
            "        return 42;",
            "    }",
            "}",
        ]
        result = find_enclosing_scope_line(lines, 5, 8, "src/Foo.java")
        assert result == 4

    def test_typescript_function(self):
        lines = [
            "import { x } from 'y';",
            "",
            "function doStuff() {",
            "  const a = 1;",
            "  return a;",
            "}",
        ]
        result = find_enclosing_scope_line(lines, 5, 8, "src/utils.ts")
        assert result == 3

    def test_no_scope_found(self):
        lines = [
            "x = 1",
            "y = 2",
            "z = 3",
        ]
        result = find_enclosing_scope_line(lines, 3, 8, "script.py")
        assert result is None

    def test_skips_non_code_files(self):
        lines = [
            "key: value",
            "other: stuff",
            "changed: true",
        ]
        result = find_enclosing_scope_line(lines, 3, 8, "config.yaml")
        assert result is None

    def test_skips_docs(self):
        lines = [
            "# Title",
            "",
            "Some text",
        ]
        result = find_enclosing_scope_line(lines, 3, 8, "README.md")
        assert result is None

    def test_max_lines_respected(self):
        lines = [
            "def far_away():",
            "    pass",
            "",
            "",
            "",
            "",
            "    x = 1",
        ]
        # max_lines=2, so won't reach the def on line 1
        result = find_enclosing_scope_line(lines, 7, 2, "app/foo.py")
        assert result is None

    def test_generic_language_with_brace(self):
        lines = [
            "func main() {",
            "    fmt.Println()",
            "    return",
            "}",
        ]
        result = find_enclosing_scope_line(lines, 3, 8, "main.go")
        assert result == 1


class TestExpandContext:
    def _make_diff(self, old_start=10, old_count=1, new_start=10, new_count=2, body="-old\n+new\n+added"):
        return (
            f"diff --git a/file.py b/file.py\n"
            f"--- a/file.py\n"
            f"+++ b/file.py\n"
            f"@@ -{old_start},{old_count} +{new_start},{new_count} @@\n"
            f"{body}"
        )

    def _make_content(self, num_lines=20):
        return "\n".join(f"line {i+1}" for i in range(num_lines))

    def test_adds_before_context(self):
        diff = self._make_diff(old_start=5, old_count=1, new_start=5, new_count=1, body="-old\n+new")
        content = self._make_content(10)
        result = expand_context(diff, content, "file.py", before=3, after=0, dynamic_context=False)
        assert " line 2" in result
        assert " line 3" in result
        assert " line 4" in result

    def test_adds_after_context(self):
        diff = self._make_diff(old_start=5, old_count=1, new_start=5, new_count=1, body="-old\n+new")
        content = self._make_content(10)
        result = expand_context(diff, content, "file.py", before=0, after=2, dynamic_context=False)
        assert " line 6" in result
        assert " line 7" in result

    def test_asymmetric_context(self):
        diff = self._make_diff(old_start=8, old_count=1, new_start=8, new_count=1, body="-old\n+new")
        content = self._make_content(15)
        result = expand_context(diff, content, "file.py", before=3, after=1, dynamic_context=False)
        # 3 lines before
        assert " line 5" in result
        assert " line 6" in result
        assert " line 7" in result
        # 1 line after
        assert " line 9" in result
        # but not 2 lines after
        assert " line 10" not in result

    def test_no_content_returns_original(self):
        diff = self._make_diff()
        result = expand_context(diff, None, "file.py")
        assert result == diff

    def test_no_hunks_returns_original(self):
        diff = "diff --git a/file.py b/file.py\n--- a/file.py\n+++ b/file.py"
        content = self._make_content()
        result = expand_context(diff, content, "file.py")
        assert result == diff

    def test_dynamic_context_finds_function(self):
        content = (
            "import os\n"
            "\n"
            "def process_data():\n"
            "    x = 1\n"
            "    y = 2\n"
            "    z = 3\n"
            "    result = x + y + z\n"
            "    return result"
        )
        # Hunk at line 7, before=1 means static window starts at line 6
        diff = self._make_diff(old_start=7, old_count=1, new_start=7, new_count=1, body="-old\n+new")
        result = expand_context(diff, content, "app/utils.py", before=1, after=0, max_dynamic_before=8, dynamic_context=True)
        # Should include the function definition
        assert " def process_data():" in result

    def test_dynamic_context_disabled(self):
        content = (
            "import os\n"
            "\n"
            "def process_data():\n"
            "    x = 1\n"
            "    y = 2\n"
            "    z = 3\n"
            "    result = x + y + z\n"
            "    return result"
        )
        diff = self._make_diff(old_start=7, old_count=1, new_start=7, new_count=1, body="-old\n+new")
        result = expand_context(diff, content, "app/utils.py", before=1, after=0, dynamic_context=False)
        # Should NOT include the function definition
        assert "def process_data():" not in result
        # But should include 1 line before
        assert " z = 3" in result

    def test_clamps_to_start_of_file(self):
        diff = self._make_diff(old_start=1, old_count=1, new_start=1, new_count=1, body="-old\n+new")
        content = self._make_content(5)
        result = expand_context(diff, content, "file.py", before=5, after=0, dynamic_context=False)
        # Should not crash, just provide what's available
        assert "+new" in result

    def test_clamps_to_end_of_file(self):
        content = self._make_content(5)
        diff = self._make_diff(old_start=5, old_count=1, new_start=5, new_count=1, body="-old\n+new")
        result = expand_context(diff, content, "file.py", before=0, after=10, dynamic_context=False)
        assert "+new" in result

    def test_hunk_header_updated(self):
        diff = self._make_diff(old_start=10, old_count=1, new_start=10, new_count=1, body="-old\n+new")
        content = self._make_content(20)
        result = expand_context(diff, content, "file.py", before=3, after=1, dynamic_context=False)
        # New start should be 10 - 3 = 7
        assert "@@ -7," in result
        assert "+7," in result


class TestExpandAllFiles:
    def test_expands_multiple_files(self):
        content = "\n".join(f"line {i+1}" for i in range(20))
        files = [
            FileReviewData(
                path="a.py",
                diff="diff --git a/a.py b/a.py\n@@ -5,1 +5,1 @@\n-old\n+new",
                content=content,
            ),
            FileReviewData(
                path="b.py",
                diff="diff --git a/b.py b/b.py\n@@ -10,1 +10,1 @@\n-old\n+new",
                content=content,
            ),
        ]
        result = expand_all_files(files, before=2, after=1, dynamic_context=False)
        assert len(result) == 2
        assert result[0].path == "a.py"
        assert result[1].path == "b.py"
        # Content preserved
        assert result[0].content == content
        # Diff was expanded
        assert " line 3" in result[0].diff
        assert " line 4" in result[0].diff

    def test_preserves_none_content(self):
        files = [
            FileReviewData(
                path="a.py",
                diff="diff --git a/a.py b/a.py\n@@ -5,1 +5,1 @@\n-old\n+new",
                content=None,
            ),
        ]
        result = expand_all_files(files, before=2, after=1)
        assert len(result) == 1
        # Original diff returned unchanged
        assert result[0].diff == files[0].diff
        assert result[0].content is None


class TestMergeOverlappingHunks:
    def _count_hunk_headers(self, diff_text: str) -> int:
        return sum(1 for line in diff_text.split("\n") if line.startswith("@@"))

    def test_adjacent_hunks_merged(self):
        content = "\n".join(f"line {i+1}" for i in range(20))
        diff = (
            "diff --git a/file.py b/file.py\n"
            "@@ -5,1 +5,1 @@\n"
            "-old1\n"
            "+new1\n"
            "@@ -7,1 +7,1 @@\n"
            "-old2\n"
            "+new2"
        )
        result = expand_context(diff, content, "file.py", before=2, after=2, dynamic_context=False)
        # Should merge into a single hunk since they overlap
        assert self._count_hunk_headers(result) == 1

    def test_distant_hunks_not_merged(self):
        content = "\n".join(f"line {i+1}" for i in range(30))
        diff = (
            "diff --git a/file.py b/file.py\n"
            "@@ -5,1 +5,1 @@\n"
            "-old1\n"
            "+new1\n"
            "@@ -20,1 +20,1 @@\n"
            "-old2\n"
            "+new2"
        )
        result = expand_context(diff, content, "file.py", before=2, after=1, dynamic_context=False)
        # Should remain as two separate hunks
        assert self._count_hunk_headers(result) == 2
