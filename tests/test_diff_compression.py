from app.copilot import FileReviewData
from app.diff_compression import (
    CompressionResult,
    compress_for_large_pr,
    detect_language,
    determine_repo_languages,
    is_deletion_only_hunk,
    is_rename_only,
    is_small_pr,
    is_test_file,
    remove_deletion_only_hunks,
    sort_files_by_language_priority,
)


def _count_tokens_fake(text: str) -> int:
    return len(text)


def _format_entry_fake(f: FileReviewData) -> str:
    return f"## {f.path}\n{f.diff}\n{f.content or ''}"


class TestDetectLanguage:
    def test_python(self):
        assert detect_language("src/main.py") == "python"
        assert detect_language("stubs/types.pyi") == "python"

    def test_jvm(self):
        assert detect_language("src/Main.java") == "jvm"
        assert detect_language("src/Main.kt") == "jvm"
        assert detect_language("build.gradle.kts") == "build-config"

    def test_typescript(self):
        assert detect_language("src/app.ts") == "typescript"
        assert detect_language("src/App.tsx") == "typescript"
        assert detect_language("utils.mjs") == "typescript"

    def test_html(self):
        assert detect_language("template.html") == "html"

    def test_css(self):
        assert detect_language("styles.scss") == "css"
        assert detect_language("app.css") == "css"

    def test_config(self):
        assert detect_language("config.yaml") == "config"
        assert detect_language("settings.toml") == "config"

    def test_docs(self):
        assert detect_language("README.md") == "docs"
        assert detect_language("notes.rst") == "docs"

    def test_build_files(self):
        assert detect_language("pom.xml") == "build-config"
        assert detect_language("angular.json") == "build-config"
        assert detect_language("nx.json") == "build-config"
        assert detect_language("package.json") == "build-config"
        assert detect_language("tsconfig.json") == "build-config"
        assert detect_language("tsconfig.app.json") == "build-config"

    def test_unknown(self):
        assert detect_language("Makefile") == "other"
        assert detect_language("data.dat") == "other"


class TestIsTestFile:
    def test_python_tests(self):
        assert is_test_file("tests/test_main.py") is True
        assert is_test_file("src/main_test.py") is True
        assert is_test_file("src/main.py") is False

    def test_java_tests(self):
        assert is_test_file("src/test/MainTest.java") is True
        assert is_test_file("src/test/MainTests.java") is True
        assert is_test_file("src/main/Main.java") is False

    def test_typescript_tests(self):
        assert is_test_file("app.spec.ts") is True
        assert is_test_file("app.test.ts") is True
        assert is_test_file("app.ts") is False

    def test_directory_pattern(self):
        assert is_test_file("src/test/Helper.java") is True
        assert is_test_file("src/__tests__/App.tsx") is True


class TestDetermineRepoLanguages:
    def test_frequency_ordering(self):
        files = [
            FileReviewData(path="a.py", diff=""),
            FileReviewData(path="b.py", diff=""),
            FileReviewData(path="c.java", diff=""),
            FileReviewData(path="d.ts", diff=""),
            FileReviewData(path="e.ts", diff=""),
            FileReviewData(path="f.ts", diff=""),
        ]
        langs = determine_repo_languages(files)
        assert langs[0] == "typescript"  # 3 files
        assert langs[1] == "python"      # 2 files
        assert langs[2] == "jvm"         # 1 file

    def test_excludes_test_files_from_count(self):
        files = [
            FileReviewData(path="src/main.py", diff=""),
            FileReviewData(path="tests/test_main.py", diff=""),
            FileReviewData(path="tests/test_other.py", diff=""),
            FileReviewData(path="src/App.ts", diff=""),
            FileReviewData(path="src/Other.ts", diff=""),
        ]
        langs = determine_repo_languages(files)
        assert langs[0] == "typescript"  # 2 non-test files
        assert langs[1] == "python"      # 1 non-test file


class TestSortFilesByLanguagePriority:
    def test_sorts_by_language_then_tokens(self):
        files = [
            FileReviewData(path="small.ts", diff="x"),
            FileReviewData(path="big.py", diff="x" * 100),
            FileReviewData(path="small.py", diff="x" * 10),
        ]
        order = ["python", "typescript"]
        result = sort_files_by_language_priority(files, order, _count_tokens_fake)
        assert result[0].path == "big.py"
        assert result[1].path == "small.py"
        assert result[2].path == "small.ts"

    def test_test_files_after_source(self):
        files = [
            FileReviewData(path="tests/test_main.py", diff="x" * 50),
            FileReviewData(path="src/main.py", diff="x" * 10),
            FileReviewData(path="src/app.ts", diff="x" * 20),
        ]
        order = ["python", "typescript"]
        result = sort_files_by_language_priority(files, order, _count_tokens_fake)
        assert result[0].path == "src/main.py"      # source python
        assert result[1].path == "src/app.ts"        # source ts
        assert result[2].path == "tests/test_main.py"  # test

    def test_deprioritized_groups_last(self):
        files = [
            FileReviewData(path="config.yaml", diff="x"),
            FileReviewData(path="src/main.py", diff="x"),
            FileReviewData(path="README.md", diff="x"),
        ]
        order = ["python", "config", "docs"]
        result = sort_files_by_language_priority(files, order, _count_tokens_fake)
        assert result[0].path == "src/main.py"
        assert result[1].path == "config.yaml"
        assert result[2].path == "README.md"


class TestHunkAnalysis:
    def test_deletion_only_hunk(self):
        hunk = ["@@ -1,3 +1,0 @@", "-line1", "-line2", "-line3"]
        assert is_deletion_only_hunk(hunk) is True

    def test_mixed_hunk(self):
        hunk = ["@@ -1,3 +1,2 @@", "-old", "+new", " context"]
        assert is_deletion_only_hunk(hunk) is False

    def test_remove_deletion_only_hunks_preserves_additions(self):
        diff = (
            "diff --git a/file.py b/file.py\n"
            "--- a/file.py\n"
            "+++ b/file.py\n"
            "@@ -1,3 +1,0 @@\n"
            "-deleted\n"
            "@@ -10,3 +10,4 @@\n"
            " context\n"
            "+added\n"
            " more context"
        )
        result = remove_deletion_only_hunks(diff)
        assert "+added" in result
        assert "-deleted" not in result

    def test_remove_deletion_only_hunks_all_deletion(self):
        diff = (
            "diff --git a/file.py b/file.py\n"
            "--- a/file.py\n"
            "+++ b/file.py\n"
            "@@ -1,3 +1,0 @@\n"
            "-line1\n"
            "-line2"
        )
        result = remove_deletion_only_hunks(diff)
        assert result == ""


class TestRenameOnly:
    def test_rename_only(self):
        diff = (
            "diff --git a/old.py b/new.py\n"
            "similarity index 100%\n"
            "rename from old.py\n"
            "rename to new.py\n"
        )
        assert is_rename_only(diff) is True

    def test_rename_with_changes(self):
        diff = (
            "diff --git a/old.py b/new.py\n"
            "similarity index 90%\n"
            "rename from old.py\n"
            "rename to new.py\n"
            "@@ -1,2 +1,3 @@\n"
            " line\n"
            "+new line\n"
        )
        assert is_rename_only(diff) is False

    def test_not_rename(self):
        diff = "diff --git a/file.py b/file.py\n+hello\n"
        assert is_rename_only(diff) is False


class TestIsSmallPr:
    def test_small_pr_fits(self):
        files = [FileReviewData(path="a.py", diff="short diff")]
        assert is_small_pr(files, 10000, "template {files}", _count_tokens_fake, _format_entry_fake) is True

    def test_large_pr_does_not_fit(self):
        files = [FileReviewData(path="a.py", diff="x" * 10000)]
        assert is_small_pr(files, 100, "template {files}", _count_tokens_fake, _format_entry_fake) is False

    def test_expansion_margin_rejects_borderline(self):
        """A PR that fits without margin but not with 1.5x expansion ratio."""
        files = [FileReviewData(path="a.py", diff="x" * 80)]
        # _format_entry_fake produces "## a.py\nxxxx...\n" — roughly len(diff) + overhead
        # With template "t {files}", overhead is len("t ") = 2
        # Entry ~ "## a.py\n" + "x"*80 + "\n" = 90 chars
        # available = 100 - 2 = 98; 90 <= 98 but 90*1.5=135 > 98
        assert is_small_pr(files, 100, "t {files}", _count_tokens_fake, _format_entry_fake) is False

    def test_no_margin_with_ratio_1(self):
        """With ratio=1.0, borderline PR should still fit."""
        files = [FileReviewData(path="a.py", diff="x" * 80)]
        assert is_small_pr(
            files, 100, "t {files}", _count_tokens_fake, _format_entry_fake,
            context_expansion_ratio=1.0,
        ) is True


class TestCompressForLargePr:
    def test_separates_deleted_and_renamed(self):
        files = [
            FileReviewData(path="active.py", diff="diff --git a/active.py b/active.py\n@@ -1 +1 @@\n+new"),
            FileReviewData(path="gone.py", diff="diff --git a/gone.py b/gone.py\n+++ /dev/null\n-old"),
            FileReviewData(
                path="moved.py",
                diff="diff --git a/old.py b/moved.py\nsimilarity index 100%\nrename from old.py\nrename to moved.py\n",
            ),
        ]
        result = compress_for_large_pr(files, 100000, "template {files}", _count_tokens_fake, _format_entry_fake)
        assert len(result.included_files) == 1
        assert result.included_files[0].path == "active.py"
        assert "gone.py" in result.deleted_file_paths
        assert "moved.py" in result.renamed_file_paths

    def test_budget_respected(self):
        files = [
            FileReviewData(path="a.py", diff="@@ -1 +1 @@\n+" + "x" * 40, content="content"),
            FileReviewData(path="b.py", diff="@@ -1 +1 @@\n+" + "y" * 40, content="content"),
            FileReviewData(path="c.py", diff="@@ -1 +1 @@\n+" + "z" * 40, content="content"),
        ]
        # Very small budget — should not fit all files
        result = compress_for_large_pr(files, 150, "t {files}", _count_tokens_fake, _format_entry_fake)
        assert len(result.included_files) + len(result.other_modified_paths) == 3
        assert len(result.other_modified_paths) > 0

    def test_deletion_only_hunks_removed(self):
        diff = (
            "diff --git a/file.py b/file.py\n"
            "--- a/file.py\n"
            "+++ b/file.py\n"
            "@@ -1,3 +1,0 @@\n"
            "-deleted only"
        )
        files = [FileReviewData(path="file.py", diff=diff)]
        result = compress_for_large_pr(files, 100000, "template {files}", _count_tokens_fake, _format_entry_fake)
        # File with only deletion hunks moves to deleted-like list
        assert len(result.included_files) == 0
        assert "file.py" in result.deleted_file_paths
