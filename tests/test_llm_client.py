import json
from unittest.mock import AsyncMock, MagicMock

import httpx
import openai
import pytest

from app.config import LLMConfig, ReviewConfig
from app.llm_client import (
    LLMClient,
    FileReviewData,
    ReviewSummary,
    _REVIEW_SYSTEM_MESSAGE,
    format_file_entry,
    _context_window_for,
    _group_files_by_token_budget,
    _merge_review_summaries,
    _parse_mention_response,
    _parse_review_response,
    _render_file_group,
    extract_path,
    is_deleted,
    is_reviewable_diff,
    split_by_file,
)


@pytest.fixture
def llm_config():
    return LLMConfig(
        model="gpt-5.3-codex",
        api_key="test-key",
        api_url="https://llm.test/v1",
    )


@pytest.fixture
def review_config():
    return ReviewConfig(
        auto_review_authors=["jan.username"],
        review_prompt_template="prompts/review.txt",
    )


def _mock_completion(content: str, prompt_tokens: int = 100, completion_tokens: int = 50):
    """Build a mock chat/completions response object."""
    usage = MagicMock()
    usage.prompt_tokens = prompt_tokens
    usage.completion_tokens = completion_tokens

    message = MagicMock()
    message.content = content
    choice = MagicMock()
    choice.message = message

    response = MagicMock()
    response.choices = [choice]
    response.usage = usage
    return response


def _user_text_from_chat_call(mock_create) -> str:
    """Extract the user text from a mocked `chat.completions.create` call."""
    call_args = mock_create.call_args
    user_block = call_args.kwargs["messages"][1]
    return user_block["content"]


class TestParseReviewResponse:
    def test_valid_json_array(self):
        content = json.dumps([
            {"file": "src/main.py", "line": 10, "severity": "issue", "comment": "Bug here"}
        ])
        findings, requirements, summary, parse_failed = _parse_review_response(content)
        assert len(findings) == 1
        assert findings[0].file == "src/main.py"
        assert findings[0].line == 10
        assert findings[0].severity == "issue"
        assert requirements == []
        assert summary == ReviewSummary()
        assert parse_failed is False

    def test_empty_array(self):
        findings, requirements, summary, parse_failed = _parse_review_response("[]")
        assert findings == []
        assert requirements == []
        assert summary == ReviewSummary()
        assert parse_failed is False

    def test_wrapped_in_code_fence(self):
        content = "```json\n[{\"file\": \"a.py\", \"line\": 1, \"severity\": \"suggestion\", \"comment\": \"test\"}]\n```"
        findings, _requirements, _, _ = _parse_review_response(content)
        assert len(findings) == 1

    def test_invalid_json(self):
        findings, requirements, summary, parse_failed = _parse_review_response("not json at all")
        assert findings == []
        # None signals extraction failure (vs [] which means "successfully
        # parsed, no requirements") so the reviewer can render the reason.
        assert requirements is None
        assert summary == ReviewSummary()
        # parse_failed distinguishes a refused/unparseable response from a
        # legitimately empty review so the reviewer posts a failure notice.
        assert parse_failed is True

    def test_refusal_text_signals_parse_failure(self):
        findings, requirements, summary, parse_failed = _parse_review_response(
            "I'm sorry, but I cannot assist with that request."
        )
        assert findings == []
        assert requirements is None
        assert summary == ReviewSummary()
        assert parse_failed is True

    def test_empty_output_signals_parse_failure(self):
        findings, requirements, summary, parse_failed = _parse_review_response("")
        assert findings == []
        assert requirements is None
        assert summary == ReviewSummary()
        assert parse_failed is True

    def test_top_level_string_signals_extraction_failure(self):
        findings, requirements, summary, parse_failed = _parse_review_response('"oops"')
        assert findings == []
        assert requirements is None
        assert summary == ReviewSummary()
        assert parse_failed is True

    def test_not_an_array(self):
        findings, _requirements, _, _ = _parse_review_response('{"file": "a.py"}')
        assert findings == []

    def test_malformed_item_skipped(self):
        content = json.dumps([
            {"file": "a.py", "line": 1, "severity": "issue", "comment": "good"},
            {"bad": "item"},
        ])
        findings, _requirements, _, _ = _parse_review_response(content)
        assert len(findings) == 1

    def test_unknown_severity_rejected(self):
        # Belt-and-braces: even if the LLM regresses past the JSON-schema enum,
        # the Pydantic Literal blocks unexpected severities so downstream stats
        # and label rendering can trust the value.
        content = json.dumps([
            {"file": "a.py", "line": 1, "severity": "critical", "comment": "stale"},
            {"file": "b.py", "line": 2, "severity": "issue", "comment": "ok"},
        ])
        findings, _requirements, _, _ = _parse_review_response(content)
        assert len(findings) == 1
        assert findings[0].file == "b.py"

    def test_object_with_findings_and_compliance_requirements(self):
        content = json.dumps({
            "findings": [
                {"file": "a.py", "line": 1, "severity": "suggestion", "comment": "test"}
            ],
            "compliance_requirements": [
                {"requirement": "Implement auth filter", "met": True},
                {"requirement": "Write tests", "met": False},
            ],
        })
        findings, requirements, _, _ = _parse_review_response(content)
        assert len(findings) == 1
        assert requirements is not None
        assert len(requirements) == 2
        assert requirements[0] == {"requirement": "Implement auth filter", "met": True}
        assert requirements[1] == {"requirement": "Write tests", "met": False}

    def test_object_with_malformed_compliance_requirements(self):
        content = json.dumps({
            "findings": [],
            "compliance_requirements": [
                {"requirement": "Valid", "met": True},
                {"bad": "item"},
                {"requirement": "Missing met field"},
                {"requirement": "Wrong met type", "met": "yes"},
            ],
        })
        findings, requirements, _, _ = _parse_review_response(content)
        assert findings == []
        assert requirements is not None
        assert len(requirements) == 1
        assert requirements[0]["requirement"] == "Valid"

    def test_summary_fields_parsed(self):
        content = json.dumps({
            "findings": [],
            "overview": "Adds retry logic to the webhook client.",
            "strengths": ["Test added for the new code path"],
            "security_performance": "None notable.",
            "test_coverage": "Existing webhook tests updated; new test for retry path.",
            "verdict": {
                "decision": "approve",
                "rationale": "Retry path is bounded and covered by a test.",
            },
        })
        _findings, _requirements, summary, _ = _parse_review_response(content)
        assert summary.overview == "Adds retry logic to the webhook client."
        assert summary.strengths == ["Test added for the new code path"]
        assert summary.security_performance == "None notable."
        assert summary.test_coverage.startswith("Existing webhook tests")
        assert summary.verdict_decision == "approve"
        assert summary.verdict_rationale == "Retry path is bounded and covered by a test."

    def test_summary_strengths_filters_non_strings(self):
        content = json.dumps({
            "findings": [],
            "overview": "x",
            "strengths": ["Valid bullet", 123, None, "Another bullet"],
            "security_performance": "None notable.",
            "test_coverage": "x",
            "verdict": {"decision": "approve", "rationale": "x"},
        })
        _, _, summary, _ = _parse_review_response(content)
        assert summary.strengths == ["Valid bullet", "Another bullet"]

    def test_summary_invalid_verdict_decision_falls_back(self):
        content = json.dumps({
            "findings": [],
            "overview": "x",
            "strengths": [],
            "security_performance": "None notable.",
            "test_coverage": "x",
            "verdict": {"decision": "looks_good_to_me", "rationale": "x"},
        })
        _, _, summary, _ = _parse_review_response(content)
        # Unknown decisions are ignored and the dataclass default ("approve") sticks.
        assert summary.verdict_decision == "approve"

    def test_vacuous_suggestion_finding_dropped(self):
        content = json.dumps([
            {
                "file": "a.py",
                "line": 10,
                "severity": "issue",
                "comment": "You finally provide useful context for error analysis.",
                "suggestion": "No fix needed—this code is actually correct for once.",
            }
        ])
        findings, _, _, _ = _parse_review_response(content)
        assert findings == []

    def test_legitimate_finding_with_real_suggestion_preserved(self):
        content = json.dumps([
            {
                "file": "a.py",
                "line": 10,
                "severity": "issue",
                "comment": "Null pointer dereference",
                "suggestion": "if user is None:\n    return None\nreturn user.name",
            }
        ])
        findings, _, _, _ = _parse_review_response(content)
        assert len(findings) == 1
        assert findings[0].suggestion == "if user is None:\n    return None\nreturn user.name"

    def test_summary_fields_missing_default_to_empty(self):
        content = json.dumps({
            "findings": [
                {"file": "a.py", "line": 1, "severity": "suggestion", "comment": "test"}
            ],
        })
        _, _, summary, _ = _parse_review_response(content)
        assert summary == ReviewSummary()

    def test_overview_empty_logs_warning(self, caplog):
        import logging
        content = json.dumps({
            "findings": [],
            "overview": "",
            "strengths": [],
            "security_performance": "None notable.",
            "test_coverage": "x",
            "verdict": {"decision": "approve", "rationale": "x"},
        })
        with caplog.at_level(logging.WARNING, logger="app.llm_client"):
            _, _, summary, _ = _parse_review_response(content)
        assert summary.overview == ""
        assert any("overview empty after parse" in msg for msg in caplog.messages)


class TestMergeReviewSummaries:
    def test_first_non_empty_wins_for_prose_fields(self):
        # Note: verdict_rationale is paired with the winning decision and is
        # covered separately by test_verdict_rationale_paired_with_winning_decision.
        a = ReviewSummary(overview="", security_performance="None notable.",
                          test_coverage="")
        b = ReviewSummary(overview="From B", security_performance="Adds a hot loop",
                          test_coverage="Tests added")
        merged = _merge_review_summaries([a, b])
        assert merged.overview == "From B"
        assert merged.security_performance == "None notable."  # first non-empty wins
        assert merged.test_coverage == "Tests added"

    def test_strengths_concatenated_and_deduped(self):
        a = ReviewSummary(strengths=["Clean rename", "Good test"])
        b = ReviewSummary(strengths=["clean rename", "Solid docs"])
        merged = _merge_review_summaries([a, b])
        assert merged.strengths == ["Clean rename", "Good test", "Solid docs"]

    def test_strengths_capped(self):
        a = ReviewSummary(strengths=[f"item {i}" for i in range(10)])
        merged = _merge_review_summaries([a])
        assert len(merged.strengths) == 4

    def test_verdict_rolls_up_to_worst(self):
        a = ReviewSummary(verdict_decision="approve")
        b = ReviewSummary(verdict_decision="request_changes")
        c = ReviewSummary(verdict_decision="approve_with_followups")
        merged = _merge_review_summaries([a, b, c])
        assert merged.verdict_decision == "request_changes"

    def test_verdict_rationale_paired_with_winning_decision(self):
        # Regression: rationale must come from the chunk that owns the worst
        # decision — not from the first non-empty chunk — so a clean chunk's
        # "ship it" rationale doesn't get attached to a request_changes verdict.
        clean = ReviewSummary(
            verdict_decision="approve",
            verdict_rationale="Rename is consistent across call sites.",
        )
        blocking = ReviewSummary(
            verdict_decision="request_changes",
            verdict_rationale="SQL injection in /users handler.",
        )
        merged = _merge_review_summaries([clean, blocking])
        assert merged.verdict_decision == "request_changes"
        assert merged.verdict_rationale == "SQL injection in /users handler."

    def test_empty_parts_returns_default(self):
        merged = _merge_review_summaries([])
        assert merged == ReviewSummary()


class TestParseMentionResponse:
    def test_plain_text_fallback(self):
        assert _parse_mention_response("Hello world") == "Hello world"

    def test_envelope_answer_only(self):
        content = json.dumps({"answer": "The function caches results.", "refs": []})
        assert _parse_mention_response(content) == "The function caches results."

    def test_envelope_with_refs(self):
        content = json.dumps({
            "answer": "It batches queries.",
            "refs": [
                {"file": "app/users.py", "line": 47},
                {"file": "app/util.py", "line": 3},
            ],
        })
        result = _parse_mention_response(content)
        assert result.startswith("It batches queries.")
        assert "**References:**" in result
        assert "`app/users.py`:47" in result
        assert "`app/util.py`:3" in result

    def test_envelope_stripped_of_code_fence(self):
        content = "```json\n" + json.dumps({"answer": "Hi"}) + "\n```"
        assert _parse_mention_response(content) == "Hi"

    def test_malformed_envelope_returns_raw(self):
        # JSON but not the expected shape — fall back to the raw string.
        raw = '{"unexpected": true}'
        assert _parse_mention_response(raw) == raw

    def test_empty(self):
        assert _parse_mention_response("") == ""

    def test_refs_ignore_malformed_entries(self):
        content = json.dumps({
            "answer": "See here.",
            "refs": [
                {"file": "a.py", "line": 1},
                {"file": "b.py"},            # no line — still keep file
                {"line": 5},                 # no file — drop
                "not a dict",                # drop
            ],
        })
        result = _parse_mention_response(content)
        assert "`a.py`:1" in result
        assert "`b.py`" in result
        assert "`b.py`:" not in result  # no line for b.py
        assert result.count("- `") == 2


class TestIsReviewableDiff:
    def test_source_files_are_reviewable(self):
        for ext in [".py", ".java", ".ts", ".tsx", ".js", ".go", ".rs", ".rb", ".c", ".cpp", ".h"]:
            diff = f"diff --git a/src/main{ext} b/src/main{ext}\n+code\n"
            assert is_reviewable_diff(diff), f"{ext} should be reviewable"

    def test_binary_extensions_skipped(self):
        for ext in [".png", ".jpg", ".pdf", ".zip", ".exe", ".jar", ".pyc"]:
            diff = f"diff --git a/assets/file{ext} b/assets/file{ext}\n+something\n"
            assert not is_reviewable_diff(diff), f"{ext} should be skipped"

    def test_json_and_lock_files_skipped(self):
        for name in ["package.json", "yarn.lock", "Pipfile.lock"]:
            diff = f"diff --git a/{name} b/{name}\n+content\n"
            assert not is_reviewable_diff(diff), f"{name} should be skipped"

    def test_diagram_sources_skipped(self):
        names = [
            "docs/architecture.puml", "sequence.puml",
            "arch.plantuml", "arch.pu", "shared.iuml",
            "docs/flow.mmd", "diagram.drawio", "diagram.dio",
            "board.excalidraw",
        ]
        for name in names:
            diff = f"diff --git a/{name} b/{name}\n+@startuml\n"
            assert not is_reviewable_diff(diff), f"{name} should be skipped"

    def test_generated_python_sources_skipped(self):
        for name in ["api_pb2.py", "gen/api_pb2_grpc.py", "src/noergler.egg-info/PKG-INFO"]:
            diff = f"diff --git a/{name} b/{name}\n+generated\n"
            assert not is_reviewable_diff(diff), f"{name} should be skipped"

    def test_handwritten_python_still_reviewable(self):
        # Neighbours of the generated-suffix rule that must stay reviewable.
        for name in ["app/pb2_helpers.py", "app/grpc_server.py", "app/egg.py"]:
            diff = f"diff --git a/{name} b/{name}\n+code\n"
            assert is_reviewable_diff(diff), f"{name} should be reviewable"

    def test_alembic_migrations_still_reviewable(self):
        # Autogenerated, but the highest-risk generated code in the repo —
        # dropped columns and non-nullable adds must reach the reviewer.
        diff = "diff --git a/alembic/versions/a1b2_add_col.py b/alembic/versions/a1b2_add_col.py\n+op.drop_column()\n"
        assert is_reviewable_diff(diff)

    def test_generated_output_skipped(self):
        for name in ["__snapshots__/App.test.js.snap", "tests/__snapshots__/x.ambr", "locale/de.po", "locale/de.mo", "msgs.xlf"]:
            diff = f"diff --git a/{name} b/{name}\n+generated\n"
            assert not is_reviewable_diff(diff), f"{name} should be skipped"

    def test_extensionless_lockfiles_skipped(self):
        for name in ["go.sum", "pnpm-lock.yaml", "vendor/go.sum"]:
            diff = f"diff --git a/{name} b/{name}\n+dep\n"
            assert not is_reviewable_diff(diff), f"{name} should be skipped"

    def test_similar_extensions_still_reviewable(self):
        # Guard against over-broad substring matching on the short extensions
        # above (.pu/.mo/.dio) — these must survive the header filter.
        for name in ["src/video.movie.py", "app/studio.py", "pkg/sum.go", "config.yaml"]:
            diff = f"diff --git a/{name} b/{name}\n+code\n"
            assert is_reviewable_diff(diff), f"{name} should be reviewable"

    def test_minified_files_skipped(self):
        for name in ["bundle.min.js", "styles.min.css"]:
            diff = f"diff --git a/dist/{name} b/dist/{name}\n+minified\n"
            assert not is_reviewable_diff(diff), f"{name} should be skipped"

    def test_binary_files_differ_marker_skipped(self):
        diff = "diff --git a/image.dat b/image.dat\nBinary files /dev/null and b/image.dat differ\n"
        assert not is_reviewable_diff(diff)

    def test_config_files_are_reviewable(self):
        for name in ["Dockerfile", "Makefile", "config.yaml", "deploy.yml", "setup.cfg"]:
            diff = f"diff --git a/{name} b/{name}\n+content\n"
            assert is_reviewable_diff(diff), f"{name} should be reviewable"

    def test_hidden_files_skipped(self):
        for name in [".gitignore", ".env", ".env.example", ".dockerignore", ".editorconfig"]:
            diff = f"diff --git a/{name} b/{name}\n+content\n"
            assert not is_reviewable_diff(diff), f"{name} should be skipped"
        # Hidden file in a subdir should also be skipped
        diff = "diff --git a/src/.gitkeep b/src/.gitkeep\n+content\n"
        assert not is_reviewable_diff(diff)

    def test_bitbucket_src_dst_format_reviewable(self):
        diff = "diff --git src://src/main/java/Foo.java dst://src/main/java/Foo.java\n+code\n"
        assert is_reviewable_diff(diff)

    def test_bitbucket_src_dst_format_skipped(self):
        diff = "diff --git src://assets/image.png dst://assets/image.png\n+binary\n"
        assert not is_reviewable_diff(diff)

    def test_build_config_extensions_skipped(self):
        for ext in [".bat", ".cmd", ".properties"]:
            diff = f"diff --git a/some/file{ext} b/some/file{ext}\n+content\n"
            assert not is_reviewable_diff(diff), f"{ext} should be skipped"

    def test_xml_is_reviewable(self):
        diff = "diff --git a/db/changelog.xml b/db/changelog.xml\n+<changeSet/>\n"
        assert is_reviewable_diff(diff)

    def test_skip_files_by_name(self):
        for name in ["gradlew", "mvnw"]:
            diff = f"diff --git a/{name} b/{name}\n+content\n"
            assert not is_reviewable_diff(diff), f"{name} should be skipped"

    def test_skip_dirs(self):
        for dir_name in ["target", ".idea", "node_modules", "build", ".gradle"]:
            diff = f"diff --git a/{dir_name}/File.java b/{dir_name}/File.java\n+code\n"
            assert not is_reviewable_diff(diff), f"files in {dir_name}/ should be skipped"

    def test_skip_hidden_dirs(self):
        diff = "diff --git a/.hidden/secret.py b/.hidden/secret.py\n+code\n"
        assert not is_reviewable_diff(diff)

    def test_normal_source_files_pass(self):
        diff = "diff --git a/src/main/App.java b/src/main/App.java\n+code\n"
        assert is_reviewable_diff(diff)

    def test_bitbucket_quoted_octal_escaped_png_skipped(self):
        diff = (
            'diff --git '
            '"src://apps/schadenmeldung-e2e/tests/e2e/__screenshots__/uvg/einsatzbetrieb.spec.ts/'
            'einsatzbetrieb-chooser-\\303\\274berpr\\303\\274fung-arbeitgeber-ueberpruefung.png" '
            '"dst://apps/schadenmeldung-e2e/tests/e2e/__screenshots__/uvg/einsatzbetrieb.spec.ts/'
            'einsatzbetrieb-chooser-\\303\\274berpr\\303\\274fung-arbeitgeber-ueberpruefung.png"\n'
        )
        assert is_reviewable_diff(diff) is False

    def test_quoted_source_file_still_reviewable(self):
        diff = (
            'diff --git '
            '"src://src/main/java/F\\303\\266o.java" '
            '"dst://src/main/java/F\\303\\266o.java"\n+code\n'
        )
        assert is_reviewable_diff(diff) is True

    def test_extension_in_middle_not_skipped(self):
        diff = "diff --git a/src/foo.png.java b/src/foo.png.java\n+code\n"
        assert is_reviewable_diff(diff) is True


class TestFileReviewData:
    def test_dataclass_fields(self):
        f = FileReviewData(path="src/main.py", diff="+hello\n", content="hello\n")
        assert f.path == "src/main.py"
        assert f.diff == "+hello\n"
        assert f.content == "hello\n"

    def test_content_defaults_to_none(self):
        f = FileReviewData(path="deleted.py", diff="-goodbye\n")
        assert f.content is None


class TestFormatFileEntry:
    def test_with_content(self):
        f = FileReviewData(path="src/main.py", diff="+hello\n", content="hello\nworld\n")
        result = format_file_entry(f)
        assert "## File: src/main.py" in result
        assert "### Full file content (new version):" in result
        assert "```py" in result
        assert "hello\nworld\n" in result
        assert "### Changes (diff: lines with `-` are REMOVED, lines with `+` are ADDED):" in result
        assert "```diff" in result

    def test_deleted_file(self):
        diff = "--- a/old.py\n+++ /dev/null\n-removed\n"
        f = FileReviewData(path="old.py", diff=diff, content=None)
        result = format_file_entry(f)
        assert "## File: old.py" in result
        assert "_(file deleted)_" in result
        assert "### Changes (diff: lines with `-` are REMOVED, lines with `+` are ADDED):" in result

    def test_content_omitted(self):
        f = FileReviewData(path="big.py", diff="+added\n", content=None)
        result = format_file_entry(f)
        assert "## File: big.py" in result
        assert "_(full file content omitted — review diff only)_" in result


class TestExtractPath:
    def test_extracts_b_path(self):
        diff = "diff --git a/old/path.py b/new/path.py\n+code\n"
        assert extract_path(diff) == "new/path.py"

    def test_returns_none_for_invalid(self):
        assert extract_path("not a diff\n") is None

    def test_handles_crlf_line_endings(self):
        diff = "diff --git a/old/path.py b/new/path.py\r\n+code\r\n"
        assert extract_path(diff) == "new/path.py"

    def test_diff_header_not_on_first_line(self):
        diff = "some preamble\ndiff --git a/old/path.py b/new/path.py\n+code\n"
        assert extract_path(diff) == "new/path.py"

    def test_fallback_via_plus_plus_plus_header(self):
        diff = "--- a/src/main.py\n+++ b/src/main.py\n@@ -1,3 +1,3 @@\n-old\n+new\n"
        assert extract_path(diff) == "src/main.py"

    def test_fallback_with_crlf(self):
        diff = "--- a/src/main.py\r\n+++ b/src/main.py\r\n@@ -1,3 +1,3 @@\r\n"
        assert extract_path(diff) == "src/main.py"

    def test_bitbucket_src_dst_format(self):
        diff = "diff --git src://build.gradle dst://build.gradle\n+code\n"
        assert extract_path(diff) == "build.gradle"

    def test_bitbucket_src_dst_nested_path(self):
        diff = "diff --git src://src/main/java/com/example/Foo.java dst://src/main/java/com/example/Foo.java\n"
        assert extract_path(diff) == "src/main/java/com/example/Foo.java"

    def test_bitbucket_dst_fallback(self):
        diff = "--- src://main.py\n+++ dst://main.py\n@@ -1,3 +1,3 @@\n"
        assert extract_path(diff) == "main.py"


class TestIsDeleted:
    def test_deleted_file(self):
        diff = "diff --git a/file.py b/file.py\ndeleted file mode 100644\n--- a/file.py\n+++ /dev/null\n"
        assert is_deleted(diff) is True

    def test_normal_file(self):
        diff = "diff --git a/file.py b/file.py\n--- a/file.py\n+++ b/file.py\n"
        assert is_deleted(diff) is False


class TestSplitByFile:
    def test_single_file(self):
        diff = "diff --git a/a.py b/a.py\n+hello\n"
        parts = split_by_file(diff)
        assert len(parts) == 1

    def test_multiple_files(self):
        diff = "diff --git a/a.py b/a.py\n+hello\ndiff --git a/b.py b/b.py\n+world\n"
        parts = split_by_file(diff)
        assert len(parts) == 2
        assert "a.py" in parts[0]
        assert "b.py" in parts[1]


class TestGroupFilesByTokenBudget:
    def test_single_file_fits(self):
        files = [FileReviewData(path="file.py", diff="+hello\n", content="hello\n")]
        template = "Review:\n{files}"
        groups, skipped = _group_files_by_token_budget(files, 80000, template)
        assert len(groups) == 1
        assert len(groups[0]) == 1
        assert groups[0][0].path == "file.py"
        assert skipped == []

    def test_multiple_files_split_by_tokens(self):
        files = [
            FileReviewData(path="a.py", diff="+line\n" * 50, content="line\n" * 50),
            FileReviewData(path="b.py", diff="+line\n" * 50, content="line\n" * 50),
        ]
        template = "Review:\n{files}"
        groups, skipped = _group_files_by_token_budget(files, 300, template)
        assert len(groups) >= 2
        assert skipped == []

    def test_oversized_single_file_skipped(self):
        files = [FileReviewData(path="huge.py", diff="+x = 1\n" * 5000, content="x = 1\n" * 5000)]
        template = "Review:\n{files}"
        groups, skipped = _group_files_by_token_budget(files, 200, template)
        assert groups == []
        assert skipped == ["huge.py"]

    def test_oversized_file_skipped_but_small_kept(self):
        files = [
            FileReviewData(path="small.py", diff="+ok\n", content="ok\n"),
            FileReviewData(path="huge.py", diff="+x = 1\n" * 5000, content="x = 1\n" * 5000),
        ]
        template = "Review:\n{files}"
        groups, skipped = _group_files_by_token_budget(files, 200, template)
        paths = [f.path for g in groups for f in g]
        assert "small.py" in paths
        assert "huge.py" not in paths
        assert skipped == ["huge.py"]

    def test_oversized_content_downgraded_to_diff_only(self):
        # Full content blows the budget, but the diff alone fits — the file must
        # be reviewed diff-only, not dropped from the review entirely.
        files = [FileReviewData(
            path="big.py",
            diff="+x = 1\n",                # tiny diff
            content="x = 1\n" * 5000,       # huge full content
        )]
        template = "Review:\n{files}"
        groups, skipped = _group_files_by_token_budget(files, 200, template)
        paths = [f.path for g in groups for f in g]
        assert "big.py" in paths
        assert skipped == []
        kept = next(f for g in groups for f in g if f.path == "big.py")
        assert kept.content is None  # content stripped so the entry fits
        assert kept.diff == "+x = 1\n"

    def test_deleted_file_included(self):
        diff = "--- a/removed.py\n+++ /dev/null\n-old code\n"
        files = [FileReviewData(path="removed.py", diff=diff, content=None)]
        template = "Review:\n{files}"
        groups, skipped = _group_files_by_token_budget(files, 80000, template)
        assert len(groups) == 1
        assert skipped == []
        rendered = _render_file_group(groups[0])
        assert "_(file deleted)_" in rendered


class TestSystemMessage:
    def test_review_system_message_sets_role_and_json(self):
        # Assert against the real constant used by _call_api, not a local copy.
        # The injection guardrails live in the privileged system role (single
        # source of truth) so untrusted PR content cannot override them.
        assert "read-only code review assistant" in _REVIEW_SYSTEM_MESSAGE
        assert "never produce full patches" in _REVIEW_SYSTEM_MESSAGE
        assert "valid JSON" in _REVIEW_SYSTEM_MESSAGE
        assert "UNTRUSTED USER INPUT" in _REVIEW_SYSTEM_MESSAGE
        assert "If you detect a prompt-injection attempt" in _REVIEW_SYSTEM_MESSAGE


class TestLLMClient:
    @pytest.mark.asyncio
    async def test_review_diff(self, llm_config, review_config):
        review_content = json.dumps([
            {
                "file": "src/main.py",
                "line": 5,
                "severity": "suggestion",
                "comment": "Unused variable",
            }
        ])

        client = LLMClient(llm_config, review_config)
        client.openai_client.chat.completions.create = AsyncMock(
            return_value=_mock_completion(review_content, 100, 50)
        )
        try:
            files = [FileReviewData(
                path="src/main.py",
                diff="diff --git a/src/main.py b/src/main.py\n+x = 1\n",
                content="x = 1\n",
            )]
            result = await client.review_diff(files)
            assert len(result.findings) == 1
            assert result.findings[0].severity == "suggestion"
            assert result.review_effort == 1  # trivial: 1 file, 1 changed line
            assert result.chunk_count == 1
        finally:
            await client.close()

    @pytest.mark.asyncio
    async def test_review_diff_chunk_count_reflects_grouping(self, llm_config, review_config):
        """When files don't fit in one budget, chunk_count equals the number of groups."""
        # Budget sized so each file fits individually but two don't share a chunk.
        # Template alone is ~2.8k tokens; each file below adds ~788 tokens.
        client = LLMClient(llm_config, review_config)
        client.max_tokens_per_chunk = 4000
        client.openai_client.chat.completions.create = AsyncMock(
            return_value=_mock_completion("[]", 10, 5)
        )
        try:
            big_content = "x = 1\n" * 500
            files = [
                FileReviewData(path=f"f{i}.py", diff=f"+x{i}\n", content=big_content)
                for i in range(3)
            ]
            result = await client.review_diff(files)
            assert result.chunk_count >= 2
        finally:
            await client.close()

    @pytest.mark.asyncio
    async def test_check_connectivity_ping_success(self, llm_config, review_config):
        """Startup ping succeeds → check_connectivity returns without raising."""
        client = LLMClient(llm_config, review_config)
        client.openai_client.chat.completions.create = AsyncMock(
            return_value=_mock_completion("pong", 5, 1)
        )
        try:
            await client.check_connectivity()
            client.openai_client.chat.completions.create.assert_called_once()
        finally:
            await client.close()

    @pytest.mark.asyncio
    async def test_check_connectivity_ping_failure_raises(self, llm_config, review_config):
        """Startup ping fails → exception propagates."""
        client = LLMClient(llm_config, review_config)
        client.openai_client.chat.completions.create = AsyncMock(
            side_effect=openai.APIStatusError(
                "No access to model",
                response=httpx.Response(403, request=httpx.Request("POST", "https://x"), text="forbidden"),
                body=None,
            )
        )
        try:
            with pytest.raises(openai.APIStatusError):
                await client.check_connectivity()
        finally:
            await client.close()

    @pytest.mark.asyncio
    async def test_chat_passes_review_response_schema(self, llm_config, review_config):
        """Review calls bind a strict json_schema via `response_format`."""
        mock_create = AsyncMock(return_value=_mock_completion("[]", 10, 5))
        client = LLMClient(llm_config, review_config)
        client.openai_client.chat.completions.create = mock_create
        try:
            files = [FileReviewData(path="a.py", diff="+x\n", content="x\n")]
            await client.review_diff(files)
            call_kwargs = mock_create.call_args.kwargs
            assert call_kwargs["response_format"]["type"] == "json_schema"
            fmt = call_kwargs["response_format"]["json_schema"]
            assert fmt["name"] == "review_response"
            assert fmt["strict"] is True
            assert "overview" in fmt["schema"]["properties"]
            assert "strengths" in fmt["schema"]["properties"]
            assert "security_performance" in fmt["schema"]["properties"]
            assert "test_coverage" in fmt["schema"]["properties"]
            assert "verdict" in fmt["schema"]["properties"]
            assert "findings" in fmt["schema"]["properties"]
        finally:
            await client.close()


class TestReasoningEffort:
    @pytest.mark.asyncio
    async def test_reasoning_omitted_when_explicitly_none(self, review_config):
        cfg = LLMConfig(
            model="gpt-5.3-codex",
            api_key="test-key",
            api_url="https://llm.test/v1",
            reasoning_effort=None,
        )
        mock_create = AsyncMock(return_value=_mock_completion("[]", 10, 5))
        client = LLMClient(cfg, review_config)
        client.openai_client.chat.completions.create = mock_create
        try:
            files = [FileReviewData(path="a.py", diff="+x\n", content="x\n")]
            await client.review_diff(files)
            assert "reasoning_effort" not in mock_create.call_args.kwargs
        finally:
            await client.close()

    @pytest.mark.asyncio
    async def test_reasoning_defaults_to_high(self, llm_config, review_config):
        assert llm_config.reasoning_effort == "high"
        mock_create = AsyncMock(return_value=_mock_completion("[]", 10, 5))
        client = LLMClient(llm_config, review_config)
        client.openai_client.chat.completions.create = mock_create
        try:
            files = [FileReviewData(path="a.py", diff="+x\n", content="x\n")]
            await client.review_diff(files)
            assert mock_create.call_args.kwargs["reasoning_effort"] == "high"
        finally:
            await client.close()

    @pytest.mark.asyncio
    async def test_reasoning_passed_when_set(self, review_config):
        cfg = LLMConfig(
            model="gpt-5.3-codex",
            api_key="test-key",
            api_url="https://llm.test/v1",
            reasoning_effort="high",
        )
        mock_create = AsyncMock(return_value=_mock_completion("[]", 10, 5))
        client = LLMClient(cfg, review_config)
        client.openai_client.chat.completions.create = mock_create
        try:
            files = [FileReviewData(path="a.py", diff="+x\n", content="x\n")]
            await client.review_diff(files)
            assert mock_create.call_args.kwargs["reasoning_effort"] == "high"
        finally:
            await client.close()

    @pytest.mark.asyncio
    async def test_reasoning_passed_on_connectivity_ping(self, review_config):
        cfg = LLMConfig(
            model="gpt-5.3-codex",
            api_key="test-key",
            api_url="https://llm.test/v1",
            reasoning_effort="low",
        )
        mock_create = AsyncMock(return_value=_mock_completion("ok", 5, 1))
        client = LLMClient(cfg, review_config)
        client.openai_client.chat.completions.create = mock_create
        try:
            await client.check_connectivity()
            assert mock_create.call_args.kwargs["reasoning_effort"] == "low"
        finally:
            await client.close()


class TestRepoInstructionsInReviewPrompt:
    @pytest.mark.asyncio
    async def test_repo_instructions_replaced_in_review_prompt(self, llm_config, review_config):
        """Verify {repo_instructions} placeholder is replaced, not left as literal text."""
        mock_create = AsyncMock(return_value=_mock_completion("[]", 100, 10))
        client = LLMClient(llm_config, review_config)
        client.openai_client.chat.completions.create = mock_create
        try:
            files = [FileReviewData(path="a.py", diff="+x\n", content="x\n")]
            await client.review_diff(files, repo_instructions="Use 4-space indent")

            prompt = _user_text_from_chat_call(mock_create)
            assert "{repo_instructions}" not in prompt
            assert "Use 4-space indent" in prompt
        finally:
            await client.close()

    @pytest.mark.asyncio
    async def test_empty_repo_instructions_clears_placeholder(self, llm_config, review_config):
        """When no repo instructions, placeholder is replaced with empty string."""
        mock_create = AsyncMock(return_value=_mock_completion("[]", 100, 10))
        client = LLMClient(llm_config, review_config)
        client.openai_client.chat.completions.create = mock_create
        try:
            files = [FileReviewData(path="a.py", diff="+x\n", content="x\n")]
            await client.review_diff(files, repo_instructions="")

            prompt = _user_text_from_chat_call(mock_create)
            assert "{repo_instructions}" not in prompt
        finally:
            await client.close()


class TestCumulativeDiffAndPostedFindingsInPrompt:
    @pytest.mark.asyncio
    async def test_cumulative_pr_diff_rendered_when_provided(self, llm_config, review_config):
        mock_create = AsyncMock(return_value=_mock_completion("[]", 100, 10))
        client = LLMClient(llm_config, review_config)
        client.openai_client.chat.completions.create = mock_create
        try:
            files = [FileReviewData(path="a.py", diff="+x\n", content="x\n")]
            await client.review_diff(
                files,
                cumulative_pr_diff="diff --git a/Foo.java b/Foo.java\n-old\n+new\n",
            )
            prompt = _user_text_from_chat_call(mock_create)
            assert "{cumulative_pr_diff}" not in prompt
            assert "Cumulative PR diff" in prompt
            assert "diff --git a/Foo.java b/Foo.java" in prompt
        finally:
            await client.close()

    @pytest.mark.asyncio
    async def test_focused_files_render_before_context_blocks(self, llm_config, review_config):
        # Per AGENTS.md: {files} must stay BEFORE {cumulative_pr_diff} and
        # {previously_posted_findings}. Files are the stable bytes across
        # re-reviews and must sit in the prefix-cache; the cumulative
        # diff/findings grow on every push and belong in the volatile suffix.
        mock_create = AsyncMock(return_value=_mock_completion("[]", 100, 10))
        client = LLMClient(llm_config, review_config)
        client.openai_client.chat.completions.create = mock_create
        try:
            files = [FileReviewData(path="subject.py", diff="+focusmarker\n", content="focusmarker\n")]
            await client.review_diff(
                files,
                cumulative_pr_diff="diff --git a/Ctx.java b/Ctx.java\n-old\n+new\n",
                previously_posted_findings=[
                    {"file_path": "src/Foo.java", "line_number": 11,
                     "severity": "issue", "comment_text": "prior finding marker"},
                ],
            )
            prompt = _user_text_from_chat_call(mock_create)
            assert prompt.index("focusmarker") < prompt.index("Cumulative PR diff")
            assert prompt.index("focusmarker") < prompt.index("Already-posted findings")
        finally:
            await client.close()

    @pytest.mark.asyncio
    async def test_cumulative_pr_diff_placeholder_cleared_when_empty(self, llm_config, review_config):
        mock_create = AsyncMock(return_value=_mock_completion("[]", 100, 10))
        client = LLMClient(llm_config, review_config)
        client.openai_client.chat.completions.create = mock_create
        try:
            files = [FileReviewData(path="a.py", diff="+x\n", content="x\n")]
            await client.review_diff(files)
            prompt = _user_text_from_chat_call(mock_create)
            assert "{cumulative_pr_diff}" not in prompt
            assert "Cumulative PR diff" not in prompt
        finally:
            await client.close()

    @pytest.mark.asyncio
    async def test_previously_posted_findings_rendered_when_provided(self, llm_config, review_config):
        mock_create = AsyncMock(return_value=_mock_completion("[]", 100, 10))
        client = LLMClient(llm_config, review_config)
        client.openai_client.chat.completions.create = mock_create
        try:
            files = [FileReviewData(path="a.py", diff="+x\n", content="x\n")]
            findings = [
                {"file_path": "src/Foo.java", "line_number": 11, "severity": "issue",
                 "comment_text": "PropertyReferenceException risk"},
            ]
            await client.review_diff(files, previously_posted_findings=findings)
            prompt = _user_text_from_chat_call(mock_create)
            assert "{previously_posted_findings}" not in prompt
            assert "Already-posted findings" in prompt
            assert "src/Foo.java:11" in prompt
            assert "PropertyReferenceException" in prompt
        finally:
            await client.close()

    @pytest.mark.asyncio
    async def test_previously_posted_findings_placeholder_cleared_when_empty(self, llm_config, review_config):
        mock_create = AsyncMock(return_value=_mock_completion("[]", 100, 10))
        client = LLMClient(llm_config, review_config)
        client.openai_client.chat.completions.create = mock_create
        try:
            files = [FileReviewData(path="a.py", diff="+x\n", content="x\n")]
            await client.review_diff(files, previously_posted_findings=None)
            prompt = _user_text_from_chat_call(mock_create)
            assert "{previously_posted_findings}" not in prompt
            assert "Already-posted findings" not in prompt
        finally:
            await client.close()


class TestComplianceInstructions:
    @pytest.mark.asyncio
    async def test_compliance_instructions_included_when_enabled_with_ticket(self, llm_config, review_config):
        mock_create = AsyncMock(return_value=_mock_completion("[]", 100, 10))
        client = LLMClient(llm_config, review_config)
        client.openai_client.chat.completions.create = mock_create
        try:
            files = [FileReviewData(path="a.py", diff="+x\n", content="x\n")]
            await client.review_diff(
                files, ticket_context="Jira ticket SEP-123", ticket_compliance_check=True,
            )
            prompt = _user_text_from_chat_call(mock_create)
            # Unique marker from COMPLIANCE_INSTRUCTIONS — the word
            # "compliance_requirements" alone now appears in the base prompt's
            # output spec, so we look for a phrase only the injected block has.
            assert "evaluate whether the code changes align" in prompt
            assert "{compliance_instructions}" not in prompt
        finally:
            await client.close()

    @pytest.mark.asyncio
    async def test_compliance_instructions_excluded_when_disabled(self, llm_config, review_config):
        mock_create = AsyncMock(return_value=_mock_completion("[]", 100, 10))
        client = LLMClient(llm_config, review_config)
        client.openai_client.chat.completions.create = mock_create
        try:
            files = [FileReviewData(path="a.py", diff="+x\n", content="x\n")]
            await client.review_diff(
                files, ticket_context="Jira ticket SEP-123", ticket_compliance_check=False,
            )
            prompt = _user_text_from_chat_call(mock_create)
            assert "evaluate whether the code changes align" not in prompt
            assert "{compliance_instructions}" not in prompt
        finally:
            await client.close()

    @pytest.mark.asyncio
    async def test_compliance_instructions_excluded_when_no_ticket_context(self, llm_config, review_config):
        mock_create = AsyncMock(return_value=_mock_completion("[]", 100, 10))
        client = LLMClient(llm_config, review_config)
        client.openai_client.chat.completions.create = mock_create
        try:
            files = [FileReviewData(path="a.py", diff="+x\n", content="x\n")]
            await client.review_diff(
                files, ticket_context="", ticket_compliance_check=True,
            )
            prompt = _user_text_from_chat_call(mock_create)
            assert "evaluate whether the code changes align" not in prompt
            assert "{compliance_instructions}" not in prompt
        finally:
            await client.close()

    @pytest.mark.asyncio
    async def test_ticket_context_always_present_regardless_of_compliance_flag(self, llm_config, review_config):
        mock_create = AsyncMock(return_value=_mock_completion("[]", 100, 10))
        client = LLMClient(llm_config, review_config)
        client.openai_client.chat.completions.create = mock_create
        try:
            files = [FileReviewData(path="a.py", diff="+x\n", content="x\n")]
            await client.review_diff(
                files, ticket_context="Jira ticket SEP-123", ticket_compliance_check=False,
            )
            prompt = _user_text_from_chat_call(mock_create)
            assert "Jira ticket SEP-123" in prompt
        finally:
            await client.close()


class TestAnswerQuestion:
    @pytest.mark.asyncio
    async def test_answer_question_with_file_data(self, llm_config, review_config):
        client = LLMClient(llm_config, review_config)
        client.openai_client.chat.completions.create = AsyncMock(
            return_value=_mock_completion("The function calculates fibonacci numbers.", 200, 30)
        )
        try:
            files = [FileReviewData(path="math.py", diff="+def fib(n):\n", content="def fib(n):\n    pass\n")]
            result = await client.answer_question("What does this do?", files)
            assert "fibonacci" in result.lower()
        finally:
            await client.close()

    @pytest.mark.asyncio
    async def test_answer_question_413_bisects_and_retries(self, llm_config, review_config):
        client = LLMClient(llm_config, review_config)

        files = [
            FileReviewData(path="a.py", diff="+a\n", content="a\n"),
            FileReviewData(path="b.py", diff="+b\n", content="b\n"),
        ]

        call_count = 0

        async def mock_call_mention_api(prompt: str):  # noqa: ARG001
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise openai.APIStatusError(
                    "too large",
                    response=httpx.Response(413, request=httpx.Request("POST", "https://x"), text="too large"),
                    body=None,
                )
            return "Answer for chunk", 50, 25

        client._call_mention_api = mock_call_mention_api
        try:
            result = await client.answer_question("What?", files)
            assert "Answer for chunk" in result
            assert call_count >= 3  # 1 failed + 2 retries
        finally:
            await client.close()

    @pytest.mark.asyncio
    async def test_answer_question_413_single_file_retries_diff_only(self, llm_config, review_config):
        client = LLMClient(llm_config, review_config)

        files = [FileReviewData(path="huge.py", diff="+x\n", content="x\n")]
        call_count = 0

        async def mock_call_mention_api(prompt: str):  # noqa: ARG001
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise openai.APIStatusError(
                    "too large",
                    response=httpx.Response(413, request=httpx.Request("POST", "https://x"), text="too large"),
                    body=None,
                )
            return "Answer with diff only", 30, 10

        client._call_mention_api = mock_call_mention_api
        try:
            result = await client.answer_question("What?", files)
            assert result == "Answer with diff only"
            assert call_count == 2
        finally:
            await client.close()

    @pytest.mark.asyncio
    async def test_answer_question_413_all_skipped_returns_fallback(self, llm_config, review_config):
        client = LLMClient(llm_config, review_config)

        files = [FileReviewData(path="huge.py", diff="+x\n", content=None)]

        async def mock_call_mention_api(prompt: str):  # noqa: ARG001
            raise openai.APIStatusError(
                "too large",
                response=httpx.Response(413, request=httpx.Request("POST", "https://x"), text="too large"),
                body=None,
            )

        client._call_mention_api = mock_call_mention_api
        try:
            result = await client.answer_question("What?", files)
            assert "couldn't process" in result.lower()
        finally:
            await client.close()


def _make_api_status_error(status_code: int, text: str = "") -> openai.APIStatusError:
    return openai.APIStatusError(
        text or f"Error {status_code}",
        response=httpx.Response(status_code, request=httpx.Request("POST", "https://x"), text=text),
        body=None,
    )


class TestReviewFileGroup413Retry:
    @pytest.mark.asyncio
    async def test_413_bisects_and_retries(self, llm_config, review_config):
        """On 413, the file group is bisected and each half retried."""
        client = LLMClient(llm_config, review_config)

        files = [
            FileReviewData(path="a.py", diff="+a\n", content="a\n"),
            FileReviewData(path="b.py", diff="+b\n", content="b\n"),
            FileReviewData(path="c.py", diff="+c\n", content="c\n"),
            FileReviewData(path="d.py", diff="+d\n", content="d\n"),
        ]

        call_count = 0
        finding_a = {"file": "a.py", "line": 1, "severity": "suggestion", "comment": "ok"}
        finding_d = {"file": "d.py", "line": 1, "severity": "suggestion", "comment": "ok"}

        async def mock_call_api(prompt: str):
            nonlocal call_count
            call_count += 1
            # First call (all 4 files) -> 413
            if call_count == 1:
                raise _make_api_status_error(413, "too large")
            # Subsequent calls succeed
            findings = []
            if "a.py" in prompt:
                findings.append(finding_a)
            if "d.py" in prompt:
                findings.append(finding_d)
            return (
                [__import__("app.models", fromlist=["ReviewFinding"]).ReviewFinding(**f) for f in findings],
                50, 25, [], ReviewSummary(), False,
            )

        client._call_api = mock_call_api
        try:
            template = client.prompt_template
            findings, pt, ct, skipped, _compliance, _summary, _to, _pf = await client._review_file_group(files, template, depth=0)
            assert len(findings) == 2
            assert {f.file for f in findings} == {"a.py", "d.py"}
            assert skipped == []
            assert call_count >= 3  # 1 failed + at least 2 retries
        finally:
            await client.close()

    @pytest.mark.asyncio
    async def test_413_single_file_retries_diff_only(self, llm_config, review_config):
        """A single file with content that triggers 413 retries with diff only."""
        client = LLMClient(llm_config, review_config)

        files = [FileReviewData(path="huge.py", diff="+x\n", content="x\n")]
        call_count = 0

        async def mock_call_api(prompt: str):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise _make_api_status_error(413, "too large")
            return [], 0, 0, [], ReviewSummary(), False

        client._call_api = mock_call_api
        try:
            template = client.prompt_template
            findings, pt, ct, skipped, _compliance, _summary, _to, _pf = await client._review_file_group(files, template, depth=0)
            assert findings == []
            assert skipped == []
            assert call_count == 2
        finally:
            await client.close()

    @pytest.mark.asyncio
    async def test_413_single_file_skipped_when_already_diff_only(self, llm_config, review_config):
        """A single file already without content that triggers 413 is skipped."""
        client = LLMClient(llm_config, review_config)

        files = [FileReviewData(path="huge.py", diff="+x\n", content=None)]

        async def mock_call_api(prompt: str):
            raise _make_api_status_error(413, "too large")

        client._call_api = mock_call_api
        try:
            template = client.prompt_template
            findings, pt, ct, skipped, _compliance, _summary, _to, _pf = await client._review_file_group(files, template, depth=0)
            assert findings == []
            assert pt == 0
            assert ct == 0
            assert skipped == ["huge.py"]
        finally:
            await client.close()

    @pytest.mark.asyncio
    async def test_413_single_file_falls_through_when_diff_only_also_fails(self, llm_config, review_config):
        """A single file that 413s with content and again with diff-only is skipped."""
        client = LLMClient(llm_config, review_config)

        files = [FileReviewData(path="huge.py", diff="+x\n", content="x\n")]

        async def mock_call_api(prompt: str):
            raise _make_api_status_error(413, "too large")

        client._call_api = mock_call_api
        try:
            template = client.prompt_template
            findings, pt, ct, skipped, _compliance, _summary, _to, _pf = await client._review_file_group(files, template, depth=0)
            assert findings == []
            assert skipped == ["huge.py"]
        finally:
            await client.close()

    @pytest.mark.asyncio
    async def test_413_max_depth_stops_recursion(self, llm_config, review_config):
        """Recursion stops at max_depth even with multiple files."""
        client = LLMClient(llm_config, review_config)

        files = [
            FileReviewData(path="a.py", diff="+a\n", content="a\n"),
            FileReviewData(path="b.py", diff="+b\n", content="b\n"),
        ]

        async def mock_call_api(prompt: str):
            raise _make_api_status_error(413, "too large")

        client._call_api = mock_call_api
        try:
            template = client.prompt_template
            findings, pt, ct, skipped, _compliance, _summary, _to, _pf = await client._review_file_group(files, template, depth=3)
            assert findings == []
            assert set(skipped) == {"a.py", "b.py"}
        finally:
            await client.close()

    @pytest.mark.asyncio
    async def test_non_413_error_propagates(self, llm_config, review_config):
        """Non-413 HTTP errors are not caught."""
        client = LLMClient(llm_config, review_config)

        files = [FileReviewData(path="a.py", diff="+a\n", content="a\n")]

        async def mock_call_api(prompt: str):
            raise _make_api_status_error(500, "server error")

        client._call_api = mock_call_api
        try:
            template = client.prompt_template
            with pytest.raises(openai.APIStatusError):
                await client._review_file_group(files, template, depth=0)
        finally:
            await client.close()


class TestEstimateReviewEffort:
    def test_trivial_change(self):
        files = [FileReviewData(path="a.py", diff="+x = 1\n", content="x = 1\n")]
        assert LLMClient._estimate_review_effort(files) == 1

    def test_small_change(self):
        files = [
            FileReviewData(path="a.py", diff="\n".join(f"+line{i}" for i in range(20)), content="x\n"),
            FileReviewData(path="b.py", diff="+fix\n", content="fix\n"),
        ]
        assert LLMClient._estimate_review_effort(files) == 2

    def test_medium_change(self):
        files = [
            FileReviewData(path=f"f{i}.py", diff="\n".join(f"+line{j}" for j in range(30)), content="x\n")
            for i in range(4)
        ]
        assert LLMClient._estimate_review_effort(files) == 3

    def test_large_change(self):
        files = [
            FileReviewData(path=f"f{i}.py", diff="\n".join(f"+line{j}" for j in range(40)), content="x\n")
            for i in range(10)
        ]
        assert LLMClient._estimate_review_effort(files) == 4

    def test_very_large_change(self):
        files = [
            FileReviewData(path=f"f{i}.py", diff="\n".join(f"+line{j}" for j in range(50)), content="x\n")
            for i in range(20)
        ]
        assert LLMClient._estimate_review_effort(files) == 5


class TestLowBudgetWarning:
    @pytest.mark.asyncio
    async def test_warns_low_effective_budget(self, llm_config, review_config, caplog):
        """When configured max_tokens_per_chunk barely clears the prompt overhead, warn."""
        client = LLMClient(llm_config, review_config)
        client.max_tokens_per_chunk = 2000
        client.openai_client.chat.completions.create = AsyncMock(
            return_value=_mock_completion("pong", 5, 1)
        )
        try:
            import logging
            with caplog.at_level(logging.WARNING, logger="app.llm_client"):
                await client.check_connectivity()
            assert any("Effective token budget" in msg for msg in caplog.messages)
        finally:
            await client.close()


class TestContextWindowAutoCap:
    def test_known_models_resolve(self):
        assert _context_window_for("gpt-5") == 272_000
        assert _context_window_for("claude-sonnet-4") == 200_000

    def test_gpt_5_5_and_5_4_use_real_million_window(self):
        # Regression: these were hardcoded to 272k (the pricing threshold), not
        # the real 1.05M context window. Now sourced from LiteLLM's static fallback.
        assert _context_window_for("gpt-5.5") == 1_050_000
        assert _context_window_for("gpt-5.4") == 1_050_000

    def test_dated_id_prefix_match(self):
        assert _context_window_for("gpt-5-2025-01-01") == 272_000
        assert _context_window_for("claude-sonnet-4-20250514") == 200_000

    def test_unknown_model_returns_none(self):
        assert _context_window_for("mystery-model-9000") is None

    def test_budget_derived_from_context_window(self, review_config):
        # gpt-4o's 128k window is below the trust threshold → flat 16k headroom.
        cfg = LLMConfig(
            model="gpt-4o",
            api_key="t",
            api_url="https://llm.test/v1",
        )
        client = LLMClient(cfg, review_config)
        assert client.max_tokens_per_chunk == 128_000 - 16_000
        assert client.context_window == 128_000

    def test_budget_for_codex_model(self, review_config):
        # gpt-5.3-codex's 272k window is just above the 256k threshold, so the
        # diminishing-trust curve applies: 256k + (272k-256k)*0.5 = 264k.
        cfg = LLMConfig(
            model="gpt-5.3-codex",
            api_key="t",
            api_url="https://llm.test/v1",
        )
        client = LLMClient(cfg, review_config)
        assert client.max_tokens_per_chunk == 264_000
        assert client.context_window == 272_000

    def test_budget_for_million_token_model(self, review_config):
        # gpt-5.5's 1.05M window degrades hard: 256k + (1050k-256k)*0.5 = 653k.
        cfg = LLMConfig(
            model="gpt-5.5",
            api_key="t",
            api_url="https://llm.test/v1",
        )
        client = LLMClient(cfg, review_config)
        assert client.context_window == 1_050_000
        assert client.max_tokens_per_chunk == 653_000

    def test_unknown_model_falls_back_to_default(self, review_config):
        cfg = LLMConfig(
            model="mystery-model-9000",
            api_key="t",
            api_url="https://llm.test/v1",
        )
        client = LLMClient(cfg, review_config)
        assert client.context_window == 128_000
        assert client.max_tokens_per_chunk == 128_000 - 16_000

    def test_413_cap_clamps_budget_below_curve(self, review_config):
        # A 413-learned cap (via the setter) takes precedence over the curve, and
        # is never undone — the getter returns min(curve, cap).
        cfg = LLMConfig(
            model="gpt-5.5",
            api_key="t",
            api_url="https://llm.test/v1",
        )
        client = LLMClient(cfg, review_config)
        assert client.max_tokens_per_chunk == 653_000
        client.max_tokens_per_chunk = 4_000  # simulate a 413 "Max size: 4,000 tokens"
        assert client.max_tokens_per_chunk == 4_000

    def test_413_cap_is_shrink_only(self, review_config):
        # The setter is intrinsically shrink-only (min), so a later, larger value
        # can never raise an already-learned cap — even bypassing the 413 guard.
        cfg = LLMConfig(
            model="gpt-5.5",
            api_key="t",
            api_url="https://llm.test/v1",
        )
        client = LLMClient(cfg, review_config)
        client.max_tokens_per_chunk = 5_000
        client.max_tokens_per_chunk = 80_000  # would-be raise — must be ignored
        assert client.max_tokens_per_chunk == 5_000


class TestParse413TokenLimit:
    @pytest.mark.asyncio
    async def test_413_with_max_size_updates_config(self, llm_config, review_config):
        """A 413 response body containing 'Max size: N tokens' updates max_tokens_per_chunk."""
        client = LLMClient(llm_config, review_config)
        client.max_tokens_per_chunk = 80000

        files = [FileReviewData(path="big.py", diff="+x\n", content=None)]

        async def mock_call_api(prompt: str):
            raise _make_api_status_error(413, "Request too large. Max size: 4,000 tokens.")

        client._call_api = mock_call_api
        try:
            template = client.prompt_template
            _, _, _, skipped, _, _, _, _ = await client._review_file_group(files, template, depth=0)
            assert skipped == ["big.py"]
            assert client.max_tokens_per_chunk == 4000
        finally:
            await client.close()

    @pytest.mark.asyncio
    async def test_413_without_max_size_leaves_config_unchanged(self, llm_config, review_config):
        """A 413 response without the 'Max size' pattern does not change max_tokens_per_chunk."""
        client = LLMClient(llm_config, review_config)
        client.max_tokens_per_chunk = 80000

        files = [FileReviewData(path="big.py", diff="+x\n", content=None)]

        async def mock_call_api(prompt: str):
            raise _make_api_status_error(413, "Request entity too large")

        client._call_api = mock_call_api
        try:
            template = client.prompt_template
            await client._review_file_group(files, template, depth=0)
            assert client.max_tokens_per_chunk == 80000
        finally:
            await client.close()

    @pytest.mark.asyncio
    async def test_413_answer_group_parses_limit(self, llm_config, review_config):
        """413 in _answer_file_group also parses and updates max_tokens_per_chunk."""
        client = LLMClient(llm_config, review_config)
        client.max_tokens_per_chunk = 80000

        files = [FileReviewData(path="big.py", diff="+x\n", content=None)]

        async def mock_call_mention_api(prompt: str):  # noqa: ARG001
            raise _make_api_status_error(413, "Max size: 5000 tokens.")

        client._call_mention_api = mock_call_mention_api
        try:
            template = "{diff}"
            _, _, _, skipped = await client._answer_file_group(files, template, depth=0)
            assert skipped == ["big.py"]
            assert client.max_tokens_per_chunk == 5000
        finally:
            await client.close()


class TestSerializationAndDeadline:
    @pytest.mark.asyncio
    async def test_concurrent_chats_are_serialized(self, llm_config, review_config):
        """Two concurrent _chat calls must not overlap inside _execute_chat_completion."""
        import asyncio
        client = LLMClient(llm_config, review_config)

        active = 0
        max_active = 0

        async def fake_create(**_kwargs):
            nonlocal active, max_active
            active += 1
            max_active = max(max_active, active)
            try:
                await asyncio.sleep(0.05)
                return _mock_completion("ok")
            finally:
                active -= 1

        client.openai_client.chat.completions.create = fake_create
        try:
            await asyncio.gather(
                client._chat("sys", "u1"),
                client._chat("sys", "u2"),
                client._chat("sys", "u3"),
            )
            assert max_active == 1, f"expected serialized calls, observed {max_active} concurrent"
        finally:
            await client.close()

    @pytest.mark.asyncio
    async def test_review_and_mention_paths_share_the_lock(
        self, llm_config, review_config,
    ):
        """answer_question (mention path) and review_diff (review path) both
        funnel through _execute_chat_completion. Concurrent calls from the
        two entry points must serialize on the same lock."""
        import asyncio
        client = LLMClient(llm_config, review_config)

        active = 0
        max_active = 0

        async def fake_create(**_kwargs):
            nonlocal active, max_active
            active += 1
            max_active = max(max_active, active)
            try:
                await asyncio.sleep(0.05)
                # Empty findings JSON: parses cleanly for the review path,
                # and answer_question doesn't validate JSON shape itself.
                return _mock_completion(
                    '{"findings": [], "compliance_requirements": [], "overview": "x", '
                    '"strengths": [], "security_performance": "None notable.", '
                    '"test_coverage": "x", "verdict": {"decision": "approve", "rationale": "x"}}'
                )
            finally:
                active -= 1

        client.openai_client.chat.completions.create = fake_create
        files = [FileReviewData(path="a.py", diff="+x\n", content="x\n")]
        try:
            await asyncio.gather(
                client.review_diff(files),
                client.answer_question("what?", files),
                client.review_diff(files),
            )
            assert max_active == 1, (
                f"review and mention paths must serialize via _execute_chat_completion, "
                f"but observed {max_active} concurrent calls"
            )
        finally:
            await client.close()

    @pytest.mark.asyncio
    async def test_hard_timeout_translates_to_apitimeout(self, llm_config, review_config, monkeypatch):
        """When the wall-clock cap fires, _chat raises openai.APITimeoutError."""
        import asyncio
        import app.llm_client as llm_module
        monkeypatch.setattr(llm_module, "INFERENCE_HARD_TIMEOUT_SECONDS", 0.05)

        client = LLMClient(llm_config, review_config)

        async def slow_create(**_kwargs):
            await asyncio.sleep(5)
            return _mock_completion("never")

        client.openai_client.chat.completions.create = slow_create
        try:
            with pytest.raises(openai.APITimeoutError):
                await client._chat("sys", "u")
        finally:
            await client.close()

    def test_async_openai_constructed_with_no_retries(self, llm_config, review_config):
        """Silent SDK retries are disabled — one attempt only."""
        client = LLMClient(llm_config, review_config)
        try:
            assert client.openai_client.max_retries == 0
        finally:
            import asyncio
            asyncio.run(client.close())


class TestReviewResultTimedOut:
    @pytest.mark.asyncio
    async def test_timed_out_chunk_sets_flag(self, llm_config, review_config):
        client = LLMClient(llm_config, review_config)

        async def raise_timeout(prompt: str):
            raise openai.APITimeoutError(request=httpx.Request("POST", "https://x"))

        client._call_api = raise_timeout
        try:
            files = [FileReviewData(path="a.py", diff="+x\n", content="x\n")]
            result = await client.review_diff(files)
            assert result.timed_out is True
            assert result.findings == []
        finally:
            await client.close()

    @pytest.mark.asyncio
    async def test_bisection_merge_propagates_timed_out(self, llm_config, review_config):
        """When one half of a 413-bisected group times out, the merged result
        carries timed_out=True even if the other half succeeded."""
        from app.models import ReviewFinding
        client = LLMClient(llm_config, review_config)

        files = [
            FileReviewData(path="a.py", diff="+a\n", content="a\n"),
            FileReviewData(path="b.py", diff="+b\n", content="b\n"),
        ]
        call_count = 0

        async def mock_call_api(prompt: str):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise _make_api_status_error(413, "too large")
            if "a.py" in prompt:
                raise openai.APITimeoutError(request=httpx.Request("POST", "https://x"))
            return (
                [ReviewFinding(file="b.py", line=1, severity="suggestion", comment="ok")],
                10, 5, [], ReviewSummary(), False,
            )

        client._call_api = mock_call_api
        try:
            template = client.prompt_template
            findings, pt, ct, skipped, _compliance, _summary, timed_out, _parse_failed = (
                await client._review_file_group(files, template, depth=0)
            )
            assert timed_out is True
            assert any(f.file == "b.py" for f in findings)
        finally:
            await client.close()

    @pytest.mark.asyncio
    async def test_no_timeout_leaves_flag_false(self, llm_config, review_config):
        from app.models import ReviewFinding
        client = LLMClient(llm_config, review_config)

        async def ok(prompt: str):
            return (
                [ReviewFinding(file="a.py", line=1, severity="suggestion", comment="ok")],
                10, 5, [], ReviewSummary(), False,
            )

        client._call_api = ok
        try:
            files = [FileReviewData(path="a.py", diff="+x\n", content="x\n")]
            result = await client.review_diff(files)
            assert result.timed_out is False
        finally:
            await client.close()


class TestReviewResultUnparseable:
    @pytest.mark.asyncio
    async def test_unparseable_chunk_sets_flag(self, llm_config, review_config):
        """A single chunk whose response could not be parsed (empty output or a
        non-JSON refusal) sets response_unparseable and yields no findings."""
        client = LLMClient(llm_config, review_config)

        async def refuse(prompt: str):
            # Mirrors _call_api's return when _parse_review_response reports a
            # refusal: no findings, compliance None, empty summary, parse_failed.
            return ([], 0, 0, None, ReviewSummary(), True)

        client._call_api = refuse
        try:
            files = [FileReviewData(path="a.py", diff="+x\n", content="x\n")]
            result = await client.review_diff(files)
            assert result.response_unparseable is True
            assert result.timed_out is False
            assert result.findings == []
        finally:
            await client.close()

    @pytest.mark.asyncio
    async def test_parseable_leaves_flag_false(self, llm_config, review_config):
        from app.models import ReviewFinding
        client = LLMClient(llm_config, review_config)

        async def ok(prompt: str):
            return (
                [ReviewFinding(file="a.py", line=1, severity="suggestion", comment="ok")],
                10, 5, [], ReviewSummary(), False,
            )

        client._call_api = ok
        try:
            files = [FileReviewData(path="a.py", diff="+x\n", content="x\n")]
            result = await client.review_diff(files)
            assert result.response_unparseable is False
        finally:
            await client.close()

    @pytest.mark.asyncio
    async def test_bisection_merge_propagates_parse_failed(self, llm_config, review_config):
        """When one half of a 413-bisected group is refused, the merged group
        result carries parse_failed=True even if the other half parsed."""
        from app.models import ReviewFinding
        client = LLMClient(llm_config, review_config)

        files = [
            FileReviewData(path="a.py", diff="+a\n", content="a\n"),
            FileReviewData(path="b.py", diff="+b\n", content="b\n"),
        ]
        call_count = 0

        async def mock_call_api(prompt: str):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise _make_api_status_error(413, "too large")
            if "a.py" in prompt:
                return ([], 0, 0, None, ReviewSummary(), True)
            return (
                [ReviewFinding(file="b.py", line=1, severity="suggestion", comment="ok")],
                10, 5, [], ReviewSummary(), False,
            )

        client._call_api = mock_call_api
        try:
            template = client.prompt_template
            findings, pt, ct, skipped, _compliance, _summary, timed_out, parse_failed = (
                await client._review_file_group(files, template, depth=0)
            )
            assert parse_failed is True
            assert timed_out is False
            assert any(f.file == "b.py" for f in findings)
        finally:
            await client.close()

