import logging
import re
from dataclasses import dataclass

from app.copilot import FileReviewData
from app.diff_compression import detect_language

logger = logging.getLogger(__name__)

_HUNK_HEADER_RE = re.compile(
    r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@"
)

_NO_DYNAMIC_LANGUAGES = frozenset({"config", "docs", "css", "html", "build-config"})

_SCOPE_PATTERNS: dict[str, re.Pattern] = {
    "python": re.compile(
        r"^\s*(?:def |class |async def )"
    ),
    "jvm": re.compile(
        r"^\s*(?:public |private |protected |static |final |abstract |default |fun |class |interface |enum |override |@)"
        r"|.*\{\s*$"
    ),
    "typescript": re.compile(
        r"^\s*(?:function |class |interface |enum |export |const \w+\s*=\s*(?:\(|async))"
        r"|.*\{\s*$"
    ),
    "other": re.compile(
        r"^\s*(?:def |func |fn )"
        r"|.*\{\s*$"
    ),
}


@dataclass
class HunkInfo:
    old_start: int
    old_count: int
    new_start: int
    new_count: int
    body_lines: list[str]


def parse_hunks(file_diff: str) -> tuple[list[str], list[HunkInfo]]:
    lines = file_diff.split("\n")
    header_lines: list[str] = []
    hunks: list[HunkInfo] = []
    current_body: list[str] = []
    current_header_match = None

    for line in lines:
        m = _HUNK_HEADER_RE.match(line)
        if m:
            if current_header_match is not None:
                hunks.append(HunkInfo(
                    old_start=int(current_header_match.group(1)),
                    old_count=int(current_header_match.group(2) or "1"),
                    new_start=int(current_header_match.group(3)),
                    new_count=int(current_header_match.group(4) or "1"),
                    body_lines=current_body,
                ))
            current_header_match = m
            current_body = []
        elif current_header_match is not None:
            current_body.append(line)
        else:
            header_lines.append(line)

    if current_header_match is not None:
        hunks.append(HunkInfo(
            old_start=int(current_header_match.group(1)),
            old_count=int(current_header_match.group(2) or "1"),
            new_start=int(current_header_match.group(3)),
            new_count=int(current_header_match.group(4) or "1"),
            body_lines=current_body,
        ))

    return header_lines, hunks


def find_enclosing_scope_line(
    file_lines: list[str],
    hunk_new_start: int,
    max_lines: int,
    path: str,
) -> int | None:
    lang = detect_language(path)
    if lang in _NO_DYNAMIC_LANGUAGES:
        return None

    pattern = _SCOPE_PATTERNS.get(lang, _SCOPE_PATTERNS["other"])

    # hunk_new_start is 1-based line number
    start_idx = hunk_new_start - 1
    search_limit = min(max_lines, start_idx)

    for offset in range(1, search_limit + 1):
        idx = start_idx - offset
        if idx < 0:
            break
        line = file_lines[idx]
        if pattern.match(line):
            return idx + 1  # return 1-based line number
    return None


_ExpandedHunk = tuple[int, int, int, int, list[str]]


def expand_context(
    file_diff: str,
    file_content: str | None,
    path: str,
    before: int = 3,
    after: int = 1,
    max_dynamic_before: int = 8,
    dynamic_context: bool = True,
) -> str:
    if file_content is None:
        return file_diff

    header_lines, hunks = parse_hunks(file_diff)
    if not hunks:
        return file_diff

    file_lines = file_content.split("\n")
    total_file_lines = len(file_lines)

    expanded: list[_ExpandedHunk] = []

    for hunk in hunks:
        # Calculate context window in new-file coordinates (1-based)
        ctx_start = max(1, hunk.new_start - before)

        # Dynamic context: search further back for enclosing scope
        if dynamic_context:
            scope_line = find_enclosing_scope_line(
                file_lines,
                ctx_start,  # search from the top of the static window
                max_dynamic_before,
                path,
            )
            if scope_line is not None and scope_line < ctx_start:
                ctx_start = scope_line

        before_count = hunk.new_start - ctx_start

        # Count actual new-file lines in the hunk body (context + added lines)
        hunk_new_line_count = sum(
            1 for line in hunk.body_lines
            if line and not line.startswith("-")
        )
        hunk_end = hunk.new_start + hunk_new_line_count
        ctx_end = min(total_file_lines, hunk_end + after - 1)
        after_count = max(0, ctx_end - hunk_end + 1)

        # Build context lines before
        before_lines = []
        for i in range(ctx_start - 1, ctx_start - 1 + before_count):
            if 0 <= i < total_file_lines:
                before_lines.append(" " + file_lines[i])

        # Build context lines after
        after_lines = []
        for i in range(hunk_end - 1, hunk_end - 1 + after_count):
            if 0 <= i < total_file_lines:
                after_lines.append(" " + file_lines[i])

        body = before_lines + hunk.body_lines + after_lines

        # Strip trailing empty line if present (artifact from splitting)
        if body and body[-1] == "":
            body = body[:-1]

        # Count old/new lines for the @@ header
        old_removed = sum(1 for ln in hunk.body_lines if ln.startswith("-"))
        old_unchanged = sum(1 for ln in hunk.body_lines if ln and not ln.startswith("+") and not ln.startswith("-"))
        new_added = sum(1 for ln in hunk.body_lines if ln.startswith("+"))

        # old_start uses the delta between old/new to stay aligned, since
        # before_count is computed in new-file coordinates and old/new can diverge
        old_new_delta = hunk.old_start - hunk.new_start
        new_new_start = hunk.new_start - before_count
        new_old_start = max(1, new_new_start + old_new_delta)
        new_old_count = before_count + old_removed + old_unchanged + after_count
        new_new_count = before_count + new_added + old_unchanged + after_count

        expanded.append((new_old_start, new_old_count, new_new_start, new_new_count, body))

    # Merge overlapping/adjacent hunks
    merged = _merge_expanded_hunks(expanded)

    # Rebuild diff
    result_lines = list(header_lines)
    for old_start, old_count, new_start, new_count, body in merged:
        result_lines.append(f"@@ -{old_start},{old_count} +{new_start},{new_count} @@")
        result_lines.extend(body)

    return "\n".join(result_lines)


def _merge_expanded_hunks(
    expanded: list[_ExpandedHunk],
) -> list[_ExpandedHunk]:
    if not expanded:
        return []

    merged: list[_ExpandedHunk] = [expanded[0]]

    for old_start, old_count, new_start, new_count, body in expanded[1:]:
        prev_old_start, prev_old_count, prev_new_start, prev_new_count, prev_body = merged[-1]
        prev_new_end = prev_new_start + prev_new_count

        if new_start <= prev_new_end:
            # Overlapping or adjacent: merge
            overlap = prev_new_end - new_start
            trimmed_body = body[overlap:] if overlap > 0 else body

            combined_body = prev_body + trimmed_body

            # Recalculate counts from the combined body
            combined_old_count = sum(
                1 for ln in combined_body
                if ln and (ln.startswith("-") or ln.startswith(" "))
            )
            combined_new_count = sum(
                1 for ln in combined_body
                if ln and (ln.startswith("+") or ln.startswith(" "))
            )

            merged[-1] = (
                prev_old_start, combined_old_count,
                prev_new_start, combined_new_count,
                combined_body,
            )
        else:
            merged.append((old_start, old_count, new_start, new_count, body))

    return merged


def expand_all_files(
    files: list[FileReviewData],
    before: int = 3,
    after: int = 1,
    max_dynamic_before: int = 8,
    dynamic_context: bool = True,
) -> list[FileReviewData]:
    result: list[FileReviewData] = []
    for f in files:
        expanded_diff = expand_context(
            f.diff, f.content, f.path,
            before=before, after=after,
            max_dynamic_before=max_dynamic_before,
            dynamic_context=dynamic_context,
        )
        result.append(FileReviewData(path=f.path, diff=expanded_diff, content=f.content))
    return result
