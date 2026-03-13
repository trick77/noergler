import logging
import re
from dataclasses import dataclass, field
from typing import Callable

from app.copilot import FileReviewData, is_deleted

logger = logging.getLogger(__name__)

EXTENSION_LANGUAGE_MAP: dict[str, str] = {
    # Python
    ".py": "python",
    ".pyi": "python",
    # JVM
    ".java": "jvm",
    ".kt": "jvm",
    ".kts": "jvm",
    ".groovy": "jvm",
    # TypeScript/Angular
    ".ts": "typescript",
    ".tsx": "typescript",
    ".js": "typescript",
    ".jsx": "typescript",
    ".mjs": "typescript",
    # HTML (Angular templates)
    ".html": "html",
    ".htm": "html",
    # CSS/Styles
    ".css": "css",
    ".scss": "css",
    ".less": "css",
    ".sass": "css",
    # Config
    ".yaml": "config",
    ".yml": "config",
    ".toml": "config",
    ".ini": "config",
    ".cfg": "config",
    # Docs
    ".md": "docs",
    ".rst": "docs",
    ".txt": "docs",
}

BUILD_FILE_MAP: dict[str, str] = {
    "pom.xml": "build-config",
    "build.gradle": "build-config",
    "build.gradle.kts": "build-config",
    "angular.json": "build-config",
    "nx.json": "build-config",
    "project.json": "build-config",
    "package.json": "build-config",
}

_LANGUAGE_PRIORITY = ["python", "jvm", "typescript", "html", "css", "build-config", "config", "docs", "other"]

_TEST_PATTERNS = re.compile(
    r"(?:"
    r"_test\.py$|test_[^/]*\.py$"
    r"|Test\.java$|Tests\.java$"
    r"|\.spec\.ts$|\.test\.ts$"
    r"|\.spec\.tsx$|\.test\.tsx$"
    r"|\.spec\.js$|\.test\.js$"
    r"|/tests?/|/__tests__/"
    r")",
    re.IGNORECASE,
)


def detect_language(path: str) -> str:
    basename = path.rsplit("/", 1)[-1] if "/" in path else path
    if basename in BUILD_FILE_MAP:
        return BUILD_FILE_MAP[basename]
    if basename.startswith("tsconfig") and basename.endswith(".json"):
        return "build-config"
    ext = ""
    dot_pos = basename.rfind(".")
    if dot_pos >= 0:
        ext = basename[dot_pos:]
    return EXTENSION_LANGUAGE_MAP.get(ext, "other")


def is_test_file(path: str) -> bool:
    return bool(_TEST_PATTERNS.search(path))


def determine_repo_languages(files: list[FileReviewData]) -> list[str]:
    lang_counts: dict[str, int] = {}
    for f in files:
        lang = detect_language(f.path)
        if not is_test_file(f.path):
            lang_counts[lang] = lang_counts.get(lang, 0) + 1
    return sorted(lang_counts, key=lambda l: (-lang_counts[l], _LANGUAGE_PRIORITY.index(l) if l in _LANGUAGE_PRIORITY else 999))


def sort_files_by_language_priority(
    files: list[FileReviewData],
    language_order: list[str],
    count_tokens_fn: Callable[[str], int],
) -> list[FileReviewData]:
    lang_rank = {lang: i for i, lang in enumerate(language_order)}
    max_rank = len(language_order)

    # Deprioritized groups always sort after source languages
    deprioritized = {"build-config", "config", "docs", "other"}

    def sort_key(f: FileReviewData) -> tuple:
        lang = detect_language(f.path)
        test = is_test_file(f.path)
        is_depri = lang in deprioritized
        base_rank = lang_rank.get(lang, max_rank)

        # Source files first, then test files of same language, then deprioritized
        if is_depri:
            group = 2
        elif test:
            group = 1
        else:
            group = 0

        text = f.diff + (f.content or "")
        tokens = count_tokens_fn(text)
        return (group, base_rank, -tokens)

    return sorted(files, key=sort_key)


def is_deletion_only_hunk(hunk_lines: list[str]) -> bool:
    for line in hunk_lines:
        if line.startswith("+") and not line.startswith("+++"):
            return False
    return True


def remove_deletion_only_hunks(file_diff: str) -> str:
    lines = file_diff.split("\n")
    header_lines: list[str] = []
    hunks: list[list[str]] = []
    current_hunk: list[str] = []

    for line in lines:
        if line.startswith("@@"):
            if current_hunk:
                hunks.append(current_hunk)
            current_hunk = [line]
        elif current_hunk:
            current_hunk.append(line)
        else:
            header_lines.append(line)

    if current_hunk:
        hunks.append(current_hunk)

    kept_hunks = [h for h in hunks if not is_deletion_only_hunk(h)]
    if not kept_hunks:
        return ""

    return "\n".join(header_lines + [line for h in kept_hunks for line in h])


def is_rename_only(file_diff: str) -> bool:
    return "similarity index 100%" in file_diff and ("rename from" in file_diff or "rename to" in file_diff)


@dataclass
class CompressionResult:
    included_files: list[FileReviewData] = field(default_factory=list)
    other_modified_paths: list[str] = field(default_factory=list)
    deleted_file_paths: list[str] = field(default_factory=list)
    renamed_file_paths: list[str] = field(default_factory=list)


def compress_for_large_pr(
    files: list[FileReviewData],
    max_tokens: int,
    prompt_template: str,
    count_tokens_fn: Callable[[str], int],
    format_entry_fn: Callable[[FileReviewData], str],
) -> CompressionResult:
    result = CompressionResult()

    deleted: list[FileReviewData] = []
    renamed: list[FileReviewData] = []
    active: list[FileReviewData] = []

    for f in files:
        if is_deleted(f.diff):
            deleted.append(f)
        elif is_rename_only(f.diff):
            renamed.append(f)
        else:
            active.append(f)

    result.deleted_file_paths = [f.path for f in deleted]
    result.renamed_file_paths = [f.path for f in renamed]

    # Remove deletion-only hunks from active files
    compressed_active: list[FileReviewData] = []
    for f in active:
        cleaned_diff = remove_deletion_only_hunks(f.diff)
        if not cleaned_diff:
            result.deleted_file_paths.append(f.path)
        else:
            compressed_active.append(FileReviewData(path=f.path, diff=cleaned_diff, content=f.content))

    # Sort by language priority
    language_order = determine_repo_languages(files)
    sorted_files = sort_files_by_language_priority(compressed_active, language_order, count_tokens_fn)

    # Calculate budget (90% of available)
    prompt_overhead = count_tokens_fn(prompt_template.replace("{files}", ""))
    budget = int((max_tokens - prompt_overhead) * 0.9)

    used_tokens = 0
    for f in sorted_files:
        entry = format_entry_fn(f)
        entry_tokens = count_tokens_fn(entry)
        if used_tokens + entry_tokens <= budget:
            result.included_files.append(f)
            used_tokens += entry_tokens
        else:
            result.other_modified_paths.append(f.path)

    logger.info(
        "Compression: %d total files — %d included, %d other_modified, %d deleted, %d renamed",
        len(files), len(result.included_files), len(result.other_modified_paths),
        len(result.deleted_file_paths), len(result.renamed_file_paths),
    )

    return result


def is_small_pr(
    files: list[FileReviewData],
    max_tokens: int,
    prompt_template: str,
    count_tokens_fn: Callable[[str], int],
    format_entry_fn: Callable[[FileReviewData], str],
    context_expansion_ratio: float = 1.5,
) -> bool:
    prompt_overhead = count_tokens_fn(prompt_template.replace("{files}", ""))
    available = max_tokens - prompt_overhead
    total = sum(count_tokens_fn(format_entry_fn(f)) for f in files)
    return total * context_expansion_ratio <= available
