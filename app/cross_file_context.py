"""Cross-file context analysis.

Extracts symbols (functions, classes) changed in the diff and finds
references to those symbols across other files in the PR.  The resulting
relationship map is injected into the LLM prompt so the model can detect
cross-file issues like mismatched signatures, missing import updates, or
broken call sites.
"""

import logging
import re
from dataclasses import dataclass, field

from app.copilot import FileReviewData
from app.context_expansion import _SCOPE_PATTERNS
from app.diff_compression import detect_language

logger = logging.getLogger(__name__)

# Patterns to extract the symbol name from a scope definition line.
_SYMBOL_NAME_PATTERNS: dict[str, re.Pattern] = {
    "python": re.compile(
        r"^\s*(?:async\s+)?(?:def|class)\s+(\w+)"
    ),
    "jvm": re.compile(
        r"(?:(?:public|private|protected|static|final|abstract|override|default)\s+)*"
        r"(?:fun|class|interface|enum|record)\s+(\w+)"
    ),
    "typescript": re.compile(
        r"(?:export\s+)?(?:default\s+)?(?:async\s+)?"
        r"(?:function|class|interface|enum|type|const)\s+(\w+)"
    ),
    "other": re.compile(
        r"(?:def|func|fn|class)\s+(\w+)"
    ),
}

# Minimum symbol name length to avoid matching single-letter variables.
_MIN_SYMBOL_LEN = 3

# Maximum references to report per symbol (to avoid flooding the prompt).
_MAX_REFS_PER_SYMBOL = 5

# Maximum total relationship lines to keep prompt overhead small.
_MAX_RELATIONSHIP_LINES = 30


@dataclass
class SymbolReference:
    """A reference to a symbol found in another file."""
    file: str
    line_number: int
    line_text: str


@dataclass
class CrossFileRelationship:
    """A changed symbol and where it's referenced in other PR files."""
    symbol: str
    defined_in: str
    references: list[SymbolReference] = field(default_factory=list)


def _extract_changed_symbols(file_data: FileReviewData) -> list[str]:
    """Extract symbol names from added/modified scope lines in the diff."""
    lang = detect_language(file_data.path)
    scope_pattern = _SCOPE_PATTERNS.get(lang, _SCOPE_PATTERNS["other"])
    name_pattern = _SYMBOL_NAME_PATTERNS.get(lang, _SYMBOL_NAME_PATTERNS["other"])

    symbols: list[str] = []
    for line in file_data.diff.splitlines():
        # Only look at added lines (new or modified code).
        if not line.startswith("+") or line.startswith("+++"):
            continue
        # Strip the leading "+" to get the actual code line.
        code_line = line[1:]
        if not scope_pattern.match(code_line):
            continue
        match = name_pattern.search(code_line)
        if match:
            name = match.group(1)
            if len(name) >= _MIN_SYMBOL_LEN and name not in symbols:
                symbols.append(name)

    return symbols


def _find_references(
    symbol: str, target_file: FileReviewData,
) -> list[SymbolReference]:
    """Find references to *symbol* in *target_file*'s content or diff."""
    refs: list[SymbolReference] = []
    # Prefer full file content; fall back to diff if content unavailable.
    text = target_file.content or target_file.diff

    # Use word-boundary regex to avoid partial matches.
    pattern = re.compile(rf"\b{re.escape(symbol)}\b")

    for line_no, line in enumerate(text.splitlines(), start=1):
        if pattern.search(line):
            # Skip lines that are the definition itself (import lines are fine).
            stripped = line.strip()
            # Skip comment-only lines
            if stripped.startswith("#") or stripped.startswith("//") or stripped.startswith("*"):
                continue
            refs.append(SymbolReference(
                file=target_file.file if hasattr(target_file, 'file') else target_file.path,
                line_number=line_no,
                line_text=stripped[:120],
            ))
            if len(refs) >= _MAX_REFS_PER_SYMBOL:
                break

    return refs


def build_cross_file_context(files: list[FileReviewData]) -> list[CrossFileRelationship]:
    """Analyse all PR files and return cross-file relationships.

    For each file with changed symbols, searches all *other* files for
    references to those symbols.
    """
    if len(files) < 2:
        return []

    # Step 1: extract changed symbols per file.
    file_symbols: dict[str, list[str]] = {}
    for f in files:
        symbols = _extract_changed_symbols(f)
        if symbols:
            file_symbols[f.path] = symbols

    if not file_symbols:
        return []

    # Step 2: for each symbol, search other files for references.
    relationships: list[CrossFileRelationship] = []
    files_by_path = {f.path: f for f in files}

    for source_path, symbols in file_symbols.items():
        for symbol in symbols:
            rel = CrossFileRelationship(symbol=symbol, defined_in=source_path)
            for target_path, target_file in files_by_path.items():
                if target_path == source_path:
                    continue
                refs = _find_references(symbol, target_file)
                rel.references.extend(refs)

            if rel.references:
                relationships.append(rel)

    logger.info(
        "Cross-file context: %d symbol(s) with cross-file references",
        len(relationships),
    )

    return relationships


def render_cross_file_context(relationships: list[CrossFileRelationship]) -> str:
    """Render relationships as a markdown section for the LLM prompt."""
    if not relationships:
        return ""

    lines = [
        "## Cross-file relationships",
        "",
        "The following symbols were changed and are referenced in other files in this PR. "
        "Pay special attention to whether callers/consumers are updated consistently.",
        "",
    ]

    total_lines = 0
    truncated = False
    for rel in relationships:
        if total_lines >= _MAX_RELATIONSHIP_LINES:
            truncated = True
            break

        lines.append(f"**`{rel.symbol}`** (changed in `{rel.defined_in}`) is referenced in:")
        total_lines += 1
        for ref in rel.references:
            if total_lines >= _MAX_RELATIONSHIP_LINES:
                truncated = True
                break
            lines.append(f"- `{ref.file}:{ref.line_number}` — `{ref.line_text}`")
            total_lines += 1
        lines.append("")

    if truncated:
        lines.append("_(additional relationships truncated)_")

    return "\n".join(lines)
