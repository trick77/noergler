"""Markdown text reflow for Bitbucket comments.

Bitbucket comments render LLM free-text prose as one continuous paragraph that
runs the full browser width, which is hard to read on wide monitors. We hard-wrap
that prose at a fixed column so no line runs full width.

Bitbucket Data Center's CommonMark renderer treats a single newline inside a
paragraph as a visible line break (verified empirically against the target
instance), so ``HARD_BREAK`` is just ``"\\n"``. If a future instance needs a
different break token (e.g. ``"  \\n"`` two trailing spaces, or ``"\\\\\\n"``
backslash), change it here — it is the single point of control.

A forced break the renderer cannot undo will double-wrap on views narrower than
``WRAP_WIDTH`` (mobile, side-by-side diff). That trade-off is accepted: the goal
is a hard cap on line width, not graceful reflow.
"""

import re
import textwrap

# Target column width. ~66 reads better than 80 for prose. Kept a module constant
# so it is trivial to promote to config later.
WRAP_WIDTH = 66

# Token inserted between wrapped pieces of a paragraph. See module docstring.
HARD_BREAK = "\n"

# Lines we never reflow: blank, fence delimiters, headings, block quotes, and
# list items (bullet or numbered). Wrapping these would mangle block structure.
_STRUCTURAL = re.compile(r"^\s*([-*+>]\s|#{1,6}\s|\d+\.\s)")

# Inline code spans — masked before wrapping so a span with internal spaces
# (e.g. `foo bar`) is never split across a break, then restored verbatim.
_CODE_SPAN = re.compile(r"`[^`]+`")
_MASK_CHAR = "\x00"


def wrap_prose(text: str, width: int = WRAP_WIDTH) -> str:
    """Hard-wrap plain prose lines at ``width``; pass structural lines through.

    Idempotent. A short string (already under ``width``) is returned unchanged.
    Headings, list items, block quotes and fenced code blocks are left intact.
    Inline code spans and long unbreakable tokens (URLs, ``path/to/file``) are
    never split.
    """
    if not text:
        return text

    out: list[str] = []
    in_fence = False
    for line in text.split("\n"):
        stripped = line.lstrip()
        if stripped.startswith("```"):
            in_fence = not in_fence
            out.append(line)
            continue
        if in_fence or not stripped or _STRUCTURAL.match(line):
            out.append(line)
            continue
        out.append(_wrap_line(line, width))
    return "\n".join(out)


def _wrap_line(line: str, width: int) -> str:
    """Wrap a single prose line, masking inline code spans first."""
    spans: list[str] = []

    def _mask(match: re.Match[str]) -> str:
        spans.append(match.group(0))
        # Same length so wrapped width stays accurate; no spaces so textwrap
        # treats it as one unbreakable token.
        return _MASK_CHAR * len(match.group(0))

    masked = _CODE_SPAN.sub(_mask, line)
    pieces = textwrap.wrap(
        masked,
        width=width,
        break_long_words=False,  # never chop a URL / config.py mid-token
        break_on_hyphens=False,  # never split well-known or path-like tokens
    )
    if not pieces:
        return line
    wrapped = HARD_BREAK.join(pieces)
    if spans:
        it = iter(spans)
        wrapped = re.sub(f"{_MASK_CHAR}+", lambda _: next(it), wrapped)
    return wrapped
