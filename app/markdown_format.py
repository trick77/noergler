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
is to cap line width, not graceful reflow. A single unbreakable token (URL,
``path/to/file``) is left to overflow rather than being chopped, so the cap is a
target, not an absolute guarantee.

Idempotency and the structural pass-through both assume ``HARD_BREAK == "\\n"``
(``wrap_prose`` re-splits on ``"\\n"``, exactly reversing the join). Changing
``HARD_BREAK`` to a multi-char token requires updating the splitter accordingly.
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
# Placeholders are index-encoded (\x00<n>\x00) so adjacent spans stay distinct
# and restore maps each back to the right span. NUL never renders, so we strip
# any pre-existing NUL from input to keep the placeholder space collision-free.
_CODE_SPAN = re.compile(r"`[^`]+`")
_PLACEHOLDER = re.compile("\x00(\\d+)\x00")


def wrap_prose(text: str, width: int = WRAP_WIDTH) -> str:
    """Hard-wrap plain prose lines at ``width``; pass structural lines through.

    Idempotent. A short string (already under ``width``) is returned unchanged.
    Headings, list items, block quotes and fenced code blocks are left intact.
    Inline code spans and long unbreakable tokens (URLs, ``path/to/file``) are
    never split.
    """
    if not text:
        return text

    # NUL never renders and would collide with our mask placeholders below.
    text = text.replace("\x00", "")

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
        idx = len(spans)
        spans.append(match.group(0))
        # Index-encoded, no spaces → textwrap keeps it as one unbreakable token,
        # and adjacent spans stay separately addressable on restore.
        return f"\x00{idx}\x00"

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
        wrapped = _PLACEHOLDER.sub(lambda m: spans[int(m.group(1))], wrapped)
    return wrapped
