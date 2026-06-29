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

Re-wrapping is idempotent with one bounded exception: a list-item continuation
line consisting solely of a single token wider than the column (e.g. a bare long
URL) loses its hang-indent on a second pass, because on re-entry such a line is
indistinguishable from plain prose and ``textwrap`` drops the leading indent.
This is cosmetic (content and code spans are untouched) and unreached in
practice — every caller wraps fresh model output exactly once.
"""

import re
import textwrap

# Target column width. Kept a module constant so it is trivial to promote to
# config later.
WRAP_WIDTH = 110

# Token inserted between wrapped pieces of a paragraph. See module docstring.
HARD_BREAK = "\n"

# Lines we never reflow: fence delimiters, headings and block quotes. Wrapping
# these would mangle block structure.
_NEVER_WRAP = re.compile(r"^\s*([>]\s|#{1,6}\s)")

# List items (bullet or numbered) — captured as (prefix, content) so the content
# can be wrapped while the marker is preserved and continuation lines are
# hang-indented under the text.
_LIST_ITEM = re.compile(r"^(\s*(?:[-*+]|\d+\.)\s+)(.*)$")

# Inline code spans — masked before wrapping so a span with internal spaces
# (e.g. `foo bar`) is never split across a break, then restored verbatim.
# Placeholders are index-encoded (\x00<n>\x00) and padded with \x01 fillers to
# the span's real length, so textwrap counts the span's true column width (an
# unpadded placeholder is far shorter than the span and makes lines overflow).
# Adjacent spans stay distinct and restore maps each back to the right span.
# Neither \x00 nor \x01 renders, so we strip any pre-existing occurrence from
# input to keep the placeholder space collision-free.
_CODE_SPAN = re.compile(r"`[^`]+`")
_PLACEHOLDER = re.compile("\x00(\\d+)\x00\x01*")


def wrap_prose(text: str, width: int = WRAP_WIDTH) -> str:
    """Hard-wrap plain prose lines at ``width``; pass structural lines through.

    Idempotent (see the module docstring for the one bounded list-item
    exception). A short string (already under ``width``) is returned unchanged.
    Headings, block quotes and fenced code blocks are left intact; list items
    are wrapped with their marker preserved and continuation lines hang-indented
    under the text. Inline code spans and long unbreakable tokens (URLs,
    ``path/to/file``) are never split.
    """
    if not text:
        return text

    # \x00 / \x01 never render and would collide with our mask placeholders.
    text = text.replace("\x00", "").replace("\x01", "")

    out: list[str] = []
    in_fence = False
    for line in text.split("\n"):
        stripped = line.lstrip()
        if stripped.startswith("```"):
            in_fence = not in_fence
            out.append(line)
            continue
        if in_fence or not stripped or _NEVER_WRAP.match(line):
            out.append(line)
            continue
        item = _LIST_ITEM.match(line)
        if item:
            prefix, content = item.group(1), item.group(2)
            out.append(_wrap_line(content, width, prefix=prefix))
        else:
            out.append(_wrap_line(line, width))
    return "\n".join(out)


def _wrap_line(line: str, width: int, prefix: str = "") -> str:
    """Wrap a single line, masking inline code spans first.

    ``prefix`` is a list marker (e.g. ``"- "`` or ``"1. "``); when given it is
    placed on the first wrapped piece and continuation lines are hang-indented
    by the same width. (Bitbucket's CommonMark renderer may collapse the leading
    spaces on the rendered ``<br>`` continuation, so the indent is a raw-markdown
    nicety; the width cap is the real goal.)
    """
    spans: list[str] = []

    def _mask(match: re.Match[str]) -> str:
        idx = len(spans)
        span = match.group(0)
        spans.append(span)
        # Index-encoded, no spaces → textwrap keeps it as one unbreakable token,
        # and adjacent spans stay separately addressable on restore. Pad with
        # \x01 to the span's real length so textwrap counts its true width.
        base = f"\x00{idx}\x00"
        return base + "\x01" * max(0, len(span) - len(base))

    masked = _CODE_SPAN.sub(_mask, line)
    pieces = textwrap.wrap(
        masked,
        width=width,
        break_long_words=False,  # never chop a URL / config.py mid-token
        break_on_hyphens=False,  # never split well-known or path-like tokens
        initial_indent=prefix,
        subsequent_indent=" " * len(prefix),
    )
    if not pieces:
        return prefix + line
    wrapped = HARD_BREAK.join(pieces)
    if spans:
        wrapped = _PLACEHOLDER.sub(lambda m: spans[int(m.group(1))], wrapped)
    return wrapped
