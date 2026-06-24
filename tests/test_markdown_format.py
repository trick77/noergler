"""Tests for app.markdown_format.wrap_prose."""

from app.markdown_format import HARD_BREAK, WRAP_WIDTH, wrap_prose


def _visible_lines(text: str) -> list[str]:
    """Split a wrapped result back into rendered lines."""
    return text.replace(HARD_BREAK, "\n").split("\n")


def test_long_prose_wraps_under_width():
    text = (
        "This change reworks the authentication flow so that tokens are "
        "refreshed lazily instead of eagerly, which avoids a thundering herd "
        "of refresh requests when many sessions expire at once."
    )
    wrapped = wrap_prose(text)
    assert wrapped != text  # something was broken up
    for line in _visible_lines(wrapped):
        assert len(line) <= WRAP_WIDTH


def test_short_string_is_noop():
    text = "Looks good."
    assert wrap_prose(text) == text


def test_empty_string():
    assert wrap_prose("") == ""


def test_idempotent():
    text = (
        "The repository layer now batches writes, which materially reduces "
        "round trips to PostgreSQL under sustained review load on large PRs."
    )
    once = wrap_prose(text)
    assert wrap_prose(once) == once


def test_long_token_not_split():
    url = "https://bitbucket.example.com/projects/FOO/repos/bar/pull-requests/42"
    text = f"See the discussion at {url} for the full rationale here."
    wrapped = wrap_prose(text)
    assert url in wrapped  # URL survived intact, never chopped mid-token


def test_code_span_with_spaces_preserved():
    text = (
        "The handler now calls `do thing` before returning, which keeps the "
        "ordering stable across retries and avoids a subtle race condition."
    )
    wrapped = wrap_prose(text)
    assert "`do thing`" in wrapped  # span not split across a break


def test_adjacent_code_spans_both_preserved():
    text = (
        "Compare `alpha` and `beta` carefully here because the ordering of "
        "those two calls genuinely matters for correctness in this path."
    )
    wrapped = wrap_prose(text)
    assert "`alpha`" in wrapped
    assert "`beta`" in wrapped


def test_stray_nul_in_input_does_not_crash():
    text = (
        "A `real` span and a stray \x00 null byte in otherwise normal prose "
        "that is deliberately long enough to force the wrapper to wrap it."
    )
    wrapped = wrap_prose(text)  # must not raise
    assert "`real`" in wrapped
    assert "\x00" not in wrapped


def test_structural_lines_pass_through():
    text = (
        "# Heading\n"
        "- a bullet item that is quite long but must stay on one structural line\n"
        "> a block quote that is also long and should not be reflowed by us here\n"
        "1. a numbered item that is similarly long and must remain untouched too"
    )
    assert wrap_prose(text) == text


def test_fenced_code_block_untouched():
    text = (
        "```\n"
        "def f(x): return x + 1  # a deliberately long comment that exceeds width here\n"
        "```"
    )
    assert wrap_prose(text) == text
