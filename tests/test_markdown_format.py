"""Tests for app.markdown_format.wrap_prose."""

from app.markdown_format import (
    HARD_BREAK,
    LIST_WRAP_WIDTH,
    WRAP_WIDTH,
    wrap_prose,
)


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


def test_code_span_counts_toward_width():
    # A long code span used to be masked to a ~3-char placeholder, so textwrap
    # undercounted the line and let it overflow. The placeholder is now padded
    # to the span's real length, so every visible line stays within the cap.
    span = "`ServiceLeistungsfallDatenschutz`"
    text = (
        f"Removes the personengruppe Fremdfall privacy workaround from {span}, "
        "including the injected feature-flag usage entirely from the service."
    )
    wrapped = wrap_prose(text)
    assert wrapped != text  # the span line was broken up
    for line in _visible_lines(wrapped):
        assert len(line) <= WRAP_WIDTH


def test_stray_control_bytes_in_input_do_not_crash():
    text = (
        "A `real` span and a stray \x00 null \x01 byte in otherwise normal "
        "prose that is deliberately long enough to force the wrapper to wrap."
    )
    wrapped = wrap_prose(text)  # must not raise
    assert "`real`" in wrapped
    assert "\x00" not in wrapped
    assert "\x01" not in wrapped


def test_headings_and_block_quotes_pass_through():
    text = (
        "# Heading\n"
        "> a block quote that is also long and should not be reflowed by us here"
    )
    assert wrap_prose(text) == text


def test_long_list_item_wraps_with_hang_indent():
    text = (
        "- Access-control logic is simplified by removing the temporary branch "
        "and relying on the existing freigaben check across the whole service."
    )
    wrapped = wrap_prose(text)
    lines = _visible_lines(wrapped)
    assert len(lines) > 1  # the long bullet was wrapped
    assert lines[0].startswith("- ")  # marker preserved on the first line
    for line in lines[1:]:
        assert line.startswith("  ")  # continuation hang-indented under text
    for line in lines:
        assert len(line) <= WRAP_WIDTH


def test_long_numbered_item_wraps():
    text = (
        "1. The updated test covers the changed Fremdfall behavior for the "
        "restricted personengruppe freigaben path and asserts the new return."
    )
    wrapped = wrap_prose(text)
    lines = _visible_lines(wrapped)
    assert len(lines) > 1
    assert lines[0].startswith("1. ")
    for line in lines:
        assert len(line) <= WRAP_WIDTH


def test_list_items_wrap_at_narrower_width():
    # A bullet whose content fits within WRAP_WIDTH but exceeds LIST_WRAP_WIDTH
    # must still wrap (proving list items use the narrower cap), and every line
    # stays within LIST_WRAP_WIDTH. The identical text as plain prose, which is
    # under WRAP_WIDTH, stays a single line — confirming the two caps differ.
    body = (
        "Access-control logic is simplified by removing the temporary branch "
        "and relying on existing checks."
    )
    assert LIST_WRAP_WIDTH < len(body) <= WRAP_WIDTH

    bullet = wrap_prose(f"- {body}")
    lines = _visible_lines(bullet)
    assert len(lines) > 1  # the bullet wrapped at the narrower cap
    for line in lines:
        assert len(line) <= LIST_WRAP_WIDTH

    assert wrap_prose(body) == body  # same text as plain prose is untouched


def test_code_span_in_list_item_expanded():
    # Placeholders for masked code spans must be restored within the list item,
    # even at the narrower list width.
    text = (
        "- The handler now calls `do thing` before returning so the ordering "
        "stays stable across retries here."
    )
    wrapped = wrap_prose(text)
    assert "`do thing`" in wrapped  # placeholder expanded, span intact
    assert "\x00" not in wrapped and "\x01" not in wrapped
    for line in _visible_lines(wrapped):
        assert len(line) <= LIST_WRAP_WIDTH


def test_paren_and_unicode_bullet_markers_wrap_at_list_width():
    # Regression: ``N)`` numbered items and the unicode bullet ``•`` used to miss
    # the list-item regex and fall through to the wider plain-prose cap, so they
    # never wrapped at LIST_WRAP_WIDTH. Every supported marker must use the
    # narrower list cap.
    body = (
        "Database migration copies existing values from arbeitsvertrag to the "
        "new schadenmeldung column before the entity mapping switches over."
    )
    assert len(body) > LIST_WRAP_WIDTH  # long enough to require wrapping
    for marker in ("- ", "* ", "+ ", "1. ", "1) ", "• "):
        wrapped = wrap_prose(f"{marker}{body}")
        lines = _visible_lines(wrapped)
        assert len(lines) > 1, marker
        for line in lines:
            assert len(line) <= LIST_WRAP_WIDTH, (marker, line)


def test_short_list_item_is_noop():
    text = "- Looks good."
    assert wrap_prose(text) == text


def test_fenced_code_block_untouched():
    text = (
        "```\n"
        "def f(x): return x + 1  # a deliberately long comment that exceeds width here\n"
        "```"
    )
    assert wrap_prose(text) == text
