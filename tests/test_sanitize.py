# (c) JFrog Ltd. (2026)

"""Unit tests for :mod:`belt._sanitize`.

Pinning behaviour here matters because every Markdown / Rich sink in
the project depends on these helpers stripping the same byte classes.
A drift to a hand-rolled regex in any caller would re-open the
injection surface that this module exists to close.
"""

from __future__ import annotations

import pytest

from belt._sanitize import sanitize, strip_ansi, strip_controls


class TestStripAnsi:
    def test_strips_csi_sequence(self):
        assert strip_ansi("\x1b[31mhello\x1b[0m") == "hello"

    def test_strips_csi_with_parameters_intermediates_and_final(self):
        # Full CSI grammar: parameter bytes [0-?], intermediate bytes [ -/],
        # final byte [@-~]. A drift to the narrower ``[0-9;]*[a-zA-Z]`` regex
        # used by some legacy callers would leak DEC-private mode set/reset
        # sequences into rendered output.
        assert strip_ansi("\x1b[?25l\x1b[?25h") == ""

    def test_strips_osc_hyperlink(self):
        # OSC 8 (terminal hyperlink) closed with BEL.
        assert strip_ansi("\x1b]8;;https://x\x07click\x1b]8;;\x07") == "click"

    def test_strips_osc_terminated_with_string_terminator(self):
        # OSC closed with ESC \\ (ST).
        assert strip_ansi("\x1b]0;title\x1b\\rest") == "rest"

    def test_returns_input_unchanged_when_no_escapes(self):
        assert strip_ansi("plain text") == "plain text"

    def test_simulate_erase_takes_segment_after_final_erase_line(self):
        # Spinner pattern: each frame writes a partial line and an erase.
        # The user only ever sees the final segment.
        spinner = "\x1b[2Kloading 1...\x1b[2Kloading 2...\x1b[2Kdone"
        assert strip_ansi(spinner, simulate_erase=True) == "done"

    def test_simulate_erase_strips_carriage_returns(self):
        assert strip_ansi("foo\rbar", simulate_erase=True) == "foobar"

    def test_simulate_erase_no_op_without_erase_line(self):
        assert strip_ansi("plain", simulate_erase=True) == "plain"

    def test_does_not_remove_unicode(self):
        # Agent names and judge reasoning often contain emoji / accented
        # characters; preserve them.
        assert strip_ansi("Goose ⚡ 1.4 - fast") == "Goose ⚡ 1.4 - fast"


class TestStripControls:
    def test_strips_c0_controls(self):
        assert strip_controls("a\x00b\x07c\x1fd") == "abcd"

    def test_strips_del(self):
        assert strip_controls("a\x7fb") == "ab"

    def test_strips_embedded_newline_and_tab(self):
        # Embedded controls would split a Markdown table row.
        assert strip_controls("col1\nstill col1\tstill col1") == "col1still col1still col1"

    def test_preserves_unicode(self):
        assert strip_controls("⚡ ✅ Goose") == "⚡ ✅ Goose"

    def test_empty_input(self):
        assert strip_controls("") == ""


class TestSanitize:
    def test_combines_strip_ansi_and_strip_controls(self):
        # ANSI escape, then a NUL byte that survives ANSI stripping.
        assert sanitize("\x1b[31mhi\x1b[0m\x00world") == "hiworld"

    def test_simulate_erase_propagates_through_pipeline(self):
        spinner = "\x1b[2K\x1b[31mready\x1b[0m"
        assert sanitize(spinner, simulate_erase=True) == "ready"

    def test_idempotent_on_clean_input(self):
        # A second pass through the pipeline must be a no-op so callers
        # can apply :func:`sanitize` defensively without risk.
        clean = "no escapes here"
        assert sanitize(sanitize(clean)) == clean


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("", ""),
        ("\x1b[31m", ""),
        ("\x00\x01\x02", ""),
        ("\x1b[31m\x00plain", "plain"),
        ("ascii only", "ascii only"),
    ],
)
def test_sanitize_table(raw: str, expected: str) -> None:
    assert sanitize(raw) == expected
