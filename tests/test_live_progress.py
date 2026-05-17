# (c) JFrog Ltd. (2026)

"""Tests for ``LiveProgress`` - the ``--progress live`` panel renderer.

These tests cover the bottom-panel ("Live Output") rendering pipeline that
consumes ``StreamEvent`` summaries and surfaces them in the integrated
Rich ``Live`` display. They drive the panel without spawning the poller
thread, so behaviour is deterministic.

What we assert:
  - Trusted framework markup (e.g. ``[green]$0.42[/green]`` from result
    events) renders as styled text - the literal ``[green]`` substring
    must never appear in the visible output.
  - Agent-controlled markup that smuggled through must surface as literal
    characters - never interpreted as Rich styling.
  - A single malformed line cannot poison the panel: other lines still
    render correctly.
  - Length-based truncation is in place so long agent stdout cannot
    blow up the panel width.
"""

from __future__ import annotations

from io import StringIO
from pathlib import Path

from rich.console import Console

from belt._safe import rich_safe
from belt.progress import LiveProgress


def _capture_panel(panel) -> str:
    """Render a Rich panel into a deterministic string for assertion.

    ``no_color=True`` strips ANSI styling so the captured text contains the
    visible characters, but any markup-parsing failure still surfaces as
    literal ``[green]`` substrings - which is exactly what we want to check.
    """
    console = Console(file=StringIO(), width=120, force_terminal=True, no_color=True)
    console.print(panel)
    return console.file.getvalue()


def _make_live(tmp_path: Path, *, max_lines: int = 20) -> LiveProgress:
    """Build a ``LiveProgress`` without starting its poller thread."""
    lp = LiveProgress(run_dir=tmp_path, console=Console(file=StringIO(), width=120), max_lines=max_lines)
    lp._scenarios_dir = tmp_path
    return lp


class TestLiveProgressTrustedMarkup:
    """``_render_result`` events carry trusted markup; the panel must render
    that markup as styled text, not as literal ``[green]`` substrings."""

    def test_result_event_renders_cost_styled(self, tmp_path: Path):
        lp = _make_live(tmp_path)
        lp._scenario_order = ["scen-A"]
        # The poller code path stores trusted-markup summaries unchanged
        # (no rich_safe escape) when ``event.summary_is_markup`` is True.
        lp._scenario_streams = {
            "scen-A": [
                f"  🔧 {rich_safe('Read(path=src/main.py)')}",
                "  ✅ result: [green]$0.4173[/green], 71.4s",
            ]
        }
        lp._scenario_pin_indices = {"scen-A": set()}

        out = _capture_panel(lp._build_stream_panel())

        assert "[green]" not in out, f"trusted markup leaked as literal text: {out!r}"
        assert "[/green]" not in out, f"trusted markup leaked as literal text: {out!r}"
        assert "$0.4173" in out
        assert "71.4s" in out
        assert "Read" in out

    def test_error_result_renders_without_leak(self, tmp_path: Path):
        lp = _make_live(tmp_path)
        lp._scenario_order = ["scen-A"]
        lp._scenario_streams = {
            "scen-A": ["  ❌ result: 1.0s, ERROR"],
        }
        lp._scenario_pin_indices = {"scen-A": set()}

        out = _capture_panel(lp._build_stream_panel())

        assert "ERROR" in out
        assert "[green]" not in out

    def test_multiple_scenarios_each_render_correctly(self, tmp_path: Path):
        lp = _make_live(tmp_path)
        lp._scenario_order = ["scen-A", "scen-B"]
        lp._scenario_streams = {
            "scen-A": ["  ✅ result: [green]$0.10[/green], 1.0s"],
            "scen-B": ["  ✅ result: [green]$0.20[/green], 2.0s"],
        }
        lp._scenario_pin_indices = {"scen-A": set(), "scen-B": set()}

        out = _capture_panel(lp._build_stream_panel())

        assert "$0.10" in out
        assert "$0.20" in out
        assert "[green]" not in out
        assert "scen-A" in out
        assert "scen-B" in out


class TestLiveProgressAttackerSurface:
    """Agent-controlled summary content must not be parsed as Rich markup."""

    def test_agent_injected_markup_does_not_style(self, tmp_path: Path):
        lp = _make_live(tmp_path)
        lp._scenario_order = ["scen-A"]
        # The poller code path passes plain (non-markup) summaries through
        # ``rich_safe`` before storing them. Simulate that exactly.
        evil = "evil_tool([red]injected[/red])"
        lp._scenario_streams = {
            "scen-A": [f"  🔧 {rich_safe(evil)}"],
        }
        lp._scenario_pin_indices = {"scen-A": set()}

        out = _capture_panel(lp._build_stream_panel())

        # Either the literal brackets survive or the markup-tokens become
        # invisible-but-not-stylized; either way no styling should be applied.
        # The key contract: the text [red]injected[/red] does NOT cause Rich
        # to render "injected" with red styling; it appears as plain text.
        assert "injected" in out
        # rich_safe escapes [ to \[, which renders as a literal [ in the
        # output - so the original brackets must surface as visible chars.
        assert "[red]" in out, f"escaped markup must show as literal text: {out!r}"


class TestLiveProgressResilience:
    """A single malformed line must not poison the entire panel."""

    def test_unbalanced_markup_does_not_break_other_lines(self, tmp_path: Path):
        lp = _make_live(tmp_path)
        lp._scenario_order = ["scen-A"]
        # A line with broken trusted-markup (unclosed tag) sneaks through.
        # The other lines must still render their content.
        lp._scenario_streams = {
            "scen-A": [
                "  ✅ result: [green]$0.10[/green], 1.0s",
                "  💬 [bold unclosed tag",  # malformed - no [/bold] anywhere
                "  💬 next line content here",
            ],
        }
        lp._scenario_pin_indices = {"scen-A": set()}

        out = _capture_panel(lp._build_stream_panel())

        # Both well-formed lines must appear in the visible output.
        assert "$0.10" in out, f"good line dropped because of nearby malformed line: {out!r}"
        assert "next line content here" in out, "malformed line poisoned the panel: " f"{out!r}"

    def test_empty_scenario_streams_renders_waiting_message(self, tmp_path: Path):
        lp = _make_live(tmp_path)
        # No scenarios registered yet.
        lp._scenario_order = []
        lp._scenario_streams = {}

        out = _capture_panel(lp._build_stream_panel())

        assert "Waiting" in out


class TestLiveProgressTruncation:
    """Long agent stdout must be truncated; truncation must not cascade
    into a panel-wide rendering failure."""

    def test_long_lines_get_clamped_with_ellipsis(self, tmp_path: Path):
        lp = _make_live(tmp_path)
        lp._scenario_order = ["scen-A"]
        long_line = "  💬 " + ("x" * 500)  # well over panel width
        lp._scenario_streams = {"scen-A": [long_line]}
        lp._scenario_pin_indices = {"scen-A": set()}

        out = _capture_panel(lp._build_stream_panel())

        # Output should contain the ellipsis indicator from _clamp.
        assert "…" in out

    def test_panel_renders_for_seven_scenarios(self, tmp_path: Path):
        """Smoke test: a realistic batch of 7 scenarios all render, none drop."""
        lp = _make_live(tmp_path, max_lines=70)
        labels = [f"scen-{i:02d}" for i in range(7)]
        lp._scenario_order = list(labels)
        lp._scenario_streams = {
            label: [
                f"  👤 {rich_safe('Run task: ' + label)}",
                f"  🔧 {rich_safe('Read(path=src/' + label + '.py)')}",
                f"  💬 {rich_safe('Reply for ' + label)}",
                f"  ✅ result: [green]${0.10 + i * 0.01:.4f}[/green], {1.0 + i * 0.5:.1f}s",
            ]
            for i, label in enumerate(labels)
        }
        lp._scenario_pin_indices = {label: set() for label in labels}

        out = _capture_panel(lp._build_stream_panel())

        # Every scenario shows up.
        for label in labels:
            assert label in out, f"missing scenario {label} in panel output"
        # No markup leaks anywhere.
        assert "[green]" not in out
        assert "[/green]" not in out
