# (c) JFrog Ltd. (2026)

"""Tests for the doctor's auth-signal hedge.

The doctor command historically rendered a green "✓" with the bare
label ``stored login`` for any agent whose credential file existed on
disk. Users (legitimately) read this as "I am authenticated", but the
:meth:`BaseAgentAdapter.check_available` contract forbids any actual
credential validation - the file may contain an expired token. These
tests pin down the new presence-only hedge plus the section footer.
"""

from __future__ import annotations

import io

from rich.console import Console

from belt.commands.doctor import CheckResult, DoctorReport, _format_check, _hedge_auth_signal, print_report


class TestHedgeAuthSignal:
    def test_stored_login_is_hedged(self) -> None:
        out = _hedge_auth_signal("stored login (~/.claude/token)")
        assert "(presence only" in out
        assert "stored login (~/.claude/token)" in out

    def test_env_signal_is_not_hedged(self) -> None:
        # Env signals are stronger than file signals (the user is
        # actively presenting a value to the process), so we don't
        # downgrade them with the same hedge.
        assert _hedge_auth_signal("env CLAUDE_API_KEY") == "env CLAUDE_API_KEY"

    def test_unknown_format_passes_through(self) -> None:
        # Defensive: unknown signal shapes (e.g. future formats) pass
        # through unchanged rather than getting a misleading hedge.
        assert _hedge_auth_signal("custom-signal") == "custom-signal"


class TestFormatCheckHedge:
    def test_hedge_appears_in_rendered_output(self) -> None:
        c = CheckResult(
            ok=True,
            label="claude-code",
            detail="ready",
            auth_signals=["stored login (~/.claude/token)"],
        )
        line = _format_check(c)
        assert "(presence only" in line


def _print_to_string(report: DoctorReport) -> str:
    buf = io.StringIO()
    console = Console(file=buf, force_terminal=False, width=120, no_color=True)
    print_report(report, console=console)
    return buf.getvalue()


class TestPrintReportFooter:
    def test_footer_appears_when_any_stored_login_signal(self) -> None:
        report = DoctorReport(
            agent_checks=[
                CheckResult(
                    ok=True,
                    label="claude-code",
                    detail="ready",
                    auth_signals=["stored login (~/.claude/token)"],
                ),
            ]
        )
        out = _print_to_string(report)
        assert "Auth signals indicate credential presence" in out
        assert "re-authenticate" in out.lower()

    def test_footer_absent_when_only_env_signals(self) -> None:
        report = DoctorReport(
            agent_checks=[
                CheckResult(
                    ok=True,
                    label="codex",
                    detail="ready",
                    auth_signals=["env OPENAI_API_KEY"],
                ),
            ]
        )
        out = _print_to_string(report)
        assert "presence" not in out

    def test_footer_absent_when_no_signals(self) -> None:
        report = DoctorReport(agent_checks=[CheckResult(ok=True, label="a", detail="ready")])
        out = _print_to_string(report)
        assert "presence" not in out
