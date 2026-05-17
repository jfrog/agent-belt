# (c) JFrog Ltd. (2026)

"""Tests for the run-phase footer's agent-error counter.

The historic ``Run: N scenarios, 0 errors`` footer counted only
``ScenarioResult.error`` (subprocess crash). Auth failures, refusals,
and rate-limits left ``error=None`` so the headline misled the user
into believing the run was clean. The footer now also surfaces
``ScenarioResult.agent_errors``.
"""

from __future__ import annotations

import io

from rich.console import Console

from belt.progress import RunnerProgress
from belt.runner.entities import ScenarioResult


def _summary_to_string(results: list[ScenarioResult]) -> str:
    buf = io.StringIO()
    progress = RunnerProgress(
        plain=True,
        console=Console(file=buf, force_terminal=False, width=120, no_color=True),
    )
    # ``summary`` only reads ``_start_time``; bypass ``start()`` (which
    # mutates several other attributes) so the test exercises just the
    # rendering branch under inspection.
    import time as _time

    progress._start_time = _time.monotonic()
    progress.summary(results)
    return buf.getvalue()


class TestRunSummary:
    def test_clean_run(self) -> None:
        results = [ScenarioResult(scenario_name="s1", group_path="g")]
        out = _summary_to_string(results)
        assert "0 errors" in out
        assert "agent error" not in out

    def test_subprocess_error_takes_precedence(self) -> None:
        # If the subprocess crashed, we surface that first - it's
        # actionable for the harness operator and usually subsumes any
        # agent-level error.
        results = [
            ScenarioResult(scenario_name="s1", group_path="g", error="boom"),
            ScenarioResult(scenario_name="s2", group_path="g", agent_errors=["authentication_failed"]),
        ]
        out = _summary_to_string(results)
        assert "1 error(s)" in out
        # Agent-error label is suppressed in this branch (one signal at a time).
        assert "agent error(s)" not in out

    def test_agent_error_when_no_subprocess_error(self) -> None:
        results = [
            ScenarioResult(scenario_name="s1", group_path="g", agent_errors=["authentication_failed"]),
            ScenarioResult(scenario_name="s2", group_path="g", agent_errors=["authentication_failed"]),
            ScenarioResult(scenario_name="s3", group_path="g"),
        ]
        out = _summary_to_string(results)
        assert "2 agent error(s)" in out
        assert "authentication_failed" in out

    def test_mixed_error_types_listed(self) -> None:
        results = [
            ScenarioResult(scenario_name="s1", group_path="g", agent_errors=["authentication_failed"]),
            ScenarioResult(scenario_name="s2", group_path="g", agent_errors=["rate_limited"]),
        ]
        out = _summary_to_string(results)
        assert "authentication_failed" in out
        assert "rate_limited" in out
