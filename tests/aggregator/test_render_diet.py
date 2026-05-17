# (c) JFrog Ltd. (2026)

"""Verbosity contract for the terminal failure renderer.

Default (``WARNING``) must produce a lean ``Failed:`` block: one
``rule:`` line per failing check and a compact ``llm: dim=score`` summary
- never the full judge reasoning paragraph, the last-500-chars response
tail, or per-evidence ``-> path`` lines. ``-v`` (``INFO``) flips
those back on.

This pin is the regression vector that produced ~840 lines of output
for a single group-level configuration error in the issue that drove
the eval-failure-UX work.
"""

from __future__ import annotations

import io

import pytest
from rich.console import Console

from belt import envvars
from belt.aggregator.render_terminal import print_terminal
from belt.entities import ScenarioScore
from belt.scorer.payloads import CheckEntry, LLMDimensionVerdict, LLMPayload, RulesPayload


@pytest.fixture(autouse=True)
def _clear_log_level_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(envvars.LOG_LEVEL, raising=False)


def _failing_score() -> ScenarioScore:
    """A scenario that fails both rules and an LLM dimension.

    Includes a deliberately long reasoning string so the test can
    cleanly distinguish the lean and verbose paths by its presence.
    """
    return ScenarioScore(
        scenario_name="s",
        group="g",
        tags=[],
        scores={
            "rules": RulesPayload(
                checks=[
                    CheckEntry(
                        turn_idx=0,
                        dimension="execution",
                        check="no_errors",
                        passed=False,
                        details="auth failure",
                    ),
                ],
                passed=False,
            ),
            "llm": LLMPayload(
                overall_pass=False,
                dimensions={
                    "correctness": LLMDimensionVerdict(
                        score="low",
                        reasoning="VERY_LONG_JUDGE_REASONING_THAT_SHOULD_NOT_LEAK_AT_WARNING",
                    ),
                },
            ),
        },
        overall_pass=False,
    )


def _capture(scores: list[ScenarioScore]) -> str:
    buf = io.StringIO()
    print_terminal(scores, run_label="run-x", console=Console(file=buf, width=120, force_terminal=False))
    return buf.getvalue()


class TestRenderDietDefaultWarning:
    """At the default ``WARNING`` level the renderer must NOT include
    judge reasoning prose or the response-tail snippet."""

    def test_omits_judge_reasoning_prose(self) -> None:
        out = _capture([_failing_score()])
        assert "VERY_LONG_JUDGE_REASONING_THAT_SHOULD_NOT_LEAK_AT_WARNING" not in out

    def test_keeps_rule_one_liner(self) -> None:
        # The lean view still shows what broke (rule one-liner). It
        # hides why (judge prose), which is one ``belt view`` away.
        out = _capture([_failing_score()])
        assert "execution/no_errors" in out

    def test_keeps_llm_score_summary(self) -> None:
        # A compact ``llm: correctness=low`` summary replaces the full
        # reasoning paragraph - the reader sees the verdict, not the
        # essay.
        out = _capture([_failing_score()])
        assert "correctness=low" in out


class TestRenderDietVerbose:
    """At ``INFO`` (``-v``) the renderer restores the prose and snippets."""

    def test_info_level_includes_judge_reasoning(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv(envvars.LOG_LEVEL, "INFO")
        out = _capture([_failing_score()])
        assert "VERY_LONG_JUDGE_REASONING_THAT_SHOULD_NOT_LEAK_AT_WARNING" in out

    def test_debug_level_includes_judge_reasoning(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # ``-vv`` (``DEBUG``) is at least as verbose as ``-v``.
        monkeypatch.setenv(envvars.LOG_LEVEL, "DEBUG")
        out = _capture([_failing_score()])
        assert "VERY_LONG_JUDGE_REASONING_THAT_SHOULD_NOT_LEAK_AT_WARNING" in out


class TestSingleLineFailureFormat:
    """Each failed scenario renders on a single line, not three.

    Pre-compaction the renderer printed scenario name, rule, and llm on
    separate lines. Two failed scenarios produced six rendered lines of
    failure detail; everything actionable now collapses onto one line
    per scenario.
    """

    def test_scenario_rule_and_llm_share_a_line(self) -> None:
        out = _capture([_failing_score()])
        # Find the line that names the scenario; it must also carry the
        # rule and the llm verdict on the same line.
        match_lines = [ln for ln in out.splitlines() if "1. s" in ln]
        assert match_lines, f"expected one '1. s' line in:\n{out}"
        scenario_line = match_lines[0]
        assert "execution/no_errors" in scenario_line
        assert "correctness=low" in scenario_line


class TestViewerFooterAndHint:
    """Aggregator footer is one ``→ belt view <dir>`` line.

    The ``(pass -v for details)`` hint piggybacks on the same line, and
    only when failures were actually hidden by the WARNING-default
    verbosity. Passing runs and ``-v`` runs both omit the hint.
    """

    def test_footer_uses_belt_view(self) -> None:
        out = _capture([_failing_score()])
        assert "→ belt view run-x" in out

    def test_hint_present_on_failure_at_warning(self) -> None:
        out = _capture([_failing_score()])
        assert "(pass -v for details)" in out

    def test_hint_absent_at_verbose(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv(envvars.LOG_LEVEL, "INFO")
        out = _capture([_failing_score()])
        # At -v the prose is already shown; the hint would be misleading.
        assert "(pass -v for details)" not in out

    def test_hint_absent_when_all_pass(self) -> None:
        # Passing scenario - nothing was hidden, no hint should appear.
        passing = ScenarioScore(scenario_name="ok", group="g", tags=[], scores={}, overall_pass=True)
        out = _capture([passing])
        assert "(pass -v for details)" not in out
