# (c) JFrog Ltd. (2026)

"""Tests for ``belt.progress.ScorerProgress.summary``.

Single-scoreboard contract: when ``belt eval`` chains the score phase
in-process, the screen must not show two scoreboards (the scorer's
``Score: X/N passed`` headline AND the aggregator's ``X/N checks
(Y%)`` footer). A ``chained`` flag threads from ``commands.eval``
through ``commands.score.main`` into ``ScorerProgress.summary``; when
set, the scorer suppresses its pass-count headline and judge-cost
subpart while keeping the cache + token subparts (which the aggregator
does not re-emit). Standalone ``belt score`` keeps the full summary.

The tests render with ``no_color=True`` so ANSI escapes do not break
substring assertions, and ``force_terminal=False`` so the rich
``Console`` does not wrap output at the test environment's TTY width.
"""

from __future__ import annotations

import io
import time
from pathlib import Path

from rich.console import Console

from belt.entities import ScenarioScore
from belt.progress import ScorerProgress


def _make_progress() -> tuple[ScorerProgress, io.StringIO]:
    buf = io.StringIO()
    console = Console(file=buf, force_terminal=False, width=200, no_color=True)
    progress = ScorerProgress(console=console)
    # ``summary`` reads ``_start_time`` (set by ``start()``) and
    # ``_mode_str`` (also set by ``start()``). The rendering branch
    # under test does not need the live/progress UI machinery that
    # ``start()`` spins up, so set the two fields directly and skip
    # the rest. Same shortcut ``tests/test_run_summary_agent_errors.py``
    # uses for ``RunnerProgress``.
    progress._start_time = time.monotonic()
    progress._mode_str = "rules + llm"
    return progress, buf


def _score(name: str, passed: bool) -> tuple[Path, ScenarioScore]:
    return (
        Path(f"/tmp/{name}"),
        ScenarioScore(scenario_name=name, group="g", overall_pass=passed),
    )


class TestScorerProgressSummaryChained:
    """``chained=True`` (called by ``belt eval`` in-process) suppresses
    the duplicate scoreboard but keeps scorer-internal stats.
    """

    def test_chained_suppresses_pass_count_headline(self) -> None:
        progress, buf = _make_progress()
        results = [_score("s1", True), _score("s2", True), _score("s3", False)]

        progress.summary(results, chained=True)

        out = buf.getvalue()
        assert "Score:" not in out, "headline must be suppressed when chained (aggregator prints the canonical one)"
        assert "passed" not in out
        assert "all passed" not in out

    def test_chained_suppresses_judge_cost_subpart(self) -> None:
        progress, buf = _make_progress()
        results = [_score("s1", True)]

        progress.summary(results, judge_cost_usd=0.1234, chained=True)

        out = buf.getvalue()
        assert "judge cost" not in out, "judge cost is duplicated by the aggregator footer when chained"
        assert "0.1234" not in out

    def test_chained_keeps_cache_and_token_subparts(self) -> None:
        progress, buf = _make_progress()
        results = [_score("s1", True)]

        progress.summary(
            results,
            cache_hits=7,
            cache_misses=3,
            prompt_tokens=1_000,
            completion_tokens=500,
            judge_cost_usd=0.01,
            chained=True,
        )

        out = buf.getvalue()
        # Aggregator does not surface these; they remain useful to the
        # operator and must survive the suppression.
        assert "cache: 7/10 hit" in out
        assert "1,500 tokens" in out


class TestScorerProgressSummaryStandalone:
    """``chained=False`` (default, standalone ``belt score``) keeps the
    full summary. Regression guard against accidentally inverting the
    flag default.
    """

    def test_default_emits_full_summary(self) -> None:
        progress, buf = _make_progress()
        results = [_score("s1", True), _score("s2", False)]

        progress.summary(
            results,
            cache_hits=2,
            cache_misses=8,
            prompt_tokens=100,
            completion_tokens=50,
            judge_cost_usd=0.42,
        )

        out = buf.getvalue()
        # Headline, judge cost, cache, tokens all present.
        assert "Score:" in out
        assert "1/2 passed" in out
        assert "judge cost" in out
        assert "0.4200" in out
        assert "cache: 2/10 hit" in out
        assert "150 tokens" in out

    def test_default_all_passed_uses_all_passed_phrase(self) -> None:
        progress, buf = _make_progress()
        results = [_score("s1", True), _score("s2", True)]

        progress.summary(results)

        out = buf.getvalue()
        assert "all passed" in out
        assert "1/2" not in out
