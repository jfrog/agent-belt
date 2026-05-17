# (c) JFrog Ltd. (2026)

"""Shared fixtures for exporter tests.

Each test composes a minimal :class:`ExportContext` from synthetic
:class:`AggregatedResults` + a couple of :class:`ScenarioScore` instances so
the exporters under test exercise their formatting / escaping logic without
touching the runner / scorer / aggregator at all (Design Principle 1: the
export phase reads files; tests can hand it the same shape in memory).

Fixtures construct typed scorer payloads directly
(:class:`belt.scorer.payloads.RulesPayload`,
:class:`belt.scorer.payloads.LLMPayload`) so the round trip through
``ScenarioScore.model_dump()`` matches what the scorer phase writes to
disk - no shape drift between fixture and production.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from belt.entities import AggregatedResults, ScenarioScore
from belt.exporter.entities import ExportContext
from belt.scorer.payloads import CheckEntry, LLMDimensionVerdict, LLMPayload, RulesPayload


def _passing_score() -> ScenarioScore:
    return ScenarioScore(
        scenario_name="alpha",
        group="g1",
        tags=["real-runnable"],
        scores={
            "rules": RulesPayload(
                passed=True,
                checks=[
                    CheckEntry(dimension="execution", check="exited_zero", passed=True),
                ],
            ),
            "llm": LLMPayload(
                overall_pass=True,
                dimensions={"correctness": LLMDimensionVerdict(score="high", reasoning="LGTM")},
            ),
        },
        overall_pass=True,
    )


def _failing_score(name: str = "beta") -> ScenarioScore:
    return ScenarioScore(
        scenario_name=name,
        group="g1",
        tags=[],
        scores={
            "rules": RulesPayload(
                passed=False,
                checks=[
                    CheckEntry(
                        dimension="execution",
                        check="exited_zero",
                        passed=False,
                        details="exit_code=2 stderr=Traceback (most recent call last)",
                    ),
                ],
            ),
            "llm": LLMPayload(
                overall_pass=False,
                dimensions={
                    "correctness": LLMDimensionVerdict(
                        score="low",
                        reasoning="Wrong output: expected 42, got <script>alert(1)</script>",
                    ),
                },
            ),
        },
        overall_pass=False,
    )


def _trial_scores() -> list[ScenarioScore]:
    """Two trials of the same scenario - one pass, one fail."""
    return [
        ScenarioScore(
            scenario_name="alpha__trial_0",
            group="g1",
            tags=["real-runnable"],
            scores={
                "rules": RulesPayload(
                    passed=True,
                    checks=[CheckEntry(dimension="execution", check="exited_zero", passed=True)],
                )
            },
            overall_pass=True,
        ),
        ScenarioScore(
            scenario_name="alpha__trial_1",
            group="g1",
            tags=["real-runnable"],
            scores={
                "rules": RulesPayload(
                    passed=False,
                    checks=[
                        CheckEntry(
                            dimension="execution",
                            check="exited_zero",
                            passed=False,
                            details="trial flake",
                        )
                    ],
                )
            },
            overall_pass=False,
        ),
    ]


@pytest.fixture
def passing_score() -> ScenarioScore:
    return _passing_score()


@pytest.fixture
def failing_score() -> ScenarioScore:
    return _failing_score()


@pytest.fixture
def mixed_scores() -> list[ScenarioScore]:
    return [_passing_score(), _failing_score()]


@pytest.fixture
def trial_scores() -> list[ScenarioScore]:
    return _trial_scores()


@pytest.fixture
def export_context(tmp_path: Path, mixed_scores: list[ScenarioScore]) -> ExportContext:
    results = AggregatedResults(
        schema_version="1",
        total=2,
        passed=1,
        failed=1,
        overall_pass=False,
        stats={"pass_rate": 0.5},
        cost_timing={
            "scenarios": [
                {
                    "scenario": "g1/alpha",
                    "passed": True,
                    "agent_cost_usd": 0.001234,
                    "total_seconds": 4.5,
                },
                {
                    "scenario": "g1/beta",
                    "passed": False,
                    "agent_cost_usd": 0.005,
                    "total_seconds": 12.0,
                },
            ]
        },
        bottom_line=["1/2 scenarios failed.", "Most common rule failure: execution/exited_zero (1x)"],
        thresholds_passed=False,
        scenarios=[s.model_dump(mode="json") for s in mixed_scores],
    )
    return ExportContext(run_dir=tmp_path, results=results, scores=mixed_scores)


@pytest.fixture
def trial_export_context(tmp_path: Path, trial_scores: list[ScenarioScore]) -> ExportContext:
    results = AggregatedResults(
        schema_version="1",
        total=2,
        passed=1,
        failed=1,
        overall_pass=False,
        cost_timing={
            "scenarios": [
                {"scenario": "g1/alpha__trial_0", "total_seconds": 1.0},
                {"scenario": "g1/alpha__trial_1", "total_seconds": 1.5},
            ]
        },
        reliability={
            "mean_pass_at_1": 0.5,
            "mean_pass_at_3": 0.875,
            "mean_pass_at_8": 1.0 - 0.5**8,
            "mean_pass_pow_1": 0.5,
            "mean_pass_pow_3": 0.125,
            "mean_pass_pow_8": 0.5**8,
        },
        scenarios=[s.model_dump(mode="json") for s in trial_scores],
    )
    return ExportContext(run_dir=tmp_path, results=results, scores=trial_scores)
