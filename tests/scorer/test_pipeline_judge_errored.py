# (c) JFrog Ltd. (2026)

"""Pipeline-level force-fail invariant for judge-errored payloads.

The scorer layer can produce an ``LLMPayload`` with ``judge_errored=True``
when the judge backend fails for infra reasons. The pipeline must then:

1. Force ``overall_pass=False`` on the scenario score, even if the rules
   scorer passed. A silent green here is the original bug from issue
   #358.
2. Surface a synthetic ``execution`` check on the rules scorer's
   ``checks`` list so every existing consumer (exporters, threshold
   gates, failure renderer) sees the failure through the same channel
   already used for missing-turn-file errors.

These tests pin both behaviours end-to-end so a regression that re-
introduces silent passing or that drops the synthetic check fails here.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from belt import _internal_envvars
from belt.constants import TURN_CLI_TEMPLATE, TURN_OUTPUT_TEMPLATE
from belt.entities import TurnOutput
from belt.scenario import GroupConfig, Scenario, Turn, TurnExpectation
from belt.scorer.base import BaseScorer
from belt.scorer.entities import ScorerResult
from belt.scorer.payloads import (
    CheckEntry,
    LLMDimensionVerdict,
    LLMPayload,
    PerTurnLLMPayload,
    RulesPayload,
    TurnVerdict,
)
from belt.scorer.pipeline import score_scenario


class _PassingRulesScorer(BaseScorer):
    """Rules scorer stub that always passes - lets us prove the LLM-side
    failure forces ``overall_pass=False`` even when rules say everything is OK.
    """

    name = "rules"

    def is_available(self) -> bool:
        return True

    def score(self, scenario: Scenario, turn_outputs):
        return ScorerResult(
            passed=True,
            data=RulesPayload(
                checks=[CheckEntry(check="x", dimension="execution", passed=True)],
                passed=True,
            ),
        )


class _JudgeErroredScorer(BaseScorer):
    """LLM scorer stub returning a verdict-less payload (``judge_errored=True``).

    Mirrors what :class:`belt.scorer.llm.scorer.LLMScorer` produces on a
    transient infra failure - the pipeline must treat the payload the
    same regardless of which concrete scorer wrote it.
    """

    name = "llm"

    def __init__(self, error_type: str = "rate_limited") -> None:
        self._error_type = error_type

    def is_available(self) -> bool:
        return True

    def score(self, scenario: Scenario, turn_outputs):
        return ScorerResult(
            passed=False,
            data=LLMPayload(
                overall_pass=False,
                dimensions={},
                judge_errored=True,
                judge_error_type=self._error_type,  # type: ignore[arg-type]
            ),
        )


class _RealLLMScorer(BaseScorer):
    """LLM scorer stub returning a real verdict - to verify pass-through."""

    name = "llm"

    def is_available(self) -> bool:
        return True

    def score(self, scenario: Scenario, turn_outputs):
        return ScorerResult(
            passed=True,
            data=LLMPayload(
                overall_pass=True,
                dimensions={"q": LLMDimensionVerdict(score="pass", reasoning="ok")},  # type: ignore[arg-type]
            ),
        )


@pytest.fixture
def setup_scenario(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path]:
    """Write a minimal but valid on-disk scenario + one turn output.

    ``score_scenario`` re-reads the scenario JSON from
    ``scenarios_root / group / name.json`` and the per-turn outputs
    from ``outcomes_root / group / name / turn_*.json``. Both layouts
    must exist for it to score a scenario. The scenarios-root override
    is set via the documented internal env var so the pipeline does not
    fall back to the bundled examples directory.
    """
    group = "g"
    name = "t"
    scenarios_root = tmp_path / "scenarios"
    outcomes_root = tmp_path / "outcomes"
    (scenarios_root / group).mkdir(parents=True)
    (outcomes_root / group / name).mkdir(parents=True)

    group_config = GroupConfig(agent="echo")
    (scenarios_root / group / "_config.json").write_text(group_config.model_dump_json(indent=2))

    scenario = Scenario(
        name=name,
        description="judge-errored pipeline test scenario",
        turns=[Turn(message="hi", expect=TurnExpectation(has_reply=True))],
    )
    (scenarios_root / group / f"{name}.json").write_text(scenario.model_dump_json(indent=2))

    outcome_dir = outcomes_root / group / name
    turn_output = TurnOutput(raw_cli="hello\n", reply_text="hello", has_reply=True)
    (outcome_dir / TURN_OUTPUT_TEMPLATE.format(0)).write_text(turn_output.model_dump_json())
    # Also write the raw CLI file so the pipeline's fallback path is happy.
    (outcome_dir / TURN_CLI_TEMPLATE.format(0)).write_text("hello\n")
    monkeypatch.setenv(_internal_envvars.SCENARIOS_ROOT, str(scenarios_root))
    return scenarios_root, outcomes_root


class TestPipelineForceFailOnJudgeErrored:
    def test_judge_errored_forces_overall_pass_false_even_with_passing_rules(
        self, setup_scenario: tuple[Path, Path]
    ) -> None:
        # Pain 1 reproduction: without the force-fail invariant, this
        # scenario would read overall_pass=True (rules pass) while the
        # judge silently dropped its verdict.
        _, outcomes_root = setup_scenario
        scorers = [_PassingRulesScorer(), _JudgeErroredScorer()]
        score = score_scenario(outcomes_root / "g" / "t", outcomes_root, scorers)
        assert score.overall_pass is False

    def test_synthetic_execution_check_appears_in_rules_payload(self, setup_scenario: tuple[Path, Path]) -> None:
        _, outcomes_root = setup_scenario
        scorers = [_PassingRulesScorer(), _JudgeErroredScorer(error_type="timeout")]
        score = score_scenario(outcomes_root / "g" / "t", outcomes_root, scorers)
        rules = score.scores.get("rules")
        assert isinstance(rules, RulesPayload)
        synthetic = [c for c in rules.checks if c.check == "llm_scorer_ran"]
        assert len(synthetic) == 1
        check = synthetic[0]
        assert check.passed is False
        assert "timeout" in check.details
        assert check.dimension == "execution"

    def test_judge_errored_payload_is_preserved_in_scores(self, setup_scenario: tuple[Path, Path]) -> None:
        # The typed field that downstream consumers key off must survive
        # the pipeline's force-fail handling, not be stripped or
        # overwritten.
        _, outcomes_root = setup_scenario
        scorers = [_PassingRulesScorer(), _JudgeErroredScorer(error_type="rate_limited")]
        score = score_scenario(outcomes_root / "g" / "t", outcomes_root, scorers)
        llm = score.scores.get("llm")
        assert isinstance(llm, LLMPayload)
        assert llm.judge_errored is True
        assert llm.judge_error_type == "rate_limited"

    def test_score_json_carries_judge_errored_field(self, setup_scenario: tuple[Path, Path]) -> None:
        # The fields must survive serialisation - downstream eval runners
        # (and the aggregator) read score.json from disk, not in-memory
        # objects.
        _, outcomes_root = setup_scenario
        scorers = [_PassingRulesScorer(), _JudgeErroredScorer(error_type="rate_limited")]
        score = score_scenario(outcomes_root / "g" / "t", outcomes_root, scorers)
        dumped = json.loads(score.model_dump_json())
        llm = dumped["scores"]["llm"]
        assert llm["judge_errored"] is True
        assert llm["judge_error_type"] == "rate_limited"
        assert llm["schema_version"] == "llm.v1"

    def test_happy_path_unchanged_no_synthetic_check(self, setup_scenario: tuple[Path, Path]) -> None:
        _, outcomes_root = setup_scenario
        scorers = [_PassingRulesScorer(), _RealLLMScorer()]
        score = score_scenario(outcomes_root / "g" / "t", outcomes_root, scorers)
        assert score.overall_pass is True
        rules = score.scores.get("rules")
        assert isinstance(rules, RulesPayload)
        assert all(c.check != "llm_scorer_ran" for c in rules.checks)


class _PartialPerTurnJudgeErroredScorer(BaseScorer):
    """Per-turn scorer stub: turn 0 voted, turn 1 errored (infra).

    Mirrors what the fixed :meth:`LLMScorer._score_per_turn` produces on
    a partial per-turn failure - the payload-level ``judge_errored`` is
    the OR over turns. The pipeline must treat the ``per_turn_llm.v1``
    payload exactly like the scenario-level ``llm.v1`` one: force-fail +
    synthetic execution check.
    """

    name = "llm"

    def is_available(self) -> bool:
        return True

    def score(self, scenario: Scenario, turn_outputs):
        return ScorerResult(
            passed=False,
            data=PerTurnLLMPayload(
                overall_pass=False,
                turns=[
                    TurnVerdict(turn_idx=0, dimensions={"q": LLMDimensionVerdict(score="high", reasoning="ok")}),  # type: ignore[arg-type]
                    TurnVerdict(turn_idx=1, dimensions={}, judge_errored=True, judge_error_type="rate_limited"),
                ],
                judge_errored=True,
                judge_error_type="rate_limited",
            ),
        )


class _CleanPerTurnScorer(BaseScorer):
    """Per-turn scorer stub with every turn voted - pass-through guard."""

    name = "llm"

    def is_available(self) -> bool:
        return True

    def score(self, scenario: Scenario, turn_outputs):
        return ScorerResult(
            passed=True,
            data=PerTurnLLMPayload(
                overall_pass=True,
                turns=[TurnVerdict(turn_idx=0, dimensions={"q": LLMDimensionVerdict(score="pass", reasoning="ok")})],  # type: ignore[arg-type]
            ),
        )


class TestPipelineForceFailOnPerTurnJudgeErrored:
    """The force-fail invariant must hold for ``per_turn_llm.v1`` too."""

    def test_partial_per_turn_error_forces_fail_and_synthetic_check(self, setup_scenario: tuple[Path, Path]) -> None:
        _, outcomes_root = setup_scenario
        scorers = [_PassingRulesScorer(), _PartialPerTurnJudgeErroredScorer()]
        score = score_scenario(outcomes_root / "g" / "t", outcomes_root, scorers)

        assert score.overall_pass is False
        rules = score.scores.get("rules")
        assert isinstance(rules, RulesPayload)
        synthetic = [c for c in rules.checks if c.check == "llm_scorer_ran"]
        assert len(synthetic) == 1
        assert synthetic[0].passed is False
        assert "rate_limited" in synthetic[0].details

        llm = score.scores.get("llm")
        assert isinstance(llm, PerTurnLLMPayload)
        assert llm.judge_errored is True

        dumped = json.loads(score.model_dump_json())
        assert dumped["scores"]["llm"]["schema_version"] == "per_turn_llm.v1"
        assert dumped["scores"]["llm"]["judge_errored"] is True
        assert dumped["scores"]["llm"]["judge_error_type"] == "rate_limited"

    def test_clean_per_turn_payload_passes_through(self, setup_scenario: tuple[Path, Path]) -> None:
        _, outcomes_root = setup_scenario
        scorers = [_PassingRulesScorer(), _CleanPerTurnScorer()]
        score = score_scenario(outcomes_root / "g" / "t", outcomes_root, scorers)
        assert score.overall_pass is True
        rules = score.scores.get("rules")
        assert isinstance(rules, RulesPayload)
        assert all(c.check != "llm_scorer_ran" for c in rules.checks)
