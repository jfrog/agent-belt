# (c) JFrog Ltd. (2026)

"""Preflight validation for per-turn judges.

Static checks must fire BEFORE any agent or judge call happens:

1. Unknown judge name in ``Turn.llm_judges`` rejects with a hint
   listing declared judges.
2. All-turns-skipped (every turn has ``skip: true`` for the only
   judge) rejects so an author cannot silently route a scenario to
   the ``all_turns_skipped`` runtime path.
3. Empty ``Turn.llm_judges`` and missing per-turn judges are no-ops -
   the preflight does not raise spuriously when nothing per-turn is
   in scope.
4. Multiple violations in one call aggregate into a single composite
   error (so the author sees every bug at once, not one-fix-then-
   rerun churn).
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from belt.agent.scoring import DimensionDef, ScoringStrategy
from belt.errors import ConfigError
from belt.scenario import Scenario, Turn, TurnExpectation, TurnJudgeOverride
from belt.scorer.entities import JudgeConfig
from belt.scorer.llm.backend import OpenAIBackend
from belt.scorer.llm.scorer import LLMScorer
from belt.scorer.pipeline import validate_per_turn_judges_against_scenarios


def _per_turn_scorer(name: str = "per_turn_judge") -> LLMScorer:
    config = JudgeConfig(model="openai/gpt-4.1-mini")
    backend = OpenAIBackend()
    strategy = ScoringStrategy(dimensions=[DimensionDef(name="correctness", description="ok?", kind="ternary")])
    with patch.object(backend, "is_available", return_value=True):
        s = LLMScorer(
            config,
            max_retries=0,
            strategy=strategy,
            backend=backend,
            resolution="turn",
        )
    s.judge_name = name
    return s


def _scenario(
    turn_overrides: list[dict | None],
    *,
    judge_name: str = "per_turn_judge",
    name: str = "demo",
) -> Scenario:
    turns: list[Turn] = []
    for i, ov in enumerate(turn_overrides):
        kw: dict = {"message": f"m{i}", "expect": TurnExpectation(has_reply=True)}
        if ov is not None:
            kw["llm_judges"] = {judge_name: TurnJudgeOverride(**ov)}
        turns.append(Turn(**kw))
    return Scenario(name=name, description="d", turns=turns)


class TestUnknownJudgeName:
    def test_typo_rejected_with_hint(self) -> None:
        scorer = _per_turn_scorer(name="per_turn_judge")
        scen = _scenario([{"instruction": "test"}], judge_name="tyop_name")

        with pytest.raises(ConfigError) as ei:
            validate_per_turn_judges_against_scenarios([scorer], [scen])
        msg = str(ei.value)
        assert "tyop_name" in msg
        # Hint lists the declared judges so the author can fix.
        assert "per_turn_judge" in msg


class TestAllSkipped:
    def test_every_turn_skipped_rejected(self) -> None:
        scorer = _per_turn_scorer()
        scen = _scenario([{"skip": True}, {"skip": True}])

        with pytest.raises(ConfigError) as ei:
            validate_per_turn_judges_against_scenarios([scorer], [scen])
        assert "per_turn_judge" in str(ei.value)


class TestQuiet:
    def test_no_per_turn_judges_is_noop(self) -> None:
        # Only scenario-level scorers in the list - preflight
        # returns silently.
        config = JudgeConfig(model="openai/gpt-4.1-mini")
        backend = OpenAIBackend()
        with patch.object(backend, "is_available", return_value=True):
            scen_lvl = LLMScorer(
                config,
                max_retries=0,
                strategy=ScoringStrategy(
                    dimensions=[DimensionDef(name="correctness", description="ok?", kind="ternary")]
                ),
                backend=backend,
                resolution="scenario",
            )

        scen = _scenario([None, None])
        validate_per_turn_judges_against_scenarios([scen_lvl], [scen])

    def test_no_overrides_is_noop(self) -> None:
        # Per-turn judge declared, scenario has zero per-turn
        # llm_judges entries: nothing to validate, no error.
        scorer = _per_turn_scorer()
        scen = _scenario([None, None])
        validate_per_turn_judges_against_scenarios([scorer], [scen])


class TestComposite:
    def test_multiple_violations_aggregated(self) -> None:
        scorer = _per_turn_scorer(name="per_turn_judge")
        bad_a = _scenario([{"skip": True}], name="bad_a")
        bad_b = _scenario([{"instruction": "x"}], judge_name="typo", name="bad_b")

        with pytest.raises(ConfigError) as ei:
            validate_per_turn_judges_against_scenarios([scorer], [bad_a, bad_b])
        msg = str(ei.value)
        # Both violations must surface in a single error so the
        # author can fix everything in one pass.
        assert "bad_a" in msg
        assert "bad_b" in msg
