# (c) JFrog Ltd. (2026)

"""Consensus all-or-nothing semantics under judge infrastructure failure.

When at least one sub-judge produced a real verdict, consensus proceeds
with the survivors and records the errored sub-judges in
``individual_verdicts`` so reviewers can see provider-specific flakiness.
When every sub-judge errored, the merged payload itself carries
``judge_errored=True`` so the aggregator partitions the scenario into the
judge-environmental axis, not the task-quality axis.

These tests fix the semantics without spinning up two LLM backends: they
build :class:`LLMScorer` stubs whose ``score`` method returns a canned
:class:`ScorerResult`, plug them into :class:`ConsensusScorer`, and pin
the merged payload shape.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

from belt.entities import TurnOutput
from belt.scenario import Scenario, Turn, TurnExpectation
from belt.scorer.entities import ScorerResult
from belt.scorer.llm.consensus import ConsensusScorer
from belt.scorer.llm.scorer import LLMScorer
from belt.scorer.payloads import LLMDimensionVerdict, LLMPayload


@pytest.fixture
def scenario() -> Scenario:
    return Scenario(
        name="t",
        description="consensus judge-error test scenario",
        turns=[Turn(message="?", expect=TurnExpectation(has_reply=True))],
    )


@pytest.fixture
def turn_outputs() -> list[TurnOutput]:
    return [TurnOutput(raw_cli="reply")]


def _stub_judge(name: str, result: ScorerResult, dim_names: list[str] | None = None) -> LLMScorer:
    """Build a minimally-faked :class:`LLMScorer` returning *result*.

    ``ConsensusScorer`` calls ``judge.score(...)`` and reads
    ``judge.judge_name`` and ``judge.strategy.dimension_names``;
    nothing else.
    """
    judge = MagicMock(spec=LLMScorer)
    judge.judge_name = name
    judge.score.return_value = result
    judge.strategy = MagicMock()
    judge.strategy.dimension_names = dim_names or ["q"]
    judge.is_available.return_value = True
    judge.on_event = None
    return judge


def _real_payload(dim: str = "q", score: str = "pass") -> LLMPayload:
    return LLMPayload(
        overall_pass=score == "pass",
        dimensions={dim: LLMDimensionVerdict(score=score, reasoning="ok")},  # type: ignore[arg-type]
    )


def _errored_payload(error_type: str = "rate_limited") -> LLMPayload:
    return LLMPayload(
        overall_pass=False,
        dimensions={},
        judge_errored=True,
        judge_error_type=error_type,  # type: ignore[arg-type]
    )


class TestConsensusJudgeErrors:
    def test_all_sub_judges_errored_yields_judge_errored_merged_payload(
        self, scenario: Scenario, turn_outputs: list[TurnOutput]
    ) -> None:
        # Two sub-judges both errored: the merged payload itself must
        # carry judge_errored=True so the aggregator partitions the
        # scenario into the judge axis.
        j1 = _stub_judge("a", ScorerResult(passed=False, data=_errored_payload("rate_limited")))
        j2 = _stub_judge("b", ScorerResult(passed=False, data=_errored_payload("timeout")))
        consensus = ConsensusScorer([j1, j2])

        result = consensus.score(scenario, turn_outputs)

        assert result is not None
        assert isinstance(result.data, LLMPayload)
        assert result.data.judge_errored is True
        # Most-common type wins; tied types break alphabetically so the
        # choice is deterministic across runs.
        assert result.data.judge_error_type in ("rate_limited", "timeout")
        # Per-judge error tokens are preserved for postmortem.
        assert result.data.individual_verdicts is not None
        assert set(result.data.individual_verdicts) == {"a", "b"}

    def test_one_judge_errored_consensus_proceeds_with_survivors(
        self, scenario: Scenario, turn_outputs: list[TurnOutput]
    ) -> None:
        # j1 produced a real verdict; j2 errored. Consensus must still
        # produce a real (non-judge-errored) merged payload that reflects
        # only the surviving verdict.
        j1 = _stub_judge("alpha", ScorerResult(passed=True, data=_real_payload(score="pass")))
        j2 = _stub_judge("beta", ScorerResult(passed=False, data=_errored_payload("timeout")))
        consensus = ConsensusScorer([j1, j2])

        result = consensus.score(scenario, turn_outputs)

        assert result is not None
        assert isinstance(result.data, LLMPayload)
        assert result.data.judge_errored is False
        # The errored judge is still recorded so reviewers can see the
        # provider flaked even though consensus succeeded.
        assert result.data.individual_verdicts is not None
        assert "beta" in result.data.individual_verdicts
        beta_payload: dict[str, Any] = result.data.individual_verdicts["beta"]
        assert beta_payload.get("judge_errored") is True

    def test_single_judge_path_unchanged_when_no_judge_erred(
        self, scenario: Scenario, turn_outputs: list[TurnOutput]
    ) -> None:
        # The pre-existing single-real-verdict short-circuit (consensus
        # of one) still returns the sub-judge's result verbatim when
        # nothing erred. This is the hot path most multi-judge runs hit.
        j1 = _stub_judge("a", ScorerResult(passed=True, data=_real_payload()))
        j2 = _stub_judge("b", ScorerResult(passed=True, data=_real_payload()))
        consensus = ConsensusScorer([j1, j2])

        result = consensus.score(scenario, turn_outputs)

        assert result is not None
        assert isinstance(result.data, LLMPayload)
        assert result.data.judge_errored is False
