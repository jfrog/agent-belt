# (c) JFrog Ltd. (2026)

"""Unit tests for scorer.consensus - ConsensusScorer voting logic."""

from __future__ import annotations

from io import StringIO
from unittest.mock import MagicMock

import pytest
from loguru import logger

from belt.entities import Scenario, ScoreLevel, ScorerResult, Turn, TurnOutput
from belt.scorer.llm.consensus import ConsensusScorer, _majority_vote_bool, _majority_vote_level
from belt.scorer.payloads import LLMDimensionVerdict, LLMPayload


def _make_scenario() -> Scenario:
    return Scenario(name="test", description="test", turns=[Turn(message="hello")])


def _make_turn_output() -> TurnOutput:
    return TurnOutput(raw_cli="output")


def _mock_judge(name: str, passed: bool, dims: dict[str, str]) -> MagicMock:
    """Create a mock LLMScorer that returns a fixed verdict."""
    judge = MagicMock()
    judge.judge_name = name
    judge.is_available.return_value = True
    # Resolution defaults to "scenario" so the mock matches the
    # behaviour the consensus block enforces (uniform across judges).
    # Tests exercising per-turn consensus override this explicitly.
    judge.resolution = "scenario"
    judge.evidence_scope = "isolated"

    dim_names = list(dims.keys())
    judge.strategy = MagicMock()
    judge.strategy.dimension_names = dim_names

    payload = LLMPayload(
        overall_pass=passed,
        dimensions={
            dim_name: LLMDimensionVerdict(score=score_val, reasoning=f"{name} says {score_val}")
            for dim_name, score_val in dims.items()
        },
    )
    judge.score.return_value = ScorerResult(passed=passed, data=payload)
    return judge


class TestMajorityVoteLevel:
    def test_unanimous_high(self):
        assert _majority_vote_level([ScoreLevel.HIGH, ScoreLevel.HIGH, ScoreLevel.HIGH]) == ScoreLevel.HIGH

    def test_majority_medium(self):
        assert _majority_vote_level([ScoreLevel.HIGH, ScoreLevel.MEDIUM, ScoreLevel.MEDIUM]) == ScoreLevel.MEDIUM

    def test_tie_breaks_pessimistic(self):
        assert _majority_vote_level([ScoreLevel.HIGH, ScoreLevel.MEDIUM]) == ScoreLevel.MEDIUM

    def test_tie_breaks_to_low(self):
        assert _majority_vote_level([ScoreLevel.LOW, ScoreLevel.HIGH]) == ScoreLevel.LOW

    def test_three_way_tie(self):
        assert _majority_vote_level([ScoreLevel.LOW, ScoreLevel.MEDIUM, ScoreLevel.HIGH]) == ScoreLevel.LOW


class TestMajorityVoteBool:
    def test_all_true(self):
        assert _majority_vote_bool([True, True, True]) is True

    def test_majority_true(self):
        assert _majority_vote_bool([True, True, False]) is True

    def test_majority_false(self):
        assert _majority_vote_bool([True, False, False]) is False

    def test_tie_breaks_false(self):
        assert _majority_vote_bool([True, False]) is False

    def test_all_false(self):
        assert _majority_vote_bool([False, False]) is False


class TestConsensusInit:
    def test_requires_at_least_two_judges(self):
        from belt.errors import ConfigError

        judge = _mock_judge("a", True, {"dim1": "high"})
        with pytest.raises(ConfigError, match="at least 2"):
            ConsensusScorer([judge])

    def test_rejects_unknown_strategy(self):
        from belt.errors import ConfigError

        judges = [_mock_judge("a", True, {}), _mock_judge("b", True, {})]
        with pytest.raises(ConfigError, match="Unknown consensus strategy"):
            ConsensusScorer(judges, strategy="bogus")

    def test_name_is_llm(self):
        judges = [_mock_judge("a", True, {}), _mock_judge("b", True, {})]
        cs = ConsensusScorer(judges)
        assert cs.name == "llm"

    def test_is_available_all_available(self):
        judges = [_mock_judge("a", True, {}), _mock_judge("b", True, {})]
        cs = ConsensusScorer(judges)
        assert cs.is_available() is True

    def test_is_available_one_down(self):
        j1 = _mock_judge("a", True, {})
        j2 = _mock_judge("b", True, {})
        j2.is_available.return_value = False
        cs = ConsensusScorer([j1, j2])
        assert cs.is_available() is False


class TestConsensusSameDimensions:
    """All judges score the same dimensions."""

    def test_unanimous_pass(self):
        judges = [
            _mock_judge("gpt4", True, {"correctness": "high", "safety": "high"}),
            _mock_judge("claude", True, {"correctness": "high", "safety": "high"}),
        ]
        cs = ConsensusScorer(judges)
        result = cs.score(_make_scenario(), [_make_turn_output()])
        assert result is not None
        assert result.passed is True
        assert result.data.dimensions["correctness"].score == "high"
        assert result.data.dimensions["safety"].score == "high"
        assert result.data.consensus_meta.disagreements == []

    def test_majority_vote_on_dimension(self):
        judges = [
            _mock_judge("a", True, {"correctness": "high"}),
            _mock_judge("b", True, {"correctness": "medium"}),
            _mock_judge("c", True, {"correctness": "high"}),
        ]
        cs = ConsensusScorer(judges)
        result = cs.score(_make_scenario(), [_make_turn_output()])
        assert result.data.dimensions["correctness"].score == "high"
        assert len(result.data.consensus_meta.disagreements) == 1

    def test_overall_pass_majority_vote(self):
        judges = [
            _mock_judge("a", True, {"dim": "high"}),
            _mock_judge("b", False, {"dim": "low"}),
            _mock_judge("c", True, {"dim": "high"}),
        ]
        cs = ConsensusScorer(judges)
        result = cs.score(_make_scenario(), [_make_turn_output()])
        assert result.passed is True

    def test_overall_fail_majority_vote(self):
        judges = [
            _mock_judge("a", True, {"dim": "high"}),
            _mock_judge("b", False, {"dim": "low"}),
            _mock_judge("c", False, {"dim": "low"}),
        ]
        cs = ConsensusScorer(judges)
        result = cs.score(_make_scenario(), [_make_turn_output()])
        assert result.passed is False


class TestConsensusDifferentDimensions:
    """Judges have partially overlapping dimensions."""

    def test_shared_voted_unique_passthrough(self):
        judges = [
            _mock_judge("gpt4", True, {"correctness": "high", "safety": "medium"}),
            _mock_judge("claude", True, {"correctness": "high", "clarity": "high"}),
        ]
        cs = ConsensusScorer(judges)
        result = cs.score(_make_scenario(), [_make_turn_output()])
        assert result.data.dimensions["correctness"].score == "high"
        assert result.data.dimensions["safety"].score == "medium"
        assert result.data.dimensions["clarity"].score == "high"
        meta = result.data.consensus_meta
        assert "correctness" in meta.shared_dimensions
        assert "safety" not in meta.shared_dimensions
        assert "clarity" not in meta.shared_dimensions

    def test_disagreement_on_shared_dimension(self):
        judges = [
            _mock_judge("a", True, {"correctness": "high", "safety": "high"}),
            _mock_judge("b", True, {"correctness": "low", "clarity": "high"}),
        ]
        cs = ConsensusScorer(judges)
        result = cs.score(_make_scenario(), [_make_turn_output()])
        assert result.data.dimensions["correctness"].score == "low"
        assert len(result.data.consensus_meta.disagreements) == 1
        assert result.data.consensus_meta.disagreements[0]["dimension"] == "correctness"


class TestConsensusStrategies:
    def test_unanimous_requires_all_same(self):
        judges = [
            _mock_judge("a", True, {"dim": "high"}),
            _mock_judge("b", True, {"dim": "medium"}),
        ]
        cs = ConsensusScorer(judges, strategy="unanimous")
        result = cs.score(_make_scenario(), [_make_turn_output()])
        assert result.data.dimensions["dim"].score == "medium"

    def test_any_takes_best(self):
        judges = [
            _mock_judge("a", True, {"dim": "low"}),
            _mock_judge("b", True, {"dim": "high"}),
        ]
        cs = ConsensusScorer(judges, strategy="any")
        result = cs.score(_make_scenario(), [_make_turn_output()])
        assert result.data.dimensions["dim"].score == "high"

    def test_unanimous_overall_pass(self):
        judges = [
            _mock_judge("a", True, {"dim": "high"}),
            _mock_judge("b", False, {"dim": "low"}),
        ]
        cs = ConsensusScorer(judges, strategy="unanimous")
        result = cs.score(_make_scenario(), [_make_turn_output()])
        assert result.passed is False

    def test_any_overall_pass(self):
        judges = [
            _mock_judge("a", True, {"dim": "high"}),
            _mock_judge("b", False, {"dim": "low"}),
        ]
        cs = ConsensusScorer(judges, strategy="any")
        result = cs.score(_make_scenario(), [_make_turn_output()])
        assert result.passed is True


class TestConsensusMetadata:
    def test_individual_verdicts_preserved(self):
        judges = [
            _mock_judge("gpt4", True, {"correctness": "high"}),
            _mock_judge("claude", True, {"correctness": "medium"}),
        ]
        cs = ConsensusScorer(judges)
        result = cs.score(_make_scenario(), [_make_turn_output()])
        iv = result.data.individual_verdicts
        assert "gpt4" in iv
        assert "claude" in iv
        assert iv["gpt4"]["dimensions"]["correctness"]["score"] == "high"
        assert iv["claude"]["dimensions"]["correctness"]["score"] == "medium"

    def test_usage_tokens_aggregated(self):
        from belt.scorer.payloads import UsageStats

        j1 = _mock_judge("a", True, {"dim": "high"})
        j1.score.return_value = ScorerResult(
            passed=True,
            data=LLMPayload(
                overall_pass=True,
                dimensions={"dim": LLMDimensionVerdict(score="high", reasoning="good")},
                usage=UsageStats(prompt_tokens=100, completion_tokens=50, total_tokens=150),
            ),
        )
        j2 = _mock_judge("b", True, {"dim": "high"})
        j2.score.return_value = ScorerResult(
            passed=True,
            data=LLMPayload(
                overall_pass=True,
                dimensions={"dim": LLMDimensionVerdict(score="high", reasoning="good")},
                usage=UsageStats(prompt_tokens=200, completion_tokens=80, total_tokens=280),
            ),
        )
        cs = ConsensusScorer([j1, j2])
        result = cs.score(_make_scenario(), [_make_turn_output()])
        assert result.data.usage.prompt_tokens == 300
        assert result.data.usage.completion_tokens == 130
        assert result.data.usage.total_tokens == 430


class TestConsensusEdgeCases:
    def test_empty_turn_outputs_returns_none(self):
        judges = [_mock_judge("a", True, {}), _mock_judge("b", True, {})]
        cs = ConsensusScorer(judges)
        assert cs.score(_make_scenario(), []) is None

    def test_single_judge_returns_passes_through(self):
        """When only one judge returns a result (other returned None), pass through."""
        j1 = _mock_judge("a", True, {"dim": "high"})
        j2 = _mock_judge("b", True, {"dim": "high"})
        j2.score.return_value = None
        cs = ConsensusScorer([j1, j2])
        result = cs.score(_make_scenario(), [_make_turn_output()])
        assert result is not None
        assert result.passed is True
        assert result.data.consensus_meta is None


class TestDimensionWarnings:
    def _capture_warnings(self, cs: ConsensusScorer) -> str:
        buf = StringIO()
        sink_id = logger.add(buf, level="WARNING", format="{message}")
        try:
            cs.emit_dimension_warnings()
        finally:
            logger.remove(sink_id)
        return buf.getvalue()

    def test_no_warning_when_identical(self):
        judges = [
            _mock_judge("a", True, {"dim1": "high", "dim2": "high"}),
            _mock_judge("b", True, {"dim1": "high", "dim2": "high"}),
        ]
        cs = ConsensusScorer(judges)
        assert "different dimensions" not in self._capture_warnings(cs)

    def test_warning_when_different(self):
        judges = [
            _mock_judge("a", True, {"dim1": "high", "dim2": "high"}),
            _mock_judge("b", True, {"dim1": "high", "dim3": "high"}),
        ]
        cs = ConsensusScorer(judges)
        output = self._capture_warnings(cs)
        assert "different dimensions" in output
        assert "dim1" in output

    def test_warning_emitted_only_once(self):
        judges = [
            _mock_judge("a", True, {"dim1": "high"}),
            _mock_judge("b", True, {"dim2": "high"}),
        ]
        cs = ConsensusScorer(judges)
        first = self._capture_warnings(cs)
        assert "different dimensions" in first
        second = self._capture_warnings(cs)
        assert "different dimensions" not in second
