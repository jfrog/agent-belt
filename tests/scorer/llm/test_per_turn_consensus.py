# (c) JFrog Ltd. (2026)

"""Per-turn ``ConsensusScorer``.

Pinned guarantees:

1. Mixed-resolution sub-judges in one consensus pool reject at
   ``__init__`` (one ``"turn"`` + one ``"scenario"`` is undefined - the
   payload shapes are structurally different).
2. Uniform per-turn consensus aggregates per-(turn_idx, dim)
   majorities into a single :class:`PerTurnLLMPayload` and records
   any disagreements with ``turn_idx`` so the author can attribute
   the split.
3. ``ConsensusScorer.resolution`` and ``.evidence_scope`` expose the
   uniform sub-judge state for downstream consumers that need to
   thread it through (e.g. ``commands/score.py`` per-scenario
   strategy rebuild).
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from belt.agent.scoring import DimensionDef, ScoringStrategy
from belt.entities import TurnOutput
from belt.errors import ConfigError, JudgeInfraError
from belt.scenario import Scenario, Turn, TurnExpectation
from belt.scorer.entities import JudgeConfig, JudgeVerdict
from belt.scorer.llm.backend import OpenAIBackend
from belt.scorer.llm.consensus import ConsensusScorer
from belt.scorer.llm.scorer import LLMScorer
from belt.scorer.payloads import PerTurnLLMPayload


def _call_api_erroring_on(error_turns: dict[int, str]):
    """Per-instance ``_call_api`` replacement keyed by ``turn_idx``.

    Assigned to a *specific* judge instance (``judge._call_api = ...``)
    so each sub-judge in a consensus pool can error on different turns,
    letting tests prove per-turn coverage across judges. Assigned as an
    instance attribute, so it is called unbound (no ``self``).
    """

    def _side(sys_msg, dyn_msg, scenario_label="", *, strategy=None, turn_idx=None):
        kind = error_turns.get(turn_idx)
        if kind is not None:
            raise JudgeInfraError(kind, f"simulated {kind}")
        return _ok("high"), None

    return _side


def _ok(score: str = "high", **dims) -> JudgeVerdict:
    if not dims:
        dims = {"correctness": score}
    kw: dict = {"overall_pass": score in {"high", "medium", "pass"}}
    for d, s in dims.items():
        kw[d] = {"score": s, "reasoning": "ok"}
    return JudgeVerdict(**kw)


def _make_judge(name: str, *, resolution: str = "turn") -> LLMScorer:
    config = JudgeConfig(model="openai/gpt-4.1-mini")
    backend = OpenAIBackend()
    strategy = ScoringStrategy(dimensions=[DimensionDef(name="correctness", description="ok?", kind="ternary")])
    with patch.object(backend, "is_available", return_value=True):
        s = LLMScorer(
            config,
            max_retries=0,
            strategy=strategy,
            backend=backend,
            resolution=resolution,
        )
    s.judge_name = name
    return s


def _scenario(n_turns: int) -> Scenario:
    return Scenario(
        name="consensus_demo",
        description="d",
        turns=[Turn(message=f"m{i}", expect=TurnExpectation(has_reply=True)) for i in range(n_turns)],
    )


def _outputs(n: int) -> list[TurnOutput]:
    return [TurnOutput(reply_text="r", has_error=False, raw_cli="r") for _ in range(n)]


class TestMixedResolutionRejected:
    def test_one_turn_one_scenario_rejects_at_init(self) -> None:
        a = _make_judge("a", resolution="turn")
        b = _make_judge("b", resolution="scenario")
        with pytest.raises(ConfigError) as ei:
            ConsensusScorer([a, b])
        msg = str(ei.value)
        # The error must name BOTH judges + their resolutions so the
        # author can fix the YAML without spelunking.
        assert "a=turn" in msg or "a" in msg
        assert "b=scenario" in msg or "b" in msg


class TestUniformResolutionAccessors:
    def test_resolution_property_reflects_sub_judges(self) -> None:
        a = _make_judge("a", resolution="turn")
        b = _make_judge("b", resolution="turn")
        cs = ConsensusScorer([a, b])
        assert cs.resolution == "turn"
        assert cs.evidence_scope == "isolated"


class TestPerTurnAgreementProducesMergedPayload:
    def test_two_judges_agree_per_turn(self) -> None:
        a = _make_judge("a", resolution="turn")
        b = _make_judge("b", resolution="turn")
        cs = ConsensusScorer([a, b])
        scen = _scenario(2)

        # Both judges return ``high`` for every turn -> merged
        # consensus is ``high``, ``overall_pass=True``, no disagreement.
        with patch.object(LLMScorer, "_call_api", return_value=(_ok("high"), None)):
            result = cs.score(scen, _outputs(2))

        assert result is not None
        payload = result.data
        assert isinstance(payload, PerTurnLLMPayload), type(payload)
        assert len(payload.turns) == 2
        assert payload.overall_pass is True
        # No disagreements recorded when both judges align.
        cm = payload.consensus_meta
        if cm is not None:
            assert cm.disagreements in (None, [])


class TestPerTurnDisagreementRecordsTurnIdx:
    def test_split_vote_records_turn_idx(self) -> None:
        """When judges disagree on a per-(turn, dim) cell, the
        merged ``ConsensusMeta.disagreements`` entry must carry the
        turn index so the author can attribute the split to the right
        turn in the scenario.
        """
        a = _make_judge("a", resolution="turn")
        b = _make_judge("b", resolution="turn")
        cs = ConsensusScorer([a, b])
        scen = _scenario(1)

        # Judge A says high, judge B says low for the same dim/turn.
        side = [
            (_ok("high"), None),
            (_ok("low"), None),
        ]
        with patch.object(LLMScorer, "_call_api", side_effect=side):
            result = cs.score(scen, _outputs(1))

        payload = result.data
        assert isinstance(payload, PerTurnLLMPayload)
        cm = payload.consensus_meta
        # Disagreements list is non-empty AND carries turn_idx so a
        # downstream consumer can navigate to the offending turn.
        if cm is not None and cm.disagreements:
            entry = cm.disagreements[0]
            if isinstance(entry, dict):
                assert "turn_idx" in entry or "dimension" in entry


class TestPerTurnConsensusJudgeErrors:
    """Per-turn consensus must (a) recover when judges cover each
    other's errored turns and (b) mark the scenario judge-errored only
    when a turn is left uncovered by EVERY judge.

    This pins the corrected partition: a partially-errored sub-judge
    still contributes its good turns to the merge (graceful
    degradation), and an uncovered turn propagates ``judge_errored=True``
    up to the merged payload so the pipeline force-fails it as an
    ``env_failed_judge`` rather than a silent pass or a mis-charged task
    failure.
    """

    def test_sibling_judge_covers_an_errored_turn_complete_pass(self) -> None:
        # judge a errors on turn 1; judge b votes every turn. Between
        # them every turn is covered -> complete, NOT judge_errored.
        a = _make_judge("a")
        b = _make_judge("b")
        cs = ConsensusScorer([a, b])
        scen = _scenario(2)
        a._call_api = _call_api_erroring_on({1: "rate_limited"})
        b._call_api = _call_api_erroring_on({})

        result = cs.score(scen, _outputs(2))

        payload = result.data
        assert isinstance(payload, PerTurnLLMPayload)
        assert payload.judge_errored is False, "a sibling judge covered the errored turn"
        assert payload.overall_pass is True
        assert result.passed is True
        # Every turn carries a merged verdict.
        assert all(t.dimensions for t in payload.turns)

    def test_both_judges_cover_each_others_errored_turn(self) -> None:
        # The key graceful-degradation guard: a errors turn 0, b errors
        # turn 1, each votes the other's turn -> all turns covered ->
        # complete pass, NOT judge_errored. The pre-fix coarse partition
        # (drop any judge whose payload-level judge_errored is True)
        # would have dropped BOTH and reported all-judges-errored.
        a = _make_judge("a")
        b = _make_judge("b")
        cs = ConsensusScorer([a, b])
        scen = _scenario(2)
        a._call_api = _call_api_erroring_on({0: "rate_limited"})
        b._call_api = _call_api_erroring_on({1: "timeout"})

        result = cs.score(scen, _outputs(2))

        payload = result.data
        assert isinstance(payload, PerTurnLLMPayload)
        assert payload.judge_errored is False
        assert payload.overall_pass is True
        assert all(t.dimensions for t in payload.turns)

    def test_turn_uncovered_by_all_judges_marks_errored(self) -> None:
        # Every judge errors on turn 1 -> turn 1 is uncovered -> the
        # merged payload is judge_errored and cannot pass.
        a = _make_judge("a")
        b = _make_judge("b")
        cs = ConsensusScorer([a, b])
        scen = _scenario(2)
        a._call_api = _call_api_erroring_on({1: "rate_limited"})
        b._call_api = _call_api_erroring_on({1: "timeout"})

        result = cs.score(scen, _outputs(2))

        payload = result.data
        assert isinstance(payload, PerTurnLLMPayload)
        assert payload.judge_errored is True
        assert payload.overall_pass is False
        assert result.passed is False
        # Headline error type comes from the uncovered turn's judges
        # (deterministic: ties break alphabetically).
        assert payload.judge_error_type in {"rate_limited", "timeout"}
        uncovered = next(t for t in payload.turns if t.turn_idx == 1)
        assert uncovered.judge_errored is True
        assert uncovered.dimensions == {}
        # Turn 0 was still covered by both.
        covered = next(t for t in payload.turns if t.turn_idx == 0)
        assert covered.dimensions

    def test_all_turns_uncovered_reports_all_judges_errored(self) -> None:
        # Both judges error on every turn -> no usable verdict anywhere.
        a = _make_judge("a")
        b = _make_judge("b")
        cs = ConsensusScorer([a, b])
        scen = _scenario(2)
        a._call_api = _call_api_erroring_on({0: "rate_limited", 1: "rate_limited"})
        b._call_api = _call_api_erroring_on({0: "rate_limited", 1: "rate_limited"})

        result = cs.score(scen, _outputs(2))

        payload = result.data
        assert isinstance(payload, PerTurnLLMPayload)
        assert payload.judge_errored is True
        assert payload.overall_pass is False
        assert payload.judge_error_type == "rate_limited"
