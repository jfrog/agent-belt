# (c) JFrog Ltd. (2026)

"""Per-turn ``LLMScorer.score`` happy path + evidence-scope behaviour.

These tests exercise the per-turn judging dispatch end-to-end with
``LLMScorer._call_api`` patched so no network call happens. They cover:

1. ``resolution="turn"`` writes a :class:`PerTurnLLMPayload` whose
   ``turns`` list has one :class:`TurnVerdict` per scenario turn.
2. ``Turn.llm_judges[<name>].instruction`` reaches the judge's dynamic
   message for that turn only - prior turns keep the scenario-level
   instruction.
3. ``evidence_scope="isolated"`` shows only the current turn in the
   prompt; ``"cumulative"`` shows turns ``[0..i]``.
4. ``skip: true`` short-circuits a turn (no API call, empty
   ``TurnVerdict``).
5. Per-turn ``dimensions`` overrides REPLACE the default rubric for
   that turn unless ``extend_default_dimensions=true``.
"""

from __future__ import annotations

from io import StringIO
from unittest.mock import patch

from loguru import logger

from belt.agent.scoring import DimensionDef, ScoringStrategy
from belt.entities import TurnOutput
from belt.errors import JudgeInfraError
from belt.scenario import Scenario, Turn, TurnExpectation, TurnJudgeOverride
from belt.scorer.entities import JudgeConfig, JudgeVerdict
from belt.scorer.llm.backend import OpenAIBackend
from belt.scorer.llm.scorer import LLMScorer
from belt.scorer.payloads import PerTurnLLMPayload, TurnVerdict


def _strategy() -> ScoringStrategy:
    return ScoringStrategy(
        dimensions=[
            DimensionDef(name="correctness", description="Is the agent reply right?", kind="ternary"),
        ]
    )


def _ok_verdict(**dims) -> JudgeVerdict:
    """Build a ``JudgeVerdict`` whose extras are ``DimensionScore`` dicts.

    Default is ``correctness=high``; callers pass kwargs like
    ``echo_fidelity="high"`` to swap dimensions or add more.
    """
    if not dims:
        dims = {"correctness": "high"}
    kwargs: dict = {"overall_pass": True}
    for name, level in dims.items():
        kwargs[name] = {"score": level, "reasoning": "ok"}
    return JudgeVerdict(**kwargs)


def _scenario_with_turns(turn_overrides: list[dict | None]) -> Scenario:
    turns: list[Turn] = []
    for i, override in enumerate(turn_overrides):
        kwargs: dict = {
            "message": f"do thing {i}",
            "expect": TurnExpectation(has_reply=True),
        }
        if override is not None:
            kwargs["llm_judges"] = {"per_turn_judge": TurnJudgeOverride(**override)}
        turns.append(Turn(**kwargs))
    return Scenario(name="per_turn_demo", description="per-turn judging fixture", turns=turns)


def _make_scorer(
    *,
    resolution: str = "turn",
    evidence_scope: str = "isolated",
    strategy: ScoringStrategy | None = None,
) -> LLMScorer:
    config = JudgeConfig(model="openai/gpt-4.1-mini", temperature=0.0, seed=2008, max_tokens=1024)
    backend = OpenAIBackend()
    with patch.object(backend, "is_available", return_value=True):
        scorer = LLMScorer(
            config,
            max_retries=0,
            strategy=strategy or _strategy(),
            backend=backend,
            resolution=resolution,
            evidence_scope=evidence_scope,
        )
    scorer.judge_name = "per_turn_judge"
    return scorer


def _dummy_outputs(n: int) -> list[TurnOutput]:
    return [TurnOutput(reply_text=f"agent reply {i}", has_error=False, raw_cli=f"agent reply {i}") for i in range(n)]


class TestPerTurnHappyPath:
    def test_writes_per_turn_payload_with_one_verdict_per_turn(self) -> None:
        scorer = _make_scorer()
        scen = _scenario_with_turns([None, None])

        with patch.object(
            LLMScorer,
            "_call_api",
            return_value=(_ok_verdict(), {"prompt_tokens": 10, "completion_tokens": 5}),
        ):
            result = scorer.score(scen, _dummy_outputs(2))

        assert result is not None
        payload = result.data
        assert isinstance(payload, PerTurnLLMPayload), type(payload)
        assert payload.schema_version == "per_turn_llm.v1"
        assert len(payload.turns) == 2
        assert all(isinstance(t, TurnVerdict) for t in payload.turns)
        assert [t.turn_idx for t in payload.turns] == [0, 1]
        assert payload.overall_pass is True
        assert payload.judge_errored is False

    def test_one_api_call_per_turn(self) -> None:
        scorer = _make_scorer()
        scen = _scenario_with_turns([None, None, None])

        with patch.object(
            LLMScorer,
            "_call_api",
            return_value=(_ok_verdict(), {"prompt_tokens": 10, "completion_tokens": 5}),
        ) as call:
            scorer.score(scen, _dummy_outputs(3))

        assert call.call_count == 3


class TestPerTurnInstructionOverride:
    def test_instruction_reaches_dynamic_message_for_that_turn_only(self) -> None:
        scorer = _make_scorer()
        scen = _scenario_with_turns([None, {"instruction": "Turn-1-only-rubric: must mention 'frog'"}])

        # Capture each call's (sys_msg, dyn_msg) so we can assert
        # what the per-turn override actually reached.
        captured: list[tuple[str, str]] = []

        def _capture(self, sys_msg, dyn_msg, *args, **kwargs):
            captured.append((sys_msg, dyn_msg))
            return _ok_verdict(), {"prompt_tokens": 10, "completion_tokens": 5}

        with patch.object(LLMScorer, "_call_api", new=_capture):
            scorer.score(scen, _dummy_outputs(2))

        # The per-turn override only reaches the judge as a fenced
        # ``<scenario_instruction>`` block for that specific turn -
        # the scenario JSON dump (rendered in every turn's prompt as
        # context) shows the override as DATA, not as an effective
        # rubric. The fence is the actual injection vector, so test
        # the fence presence.
        assert len(captured) == 2
        assert "<scenario_instruction>\nTurn-1-only-rubric" not in captured[0][1]
        assert "<scenario_instruction>\nTurn-1-only-rubric" in captured[1][1]


class TestEvidenceScope:
    def test_isolated_shows_only_current_turn(self) -> None:
        scorer = _make_scorer(evidence_scope="isolated")
        scen = _scenario_with_turns([None, None, None])
        captured: list[str] = []

        def _capture(self, sys_msg, dyn_msg, *args, **kwargs):
            captured.append(dyn_msg)
            return _ok_verdict(), {"prompt_tokens": 10, "completion_tokens": 5}

        with patch.object(LLMScorer, "_call_api", new=_capture):
            scorer.score(scen, _dummy_outputs(3))

        turn2 = captured[2]
        # The turn-N rendering uses "### Turn N" headers; isolated
        # means only the current turn header appears.
        assert "### Turn 0" not in turn2
        assert "### Turn 1" not in turn2
        assert "### Turn 2" in turn2

    def test_cumulative_shows_all_turns_up_to_current(self) -> None:
        scorer = _make_scorer(evidence_scope="cumulative")
        scen = _scenario_with_turns([None, None, None])
        captured: list[str] = []

        def _capture(self, sys_msg, dyn_msg, *args, **kwargs):
            captured.append(dyn_msg)
            return _ok_verdict(), {"prompt_tokens": 10, "completion_tokens": 5}

        with patch.object(LLMScorer, "_call_api", new=_capture):
            scorer.score(scen, _dummy_outputs(3))

        turn2 = captured[2]
        assert "### Turn 0" in turn2
        assert "### Turn 1" in turn2
        assert "### Turn 2" in turn2


class TestSkip:
    def test_skip_true_short_circuits_turn(self) -> None:
        scorer = _make_scorer()
        scen = _scenario_with_turns([None, {"skip": True}, None])

        with patch.object(
            LLMScorer,
            "_call_api",
            return_value=(_ok_verdict(), {"prompt_tokens": 10, "completion_tokens": 5}),
        ) as call:
            result = scorer.score(scen, _dummy_outputs(3))

        # Backend only got two calls (turn 1 skipped).
        assert call.call_count == 2
        payload = result.data
        assert isinstance(payload, PerTurnLLMPayload)
        skipped = next(t for t in payload.turns if t.turn_idx == 1)
        assert skipped.dimensions == {}
        assert skipped.judge_errored is False


class TestDimensionOverride:
    def test_replaces_default_dimensions_when_extend_false(self) -> None:
        scorer = _make_scorer()
        scen = _scenario_with_turns(
            [
                None,
                {"dimensions": [{"name": "echo_fidelity", "description": "Did it echo?", "kind": "ternary"}]},
            ]
        )
        custom = _ok_verdict(echo_fidelity="high")

        # Return correctness for turn 0, echo_fidelity for turn 1.
        side: list = [
            (_ok_verdict(), {"prompt_tokens": 10, "completion_tokens": 5}),
            (custom, {"prompt_tokens": 10, "completion_tokens": 5}),
        ]
        with patch.object(LLMScorer, "_call_api", side_effect=side):
            result = scorer.score(scen, _dummy_outputs(2))

        payload = result.data
        assert isinstance(payload, PerTurnLLMPayload)
        assert list(payload.turns[0].dimensions) == ["correctness"]
        assert list(payload.turns[1].dimensions) == ["echo_fidelity"]

    def test_extend_default_dimensions_merges_with_strategy(self) -> None:
        scorer = _make_scorer()
        scen = _scenario_with_turns(
            [
                {
                    "extend_default_dimensions": True,
                    "dimensions": [{"name": "extra", "description": "Extra dim", "kind": "ternary"}],
                }
            ]
        )
        merged = _ok_verdict(correctness="high", extra="high")
        with patch.object(
            LLMScorer,
            "_call_api",
            return_value=(merged, {"prompt_tokens": 10, "completion_tokens": 5}),
        ):
            result = scorer.score(scen, _dummy_outputs(1))

        payload = result.data
        assert isinstance(payload, PerTurnLLMPayload)
        dims = set(payload.turns[0].dimensions)
        assert {"correctness", "extra"} <= dims


class TestPerTurnJudgeErrors:
    """Per-turn judge infrastructure failures must not silently pass.

    The single-judge per-turn payload's ``judge_errored`` is the OR of
    every turn's flag (the documented :class:`TurnVerdict` contract). A
    transient infra error (rate-limit / timeout / network / parse
    failure) on ANY turn leaves that turn unjudged, so the scenario's
    verdict is incomplete and MUST be marked ``judge_errored=True`` with
    ``overall_pass=False`` - identical to the scenario-level path
    (``_score_scenario`` -> ``_judge_errored_payload``) and the
    SCORING.md 2.5.1 guarantee that a flaky provider never turns into a
    green. Regression guard for the silent-pass hole where one errored
    turn among several voting turns let the scenario pass.
    """

    @staticmethod
    def _side_for(error_turns: dict[int, str]):
        """Build a ``_call_api`` replacement: error on the given turns.

        ``error_turns`` maps ``turn_idx`` to an error kind: a
        :data:`belt.scorer.entities.JUDGE_ERROR_TYPES` token raises
        :class:`JudgeInfraError`; the literal ``"parse"`` returns
        ``(None, None)`` to simulate a schema-violating model reply.
        Other turns vote ``high``.
        """

        def _side(self, sys_msg, dyn_msg, scenario_label="", *, strategy=None, turn_idx=None):
            kind = error_turns.get(turn_idx)
            if kind == "parse":
                return None, None
            if kind is not None:
                raise JudgeInfraError(kind, f"simulated {kind}")
            return _ok_verdict(), {"prompt_tokens": 5, "completion_tokens": 2}

        return _side

    def test_one_turn_infra_error_marks_payload_errored_and_fails(self) -> None:
        scorer = _make_scorer()
        scen = _scenario_with_turns([None, None, None])
        with patch.object(LLMScorer, "_call_api", new=self._side_for({1: "rate_limited"})):
            result = scorer.score(scen, _dummy_outputs(3))

        payload = result.data
        assert isinstance(payload, PerTurnLLMPayload)
        assert payload.judge_errored is True, "one errored turn must taint the payload"
        assert payload.judge_error_type == "rate_limited"
        assert payload.overall_pass is False
        assert result.passed is False
        errored = next(t for t in payload.turns if t.turn_idx == 1)
        assert errored.judge_errored is True
        assert errored.dimensions == {}
        # Sibling turns still recorded their real verdicts.
        for idx in (0, 2):
            tv = next(t for t in payload.turns if t.turn_idx == idx)
            assert tv.judge_errored is False
            assert tv.dimensions

    def test_one_turn_parse_failure_classifies_as_other(self) -> None:
        scorer = _make_scorer()
        scen = _scenario_with_turns([None, None])
        with patch.object(LLMScorer, "_call_api", new=self._side_for({0: "parse"})):
            result = scorer.score(scen, _dummy_outputs(2))

        payload = result.data
        assert isinstance(payload, PerTurnLLMPayload)
        assert payload.judge_errored is True
        assert payload.judge_error_type == "other"
        assert payload.overall_pass is False

    def test_first_error_type_is_the_headline(self) -> None:
        scorer = _make_scorer()
        scen = _scenario_with_turns([None, None, None])
        # turn 0 -> timeout (first), turn 1 votes, turn 2 -> rate_limited.
        with patch.object(LLMScorer, "_call_api", new=self._side_for({0: "timeout", 2: "rate_limited"})):
            result = scorer.score(scen, _dummy_outputs(3))

        payload = result.data
        assert isinstance(payload, PerTurnLLMPayload)
        assert payload.judge_errored is True
        assert payload.judge_error_type == "timeout"
        assert payload.overall_pass is False

    def test_all_turns_errored_uses_first_error_not_all_skipped(self) -> None:
        scorer = _make_scorer()
        scen = _scenario_with_turns([None, None])
        with patch.object(LLMScorer, "_call_api", new=self._side_for({0: "timeout", 1: "rate_limited"})):
            result = scorer.score(scen, _dummy_outputs(2))

        payload = result.data
        assert isinstance(payload, PerTurnLLMPayload)
        assert payload.judge_errored is True
        # Real infra error wins the headline over the all-skip fallback.
        assert payload.judge_error_type == "timeout"
        assert payload.overall_pass is False

    def test_clean_run_is_not_falsely_marked_errored(self) -> None:
        # No-false-negative guard: a fully-voted run must keep
        # judge_errored=False / overall_pass=True. Proves the fix does
        # not over-trigger.
        scorer = _make_scorer()
        scen = _scenario_with_turns([None, None, None])
        with patch.object(LLMScorer, "_call_api", new=self._side_for({})):
            result = scorer.score(scen, _dummy_outputs(3))

        payload = result.data
        assert isinstance(payload, PerTurnLLMPayload)
        assert payload.judge_errored is False
        assert payload.overall_pass is True
        assert result.passed is True

    def test_skip_is_not_treated_as_error(self) -> None:
        # A skipped turn is intentional, not an infra failure: a run with
        # a skip + real votes must NOT be marked judge_errored.
        scorer = _make_scorer()
        scen = _scenario_with_turns([None, {"skip": True}, None])
        with patch.object(LLMScorer, "_call_api", new=self._side_for({})):
            result = scorer.score(scen, _dummy_outputs(3))

        payload = result.data
        assert isinstance(payload, PerTurnLLMPayload)
        assert payload.judge_errored is False
        assert payload.overall_pass is True


class TestGenericPerTurnDimsWarning:
    """A per-turn judge with no declared dimensions falls back to the
    generic whole-run dimensions, which are degenerate on a single
    ``isolated`` turn. belt must warn once per judge when that actually
    happens - and must NOT warn when the judge declares dimensions, when
    the turn overrides dimensions, or under ``cumulative`` scope.
    """

    @staticmethod
    def _generic_scorer(evidence_scope: str = "isolated") -> LLMScorer:
        # No strategy => lazy fallback to the generic whole-run dims.
        backend = OpenAIBackend()
        with patch.object(backend, "is_available", return_value=True):
            s = LLMScorer(
                JudgeConfig(model="openai/gpt-4.1-mini"),
                max_retries=0,
                strategy=None,
                backend=backend,
                resolution="turn",
                evidence_scope=evidence_scope,
            )
        s.judge_name = "per_turn_judge"
        return s

    @staticmethod
    def _run_capture(scorer: LLMScorer, scen: Scenario, n: int) -> str:
        buf = StringIO()
        sink = logger.add(buf, level="WARNING", format="{message}")
        try:
            with patch.object(
                LLMScorer,
                "_call_api",
                return_value=(_ok_verdict(execution="high"), {"prompt_tokens": 1, "completion_tokens": 1}),
            ):
                scorer.score(scen, _dummy_outputs(n))
        finally:
            logger.remove(sink)
        return buf.getvalue()

    def test_warns_on_generic_fallback_isolated(self) -> None:
        out = self._run_capture(self._generic_scorer("isolated"), _scenario_with_turns([None, None]), 2)
        assert "generic whole-run dimensions" in out
        assert "per_turn_judge" in out

    def test_warns_only_once_per_judge(self) -> None:
        out = self._run_capture(self._generic_scorer("isolated"), _scenario_with_turns([None, None, None]), 3)
        assert out.count("generic whole-run dimensions") == 1

    def test_no_warning_when_judge_declares_dimensions(self) -> None:
        # _make_scorer passes an explicit strategy -> no fallback.
        out = self._run_capture(_make_scorer(), _scenario_with_turns([None, None]), 2)
        assert "generic whole-run dimensions" not in out

    def test_no_warning_under_cumulative_scope(self) -> None:
        out = self._run_capture(self._generic_scorer("cumulative"), _scenario_with_turns([None, None]), 2)
        assert "generic whole-run dimensions" not in out

    def test_no_warning_when_every_turn_overrides_dimensions(self) -> None:
        dim = {"dimensions": [{"name": "exact", "description": "exact?", "kind": "binary"}]}
        out = self._run_capture(self._generic_scorer("isolated"), _scenario_with_turns([dim, dim]), 2)
        assert "generic whole-run dimensions" not in out
