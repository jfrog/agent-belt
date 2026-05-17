# (c) JFrog Ltd. (2026)

"""Regression tests for the LLM-judge infrastructure-failure axis.

The aggregator must surface judge-infra failures as their own axis so
that four classes of correctness bug cannot recur:

1. False green: rules pass, every dimension's judge silently dropped a
   verdict, ``overall_pass=True`` would otherwise read as a real green.
2. Wrong-axis red: rules fail, attribution would otherwise blame the
   agent's task quality instead of the unreachable judge.
3. Inconsistent visibility: judge failures must appear in the headline
   and the failure list, not only in the rules execution block.
4. Every consumer reinventing detection: downstream eval runners must be
   able to read one typed field (``LLMPayload.judge_errored``) instead
   of regex-matching grader evidence strings.

These tests pin the contract end-to-end so a future refactor that
re-introduces silent ``None`` returns from the scorer or that drops the
``judge_errored`` field from the payload fails loudly here.
"""

from __future__ import annotations

from pathlib import Path

from belt.aggregator.stats import (
    build_bottom_line,
    build_task_quality_split,
    collect_agent_errors,
    collect_judge_errors,
)
from belt.constants import SCHEMA_VERSION, TURN_OUTPUT_TEMPLATE
from belt.entities import AUTHENTICATION_FAILED, ScenarioScore, TurnOutput
from belt.scorer.entities import JUDGE_ERROR_TYPES
from belt.scorer.payloads import CheckEntry, LLMDimensionVerdict, LLMPayload, RulesPayload

# ── fixtures ──


def _make_score(
    group: str,
    name: str,
    *,
    overall_pass: bool,
    rules_passed: bool = True,
    judge_errored: bool = False,
    judge_error_type: str | None = None,
    llm_dimensions: dict[str, str] | None = None,
) -> ScenarioScore:
    """Build a ``ScenarioScore`` with rules + optional LLM payload.

    ``llm_dimensions`` is a map of ``dim -> verdict_token`` for happy-path
    scenarios; pass it as None to skip the LLM scorer entirely.
    ``judge_errored`` produces a verdict-less LLM payload (mirrors what
    the scorer writes on infra failure).
    """
    scores: dict = {
        "rules": RulesPayload(
            checks=[CheckEntry(check="x", dimension="execution", passed=rules_passed)],
            passed=rules_passed,
        )
    }
    if judge_errored:
        scores["llm"] = LLMPayload(
            overall_pass=False,
            dimensions={},
            judge_errored=True,
            judge_error_type=judge_error_type or "other",  # type: ignore[arg-type]
        )
    elif llm_dimensions:
        scores["llm"] = LLMPayload(
            overall_pass=overall_pass,
            dimensions={dim: LLMDimensionVerdict(score=score, reasoning="ok") for dim, score in llm_dimensions.items()},
        )
    return ScenarioScore(
        schema_version=SCHEMA_VERSION,
        group=group,
        scenario_name=name,
        overall_pass=overall_pass,
        scores=scores,
    )


def _write_turn_output(outcome_dir: Path, idx: int, **fields) -> None:
    outcome_dir.mkdir(parents=True, exist_ok=True)
    to = TurnOutput(**fields)
    (outcome_dir / TURN_OUTPUT_TEMPLATE.format(idx)).write_text(to.model_dump_json())


# ── pain 4: typed contract for downstream consumers ──


class TestJudgeErroredField:
    def test_default_payload_is_not_errored(self) -> None:
        payload = LLMPayload(overall_pass=True, dimensions={})
        assert payload.judge_errored is False
        assert payload.judge_error_type is None

    def test_judge_error_type_accepts_every_documented_token(self) -> None:
        for token in JUDGE_ERROR_TYPES:
            payload = LLMPayload(
                overall_pass=False,
                dimensions={},
                judge_errored=True,
                judge_error_type=token,  # type: ignore[arg-type]
            )
            assert payload.judge_error_type == token

    def test_payload_roundtrips_through_json(self) -> None:
        original = LLMPayload(
            overall_pass=False,
            dimensions={},
            judge_errored=True,
            judge_error_type="rate_limited",
        )
        dumped = original.model_dump(mode="json")
        roundtripped = LLMPayload.model_validate(dumped)
        assert roundtripped.judge_errored is True
        assert roundtripped.judge_error_type == "rate_limited"


# ── pain 1: false-green prevention ──


class TestFalseGreenPrevention:
    def test_collect_judge_errors_flags_silent_passes(self) -> None:
        # The trap: rules-only scenario with judge_errored payload would
        # report ``overall_pass=True`` if the pipeline did not force a
        # synthetic check. ``collect_judge_errors`` is the structural
        # signal a downstream consumer can use without parsing strings.
        scores = [
            _make_score("g", "silent_a", overall_pass=True, judge_errored=True, judge_error_type="rate_limited"),
        ]
        judge_errors = collect_judge_errors(scores)
        assert judge_errors is not None
        assert judge_errors["scenarios_with_errors"] == 1
        assert judge_errors["by_error_type"] == {"rate_limited": 1}
        assert judge_errors["per_scenario"][0]["scenario"] == "g/silent_a"
        assert judge_errors["per_scenario"][0]["error_type"] == "rate_limited"

    def test_partition_subtracts_judge_failures_from_pass_rate_denominator(self) -> None:
        # 2 scenarios attempted: one clean, one judge-errored. The
        # task-quality denominator must be ``completed=1`` (not 2), so
        # downstream gating reads "1/1 (100%) task quality" plus
        # "1 judge env failure" rather than "1/2 (50%) failed".
        scores = [
            _make_score("g", "ok", overall_pass=True),
            _make_score("g", "judge_died", overall_pass=False, judge_errored=True),
        ]
        judge_errors = collect_judge_errors(scores)
        split = build_task_quality_split(scores, [], judge_per_scenario=(judge_errors or {}).get("per_scenario", []))
        assert split is not None
        assert split["attempted"] == 2
        assert split["env_failed_judge"] == 1
        assert split["env_failed_agent"] == 0
        assert split["completed"] == 1
        assert split["passed"] == 1
        assert split["task_failed"] == 0


# ── pain 2: wrong-axis-red prevention ──


class TestWrongAxisRedPrevention:
    def test_judge_failure_does_not_inflate_task_failure_count(self) -> None:
        # Without the judge axis, a judge timeout would manifest as
        # ``task_failed += 1`` (the agent ran, rules passed, but
        # overall_pass=False because of the synthetic check). The split
        # must attribute it to the judge axis instead.
        scores = [
            _make_score("g", "clean_ok", overall_pass=True),
            _make_score("g", "task_fail", overall_pass=False, rules_passed=False),
            _make_score(
                "g",
                "judge_timeout",
                overall_pass=False,
                rules_passed=True,
                judge_errored=True,
                judge_error_type="timeout",
            ),
        ]
        judge_errors = collect_judge_errors(scores)
        split = build_task_quality_split(scores, [], judge_per_scenario=(judge_errors or {}).get("per_scenario", []))
        assert split is not None
        assert split["env_failed_judge"] == 1
        assert split["env_failed_agent"] == 0
        # The only task failure is the rules-failed scenario; the
        # judge-timeout one must not double-count.
        assert split["task_failed"] == 1
        assert split["completed"] == 2

    def test_agent_axis_wins_over_judge_axis_on_overlap(self) -> None:
        # When a scenario's agent errored AND its judge would have
        # errored, the agent axis attributes it. The judge could not
        # have voted on an agent that never ran, so attributing it to
        # the judge would mislead the operator.
        scores = [
            _make_score(
                "g",
                "both",
                overall_pass=False,
                judge_errored=True,
                judge_error_type="timeout",
            ),
        ]
        agent_per = [
            {
                "scenario": "g/both",
                "passed": False,
                "vacuous_pass": False,
                "error_types": [AUTHENTICATION_FAILED],
                "first_reply_text": None,
            }
        ]
        judge_errors = collect_judge_errors(scores)
        split = build_task_quality_split(
            scores,
            agent_per,
            judge_per_scenario=(judge_errors or {}).get("per_scenario", []),
        )
        assert split is not None
        assert split["env_failed_agent"] == 1
        assert split["env_failed_judge"] == 0
        assert split["env_failed"] == 1


# ── pain 3: visibility in the headline ──


class TestHeadlineVisibility:
    def test_judge_only_run_produces_three_axis_headline(self) -> None:
        scores = [
            _make_score("g", "ok", overall_pass=True),
            _make_score("g", "judge_died", overall_pass=False, judge_errored=True, judge_error_type="rate_limited"),
        ]
        judge_errors = collect_judge_errors(scores)
        ae = collect_agent_errors(Path("/tmp"), scores, judge_errors=judge_errors)
        # Even with no agent errors, the partition is computed so the
        # bottom line carries the judge-axis count.
        assert ae is not None
        assert "task_quality" in ae
        lines = build_bottom_line(scores, agent_errors=ae, judge_errors=judge_errors)
        assert lines[0].startswith("1/1 task quality")
        assert "1 judge env failure" in lines[0]
        # A dedicated judge-infra detail line follows the headline.
        assert any("LLM judge infrastructure failure" in line for line in lines)

    def test_judge_errors_dropped_when_no_scenarios_affected(self) -> None:
        scores = [_make_score("g", "ok", overall_pass=True)]
        assert collect_judge_errors(scores) is None

    def test_mixed_axes_both_visible(self, tmp_path: Path) -> None:
        # Agent-axis failure and judge-axis failure on different scenarios.
        _write_turn_output(
            tmp_path / "g" / "auth_blocked",
            0,
            raw_cli="x",
            reply_text="Not logged in",
            has_error=True,
            error_type=AUTHENTICATION_FAILED,
        )
        scores = [
            _make_score("g", "ok", overall_pass=True),
            _make_score("g", "auth_blocked", overall_pass=False, rules_passed=False),
            _make_score(
                "g",
                "judge_died",
                overall_pass=False,
                judge_errored=True,
                judge_error_type="rate_limited",
            ),
        ]
        judge_errors = collect_judge_errors(scores)
        ae = collect_agent_errors(tmp_path, scores, judge_errors=judge_errors)
        assert ae is not None
        split = ae["task_quality"]
        assert split["env_failed_agent"] == 1
        assert split["env_failed_judge"] == 1
        assert split["env_failed"] == 2
        lines = build_bottom_line(scores, agent_errors=ae, judge_errors=judge_errors)
        # Both axes named in the headline.
        assert "1 agent env failure" in lines[0]
        assert "1 judge env failure" in lines[0]
