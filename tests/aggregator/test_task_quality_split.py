# (c) JFrog Ltd. (2026)

"""Tests for the task-quality vs environmental-health headline split.

Covers ``build_task_quality_split`` (the partition helper),
``collect_agent_errors`` (which threads the split into the
``agent_errors`` block), and ``build_bottom_line`` (which renders the
new lead headline when at least one scenario is environment-blocked).

Includes a partition-invariant parity test that pins
``ENVIRONMENTAL_ERROR_TYPES`` and ``TASK_ERROR_TYPES`` as a disjoint
cover of ``ERROR_TYPES``: adding a new canonical error token requires
placing it in exactly one bucket, or this test fails loudly.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from belt.aggregator.stats import build_bottom_line, build_task_quality_split, collect_agent_errors
from belt.constants import SCHEMA_VERSION, TURN_OUTPUT_TEMPLATE
from belt.entities import (
    AUTHENTICATION_FAILED,
    ENVIRONMENTAL_ERROR_TYPES,
    ERROR_TYPES,
    MODEL_UNAVAILABLE,
    RATE_LIMITED,
    REFUSED,
    TASK_ERROR_TYPES,
    TIMEOUT,
    UNKNOWN,
    ScenarioScore,
    TurnOutput,
)
from belt.scorer.payloads import CheckEntry, RulesPayload

# ── fixtures ──


def _write_turn_output(outcome_dir: Path, idx: int, **fields) -> None:
    outcome_dir.mkdir(parents=True, exist_ok=True)
    to = TurnOutput(**fields)
    (outcome_dir / TURN_OUTPUT_TEMPLATE.format(idx)).write_text(to.model_dump_json())


def _make_score(group: str, name: str, *, overall_pass: bool, passed_check: bool = True) -> ScenarioScore:
    rules = RulesPayload(
        checks=[CheckEntry(check="x", dimension="execution", passed=passed_check)],
        passed=passed_check,
    )
    return ScenarioScore(
        schema_version=SCHEMA_VERSION,
        group=group,
        scenario_name=name,
        overall_pass=overall_pass,
        scores={"rules": rules},
    )


def _per_scenario_entry(group: str, name: str, *, passed: bool, error_types: list[str]) -> dict:
    return {
        "scenario": f"{group}/{name}",
        "passed": passed,
        "vacuous_pass": passed,
        "error_types": error_types,
        "first_reply_text": None,
    }


# ── partition invariant ──


class TestErrorTypePartition:
    def test_environmental_and_task_are_disjoint(self) -> None:
        assert ENVIRONMENTAL_ERROR_TYPES.isdisjoint(TASK_ERROR_TYPES)

    def test_environmental_and_task_cover_all_error_types(self) -> None:
        assert ENVIRONMENTAL_ERROR_TYPES | TASK_ERROR_TYPES == ERROR_TYPES

    def test_each_token_belongs_to_exactly_one_bucket(self) -> None:
        # Defence in depth: spell out the expected bucket for each
        # canonical token so a typo in either constant fails this test
        # rather than silently rebalancing the partition.
        assert AUTHENTICATION_FAILED in ENVIRONMENTAL_ERROR_TYPES
        assert RATE_LIMITED in ENVIRONMENTAL_ERROR_TYPES
        assert TIMEOUT in ENVIRONMENTAL_ERROR_TYPES
        assert MODEL_UNAVAILABLE in ENVIRONMENTAL_ERROR_TYPES
        assert REFUSED in TASK_ERROR_TYPES
        assert UNKNOWN in TASK_ERROR_TYPES


# ── build_task_quality_split helper ──


class TestBuildTaskQualitySplit:
    def test_returns_none_when_no_environmental_errors(self) -> None:
        # All errors are task-bucket -> existing single-axis headline
        # is already correct; helper signals "no split needed".
        scores = [_make_score("g", "a", overall_pass=False)]
        per = [_per_scenario_entry("g", "a", passed=False, error_types=[REFUSED])]
        assert build_task_quality_split(scores, per) is None

    def test_returns_none_when_no_errors_at_all(self) -> None:
        scores = [_make_score("g", "a", overall_pass=True)]
        assert build_task_quality_split(scores, []) is None

    def test_single_environmental_failure(self) -> None:
        scores = [
            _make_score("g", "ok", overall_pass=True),
            _make_score("g", "env_blocked", overall_pass=False),
        ]
        per = [_per_scenario_entry("g", "env_blocked", passed=False, error_types=[AUTHENTICATION_FAILED])]
        split = build_task_quality_split(scores, per)
        assert split == {
            "attempted": 2,
            "env_failed": 1,
            "env_failed_agent": 1,
            "env_failed_judge": 0,
            "completed": 1,
            "passed": 1,
            "task_failed": 0,
            "pct": 100.0,
        }

    def test_mixed_env_and_task_failures(self) -> None:
        scores = [
            _make_score("g", "clean_pass", overall_pass=True),
            _make_score("g", "task_fail", overall_pass=False),
            _make_score("g", "env_blocked", overall_pass=False),
        ]
        per = [
            # Only env_blocked carries an environmental error type.
            _per_scenario_entry("g", "task_fail", passed=False, error_types=[REFUSED]),
            _per_scenario_entry("g", "env_blocked", passed=False, error_types=[TIMEOUT]),
        ]
        split = build_task_quality_split(scores, per)
        assert split == {
            "attempted": 3,
            "env_failed": 1,
            "env_failed_agent": 1,
            "env_failed_judge": 0,
            # 3 attempted - 1 env-blocked = 2 completed.
            "completed": 2,
            # of the 2 completed, 1 passed.
            "passed": 1,
            "task_failed": 1,
            "pct": 50.0,
        }

    def test_all_scenarios_env_blocked_completed_zero(self) -> None:
        scores = [
            _make_score("g", "a", overall_pass=False),
            _make_score("g", "b", overall_pass=True),  # vacuous pass
        ]
        per = [
            _per_scenario_entry("g", "a", passed=False, error_types=[RATE_LIMITED]),
            _per_scenario_entry("g", "b", passed=True, error_types=[AUTHENTICATION_FAILED]),
        ]
        split = build_task_quality_split(scores, per)
        # Every attempt was env-blocked -> nothing was completed, so
        # task quality is undefined and the renderer must show "N/A".
        assert split == {
            "attempted": 2,
            "env_failed": 2,
            "env_failed_agent": 2,
            "env_failed_judge": 0,
            "completed": 0,
            "passed": 0,
            "task_failed": 0,
            "pct": None,
        }

    def test_vacuous_pass_in_env_blocked_does_not_count_as_passed(self) -> None:
        # A scenario where rules pass but the agent was env-blocked is
        # never a real pass; the split must not credit it.
        scores = [
            _make_score("g", "ok", overall_pass=True),
            _make_score("g", "vacuous", overall_pass=True),
        ]
        per = [_per_scenario_entry("g", "vacuous", passed=True, error_types=[TIMEOUT])]
        split = build_task_quality_split(scores, per)
        assert split == {
            "attempted": 2,
            "env_failed": 1,
            "env_failed_agent": 1,
            "env_failed_judge": 0,
            "completed": 1,
            # Only the genuinely-clean ``g/ok`` scores as passed.
            "passed": 1,
            "task_failed": 0,
            "pct": 100.0,
        }

    def test_pct_rounded_to_one_decimal(self) -> None:
        scores = [_make_score("g", f"s{i}", overall_pass=(i < 5)) for i in range(7)]
        # 5 of 7 pass. One env-blocked failure -> completed=6, passed=5.
        # 5/6 = 0.8333... -> 83.3% (one decimal).
        per = [_per_scenario_entry("g", "s5", passed=False, error_types=[AUTHENTICATION_FAILED])]
        split = build_task_quality_split(scores, per)
        assert split is not None
        assert split["completed"] == 6
        assert split["passed"] == 5
        assert split["pct"] == 83.3


# ── collect_agent_errors threads the split into the dict ──


class TestCollectAgentErrorsTaskQuality:
    def test_task_quality_present_for_environmental_errors(self, tmp_path: Path) -> None:
        _write_turn_output(
            tmp_path / "g" / "blocked",
            0,
            raw_cli="x",
            reply_text="Not logged in",
            has_error=True,
            error_type=AUTHENTICATION_FAILED,
        )
        scores = [_make_score("g", "blocked", overall_pass=False, passed_check=False)]
        ae = collect_agent_errors(tmp_path, scores)
        assert ae is not None
        assert "task_quality" in ae
        assert ae["task_quality"]["env_failed"] == 1
        assert ae["task_quality"]["completed"] == 0

    def test_task_quality_absent_for_task_only_errors(self, tmp_path: Path) -> None:
        _write_turn_output(
            tmp_path / "g" / "refused_one",
            0,
            raw_cli="x",
            reply_text="Sorry, I can't help with that.",
            has_error=True,
            error_type=REFUSED,
        )
        scores = [_make_score("g", "refused_one", overall_pass=False, passed_check=False)]
        ae = collect_agent_errors(tmp_path, scores)
        assert ae is not None
        # ``refused`` is a TASK error: the legacy single-axis headline
        # already conveys it, so no split is attached.
        assert "task_quality" not in ae


# ── build_bottom_line renders the split ──


class TestBuildBottomLineSplit:
    def _ae(self, *, env_failed: int, completed: int, passed: int, task_failed: int, pct: float | None) -> dict:
        return {
            "scenarios_with_errors": env_failed,
            "scenarios_total": completed + env_failed,
            "vacuous_passes": 0,
            "by_error_type": {AUTHENTICATION_FAILED: env_failed} if env_failed else {},
            "per_scenario": [],
            "task_quality": {
                "attempted": completed + env_failed,
                "env_failed": env_failed,
                "completed": completed,
                "passed": passed,
                "task_failed": task_failed,
                "pct": pct,
            },
        }

    def test_split_headline_replaces_legacy_headline(self) -> None:
        scores = [
            _make_score("g", "ok", overall_pass=True),
            _make_score("g", "blocked", overall_pass=False, passed_check=False),
        ]
        ae = self._ae(env_failed=1, completed=1, passed=1, task_failed=0, pct=100.0)
        lines = build_bottom_line(scores, agent_errors=ae)
        # Three-axis headline: env failures are now split into agent-env
        # and judge-env so a reader can attribute infra blame correctly.
        assert lines[0] == "1/1 task quality (100.0%) - 1 agent env failure - 0 agent task failures"
        # Legacy single-axis line must NOT be present alongside the split.
        assert not any(line.endswith(" scenarios failed.") for line in lines)

    def test_split_headline_pluralises_correctly(self) -> None:
        scores = [_make_score("g", f"s{i}", overall_pass=(i == 0)) for i in range(5)]
        ae = self._ae(env_failed=2, completed=3, passed=1, task_failed=2, pct=33.3)
        lines = build_bottom_line(scores, agent_errors=ae)
        assert lines[0].startswith("1/3 task quality (33.3%)")
        # Plural forms when count != 1. The agent-env axis carries them
        # because this test's fixture loads only the agent side.
        assert "2 agent env failures" in lines[0]
        assert "2 agent task failures" in lines[0]

    def test_split_headline_handles_zero_completed(self) -> None:
        scores = [_make_score("g", "blocked", overall_pass=False, passed_check=False)]
        ae = self._ae(env_failed=1, completed=0, passed=0, task_failed=0, pct=None)
        lines = build_bottom_line(scores, agent_errors=ae)
        assert lines[0].startswith("0/0 task quality (N/A)")
        assert "1 agent env failure" in lines[0]

    def test_split_does_not_appear_without_env_failures(self) -> None:
        scores = [_make_score("g", "task_fail", overall_pass=False, passed_check=False)]
        # Mimic ``collect_agent_errors`` for a refused-only run: no
        # ``task_quality`` key on the dict at all.
        ae = {
            "scenarios_with_errors": 1,
            "scenarios_total": 1,
            "vacuous_passes": 0,
            "by_error_type": {REFUSED: 1},
            "per_scenario": [],
        }
        lines = build_bottom_line(scores, agent_errors=ae)
        assert lines[0] == "1/1 scenarios failed."


# ── Smoke against the documented argument shape ──


class TestPublicShape:
    def test_split_keys_are_documented_set(self) -> None:
        # Pin the public dict shape so adding a new key elsewhere can't
        # silently miss the rendering / card / docs.
        scores = [
            _make_score("g", "ok", overall_pass=True),
            _make_score("g", "blocked", overall_pass=False),
        ]
        per = [_per_scenario_entry("g", "blocked", passed=False, error_types=[TIMEOUT])]
        split = build_task_quality_split(scores, per)
        assert split is not None
        assert set(split.keys()) == {
            "attempted",
            "env_failed",
            "env_failed_agent",
            "env_failed_judge",
            "completed",
            "passed",
            "task_failed",
            "pct",
        }

    @pytest.mark.parametrize("env_token", sorted(ENVIRONMENTAL_ERROR_TYPES))
    def test_every_environmental_token_triggers_split(self, env_token: str) -> None:
        scores = [_make_score("g", "a", overall_pass=False)]
        per = [_per_scenario_entry("g", "a", passed=False, error_types=[env_token])]
        assert build_task_quality_split(scores, per) is not None

    @pytest.mark.parametrize("task_token", sorted(TASK_ERROR_TYPES))
    def test_every_task_token_skips_split(self, task_token: str) -> None:
        scores = [_make_score("g", "a", overall_pass=False)]
        per = [_per_scenario_entry("g", "a", passed=False, error_types=[task_token])]
        assert build_task_quality_split(scores, per) is None
