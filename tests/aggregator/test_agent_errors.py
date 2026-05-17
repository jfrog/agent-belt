# (c) JFrog Ltd. (2026)

"""Tests for the aggregator's agent-error surfacing.

Covers :func:`collect_agent_errors`, :func:`build_bottom_line`,
:func:`_agent_error_context`, and the ``print_terminal`` integration
that surfaces vacuous passes in the failures section.

The tests use small on-disk fixtures (turn output JSONs) rather than
mocks so the file-based phase contract is exercised end-to-end at
the aggregator boundary.
"""

from __future__ import annotations

from pathlib import Path

from belt.aggregator.stats import build_bottom_line, collect_agent_errors, load_turn_outputs_for_scenario
from belt.constants import SCHEMA_VERSION, TURN_OUTPUT_TEMPLATE
from belt.entities import ScenarioScore, TurnOutput
from belt.scorer.payloads import CheckEntry, RulesPayload


def _write_turn_output(outcome_dir: Path, idx: int, **fields) -> None:
    outcome_dir.mkdir(parents=True, exist_ok=True)
    to = TurnOutput(**fields)
    (outcome_dir / TURN_OUTPUT_TEMPLATE.format(idx)).write_text(to.model_dump_json())


def _make_score(group: str, name: str, *, overall_pass: bool, checks: list[CheckEntry]) -> ScenarioScore:
    rules = RulesPayload(checks=checks, passed=all(c.passed for c in checks))
    return ScenarioScore(
        schema_version=SCHEMA_VERSION,
        group=group,
        scenario_name=name,
        overall_pass=overall_pass,
        scores={"rules": rules},
    )


class TestLoadTurnOutputs:
    def test_loads_in_order_and_stops_at_gap(self, tmp_path: Path) -> None:
        d = tmp_path / "scenario"
        _write_turn_output(d, 0, raw_cli="t0", reply_text="r0")
        _write_turn_output(d, 1, raw_cli="t1", reply_text="r1")
        # Skip 2; loader must stop here, not jump to 3.
        _write_turn_output(d, 3, raw_cli="t3", reply_text="r3")
        outputs = load_turn_outputs_for_scenario(d)
        assert [o.reply_text for o in outputs] == ["r0", "r1"]

    def test_returns_empty_when_dir_empty(self, tmp_path: Path) -> None:
        assert load_turn_outputs_for_scenario(tmp_path) == []


class TestCollectAgentErrors:
    def test_returns_none_when_no_scenario_errored(self, tmp_path: Path) -> None:
        d = tmp_path / "g" / "s"
        _write_turn_output(d, 0, raw_cli="ok", reply_text="hello", has_error=False)
        score = _make_score(
            "g",
            "s",
            overall_pass=True,
            checks=[CheckEntry(check="x", dimension="execution", passed=True)],
        )
        assert collect_agent_errors(tmp_path, [score]) is None

    def test_counts_agent_errors_and_vacuous_passes(self, tmp_path: Path) -> None:
        # Scenario A: rules failed AND agent errored - normal failure.
        _write_turn_output(
            tmp_path / "g" / "a",
            0,
            raw_cli="x",
            reply_text="Not logged in · Please run /login",
            has_error=True,
            error_type="authentication_failed",
        )
        # Scenario B: rules passed BUT agent errored - vacuous pass.
        _write_turn_output(
            tmp_path / "g" / "b",
            0,
            raw_cli="x",
            reply_text="Not logged in",
            has_error=True,
            error_type="authentication_failed",
        )
        # Scenario C: clean pass - must not appear in any count.
        _write_turn_output(
            tmp_path / "g" / "c",
            0,
            raw_cli="x",
            reply_text="great answer",
            has_error=False,
        )
        scores = [
            _make_score(
                "g", "a", overall_pass=False, checks=[CheckEntry(check="x", dimension="execution", passed=False)]
            ),
            _make_score(
                "g", "b", overall_pass=True, checks=[CheckEntry(check="x", dimension="execution", passed=True)]
            ),
            _make_score(
                "g", "c", overall_pass=True, checks=[CheckEntry(check="x", dimension="execution", passed=True)]
            ),
        ]
        ae = collect_agent_errors(tmp_path, scores)
        assert ae is not None
        assert ae["scenarios_with_errors"] == 2
        assert ae["scenarios_total"] == 3
        assert ae["vacuous_passes"] == 1
        assert ae["by_error_type"] == {"authentication_failed": 2}
        # per_scenario carries the first reply text so terminal renderer
        # can show the agent's own message (instead of buried in JSON).
        per = {s["scenario"]: s for s in ae["per_scenario"]}
        assert per["g/b"]["vacuous_pass"] is True
        assert per["g/a"]["vacuous_pass"] is False
        assert "logged in" in per["g/a"]["first_reply_text"].lower()

    def test_unknown_fallback_when_error_type_missing(self, tmp_path: Path) -> None:
        # Adapter set has_error=True but didn't classify (older runs / new pattern).
        _write_turn_output(tmp_path / "g" / "s", 0, raw_cli="x", has_error=True)
        score = _make_score(
            "g",
            "s",
            overall_pass=False,
            checks=[CheckEntry(check="x", dimension="execution", passed=False)],
        )
        ae = collect_agent_errors(tmp_path, [score])
        assert ae["by_error_type"] == {"unknown": 1}

    def test_remediation_added_when_agent_named(self, tmp_path: Path) -> None:
        _write_turn_output(
            tmp_path / "g" / "s",
            0,
            raw_cli="x",
            reply_text="Not logged in",
            has_error=True,
            error_type="authentication_failed",
        )
        score = _make_score(
            "g",
            "s",
            overall_pass=False,
            checks=[CheckEntry(check="x", dimension="execution", passed=False)],
        )
        ae = collect_agent_errors(tmp_path, [score], agent_name="claude-code")
        assert "claude login" in ae["remediation"]

        ae_no_agent = collect_agent_errors(tmp_path, [score])
        assert "remediation" not in ae_no_agent


class TestBuildBottomLine:
    def test_clean_pass_unchanged(self) -> None:
        scores = [
            _make_score("g", "s", overall_pass=True, checks=[CheckEntry(check="x", dimension="execution", passed=True)])
        ]
        lines = build_bottom_line(scores)
        assert lines == ["All 1 scenarios passed."]

    def test_agent_error_headline_precedes_rule_headline(self) -> None:
        # Authentication errors are environmental: the new headline is
        # the task-quality / environmental-health split. The single-axis
        # "M/N scenarios failed" line is dropped because the split tells
        # that story already, with the right denominator.
        scores = [
            _make_score(
                "g", "a", overall_pass=False, checks=[CheckEntry(check="x", dimension="execution", passed=False)]
            ),
        ]
        ae = {
            "scenarios_with_errors": 1,
            "scenarios_total": 1,
            "vacuous_passes": 0,
            "by_error_type": {"authentication_failed": 1},
            "per_scenario": [
                {
                    "scenario": "g/a",
                    "passed": False,
                    "vacuous_pass": False,
                    "error_types": ["authentication_failed"],
                    "first_reply_text": "Not logged in",
                }
            ],
            "task_quality": {
                "attempted": 1,
                "env_failed": 1,
                "completed": 0,
                "passed": 0,
                "task_failed": 0,
                "pct": None,
            },
            "remediation": "Re-authenticate the Claude Code CLI: run `claude login`.",
        }
        lines = build_bottom_line(scores, agent_errors=ae)
        joined = "\n".join(lines)
        # Lead headline is the task-quality split.
        assert lines[0].startswith("0/0 task quality")
        assert "1 agent env failure" in lines[0]
        assert "0 agent task failures" in lines[0]
        # Per-type detail line still follows.
        assert "Agent error in 1/1" in lines[1]
        assert "claude login" in lines[1]
        # No "M/N scenarios failed." line is emitted when the split is
        # active (the split already conveys it).
        assert not any("scenarios failed." in line for line in lines), joined

    def test_vacuous_pass_warning(self) -> None:
        # All rules pass but agent had errors. With auth (environmental)
        # the split-headline appears with completed=0 (everything blocked)
        # and the vacuous-pass warning still surfaces.
        scores = [
            _make_score(
                "g", "a", overall_pass=True, checks=[CheckEntry(check="x", dimension="execution", passed=True)]
            ),
        ]
        ae = {
            "scenarios_with_errors": 1,
            "scenarios_total": 1,
            "vacuous_passes": 1,
            "by_error_type": {"authentication_failed": 1},
            "per_scenario": [
                {
                    "scenario": "g/a",
                    "passed": True,
                    "vacuous_pass": True,
                    "error_types": ["authentication_failed"],
                    "first_reply_text": "Not logged in",
                }
            ],
            "task_quality": {
                "attempted": 1,
                "env_failed": 1,
                "completed": 0,
                "passed": 0,
                "task_failed": 0,
                "pct": None,
            },
        }
        lines = build_bottom_line(scores, agent_errors=ae)
        joined = "\n".join(lines)
        assert lines[0].startswith("0/0 task quality")
        assert "1 agent env failure" in lines[0]
        assert any("vacuous" in line.lower() for line in lines), joined

    def test_task_only_errors_use_legacy_format(self) -> None:
        # ``refused`` and ``unknown`` are TASK errors, not environmental.
        # The existing single-axis headline is preserved (no split).
        scores = [
            _make_score(
                "g", "a", overall_pass=False, checks=[CheckEntry(check="x", dimension="execution", passed=False)]
            ),
        ]
        ae = {
            "scenarios_with_errors": 1,
            "scenarios_total": 1,
            "vacuous_passes": 0,
            "by_error_type": {"refused": 1},
            "per_scenario": [
                {
                    "scenario": "g/a",
                    "passed": False,
                    "vacuous_pass": False,
                    "error_types": ["refused"],
                    "first_reply_text": "I can't help with that.",
                }
            ],
            # Note: build_task_quality_split returns None for task-only
            # error sets, so collect_agent_errors does not attach
            # ``task_quality`` to the dict.
        }
        lines = build_bottom_line(scores, agent_errors=ae)
        # Legacy single-axis headline kept verbatim.
        assert lines[0] == "1/1 scenarios failed."
        assert "Agent error in 1/1" in lines[1]
        assert "refused" in lines[1]
