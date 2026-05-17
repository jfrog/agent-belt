# (c) JFrog Ltd. (2026)

"""Tests for aggregator threshold parsing, validation, counting, and enforcement."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest

from belt.commands.aggregate import (
    ThresholdCheck,
    _compute_partial_score,
    _evidence_paths,
    _extract_error_lines,
    _failure_context,
    build_result_table,
    build_stats,
    compute_reliability,
    count_llm_failures,
    count_rules_failures,
    discover_llm_dimensions,
    enforce_thresholds,
    parse_threshold,
    validate_thresholds,
)
from belt.entities import ScenarioScore

# ── parse_threshold ──


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("rules/execution:0", ("rules", "execution", 0.0)),
        ("rules/trajectory:10", ("rules", "trajectory", 10.0)),
        ("llm/response_quality:15.5", ("llm", "response_quality", 15.5)),
        ("rules/execution:100", ("rules", "execution", 100.0)),
    ],
)
def test_parse_threshold_valid(raw: str, expected: tuple) -> None:
    assert parse_threshold(raw) == expected


@pytest.mark.parametrize(
    "raw",
    [
        "noslash",
        "rules/execution",
        "rules:10",
        "rules/execution:abc",
        "rules/execution:-1",
        "rules/execution:101",
    ],
)
def test_parse_threshold_invalid(raw: str) -> None:
    with pytest.raises(argparse.ArgumentTypeError):
        parse_threshold(raw)


# ── validate_thresholds ──


def test_validate_thresholds_valid(capsys: pytest.CaptureFixture) -> None:
    validate_thresholds({"rules": {"execution": 0, "trajectory": 10}, "llm": {"response_quality": 15}})


def test_validate_thresholds_bad_mode() -> None:
    from belt.errors import ConfigError

    with pytest.raises(ConfigError, match="Unknown mode"):
        validate_thresholds({"cow": {"horse": 50}})


def test_validate_thresholds_rules_dims_accepted_any() -> None:
    """Rules dimensions are validated at enforcement time, not upfront."""
    validate_thresholds({"rules": {"custom_dimension": 10}})


def test_validate_thresholds_bad_llm_dim() -> None:
    from belt.errors import ConfigError

    with pytest.raises(ConfigError, match="unknown dimension"):
        validate_thresholds({"llm": {"bogus": 10}}, known_llm_dims={"execution", "trajectory"})


def test_validate_thresholds_llm_no_known_dims_accepts_any() -> None:
    """Without known_llm_dims, any LLM dimension is accepted (validated at enforcement)."""
    validate_thresholds({"llm": {"anything_goes": 10}})


# ── Helpers ──


def _score(name: str, group: str, rules_checks: list[dict], llm: dict | None = None) -> ScenarioScore:
    from belt.scorer.payloads import LLMDimensionVerdict, LLMPayload, RulesPayload, UsageStats

    scores: dict = {}
    rules_passed_flag = True
    if rules_checks:
        rules_passed_flag = all(c["passed"] for c in rules_checks)
        scores["rules"] = RulesPayload(
            checks=rules_checks,
            passed=rules_passed_flag,
        )
    llm_overall = True
    if llm:
        meta_keys = ("overall_pass", "usage", "consensus_meta", "individual_verdicts", "schema_version")
        dims = {
            k: (LLMDimensionVerdict(**v) if isinstance(v, dict) and "score" in v else v)
            for k, v in llm.items()
            if k not in meta_keys
        }
        llm_overall = bool(llm.get("overall_pass", True))
        usage = llm.get("usage")
        scores["llm"] = LLMPayload(
            overall_pass=llm_overall,
            dimensions=dims,
            usage=UsageStats(**usage) if isinstance(usage, dict) else None,
        )
    overall = rules_passed_flag and llm_overall
    return ScenarioScore(scenario_name=name, group=group, scores=scores, overall_pass=overall)


def _check(dim: str, check: str, passed: bool) -> dict:
    return {"dimension": dim, "check": check, "passed": passed}


# ── count_rules_failures ──


def test_count_rules_failures_all_pass() -> None:
    scores = [
        _score("a", "g1", [_check("execution", "no_errors", True), _check("trajectory", "tool_invoked(x)", True)]),
        _score("b", "g1", [_check("execution", "no_errors", True)]),
    ]
    result = count_rules_failures(scores)
    assert result["execution"] == (0, 2)
    assert result["trajectory"] == (0, 1)


def test_count_rules_failures_mixed() -> None:
    scores = [
        _score("a", "g1", [_check("execution", "no_errors", True), _check("trajectory", "tool_invoked(x)", False)]),
        _score("b", "g1", [_check("execution", "no_errors", False)]),
        _score("c", "g1", [_check("execution", "no_errors", True), _check("trajectory", "tool_invoked(y)", True)]),
    ]
    result = count_rules_failures(scores)
    assert result["execution"] == (1, 3)
    assert result["trajectory"] == (1, 2)


def test_count_rules_multiple_checks_same_dim_one_scenario() -> None:
    """A scenario with 2 trajectory checks, one passing one failing = 1 failed scenario."""
    scores = [
        _score(
            "a",
            "g1",
            [
                _check("trajectory", "tool_invoked(x)", True),
                _check("trajectory", "tool_invoked(y)", False),
            ],
        ),
    ]
    result = count_rules_failures(scores)
    assert result["trajectory"] == (1, 1)


# ── count_llm_failures ──


def test_count_llm_failures_low_only() -> None:
    scores = [
        _score("a", "g1", [], llm={"execution": {"score": "high"}, "response_quality": {"score": "low"}}),
        _score("b", "g1", [], llm={"execution": {"score": "low"}, "response_quality": {"score": "high"}}),
    ]
    result = count_llm_failures(scores, {"low"})
    assert result["execution"] == (1, 2)
    assert result["response_quality"] == (1, 2)


def test_count_llm_failures_low_and_medium() -> None:
    scores = [
        _score("a", "g1", [], llm={"execution": {"score": "medium"}}),
    ]
    result = count_llm_failures(scores, {"low", "medium"})
    assert result["execution"] == (1, 1)


def test_count_llm_failures_medium_not_counted_by_default() -> None:
    scores = [
        _score("a", "g1", [], llm={"execution": {"score": "medium"}}),
    ]
    result = count_llm_failures(scores, {"low"})
    assert result["execution"] == (0, 1)


# ── enforce_thresholds ──


def test_enforce_all_pass() -> None:
    lines, passed, checks = enforce_thresholds(
        {"rules": {"execution": 0}},
        {"execution": (0, 20)},
        {},
    )
    assert passed is True
    assert "0.0% failed" in lines[0]
    assert len(checks) == 1
    assert checks[0].passed is True
    assert checks[0].actual_pct == 0.0


def test_enforce_over_threshold() -> None:
    lines, passed, checks = enforce_thresholds(
        {"rules": {"trajectory": 5}},
        {"trajectory": (3, 20)},
        {},
    )
    assert passed is False
    assert "15.0% failed" in lines[0]
    assert checks[0].passed is False
    assert checks[0].actual_pct == 15.0
    assert checks[0].max_pct == 5.0


def test_enforce_exact_boundary() -> None:
    _lines, passed, _checks = enforce_thresholds(
        {"rules": {"trajectory": 10}},
        {"trajectory": (2, 20)},
        {},
    )
    assert passed is True


def test_enforce_missing_dimension_is_zero() -> None:
    """Threshold on a dimension with no data = 0/0 = 0% = passes."""
    lines, passed, checks = enforce_thresholds(
        {"rules": {"review": 0}},
        {},
        {},
    )
    assert passed is True
    assert "0.0% failed" in lines[0]
    assert checks[0].dimension == "rules/review"


# ── _extract_error_lines ──


def test_extract_error_lines_traceback() -> None:
    cli = "some output\nTraceback (most recent call last):\n  File foo.py\nValueError: bad\n"
    lines = _extract_error_lines(cli)
    assert lines[0] == "Traceback (most recent call last):"
    assert "ValueError: bad" in lines


def test_extract_error_lines_runtimeerror() -> None:
    cli = "RuntimeError: agent failed on turn 0 - timed out after 180s\n"
    lines = _extract_error_lines(cli)
    assert len(lines) == 1
    assert "RuntimeError" in lines[0]


def test_extract_error_lines_none_found() -> None:
    cli = "all good here\nno problems at all\n"
    assert _extract_error_lines(cli) == []


# ── _failure_context ──


def test_failure_context_no_errors(tmp_path: Path) -> None:
    """execution/no_errors failure shows actual error text from CLI file."""
    (tmp_path / "g" / "s").mkdir(parents=True)
    cli = "some output\nRuntimeError: agent failed on turn 0 - timed out after 180s\nmore stuff\n"
    (tmp_path / "g" / "s" / "turn_0_cli.txt").write_text(cli)

    score = ScenarioScore(
        scenario_name="s",
        group="g",
        scores={
            "rules": {
                "schema_version": "rules.v1",
                "checks": [
                    {"dimension": "execution", "check": "no_errors", "passed": False, "details": "", "turn_idx": 0}
                ],
                "passed": False,
            }
        },
        overall_pass=False,
    )
    lines = _failure_context(score, tmp_path / "g" / "s")
    assert any("RuntimeError" in line for line in lines)
    assert not any("│" in line or "└" in line for line in lines)


def test_failure_context_tool_invoked(tmp_path: Path) -> None:
    """trajectory/tool_invoked failure shows actual tools from TurnOutput."""
    outcome = tmp_path / "g" / "s"
    outcome.mkdir(parents=True)
    (outcome / "turn_0_cli.txt").write_text("some output")
    turn_output = {
        "raw_cli": "some output",
        "tool_calls": [
            {"name": "search_artifacts", "call_id": "c1", "args": {}},
            {"name": "reply_to_user", "call_id": "c2", "args": {}},
        ],
    }
    (outcome / "turn_0_output.json").write_text(json.dumps(turn_output))

    score = ScenarioScore(
        scenario_name="s",
        group="g",
        scores={
            "rules": {
                "schema_version": "rules.v1",
                "checks": [
                    {
                        "dimension": "trajectory",
                        "check": "tool_invoked(manage_environments)",
                        "passed": False,
                        "details": "",
                        "turn_idx": 0,
                    }
                ],
                "passed": False,
            }
        },
        overall_pass=False,
    )
    lines = _failure_context(score, outcome)
    assert any("search_artifacts" in line for line in lines)
    assert any("reply_to_user" in line for line in lines)


def test_failure_context_contains(tmp_path: Path) -> None:
    """response/contains failure shows actual response tail."""
    outcome = tmp_path / "g" / "s"
    outcome.mkdir(parents=True)
    (outcome / "turn_0_cli.txt").write_text("here is the actual response text from cli")

    score = ScenarioScore(
        scenario_name="s",
        group="g",
        scores={
            "rules": {
                "schema_version": "rules.v1",
                "checks": [
                    {
                        "dimension": "response",
                        "check": "contains(load-test)",
                        "passed": False,
                        "details": "",
                        "turn_idx": 0,
                    }
                ],
                "passed": False,
            }
        },
        overall_pass=False,
    )
    lines = _failure_context(score, outcome)
    assert any("actual response text" in line for line in lines)


def test_failure_context_missing_turn_idx() -> None:
    """Checks without turn_idx (old scores) don't crash."""
    score = ScenarioScore(
        scenario_name="s",
        group="g",
        scores={
            "rules": {
                "schema_version": "rules.v1",
                "checks": [{"dimension": "execution", "check": "no_errors", "passed": False, "details": ""}],
                "passed": False,
            }
        },
        overall_pass=False,
    )
    lines = _failure_context(score, Path("/nonexistent"))
    assert lines == []


def test_failure_context_no_failures() -> None:
    """All-pass scores produce no context lines."""
    score = ScenarioScore(
        scenario_name="s",
        group="g",
        scores={
            "rules": {
                "schema_version": "rules.v1",
                "checks": [
                    {"dimension": "execution", "check": "no_errors", "passed": True, "details": "", "turn_idx": 0}
                ],
                "passed": True,
            }
        },
        overall_pass=True,
    )
    assert _failure_context(score, Path("/nonexistent")) == []


# ── _evidence_paths ──


def test_evidence_paths_returns_existing_files(tmp_path: Path) -> None:
    """Returns only files that exist for failing turns."""
    outcome = tmp_path / "g" / "s"
    outcome.mkdir(parents=True)
    (outcome / "turn_0_output.json").write_text("{}")
    (outcome / "turn_0_cli.txt").write_text("output")

    score = ScenarioScore(
        scenario_name="s",
        group="g",
        scores={
            "rules": {
                "schema_version": "rules.v1",
                "checks": [
                    {"dimension": "execution", "check": "no_errors", "passed": False, "details": "", "turn_idx": 0}
                ],
                "passed": False,
            }
        },
        overall_pass=False,
    )
    paths = _evidence_paths(score, outcome)
    assert len(paths) == 2
    assert paths[0].name == "turn_0_output.json"
    assert paths[1].name == "turn_0_cli.txt"


def test_evidence_paths_no_failures() -> None:
    """All-pass scores produce no evidence paths."""
    score = ScenarioScore(
        scenario_name="s",
        group="g",
        scores={
            "rules": {
                "schema_version": "rules.v1",
                "checks": [
                    {"dimension": "execution", "check": "no_errors", "passed": True, "details": "", "turn_idx": 0}
                ],
                "passed": True,
            }
        },
        overall_pass=True,
    )
    assert _evidence_paths(score, Path("/nonexistent")) == []


# ── print_terminal ──


def _capture_print_terminal(scores, run_label="test-run", **kwargs):
    """Helper: call print_terminal with a plain Console and capture output."""
    import io

    from rich.console import Console

    from belt.commands.aggregate import print_terminal

    buf = io.StringIO()
    con = Console(file=buf, no_color=True, width=120)
    print_terminal(scores, run_label, console=con, **kwargs)
    return buf.getvalue()


def test_print_terminal_results_header() -> None:
    """Phase header should say 'Results'."""
    scores = [_score("a", "g1", [_check("execution", "no_errors", True)])]
    out = _capture_print_terminal(scores)
    assert "Results" in out


def test_print_terminal_no_run_label_in_panel() -> None:
    """Run label should appear at the bottom, not inside the panel."""
    scores = [_score("a", "g1", [_check("execution", "no_errors", True)])]
    out = _capture_print_terminal(scores, run_label="outcomes/20260322-143022")
    assert "outcomes/20260322-143022" in out
    assert "run:" not in out


def test_print_terminal_no_ci_gate_hint_removed() -> None:
    """'No CI gate' noise removed - thresholds section only shown when set."""
    scores = [_score("a", "g1", [_check("execution", "no_errors", True)])]
    out = _capture_print_terminal(scores)
    assert "No CI gate" not in out


def test_print_terminal_cost_timing_in_footer() -> None:
    """Cost and timing should be surfaced in the panel footer."""
    scores = [_score("a", "g1", [_check("execution", "no_errors", True)])]
    cost_timing = {"total_cost_usd": 0.0432, "total_seconds": 95.0}
    out = _capture_print_terminal(scores, cost_timing=cost_timing)
    assert "$0.0432" in out
    assert "1m35s" in out


def test_print_terminal_no_cost_when_absent() -> None:
    """Cost line should not appear when no cost data."""
    scores = [_score("a", "g1", [_check("execution", "no_errors", True)])]
    out = _capture_print_terminal(scores)
    assert "$" not in out


# ── _extract_scorer_usage ──


def test_extract_scorer_usage_from_llm() -> None:
    from belt.commands.aggregate import _extract_scorer_usage

    s = _score(
        "a",
        "g1",
        [],
        llm={
            "overall_pass": True,
            "dim": {"score": "high", "reasoning": "ok"},
            "usage": {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150},
        },
    )
    usage = _extract_scorer_usage(s)
    assert usage == {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150}


def test_extract_scorer_usage_none_when_no_llm() -> None:
    from belt.commands.aggregate import _extract_scorer_usage

    s = _score("a", "g1", [_check("execution", "no_errors", True)])
    assert _extract_scorer_usage(s) is None


def test_extract_scorer_usage_none_when_no_usage_field() -> None:
    from belt.commands.aggregate import _extract_scorer_usage

    s = _score("a", "g1", [], llm={"overall_pass": True, "dim": {"score": "high", "reasoning": "ok"}})
    assert _extract_scorer_usage(s) is None


# ── build_result_table ──


def test_result_table_rules_only() -> None:
    """Rules-only scores produce a table with Scenario + Rules columns."""
    scores = [
        _score("a", "g1", [_check("execution", "no_errors", True), _check("trajectory", "tool_invoked(x)", True)]),
        _score("b", "g1", [_check("execution", "no_errors", False)]),
    ]
    lines = build_result_table(scores)
    assert any("Scenario" in line and "Rules" in line for line in lines)
    assert any("g1/a" in line and "2/2" in line for line in lines)
    assert any("g1/b" in line and "0/1" in line for line in lines)


def test_result_table_with_llm_dims() -> None:
    """LLM dimension scores appear as columns - visible even when all pass."""
    scores = [
        _score(
            "a",
            "g1",
            [_check("execution", "no_errors", True)],
            llm={"quality": {"score": "high"}, "security": {"score": "medium"}, "overall_pass": True},
        ),
    ]
    lines = build_result_table(scores)
    header = lines[0]
    assert "quality" in header
    assert "security" in header
    assert any("✅ high" in line and "⚠️ medium" in line for line in lines)


def test_result_table_empty() -> None:
    assert build_result_table([]) == []


def test_result_table_no_llm_no_rules() -> None:
    """Scores with neither rules nor LLM still produce a table."""
    scores = [ScenarioScore(scenario_name="x", group="g", scores={}, overall_pass=True)]
    lines = build_result_table(scores)
    assert any("g/x" in line for line in lines)


# ── discover_llm_dimensions ──


def test_discover_llm_dimensions_empty() -> None:
    assert discover_llm_dimensions([]) == []


def test_discover_llm_dimensions_from_scores() -> None:
    scores = [
        _score(
            "a", "g1", [], llm={"execution": {"score": "high"}, "custom_dim": {"score": "low"}, "overall_pass": True}
        ),
        _score(
            "b", "g1", [], llm={"execution": {"score": "medium"}, "another": {"score": "high"}, "overall_pass": True}
        ),
    ]
    dims = discover_llm_dimensions(scores)
    assert dims == ["another", "custom_dim", "execution"]


def test_discover_llm_dimensions_excludes_overall_pass() -> None:
    scores = [
        _score("a", "g1", [], llm={"overall_pass": True, "execution": {"score": "high"}}),
    ]
    dims = discover_llm_dimensions(scores)
    assert "overall_pass" not in dims
    assert dims == ["execution"]


def test_count_llm_failures_dynamic_dimensions() -> None:
    """count_llm_failures discovers dimensions from data, not a hardcoded list."""
    scores = [
        _score("a", "g1", [], llm={"custom": {"score": "low"}, "overall_pass": False}),
        _score("b", "g1", [], llm={"custom": {"score": "high"}, "overall_pass": True}),
    ]
    result = count_llm_failures(scores, {"low"})
    assert result["custom"] == (1, 2)


# ── ThresholdCheck structured data ──


def test_threshold_check_display_line() -> None:
    tc = ThresholdCheck(dimension="rules/execution", actual_pct=5.0, max_pct=10.0, passed=True)
    line = tc.display_line()
    assert "5.0% failed" in line
    assert "max 10%" in line


def test_enforce_returns_structured_checks() -> None:
    """enforce_thresholds returns ThresholdCheck objects alongside display lines."""
    _lines, _passed, checks = enforce_thresholds(
        {"rules": {"execution": 0, "trajectory": 50}},
        {"execution": (1, 10), "trajectory": (2, 10)},
        {},
    )
    assert len(checks) == 2
    exec_check = next(c for c in checks if c.dimension == "rules/execution")
    assert exec_check.actual_pct == 10.0
    assert exec_check.max_pct == 0.0
    assert exec_check.passed is False
    traj_check = next(c for c in checks if c.dimension == "rules/trajectory")
    assert traj_check.actual_pct == 20.0
    assert traj_check.passed is True


# ── Partial credit ──


def test_partial_score_all_pass() -> None:
    scores = [
        _score("a", "g", [_check("execution", "no_errors", True), _check("trajectory", "tool(x)", True)]),
        _score("b", "g", [_check("execution", "no_errors", True)]),
    ]
    result = _compute_partial_score(scores)
    assert result is not None
    assert result["checks_passed"] == 3
    assert result["checks_total"] == 3
    assert result["partial_score"] == 1.0


def test_partial_score_mixed() -> None:
    scores = [
        _score("a", "g", [_check("execution", "no_errors", True), _check("trajectory", "tool(x)", False)]),
        _score("b", "g", [_check("execution", "no_errors", False)]),
    ]
    result = _compute_partial_score(scores)
    assert result is not None
    assert result["checks_passed"] == 1
    assert result["checks_total"] == 3
    assert abs(result["partial_score"] - 1 / 3) < 0.001


def test_partial_score_no_rules() -> None:
    scores = [_score("a", "g", [], llm={"execution": {"score": "high"}, "overall_pass": True})]
    assert _compute_partial_score(scores) is None


def test_build_stats_includes_partial_score() -> None:
    scores = [
        _score("a", "g", [_check("execution", "no_errors", True), _check("trajectory", "tool(x)", False)]),
    ]
    stats = build_stats(scores)
    assert stats["checks_passed"] == 1
    assert stats["checks_total"] == 2
    assert stats["partial_score"] == 0.5


def test_print_terminal_shows_partial_score() -> None:
    scores = [
        _score("a", "g", [_check("execution", "no_errors", True), _check("trajectory", "tool(x)", False)]),
        _score("b", "g", [_check("execution", "no_errors", True)]),
    ]
    out = _capture_print_terminal(scores)
    assert "2/3 checks" in out


# ── pass@k and pass^k reliability ──


def test_reliability_no_trials() -> None:
    """Without __trial_ suffix, compute_reliability returns None."""
    scores = [
        _score("math", "g", [_check("execution", "no_errors", True)]),
        _score("code", "g", [_check("execution", "no_errors", True)]),
    ]
    assert compute_reliability(scores) is None


def test_reliability_all_pass() -> None:
    scores = [
        ScenarioScore(scenario_name="math__trial_0", group="g", scores={}, overall_pass=True),
        ScenarioScore(scenario_name="math__trial_1", group="g", scores={}, overall_pass=True),
        ScenarioScore(scenario_name="math__trial_2", group="g", scores={}, overall_pass=True),
    ]
    r = compute_reliability(scores)
    assert r is not None
    assert r["mean_pass_at_1"] == 1.0
    assert r["mean_pass_at_3"] == 1.0
    assert r["mean_pass_at_8"] == 1.0
    assert r["mean_pass_pow_3"] == 1.0
    assert r["mean_pass_pow_8"] == 1.0
    assert len(r["scenarios"]) == 1
    assert r["scenarios"][0]["scenario"] == "g/math"
    assert r["scenarios"][0]["trials"] == 3


def test_reliability_partial() -> None:
    scores = [
        ScenarioScore(scenario_name="math__trial_0", group="g", scores={}, overall_pass=True),
        ScenarioScore(scenario_name="math__trial_1", group="g", scores={}, overall_pass=False),
    ]
    r = compute_reliability(scores)
    assert r is not None
    assert r["mean_pass_at_1"] == 0.5
    assert r["mean_pass_at_3"] == pytest.approx(1.0 - 0.5**3, abs=0.0001)
    assert r["mean_pass_at_8"] == pytest.approx(1.0 - 0.5**8, abs=0.0001)
    assert r["mean_pass_pow_1"] == 0.5
    assert r["mean_pass_pow_3"] == pytest.approx(0.5**3, abs=0.0001)
    assert r["mean_pass_pow_8"] == pytest.approx(0.5**8, abs=0.0001)
    s = r["scenarios"][0]
    assert s["pass_at_3"] == pytest.approx(0.875, abs=0.0001)
    assert s["pass_pow_3"] == pytest.approx(0.125, abs=0.0001)


def test_reliability_multiple_scenarios() -> None:
    scores = [
        ScenarioScore(scenario_name="a__trial_0", group="g", scores={}, overall_pass=True),
        ScenarioScore(scenario_name="a__trial_1", group="g", scores={}, overall_pass=True),
        ScenarioScore(scenario_name="b__trial_0", group="g", scores={}, overall_pass=True),
        ScenarioScore(scenario_name="b__trial_1", group="g", scores={}, overall_pass=False),
    ]
    r = compute_reliability(scores)
    assert r is not None
    assert len(r["scenarios"]) == 2
    a = next(s for s in r["scenarios"] if s["scenario"] == "g/a")
    b = next(s for s in r["scenarios"] if s["scenario"] == "g/b")
    assert a["pass_rate"] == 1.0
    assert b["pass_rate"] == 0.5
    assert r["mean_pass_rate"] == 0.75
