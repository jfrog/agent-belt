# (c) JFrog Ltd. (2026)

"""Tests for ``belt compare`` - cross-agent comparison logic."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from belt.commands.compare import (
    _discover_dimensions,
    _extract_label,
    _scenario_key,
    build_markdown,
    compare,
    load_results,
    main,
    print_terminal,
)


def _results(
    scenarios: list[dict],
    pass_rate: float = 1.0,
    cost_timing: dict | None = None,
) -> dict:
    r: dict = {
        "total": len(scenarios),
        "passed": sum(1 for s in scenarios if s.get("overall_pass")),
        "failed": sum(1 for s in scenarios if not s.get("overall_pass")),
        "overall_pass": all(s.get("overall_pass") for s in scenarios),
        "stats": {"pass_rate": pass_rate},
        "scenarios": scenarios,
    }
    if cost_timing:
        r["cost_timing"] = cost_timing
    return r


def _scenario(
    name: str,
    group: str,
    overall_pass: bool = True,
    llm: dict | None = None,
    rules_checks: list[dict] | None = None,
) -> dict:
    """Build a serialised ``ScenarioScore`` dict matching what the aggregator writes.

    The legacy flat shape (``{"execution": {"score": "high"}, "overall_pass": True}``)
    is normalised here into the canonical envelope - ``schema_version``, top-level
    ``overall_pass``, and dimension verdicts under ``dimensions`` - so each test can
    keep specifying just the dimension data.
    """
    scores: dict = {}
    if llm:
        meta_keys = ("overall_pass", "usage", "consensus_meta", "individual_verdicts", "schema_version")
        dims = {k: v for k, v in llm.items() if k not in meta_keys}
        scores["llm"] = {
            "schema_version": "llm.v1",
            "overall_pass": bool(llm.get("overall_pass", overall_pass)),
            "dimensions": dims,
        }
    if rules_checks:
        scores["rules"] = {
            "schema_version": "rules.v1",
            "checks": rules_checks,
            "passed": all(c["passed"] for c in rules_checks),
        }
    return {"scenario_name": name, "group": group, "scores": scores, "overall_pass": overall_pass}


class TestScenarioKey:
    def test_basic(self) -> None:
        assert _scenario_key({"group": "g1", "scenario_name": "s1"}) == "g1/s1"


class TestDiscoverDimensions:
    def test_discovers_llm_dims(self) -> None:
        r = _results(
            [
                _scenario(
                    "s1", "g1", llm={"execution": {"score": "high"}, "custom": {"score": "low"}, "overall_pass": True}
                ),
            ]
        )
        _, llm = _discover_dimensions(r)
        assert llm == {"execution", "custom"}

    def test_discovers_rules_dims(self) -> None:
        r = _results(
            [
                _scenario("s1", "g1", rules_checks=[{"dimension": "execution", "check": "no_errors", "passed": True}]),
            ]
        )
        rules, _ = _discover_dimensions(r)
        assert "execution" in rules


class TestCompare:
    def test_identical_results(self) -> None:
        s = [_scenario("s1", "g1", llm={"execution": {"score": "high"}, "overall_pass": True})]
        ra = _results(s)
        rb = _results(s)
        comp = compare(ra, rb, "run-a", "run-b")
        assert comp["total_a"] == comp["total_b"] == 1
        assert len(comp["scenarios"]) == 1
        for dd in comp["scenarios"][0]["dimensions"]:
            assert dd["delta"] == 0

    def test_detects_regression(self) -> None:
        sa = [_scenario("s1", "g1", llm={"execution": {"score": "high"}, "overall_pass": True})]
        sb = [_scenario("s1", "g1", llm={"execution": {"score": "low"}, "overall_pass": False})]
        comp = compare(_results(sa), _results(sb, 0.0), "a", "b")
        dims = comp["scenarios"][0]["dimensions"]
        exec_dim = next(d for d in dims if d["dimension"] == "execution")
        assert exec_dim["delta"] < 0  # regression

    def test_detects_improvement(self) -> None:
        sa = [_scenario("s1", "g1", llm={"execution": {"score": "low"}, "overall_pass": False})]
        sb = [_scenario("s1", "g1", llm={"execution": {"score": "high"}, "overall_pass": True})]
        comp = compare(_results(sa, 0.0), _results(sb), "a", "b")
        dims = comp["scenarios"][0]["dimensions"]
        exec_dim = next(d for d in dims if d["dimension"] == "execution")
        assert exec_dim["delta"] > 0  # improvement

    def test_scenarios_only_in_a(self) -> None:
        sa = [_scenario("s1", "g1"), _scenario("s2", "g1")]
        sb = [_scenario("s1", "g1")]
        comp = compare(_results(sa), _results(sb), "a", "b")
        assert len(comp["scenarios"]) == 2
        s2 = next(s for s in comp["scenarios"] if s["key"] == "g1/s2")
        assert s2["in_a"] is True
        assert s2["in_b"] is False

    def test_shared_dims_only(self) -> None:
        """Only dimensions present in both results are compared."""
        sa = [
            _scenario(
                "s1", "g1", llm={"execution": {"score": "high"}, "custom_a": {"score": "high"}, "overall_pass": True}
            )
        ]
        sb = [
            _scenario(
                "s1", "g1", llm={"execution": {"score": "high"}, "custom_b": {"score": "low"}, "overall_pass": True}
            )
        ]
        comp = compare(_results(sa), _results(sb), "a", "b")
        assert comp["shared_llm_dims"] == ["execution"]
        assert len(comp["scenarios"][0]["dimensions"]) == 1

    def test_empty_results(self) -> None:
        comp = compare(_results([]), _results([]), "a", "b")
        assert comp["scenarios"] == []

    def test_normalizes_keys_when_different_roots(self) -> None:
        """Runs with different roots (single-group vs multi-group) should still match."""
        sa = [_scenario("s1", "", overall_pass=True)]
        sb = [_scenario("s1", "claude-code", overall_pass=True)]
        comp = compare(_results(sa), _results(sb), "a", "b")
        shared = [s for s in comp["scenarios"] if s["in_a"] and s["in_b"]]
        assert len(shared) == 1

    def test_zero_overlap_not_normalized_when_truly_different(self) -> None:
        """When scenarios genuinely differ, no normalization occurs."""
        sa = [_scenario("alpha", "g1")]
        sb = [_scenario("beta", "g2")]
        comp = compare(_results(sa), _results(sb), "a", "b")
        shared = [s for s in comp["scenarios"] if s["in_a"] and s["in_b"]]
        assert len(shared) == 0


class TestOutput:
    def test_terminal_warns_zero_overlap(self, capsys: pytest.CaptureFixture) -> None:
        sa = [_scenario("alpha", "g1")]
        sb = [_scenario("beta", "g2")]
        comp = compare(_results(sa), _results(sb), "a", "b")
        print_terminal(comp)
        out = capsys.readouterr().err
        assert "Zero shared scenarios" in out

    def test_terminal_no_crash(self, capsys: pytest.CaptureFixture) -> None:
        sa = [_scenario("s1", "g1", llm={"execution": {"score": "high"}, "overall_pass": True})]
        sb = [_scenario("s1", "g1", llm={"execution": {"score": "low"}, "overall_pass": False})]
        comp = compare(_results(sa), _results(sb, 0.0), "run-a", "run-b")
        print_terminal(comp)
        out = capsys.readouterr().err
        assert "Comparison" in out
        assert "Regressions" in out

    def test_markdown_output(self) -> None:
        sa = [_scenario("s1", "g1", llm={"execution": {"score": "high"}, "overall_pass": True})]
        sb = [_scenario("s1", "g1", llm={"execution": {"score": "high"}, "overall_pass": True})]
        comp = compare(_results(sa), _results(sb), "a", "b")
        md = build_markdown(comp)
        assert "## " in md
        assert "Pass rate" in md


class TestLoadResults:
    def test_load_valid(self, tmp_path: Path) -> None:
        p = tmp_path / "results.json"
        p.write_text(json.dumps({"total": 1, "scenarios": []}))
        result = load_results(p)
        assert result["total"] == 1

    def test_load_missing_raises(self, tmp_path: Path) -> None:
        from belt.errors import BeltError

        with pytest.raises(BeltError, match="Failed to load"):
            load_results(tmp_path / "nonexistent.json")


class TestMainZeroOverlap:
    def test_exits_nonzero_on_zero_overlap(self, tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
        ra = _results([_scenario("alpha", "g1")])
        rb = _results([_scenario("beta", "g2")])
        a_path = tmp_path / "a.json"
        b_path = tmp_path / "b.json"
        a_path.write_text(json.dumps(ra))
        b_path.write_text(json.dumps(rb))
        rc = main([str(a_path), str(b_path)])
        assert rc == 1
        out = capsys.readouterr().err
        assert "Zero shared scenarios" in out

    def test_exits_zero_on_shared_scenarios(self, tmp_path: Path) -> None:
        s = [_scenario("s1", "g1")]
        ra = _results(s)
        rb = _results(s)
        a_path = tmp_path / "a.json"
        b_path = tmp_path / "b.json"
        a_path.write_text(json.dumps(ra))
        b_path.write_text(json.dumps(rb))
        rc = main([str(a_path), str(b_path)])
        assert rc == 0


class TestCostTiming:
    def test_compare_includes_cost_timing(self) -> None:
        ct_a = {"total_cost_usd": 0.05, "mean_cost_usd": 0.025, "mean_seconds": 12.5}
        ct_b = {"total_cost_usd": 0.10, "mean_cost_usd": 0.050, "mean_seconds": 8.0}
        s = [_scenario("s1", "g1")]
        comp = compare(_results(s, cost_timing=ct_a), _results(s, cost_timing=ct_b), "a", "b")
        assert comp["cost_a"] == 0.05
        assert comp["cost_b"] == 0.10
        assert comp["time_a"] == 12.5
        assert comp["time_b"] == 8.0

    def test_terminal_shows_cost(self, capsys: pytest.CaptureFixture) -> None:
        ct_a = {"total_cost_usd": 0.05, "mean_cost_usd": 0.025, "mean_seconds": 12.5}
        ct_b = {"total_cost_usd": 0.10, "mean_cost_usd": 0.050, "mean_seconds": 8.0}
        s = [_scenario("s1", "g1")]
        comp = compare(_results(s, cost_timing=ct_a), _results(s, cost_timing=ct_b), "a", "b")
        print_terminal(comp)
        out = capsys.readouterr().err
        assert "Cost:" in out
        assert "Avg time:" in out

    def test_markdown_shows_cost(self) -> None:
        ct_a = {"total_cost_usd": 0.05, "mean_seconds": 12.5}
        ct_b = {"total_cost_usd": 0.10, "mean_seconds": 8.0}
        s = [_scenario("s1", "g1")]
        comp = compare(_results(s, cost_timing=ct_a), _results(s, cost_timing=ct_b), "a", "b")
        md = build_markdown(comp)
        assert "Total cost" in md
        assert "Avg time" in md

    def test_no_cost_timing_graceful(self) -> None:
        s = [_scenario("s1", "g1")]
        comp = compare(_results(s), _results(s), "a", "b")
        assert comp["cost_a"] is None
        assert comp["time_b"] is None


class TestExtractLabel:
    def test_with_outcomes_path(self) -> None:
        label = _extract_label(Path("evaluation-tests/outcomes/run-123/results.json"))
        assert "run-123" in label

    def test_simple_path(self) -> None:
        label = _extract_label(Path("/tmp/my_results.json"))
        assert label == "my_results"
