# (c) JFrog Ltd. (2026)

"""Tests for the terminal results viewer (belt view)."""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest
from rich.console import Console

from belt.commands.view import (
    _count_turns,
    _fmt_args,
    _fmt_cost,
    _fmt_secs,
    find_latest_outcomes_dir,
    load_summary,
    show_summary_table,
    view_cmd,
)

# ── Fixtures ──


def _make_score_json(
    group: str,
    scenario_name: str,
    overall_pass: bool = True,
    rules_checks: list[dict] | None = None,
    llm_dims: dict[str, dict] | None = None,
) -> dict:
    """Return a minimal ScenarioScore dict suitable for JSON serialisation."""
    scores: dict = {}
    if rules_checks is not None:
        scores["rules"] = {
            "schema_version": "rules.v1",
            "checks": rules_checks,
            "passed": overall_pass,
        }
    if llm_dims is not None:
        # Caller provides the raw shape (with or without ``overall_pass`` /
        # ``usage`` mixed in alongside dimension dicts) - normalise into the
        # typed ``llm.v1`` envelope so the file matches what the LLM scorer
        # writes today.
        dims = {
            k: v
            for k, v in llm_dims.items()
            if k not in ("overall_pass", "usage", "consensus_meta", "individual_verdicts")
        }
        scores["llm"] = {
            "schema_version": "llm.v1",
            "overall_pass": llm_dims.get("overall_pass", overall_pass),
            "dimensions": dims,
        }
        if "usage" in llm_dims:
            scores["llm"]["usage"] = llm_dims["usage"]
    return {
        "schema_version": "1",
        "scenario_name": scenario_name,
        "group": group,
        "scores": scores,
        "overall_pass": overall_pass,
    }


def _make_turn_output_json(
    cost_usd: float | None = None,
    total_secs: float | None = None,
    reply_text: str = "ok",
    tool_calls: list[dict] | None = None,
) -> dict:
    timing = {}
    if total_secs is not None:
        timing = {"total": total_secs}
    return {
        "raw_cli": "$ agent hello\nok\n",
        "reply_text": reply_text,
        "tool_calls": tool_calls or [],
        "has_reply": True,
        "cost_usd": cost_usd,
        "timing": timing if timing else None,
    }


def _write_scenario(
    base_dir: Path,
    group: str,
    scenario_name: str,
    overall_pass: bool = True,
    cost_usd: float | None = None,
    total_secs: float | None = None,
    num_turns: int = 1,
    rules_checks: list[dict] | None = None,
    llm_dims: dict[str, dict] | None = None,
) -> Path:
    """Write scenario artifacts to *base_dir/group/scenario_name/* and return the outcome dir."""
    outcome_dir = base_dir / group / scenario_name
    outcome_dir.mkdir(parents=True, exist_ok=True)

    score = _make_score_json(group, scenario_name, overall_pass, rules_checks, llm_dims)
    (outcome_dir / "score.json").write_text(json.dumps(score))

    for i in range(num_turns):
        turn = _make_turn_output_json(cost_usd=cost_usd, total_secs=total_secs)
        (outcome_dir / f"turn_{i}_output.json").write_text(json.dumps(turn))
        (outcome_dir / f"turn_{i}_cli.txt").write_text(f"$ agent prompt{i}\nresult{i}\n")

    return outcome_dir


# ── find_latest_outcomes_dir ──


class TestFindLatestOutcomesDir:
    def test_returns_none_when_base_missing(self, tmp_path: Path) -> None:
        missing = tmp_path / "nonexistent"
        assert find_latest_outcomes_dir(missing) is None

    def test_returns_none_when_empty(self, tmp_path: Path) -> None:
        assert find_latest_outcomes_dir(tmp_path) is None

    def test_returns_single_dir(self, tmp_path: Path) -> None:
        run_dir = tmp_path / "20260101-120000"
        run_dir.mkdir()
        result = find_latest_outcomes_dir(tmp_path)
        assert result == run_dir

    def test_returns_most_recently_modified(self, tmp_path: Path) -> None:
        old = tmp_path / "old"
        new = tmp_path / "new"
        old.mkdir()
        time.sleep(0.01)  # ensure mtime difference
        new.mkdir()
        result = find_latest_outcomes_dir(tmp_path)
        assert result == new

    def test_skips_dotfiles(self, tmp_path: Path) -> None:
        hidden = tmp_path / ".hidden"
        visible = tmp_path / "run1"
        hidden.mkdir()
        time.sleep(0.01)
        visible.mkdir()
        result = find_latest_outcomes_dir(tmp_path)
        assert result == visible

    def test_skips_files(self, tmp_path: Path) -> None:
        (tmp_path / "results.json").write_text("{}")
        run_dir = tmp_path / "run1"
        run_dir.mkdir()
        result = find_latest_outcomes_dir(tmp_path)
        assert result == run_dir


# ── load_summary ──


class TestLoadSummary:
    def test_empty_dir_returns_empty(self, tmp_path: Path) -> None:
        assert load_summary(tmp_path) == []

    def test_single_scenario(self, tmp_path: Path) -> None:
        _write_scenario(tmp_path, "mygroup", "myscenario", overall_pass=True, cost_usd=0.005, total_secs=3.2)
        summaries = load_summary(tmp_path)
        assert len(summaries) == 1
        s = summaries[0]
        assert s["scenario"] == "mygroup/myscenario"
        assert s["pass"] is True
        assert s["cost_usd"] == pytest.approx(0.005)
        assert s["total_secs"] == pytest.approx(3.2)

    def test_rules_check_summary(self, tmp_path: Path) -> None:
        checks = [
            {"dimension": "execution", "check": "no_errors", "passed": True},
            {"dimension": "trajectory", "check": "tool_invoked(Edit)", "passed": False, "details": "missing Edit"},
        ]
        _write_scenario(tmp_path, "g", "s", overall_pass=False, rules_checks=checks)
        summaries = load_summary(tmp_path)
        assert summaries[0]["rules"] == "1/2"

    def test_rules_no_checks_shows_dash(self, tmp_path: Path) -> None:
        _write_scenario(tmp_path, "g", "s", overall_pass=True)
        summaries = load_summary(tmp_path)
        assert summaries[0]["rules"] == "-"

    def test_llm_dims_parsed(self, tmp_path: Path) -> None:
        llm = {
            "execution": {"score": "high", "reasoning": "Good execution"},
            "response_quality": {"score": "medium", "reasoning": "OK"},
        }
        _write_scenario(tmp_path, "g", "s", overall_pass=True, llm_dims=llm)
        summaries = load_summary(tmp_path)
        dims = summaries[0]["llm_dims"]
        assert dims.get("execution") == "high"
        assert dims.get("response_quality") == "medium"

    def test_llm_meta_keys_excluded(self, tmp_path: Path) -> None:
        llm = {
            "overall_pass": True,
            "usage": {"prompt_tokens": 100},
            "execution": {"score": "high", "reasoning": "ok"},
        }
        _write_scenario(tmp_path, "g", "s", overall_pass=True, llm_dims=llm)
        summaries = load_summary(tmp_path)
        dims = summaries[0]["llm_dims"]
        assert "overall_pass" not in dims
        assert "usage" not in dims
        assert "execution" in dims

    def test_multiple_scenarios(self, tmp_path: Path) -> None:
        _write_scenario(tmp_path, "g", "s1")
        _write_scenario(tmp_path, "g", "s2")
        summaries = load_summary(tmp_path)
        assert len(summaries) == 2
        names = {s["scenario"] for s in summaries}
        assert names == {"g/s1", "g/s2"}

    def test_cost_accumulated_across_turns(self, tmp_path: Path) -> None:
        _write_scenario(tmp_path, "g", "multi", cost_usd=0.001, num_turns=3)
        summaries = load_summary(tmp_path)
        assert summaries[0]["cost_usd"] == pytest.approx(0.003)

    def test_no_cost_returns_none(self, tmp_path: Path) -> None:
        _write_scenario(tmp_path, "g", "s", cost_usd=None)
        summaries = load_summary(tmp_path)
        assert summaries[0]["cost_usd"] is None

    def test_corrupt_score_json_skipped(self, tmp_path: Path) -> None:
        bad_dir = tmp_path / "g" / "bad"
        bad_dir.mkdir(parents=True)
        (bad_dir / "score.json").write_text("not valid json {{{")
        summaries = load_summary(tmp_path)
        assert summaries == []

    def test_reads_cost_timing_from_results_json(self, tmp_path: Path) -> None:
        """When ``results.json`` is present, cost/timing come from it - no turn re-parse."""
        # Write a scenario with bogus turn cost; results.json's value should win.
        outcome_dir = _write_scenario(tmp_path, "g", "s1", cost_usd=99.99, total_secs=999.0)
        results = {
            "schema_version": 1,
            "cost_timing": {
                "scenarios": [
                    {"scenario": "g/s1", "agent_cost_usd": 0.042, "total_seconds": 7.5},
                ]
            },
        }
        (tmp_path / "results.json").write_text(json.dumps(results))

        # Sanity: the turn file does exist (so the fallback would otherwise read it).
        assert (outcome_dir / "turn_0_output.json").exists()

        summaries = load_summary(tmp_path)
        assert len(summaries) == 1
        assert summaries[0]["cost_usd"] == pytest.approx(0.042)
        assert summaries[0]["total_secs"] == pytest.approx(7.5)

    def test_falls_back_to_turn_files_when_results_missing(self, tmp_path: Path) -> None:
        """No ``results.json`` (e.g. score-only run): walk turn_*_output.json."""
        _write_scenario(tmp_path, "g", "s1", cost_usd=0.01, total_secs=4.0, num_turns=2)
        summaries = load_summary(tmp_path)
        assert summaries[0]["cost_usd"] == pytest.approx(0.02)
        assert summaries[0]["total_secs"] == pytest.approx(8.0)

    def test_falls_back_when_results_json_malformed(self, tmp_path: Path) -> None:
        """Malformed ``results.json`` does not break ``view``; falls back to turn walk."""
        _write_scenario(tmp_path, "g", "s1", cost_usd=0.01, total_secs=4.0)
        (tmp_path / "results.json").write_text("not valid json {")
        summaries = load_summary(tmp_path)
        assert summaries[0]["cost_usd"] == pytest.approx(0.01)
        assert summaries[0]["total_secs"] == pytest.approx(4.0)

    def test_results_json_partial_fallback_per_scenario(self, tmp_path: Path) -> None:
        """``results.json`` may list a subset; missing scenarios fall back to turn files."""
        _write_scenario(tmp_path, "g", "indexed", cost_usd=0.99, total_secs=99.0)
        _write_scenario(tmp_path, "g", "unindexed", cost_usd=0.05, total_secs=2.0)
        results = {
            "cost_timing": {
                "scenarios": [
                    {"scenario": "g/indexed", "agent_cost_usd": 0.001, "total_seconds": 0.5},
                ]
            }
        }
        (tmp_path / "results.json").write_text(json.dumps(results))
        by_name = {s["name"]: s for s in load_summary(tmp_path)}
        assert by_name["indexed"]["cost_usd"] == pytest.approx(0.001)
        assert by_name["indexed"]["total_secs"] == pytest.approx(0.5)
        assert by_name["unindexed"]["cost_usd"] == pytest.approx(0.05)
        assert by_name["unindexed"]["total_secs"] == pytest.approx(2.0)


# ── _count_turns ──


class TestCountTurns:
    def test_zero_turns(self, tmp_path: Path) -> None:
        assert _count_turns(tmp_path) == 0

    def test_sequential_turns(self, tmp_path: Path) -> None:
        for i in range(3):
            (tmp_path / f"turn_{i}_output.json").write_text("{}")
        assert _count_turns(tmp_path) == 3

    def test_gap_stops_count(self, tmp_path: Path) -> None:
        # turn_0 and turn_2 exist but not turn_1 - count stops at 1
        (tmp_path / "turn_0_output.json").write_text("{}")
        (tmp_path / "turn_2_output.json").write_text("{}")
        assert _count_turns(tmp_path) == 1


# ── Formatting helpers ──


class TestFormatHelpers:
    def test_fmt_cost_none(self) -> None:
        assert _fmt_cost(None) == "-"

    def test_fmt_cost_zero(self) -> None:
        assert _fmt_cost(0.0) == "$0.0000"

    def test_fmt_cost_nonzero(self) -> None:
        result = _fmt_cost(0.1234)
        assert result == "$0.1234"

    def test_fmt_secs_none(self) -> None:
        assert _fmt_secs(None) == "-"

    def test_fmt_secs_sub_minute(self) -> None:
        assert _fmt_secs(5.7) == "5.7s"

    def test_fmt_secs_minutes(self) -> None:
        result = _fmt_secs(95.0)
        assert result == "1m35s"

    def test_fmt_args_empty(self) -> None:
        assert _fmt_args({}) == ""

    def test_fmt_args_simple(self) -> None:
        result = _fmt_args({"path": "src/main.py"})
        assert "path=src/main.py" in result

    def test_fmt_args_truncated(self) -> None:
        long_val = "x" * 200
        result = _fmt_args({"key": long_val})
        assert len(result) <= 130  # 120-char limit + some key overhead


# ── show_summary_table ──


class TestShowSummaryTable:
    """show_summary_table renders without exceptions and includes key text."""

    def _render(self, summaries: list[dict]) -> str:
        con = Console(width=120, no_color=True, force_terminal=False)
        with con.capture() as cap:
            show_summary_table(summaries, console=con)
        return cap.get()

    def test_renders_without_error(self, tmp_path: Path) -> None:
        _write_scenario(tmp_path, "g", "s", overall_pass=True)
        summaries = load_summary(tmp_path)
        output = self._render(summaries)
        assert "g/s" in output

    def test_pass_fail_shown(self, tmp_path: Path) -> None:
        _write_scenario(tmp_path, "g", "pass_s", overall_pass=True)
        _write_scenario(tmp_path, "g", "fail_s", overall_pass=False)
        summaries = load_summary(tmp_path)
        output = self._render(summaries)
        assert "pass" in output.lower()
        assert "fail" in output.lower()

    def test_empty_summaries(self) -> None:
        # Should not raise even with no data
        output = self._render([])
        assert output  # non-empty (table header still renders)

    def test_cost_displayed(self, tmp_path: Path) -> None:
        _write_scenario(tmp_path, "g", "s", cost_usd=0.0042)
        summaries = load_summary(tmp_path)
        output = self._render(summaries)
        assert "0.0042" in output


# ── view_cmd ──


class TestViewCmd:
    def test_missing_dir_returns_1(self, tmp_path: Path) -> None:
        missing = str(tmp_path / "nonexistent")
        result = view_cmd(missing, non_interactive=True)
        assert result == 1

    def test_no_scores_returns_1(self, tmp_path: Path) -> None:
        run_dir = tmp_path / "run1"
        run_dir.mkdir()
        result = view_cmd(str(run_dir), non_interactive=True)
        assert result == 1

    def test_with_scores_returns_0(self, tmp_path: Path) -> None:
        run_dir = tmp_path / "run1"
        run_dir.mkdir()
        _write_scenario(run_dir, "g", "s")
        result = view_cmd(str(run_dir), non_interactive=True)
        assert result == 0

    def test_auto_discovers_latest(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """When no dir given, find_latest_outcomes_dir should be called."""
        import belt.commands.view as viewer_mod

        run_dir = tmp_path / "run1"
        run_dir.mkdir()
        _write_scenario(run_dir, "g", "s")

        monkeypatch.setattr(viewer_mod, "OUTCOMES_ROOT", tmp_path)
        result = view_cmd(None, non_interactive=True)
        assert result == 0

    def test_no_outcomes_root_returns_1(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """When no dir given and outcomes root is empty, return 1."""
        import belt.commands.view as viewer_mod

        empty_root = tmp_path / "empty_outcomes"
        empty_root.mkdir()
        monkeypatch.setattr(viewer_mod, "OUTCOMES_ROOT", empty_root)
        result = view_cmd(None, non_interactive=True)
        assert result == 1
