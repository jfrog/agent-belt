# (c) JFrog Ltd. (2026)

"""Tests for ``belt.scorer.dry_run`` (rules-mode preview, payload preview)."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from belt.scorer.dry_run import handle_dry_run


def _write_scenario_and_turns(
    base: Path,
    *,
    scenario_name: str = "demo",
    expectations: dict | None = None,
) -> tuple[Path, Path]:
    """Create a scenarios dir + outcome dir pair pointing at one scenario."""
    scenarios_root = base / "scenarios"
    group_dir = scenarios_root / "g"
    group_dir.mkdir(parents=True)
    (group_dir / "_config.json").write_text(json.dumps({"agent": "claude-code"}))
    expect = expectations or {}
    scenario = {
        "name": scenario_name,
        "description": "test",
        "turns": [{"message": "hello", "expect": expect}],
    }
    (group_dir / f"{scenario_name}.json").write_text(json.dumps(scenario))

    outcomes_root = base / "outcomes"
    outcome_dir = outcomes_root / "g" / scenario_name
    outcome_dir.mkdir(parents=True)
    (outcome_dir / "turn_0_cli.txt").write_text("agent stdout")
    # ``map_to_scenario`` reads ``run_meta.json`` to locate the scenarios
    # root; without it the lookup falls back to the global examples dir.
    (outcomes_root / "run_meta.json").write_text(json.dumps({"scenarios_root": str(scenarios_root)}))
    return outcomes_root, outcome_dir


class TestHandleDryRunNoScorers:
    def test_returns_1_when_no_scorers_at_all(self, tmp_path: Path, capsys) -> None:
        outcomes_root, outcome_dir = _write_scenario_and_turns(tmp_path)
        rc = handle_dry_run(llm_scorer=None, outcome_dir=outcome_dir, outcomes_root=outcomes_root, scorers=None)
        assert rc == 1
        out = capsys.readouterr().err
        assert "rules" in out and "llm" in out


class TestHandleDryRunRulesMode:
    """``--modes rules --dry-run`` previews scenario expectations - no LLM required."""

    def test_rules_only_succeeds_without_llm_scorer(self, tmp_path: Path, capsys) -> None:
        outcomes_root, outcome_dir = _write_scenario_and_turns(tmp_path)
        rules_scorer = MagicMock()
        rules_scorer.name = "rules"
        rc = handle_dry_run(
            llm_scorer=None,
            outcome_dir=outcome_dir,
            outcomes_root=outcomes_root,
            scorers=[rules_scorer],
        )
        assert rc == 0
        out = capsys.readouterr().err
        assert "Mode: rules" in out
        assert "RULE EXPECTATIONS" in out

    def test_rules_preview_lists_populated_expectation_fields(self, tmp_path: Path, capsys) -> None:
        outcomes_root, outcome_dir = _write_scenario_and_turns(
            tmp_path,
            expectations={"contains": ["hello"], "max_tool_calls": 3},
        )
        rules_scorer = MagicMock()
        rules_scorer.name = "rules"
        handle_dry_run(
            llm_scorer=None,
            outcome_dir=outcome_dir,
            outcomes_root=outcomes_root,
            scorers=[rules_scorer],
        )
        out = capsys.readouterr().err
        # Both populated fields should appear; defaults like ``no_errors=True`` should not.
        assert "contains" in out
        assert "max_tool_calls" in out
        assert "no_errors" not in out

    def test_rules_preview_no_expectations_shows_defaults_only(self, tmp_path: Path, capsys) -> None:
        outcomes_root, outcome_dir = _write_scenario_and_turns(tmp_path)
        rules_scorer = MagicMock()
        rules_scorer.name = "rules"
        handle_dry_run(
            llm_scorer=None,
            outcome_dir=outcome_dir,
            outcomes_root=outcomes_root,
            scorers=[rules_scorer],
        )
        out = capsys.readouterr().err
        assert "defaults only" in out


class TestHandleDryRunSelector:
    """The selector is wired by ``commands/score.py::_select_dry_run_outcome`` -
    here we only assert the handler accepts whatever outcome dir it gets."""

    def test_handler_uses_provided_outcome_dir(self, tmp_path: Path, capsys) -> None:
        outcomes_root, outcome_dir = _write_scenario_and_turns(tmp_path, scenario_name="alpha")
        rules_scorer = MagicMock()
        rules_scorer.name = "rules"
        handle_dry_run(
            llm_scorer=None,
            outcome_dir=outcome_dir,
            outcomes_root=outcomes_root,
            scorers=[rules_scorer],
        )
        out = capsys.readouterr().err
        assert "alpha" in out


class TestSelectDryRunOutcome:
    """Selector logic in ``commands/score.py``."""

    @pytest.fixture
    def outcomes(self, tmp_path: Path) -> tuple[Path, list[Path]]:
        outcomes_root = tmp_path / "outcomes"
        dirs = [
            outcomes_root / "agents" / "claude-code" / "read-only" / "simple_math",
            outcomes_root / "agents" / "cursor" / "read-only" / "simple_math",
            outcomes_root / "agents" / "claude-code" / "read-only" / "explain_code",
        ]
        for d in dirs:
            d.mkdir(parents=True)
        return outcomes_root, dirs

    def test_empty_selector_returns_first(self, outcomes) -> None:
        from belt.commands.score import _select_dry_run_outcome

        outcomes_root, dirs = outcomes
        result = _select_dry_run_outcome(dirs, outcomes_root, "")
        assert result == dirs[0]

    def test_unique_substring_matches(self, outcomes) -> None:
        from belt.commands.score import _select_dry_run_outcome

        outcomes_root, dirs = outcomes
        result = _select_dry_run_outcome(dirs, outcomes_root, "explain_code")
        assert result == dirs[2]

    def test_no_match_returns_none(self, outcomes, capsys) -> None:
        from belt.commands.score import _select_dry_run_outcome

        outcomes_root, dirs = outcomes
        result = _select_dry_run_outcome(dirs, outcomes_root, "nonexistent")
        assert result is None
        assert "no scenario matches" in capsys.readouterr().err

    def test_ambiguous_selector_returns_none_and_lists(self, outcomes, capsys) -> None:
        from belt.commands.score import _select_dry_run_outcome

        outcomes_root, dirs = outcomes
        result = _select_dry_run_outcome(dirs, outcomes_root, "simple_math")
        assert result is None
        out = capsys.readouterr().err
        assert "matches 2 scenarios" in out
        assert "claude-code" in out
        assert "cursor" in out


class TestHandleDryRunLLMPreview:
    """LLM payload preview behaviour."""

    def _build_judge(self, judge_name: str, dim_names: list[str]):
        from belt.agent.scoring import DimensionDef, ScoringStrategy
        from belt.scorer import LLMScorer
        from belt.scorer.entities import JudgeConfig
        from belt.scorer.llm.backend import OpenAIBackend

        strategy = ScoringStrategy(dimensions=[DimensionDef(name=n, description=n) for n in dim_names])
        judge = LLMScorer(
            config=JudgeConfig(model="openai/gpt-4.1"),
            backend=OpenAIBackend(),
            skip_availability=True,
            strategy=strategy,
        )
        judge.judge_name = judge_name
        return judge

    def test_dimensions_render_in_strategy_order(self, tmp_path: Path, capsys) -> None:
        outcomes_root, outcome_dir = _write_scenario_and_turns(tmp_path)
        judge = self._build_judge("solo", ["beta", "alpha", "gamma"])
        handle_dry_run(
            llm_scorer=judge,
            outcome_dir=outcome_dir,
            outcomes_root=outcomes_root,
            scorers=[judge],
        )
        out = capsys.readouterr().err
        assert "Dimensions:  beta, alpha, gamma" in out

    def test_default_judge_name_omits_suffix(self, tmp_path: Path, capsys) -> None:
        outcomes_root, outcome_dir = _write_scenario_and_turns(tmp_path)
        judge = self._build_judge("llm", ["alpha"])  # "llm" is the default name
        handle_dry_run(
            llm_scorer=judge,
            outcome_dir=outcome_dir,
            outcomes_root=outcomes_root,
            scorers=[judge],
        )
        out = capsys.readouterr().err
        assert "Mode: llm\n" in out
        assert "(judge:" not in out

    def test_each_consensus_judge_gets_its_own_preview_block(self, tmp_path: Path, capsys) -> None:
        from belt.scorer import ConsensusScorer

        outcomes_root, outcome_dir = _write_scenario_and_turns(tmp_path)
        judge_a = self._build_judge("judge_a", ["dim_a"])
        judge_b = self._build_judge("judge_b", ["dim_b"])
        consensus = ConsensusScorer([judge_a, judge_b], strategy="majority")

        handle_dry_run(
            llm_scorer=judge_a,
            outcome_dir=outcome_dir,
            outcomes_root=outcomes_root,
            scorers=[consensus],
        )
        out = capsys.readouterr().err
        assert "judge: judge_a" in out
        assert "judge: judge_b" in out
        assert "Dimensions:  dim_a" in out
        assert "Dimensions:  dim_b" in out
