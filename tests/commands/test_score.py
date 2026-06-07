# (c) JFrog Ltd. (2026)

"""Tests for scorer/cli.py - TurnOutput loading and score_scenario."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

from belt.commands.score import _parse_dimension_defs, _scenarios_root, map_to_scenario, score_scenario
from belt.entities import Scenario, Turn, TurnExpectation, TurnOutput


def _make_scenario_file(scenarios_dir: Path, group: str, name: str, turns: int = 1) -> None:
    scenario = Scenario(
        name=name,
        description="test",
        turns=[Turn(message=f"Turn {i}", expect=TurnExpectation()) for i in range(turns)],
    )
    path = scenarios_dir / group / f"{name}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(scenario.model_dump_json(indent=2))


def _write_turn_output(outcome_dir: Path, turn_idx: int, **kwargs: object) -> None:
    to = TurnOutput(
        raw_cli=kwargs.get("raw_cli", "clean output"), **{k: v for k, v in kwargs.items() if k != "raw_cli"}
    )
    (outcome_dir / f"turn_{turn_idx}_output.json").write_text(to.model_dump_json(indent=2))


def _write_raw_cli(outcome_dir: Path, turn_idx: int, content: str = "raw cli output") -> None:
    (outcome_dir / f"turn_{turn_idx}_cli.txt").write_text(content)


class TestScenariosRoot:
    """Verifies _scenarios_root resolution from run_meta.json."""

    def test_reads_from_run_meta_json(self, tmp_path: Path):
        run_dir = tmp_path / "outcomes" / "20260322-120000"
        run_dir.mkdir(parents=True)
        (run_dir / "run_meta.json").write_text(json.dumps({"scenarios_root": "/abs/path/scenarios"}))

        with patch.dict("os.environ", {}, clear=False):
            import os

            os.environ.pop("_BELT_SCENARIOS_ROOT", None)
            result = _scenarios_root(run_dir)

        assert result == Path("/abs/path/scenarios")

    def test_env_var_takes_precedence(self, tmp_path: Path):
        run_dir = tmp_path / "outcomes" / "20260322-120000"
        run_dir.mkdir(parents=True)
        (run_dir / "run_meta.json").write_text(json.dumps({"scenarios_root": "/from/meta"}))

        with patch.dict("os.environ", {"_BELT_SCENARIOS_ROOT": "/from/env"}):
            result = _scenarios_root(run_dir)

        assert result == Path("/from/env")

    def test_fallback_when_no_meta(self, tmp_path: Path):
        run_dir = tmp_path / "outcomes" / "empty"
        run_dir.mkdir(parents=True)

        with patch.dict("os.environ", {}, clear=False):
            import os

            os.environ.pop("_BELT_SCENARIOS_ROOT", None)
            result = _scenarios_root(run_dir)

        from belt.constants import SCENARIOS_DIR

        assert result == SCENARIOS_DIR


class TestScoreScenarioWithTurnOutput:
    """Verifies that score_scenario loads TurnOutput files when available."""

    def test_loads_from_output_json(self, tmp_path: Path):
        scenarios_dir = tmp_path / "scenarios"
        outcomes_root = tmp_path / "outcomes" / "run1"
        _make_scenario_file(scenarios_dir, "group1", "test_scenario")

        outcome_dir = outcomes_root / "group1" / "test_scenario"
        outcome_dir.mkdir(parents=True)
        _write_turn_output(outcome_dir, 0, raw_cli="│ 1.0s reply_to_user\n│ Hello!", has_reply=True)

        from belt.scorer.rules import RuleBasedScorer

        with patch("belt.scorer.scenario_map.SCENARIOS_DIR", scenarios_dir):
            result = score_scenario(outcome_dir, outcomes_root, [RuleBasedScorer()])

        assert result.overall_pass is True
        assert "rules" in result.scores

    def test_falls_back_to_raw_files(self, tmp_path: Path):
        scenarios_dir = tmp_path / "scenarios"
        outcomes_root = tmp_path / "outcomes" / "run1"
        scenario = Scenario(
            name="test_scenario",
            description="test",
            turns=[Turn(message="Turn 0", expect=TurnExpectation(has_reply=False))],
        )
        path = scenarios_dir / "group1" / "test_scenario.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(scenario.model_dump_json(indent=2))

        outcome_dir = outcomes_root / "group1" / "test_scenario"
        outcome_dir.mkdir(parents=True)
        _write_raw_cli(outcome_dir, 0, "clean output without errors")

        from belt.scorer.rules import RuleBasedScorer

        with patch("belt.scorer.scenario_map.SCENARIOS_DIR", scenarios_dir):
            result = score_scenario(outcome_dir, outcomes_root, [RuleBasedScorer()])

        assert result.overall_pass is True

    def test_fallback_includes_state(self, tmp_path: Path):
        """Raw-file fallback loads state JSON into raw_state for downstream use."""
        scenarios_dir = tmp_path / "scenarios"
        outcomes_root = tmp_path / "outcomes" / "run1"
        scenario = Scenario(
            name="test_scenario",
            description="test",
            turns=[Turn(message="Turn 0", expect=TurnExpectation(has_reply=False))],
        )
        path = scenarios_dir / "group1" / "test_scenario.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(scenario.model_dump_json(indent=2))

        outcome_dir = outcomes_root / "group1" / "test_scenario"
        outcome_dir.mkdir(parents=True)
        _write_raw_cli(outcome_dir, 0, "clean output")
        state = {
            "values": {
                "messages": [
                    {"type": "ai", "tool_calls": [{"name": "reply_to_user", "id": "c1", "args": {"message": "hi"}}]}
                ]
            }
        }
        (outcome_dir / "turn_0_state.json").write_text(json.dumps(state))

        from belt.scorer.rules import RuleBasedScorer

        with patch("belt.scorer.scenario_map.SCENARIOS_DIR", scenarios_dir):
            result = score_scenario(outcome_dir, outcomes_root, [RuleBasedScorer()])

        assert result.overall_pass is True

    def test_missing_turn_file_creates_check(self, tmp_path: Path):
        scenarios_dir = tmp_path / "scenarios"
        outcomes_root = tmp_path / "outcomes" / "run1"
        _make_scenario_file(scenarios_dir, "group1", "test_scenario", turns=2)

        outcome_dir = outcomes_root / "group1" / "test_scenario"
        outcome_dir.mkdir(parents=True)
        _write_turn_output(outcome_dir, 0, raw_cli="good", has_reply=True)

        from belt.scorer.rules import RuleBasedScorer

        with patch("belt.scorer.scenario_map.SCENARIOS_DIR", scenarios_dir):
            result = score_scenario(outcome_dir, outcomes_root, [RuleBasedScorer()])

        assert result.overall_pass is False
        checks = result.scores["rules"].checks
        missing = [c for c in checks if "turn_1_exists" in c.check]
        assert len(missing) == 1
        assert missing[0].passed is False


# ── map_to_scenario trial suffix stripping ──


class TestMapToScenarioTrials:
    def test_no_trial_suffix(self, tmp_path: Path) -> None:
        """Regular scenario name maps normally."""
        outcomes = tmp_path / "outcomes"
        outcome_dir = outcomes / "mygroup" / "my_scenario"
        outcome_dir.mkdir(parents=True)
        scenarios_dir = tmp_path / "scenarios"
        with patch("belt.scorer.scenario_map.SCENARIOS_DIR", scenarios_dir):
            result = map_to_scenario(outcome_dir, outcomes)
        assert result == scenarios_dir / "mygroup" / "my_scenario.json"

    def test_trial_suffix_stripped(self, tmp_path: Path) -> None:
        """__trial_N suffix is stripped to find the original scenario file."""
        outcomes = tmp_path / "outcomes"
        outcome_dir = outcomes / "mygroup" / "my_scenario__trial_3"
        outcome_dir.mkdir(parents=True)
        scenarios_dir = tmp_path / "scenarios"
        with patch("belt.scorer.scenario_map.SCENARIOS_DIR", scenarios_dir):
            result = map_to_scenario(outcome_dir, outcomes)
        assert result == scenarios_dir / "mygroup" / "my_scenario.json"

    def test_trial_suffix_multi_digit(self, tmp_path: Path) -> None:
        outcomes = tmp_path / "outcomes"
        outcome_dir = outcomes / "g" / "s__trial_42"
        outcome_dir.mkdir(parents=True)
        scenarios_dir = tmp_path / "scenarios"
        with patch("belt.scorer.scenario_map.SCENARIOS_DIR", scenarios_dir):
            result = map_to_scenario(outcome_dir, outcomes)
        assert result == scenarios_dir / "g" / "s.json"


# ── _parse_dimension_defs ──


class TestParseDimensionDefs:
    def test_string_shorthand(self) -> None:
        result = _parse_dimension_defs(["tool_selection", "safety"])
        assert len(result) == 2
        assert result[0].name == "tool_selection"
        assert result[0].description == "Tool Selection"
        assert result[1].name == "safety"

    def test_full_dict(self) -> None:
        result = _parse_dimension_defs(
            [
                {
                    "name": "accuracy",
                    "description": "Was the answer correct?",
                    "high": "fully correct",
                    "medium": "partially correct",
                    "low": "wrong",
                }
            ]
        )
        assert len(result) == 1
        assert result[0].name == "accuracy"
        assert result[0].high == "fully correct"
        assert result[0].low == "wrong"

    def test_mixed(self) -> None:
        result = _parse_dimension_defs(
            [
                "simple_dim",
                {"name": "rich_dim", "description": "detailed"},
            ]
        )
        assert len(result) == 2
        assert result[0].name == "simple_dim"
        assert result[1].name == "rich_dim"
        assert result[1].description == "detailed"

    def test_empty(self) -> None:
        assert _parse_dimension_defs([]) == []


# ── extend_defaults in _build_scorers ──


class TestBuildScorersExtendDefaults:
    def test_extend_defaults_merges_with_generic(self) -> None:
        from belt.agent.scoring import GENERIC_DIMENSIONS
        from belt.commands.score import _build_scorers
        from belt.scorer.config_schema import JudgeDef

        with patch("belt.scorer.pipeline.load_scorer_config") as mock_load:
            mock_load.return_value = (
                [
                    JudgeDef(
                        name="test_judge",
                        model="openai/gpt-4.1",
                        extend_defaults=True,
                        dimensions=[{"name": "custom_dim", "description": "My custom dimension"}],
                    )
                ],
                None,
            )
            _, llm_scorers = _build_scorers("llm", {}, scorer_config_path="fake.yaml", skip_availability=True)
            assert len(llm_scorers) == 1
            dim_names = llm_scorers[0].strategy.dimension_names
            for gd in GENERIC_DIMENSIONS:
                assert gd.name in dim_names
            assert "custom_dim" in dim_names

    def test_no_extend_replaces_defaults(self) -> None:
        from belt.commands.score import _build_scorers
        from belt.scorer.config_schema import JudgeDef

        with patch("belt.scorer.pipeline.load_scorer_config") as mock_load:
            mock_load.return_value = (
                [
                    JudgeDef(
                        name="test_judge",
                        model="openai/gpt-4.1",
                        dimensions=[{"name": "only_this"}],
                    )
                ],
                None,
            )
            _, llm_scorers = _build_scorers("llm", {}, scorer_config_path="fake.yaml", skip_availability=True)
            assert llm_scorers[0].strategy.dimension_names == ["only_this"]
