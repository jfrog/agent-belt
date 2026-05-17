# (c) JFrog Ltd. (2026)

"""Integration tests - exercise the full flow from agent through orchestrator to scorer.

Uses a stub agent (no real CLI calls) to test the wiring end-to-end:
agent.setup → execute → fetch_results → orchestrator artifact writing → scorer loading.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from belt.agent.base import AgentNotAvailableError, BaseAgentAdapter
from belt.agent.registry import get_agent_class
from belt.agent.scoring import default_scoring_strategy
from belt.commands.score import score_scenario
from belt.config import _find_config_file, load_judge_config
from belt.entities import AgentConfig, GroupConfig, Scenario, ScenarioScore, Turn, TurnExpectation, TurnOutput
from belt.runner.orchestrator import build_agent_config, run_scenario_turns
from belt.scorer.rules import RuleBasedScorer

# ── Stub agent ──


class StubAgentAdapter(BaseAgentAdapter):
    """Predictable agent for testing - returns canned responses."""

    def __init__(self, **_kwargs: Any):
        self._turn_idx = 0

    def setup(self, config: AgentConfig) -> None:
        pass

    def execute(self, message: str, flags: list[str]) -> str:
        self._turn_idx += 1
        return f"StubAgentAdapter response to: {message}"

    def fetch_results(self, raw_output: str) -> TurnOutput:
        return TurnOutput(
            raw_cli=raw_output,
            reply_text=raw_output,
            has_reply=True,
            has_error=False,
        )

    def teardown(self) -> None:
        pass

    def metadata(self) -> dict[str, Any] | None:
        return {"stub": True, "turns": self._turn_idx}


class FailingAgentAdapter(BaseAgentAdapter):
    """Agent that fails on execute - for error path testing."""

    def __init__(self, **_kwargs: Any):
        pass

    @classmethod
    def check_available(cls) -> None:
        raise AgentNotAvailableError("failing", "intentionally unavailable", "this is a test")

    def setup(self, config: AgentConfig) -> None:
        pass

    def execute(self, message: str, flags: list[str]) -> str:
        raise RuntimeError("CLI tool crashed")

    def fetch_results(self, raw_output: str) -> TurnOutput:
        raise NotImplementedError

    def teardown(self) -> None:
        pass


# ── Fixtures ──


@pytest.fixture
def simple_scenario() -> Scenario:
    return Scenario(
        name="test_scenario",
        description="A test scenario",
        turns=[
            Turn(
                message="Hello, what can you do?",
                expect=TurnExpectation(no_errors=True, has_reply=True),
            ),
            Turn(
                message="Now do something else",
                expect=TurnExpectation(no_errors=True, has_reply=True, contains=["something"]),
            ),
        ],
    )


@pytest.fixture
def group_config() -> GroupConfig:
    return GroupConfig(agent="stub")


# ── Integration: Orchestrator → Artifact Writing ──


class TestOrchestratorFlow:
    def test_full_scenario_writes_artifacts(self, tmp_path: Path, simple_scenario: Scenario, group_config: GroupConfig):
        agent = StubAgentAdapter()
        config = build_agent_config(group_config, simple_scenario, shared_state=None)
        outcome_dir = tmp_path / "group" / "test_scenario"

        result = run_scenario_turns(agent, simple_scenario, outcome_dir, config)

        assert result.turns_completed == 2
        assert result.error is None
        assert result.agent_metadata == {"stub": True, "turns": 2}

        for i in range(2):
            assert (outcome_dir / f"turn_{i}_cli.txt").exists()
            assert (outcome_dir / f"turn_{i}_output.json").exists()
            output = TurnOutput.model_validate_json((outcome_dir / f"turn_{i}_output.json").read_text())
            assert output.has_reply is True
            assert output.has_error is False

    def test_failing_agent_writes_sentinel(self, tmp_path: Path, simple_scenario: Scenario, group_config: GroupConfig):
        agent = FailingAgentAdapter()
        config = build_agent_config(group_config, simple_scenario, shared_state=None)
        outcome_dir = tmp_path / "group" / "test_scenario"

        result = run_scenario_turns(agent, simple_scenario, outcome_dir, config)

        assert result.turns_completed == 0
        sentinel = outcome_dir / "turn_0_cli.txt"
        assert sentinel.exists()
        assert "RuntimeError" in sentinel.read_text()


# ── Integration: Orchestrator → Scorer ──


class TestScorerFlow:
    def test_rule_scorer_on_orchestrator_output(
        self, tmp_path: Path, simple_scenario: Scenario, group_config: GroupConfig
    ):
        agent = StubAgentAdapter()
        config = build_agent_config(group_config, simple_scenario, shared_state=None)

        outcomes_root = tmp_path / "outcomes"
        scenario_dir = tmp_path / "scenarios" / "group"
        scenario_dir.mkdir(parents=True)

        outcome_dir = outcomes_root / "group" / "test_scenario"
        run_scenario_turns(agent, simple_scenario, outcome_dir, config)

        scenario_path = scenario_dir / "test_scenario.json"
        scenario_path.write_text(simple_scenario.model_dump_json(indent=2))

        config_path = scenario_dir / "_config.json"
        config_path.write_text(group_config.model_dump_json(indent=2))

        scorers = [RuleBasedScorer()]

        with patch("belt.scorer.scenario_map.SCENARIOS_DIR", scenario_dir.parent):
            score = score_scenario(outcome_dir, outcomes_root, scorers)

        assert isinstance(score, ScenarioScore)
        assert score.scenario_name == "test_scenario"
        assert "rules" in score.scores

        rules_data = score.scores["rules"]
        checks = rules_data.checks
        assert len(checks) > 0

    def test_missing_turn_detected_by_scorer(self, tmp_path: Path, group_config: GroupConfig):
        """Scorer detects missing turn files and reports them."""
        scenario = Scenario(
            name="missing_turn",
            description="Scenario where turn 1 is missing",
            turns=[
                Turn(message="Turn 0"),
                Turn(message="Turn 1"),
            ],
        )

        outcomes_root = tmp_path / "outcomes"
        outcome_dir = outcomes_root / "group" / "missing_turn"
        outcome_dir.mkdir(parents=True)

        (outcome_dir / "turn_0_cli.txt").write_text("turn 0 output")

        scenario_dir = tmp_path / "scenarios" / "group"
        scenario_dir.mkdir(parents=True)
        (scenario_dir / "missing_turn.json").write_text(scenario.model_dump_json(indent=2))
        (scenario_dir / "_config.json").write_text(group_config.model_dump_json(indent=2))

        scorers = [RuleBasedScorer()]

        with patch("belt.scorer.scenario_map.SCENARIOS_DIR", scenario_dir.parent):
            score = score_scenario(outcome_dir, outcomes_root, scorers)

        assert score.overall_pass is False


# ── Fail-fast: check_available ──


class TestCheckAvailable:
    def test_base_agent_is_always_available(self):
        BaseAgentAdapter.check_available()

    def test_failing_agent_raises(self):
        with pytest.raises(AgentNotAvailableError) as exc_info:
            FailingAgentAdapter.check_available()
        assert "intentionally unavailable" in str(exc_info.value)
        assert exc_info.value.agent_name == "failing"
        assert exc_info.value.suggestion == "this is a test"

    def test_claude_check_when_not_installed(self):
        with patch("shutil.which", return_value=None):
            with pytest.raises(AgentNotAvailableError):
                from belt.agent.claude_code import ClaudeCodeAgentAdapter

                ClaudeCodeAgentAdapter.check_available()


# ── Agent Registry ──


class TestRegistryIntegration:
    def test_all_registered_agents_importable(self):
        # Only check the built-in registry, not entry-point-installed third-party
        # plugins. A locally-installed plugin that imports from a stale
        # ``belt.*`` path (e.g., after a refactor) shouldn't break this
        # repo's test suite.
        from belt.agent.registry import _AGENT_REGISTRY

        names = sorted(_AGENT_REGISTRY)
        assert len(names) >= 5, f"Expected at least 5 built-in agents, got {names}"
        for name in names:
            cls = get_agent_class(name)
            assert issubclass(cls, BaseAgentAdapter)

    def test_unknown_agent_raises(self):
        from belt.errors import ConfigError

        with pytest.raises(ConfigError, match="Unknown agent"):
            get_agent_class("nonexistent")


# ── Config layering ──


class TestConfigLayering:
    def test_no_layer_supplies_model_raises(self, tmp_path: Path):
        # ``model`` is required and there is no built-in default.
        # An isolated config path with no env / cli overrides surfaces the
        # three-source ConfigError instead of silently picking ``openai/...``.
        from belt.errors import ConfigError

        with pytest.raises(ConfigError) as exc_info:
            load_judge_config(config_path=tmp_path / "nonexistent.yaml")
        msg = str(exc_info.value)
        assert "BELT_LLM_MODEL" in msg
        assert "belt.yaml" in msg
        assert "--scorer-arg model=" in msg

    def test_defaults_for_non_required_fields(self, tmp_path: Path):
        # Numeric / boolean fields still have defaults; only ``model`` is required.
        config = load_judge_config(
            config_path=tmp_path / "nonexistent.yaml",
            cli_overrides={"model": "openai/gpt-5.4-mini"},
        )
        assert config.model == "openai/gpt-5.4-mini"
        assert config.temperature == 0.0
        assert config.seed == 2008

    def test_env_var_override(self):
        with patch.dict("os.environ", {"BELT_LLM_MODEL": "my-model", "BELT_LLM_PROVIDER": "openai"}):
            config = load_judge_config()
            assert config.model == "my-model"
            assert config.provider == "openai"

    def test_cli_overrides_env(self):
        with patch.dict("os.environ", {"BELT_LLM_MODEL": "env-model"}):
            config = load_judge_config(cli_overrides={"model": "cli-model"})
            assert config.model == "cli-model"

    def test_yaml_config(self, tmp_path: Path):
        yaml_content = "llm:\n  model: yaml-model\n  temperature: 0.7\n"
        config_path = tmp_path / "belt.yaml"
        config_path.write_text(yaml_content)
        config = load_judge_config(config_path=config_path)
        assert config.model == "yaml-model"
        assert config.temperature == 0.7

    def test_yaml_overridden_by_env(self, tmp_path: Path):
        yaml_content = "llm:\n  model: yaml-model\n"
        config_path = tmp_path / "belt.yaml"
        config_path.write_text(yaml_content)
        with patch.dict("os.environ", {"BELT_LLM_MODEL": "env-model"}):
            config = load_judge_config(config_path=config_path)
            assert config.model == "env-model"

    def test_find_config_file(self, tmp_path: Path):
        (tmp_path / "belt.yaml").write_text("llm:\n  model: found\n")

        found = _find_config_file(tmp_path)
        assert found is not None
        assert found.name == "belt.yaml"

        # Walks up from nested dir and finds the yaml in an ancestor
        subdir = tmp_path / "sub" / "dir"
        subdir.mkdir(parents=True)
        found = _find_config_file(subdir)
        assert found is not None
        assert found == tmp_path / "belt.yaml"


# ── Scoring Strategy in multi-agent context ──


class TestScoringStrategyIntegration:
    def test_generic_has_4_dimensions(self):
        strategy = default_scoring_strategy()
        assert len(strategy.dimensions) == 4

    def test_schema_is_valid_json_schema(self):
        strategy = default_scoring_strategy()
        schema = strategy.build_schema()
        assert schema["type"] == "object"
        assert "overall_pass" in schema["required"]
        assert "$defs" in schema
