# (c) JFrog Ltd. (2026)

"""End-to-end integration test: workspace isolation through the full pipeline.

Simulates what happens when `belt run` executes code-editing scenarios
with workspace isolation. Uses a synthetic agent that makes real file edits,
then verifies the orchestrator + scorer produce correct results.
"""

from __future__ import annotations

import json
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from belt.agent.base import BaseAgentAdapter
from belt.entities import AgentConfig, GroupConfig, Scenario, StateExpectation, Turn, TurnExpectation, TurnOutput
from belt.runner.orchestrator import build_agent_config, run_scenario_turns
from belt.runner.workspace import WorkspaceManager
from belt.scorer.rules import RuleBasedScorer


def _init_fixture_repo(path: Path) -> None:
    """Create a minimal fixture repo matching examples/fixtures/sample-project."""
    path.mkdir(parents=True, exist_ok=True)
    (path / "src").mkdir()
    (path / "src" / "calculator.py").write_text("def add(a, b): return a + b\ndef divide(a, b): return a / b\n")
    (path / "README.md").write_text("# Sample\n")
    subprocess.run(["git", "init"], cwd=path, capture_output=True, check=True)
    subprocess.run(["git", "config", "user.email", "test@test.com"], cwd=path, capture_output=True, check=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=path, capture_output=True, check=True)
    subprocess.run(["git", "add", "."], cwd=path, capture_output=True, check=True)
    subprocess.run(["git", "commit", "-m", "init"], cwd=path, capture_output=True, check=True)


class CodeEditingAgentAdapter(BaseAgentAdapter):
    """Simulates an agent that edits code in the workspace."""

    def __init__(self, action: str = "add_tests"):
        self._action = action
        self._workspace_dir: str | None = None

    def setup(self, config: AgentConfig) -> None:
        self._workspace_dir = config.workspace_dir

    def execute(self, message: str, flags: list[str]) -> str:
        ws = Path(self._workspace_dir) if self._workspace_dir else Path.cwd()

        if self._action == "add_tests":
            tests_dir = ws / "tests"
            tests_dir.mkdir(exist_ok=True)
            (tests_dir / "test_calculator.py").write_text(
                "from src.calculator import add, divide\n\n"
                "def test_add():\n    assert add(1, 2) == 3\n\n"
                "def test_divide():\n    assert divide(10, 2) == 5\n\n"
                "def test_divide_zero():\n    import pytest\n"
                "    with pytest.raises(ZeroDivisionError):\n"
                "        divide(1, 0)\n"
            )
            return "Created tests/test_calculator.py with 3 test functions"

        elif self._action == "fix_bug":
            calc = ws / "src" / "calculator.py"
            calc.write_text(
                "def add(a, b): return a + b\n"
                "def divide(a, b):\n"
                "    if b == 0:\n"
                '        raise ValueError("Cannot divide by zero")\n'
                "    return a / b\n"
            )
            return "Fixed divide() to raise ValueError on zero divisor"

        return "no-op"

    def fetch_results(self, raw_output: str) -> TurnOutput:
        return TurnOutput(raw_cli=raw_output, reply_text=raw_output, has_reply=True)

    def teardown(self) -> None:
        pass


class TestE2EAddTests:
    """Full pipeline: agent edits → worktree captures → scorer verifies."""

    def test_add_tests_scenario_passes_rules(self, tmp_path: Path):
        repo = tmp_path / "repo"
        _init_fixture_repo(repo)

        scenario = Scenario(
            name="add_tests",
            description="Add tests for calculator",
            turns=[
                Turn(
                    message="Add tests for calculator",
                    expect=TurnExpectation(
                        no_errors=True,
                        has_reply=True,
                        files_modified_any=["tests/test_calculator.py"],
                        git_diff_contains=["def test_"],
                    ),
                    state_expect=StateExpectation(
                        files_exist=["tests/test_calculator.py"],
                        files_contain={"tests/test_calculator.py": "def test_"},
                    ),
                ),
            ],
        )

        mgr = WorkspaceManager(repo)
        agent = CodeEditingAgentAdapter(action="add_tests")
        config = build_agent_config(GroupConfig(agent="test"), scenario, None)
        outcome = tmp_path / "outcome"

        result = run_scenario_turns(agent, scenario, outcome, config, workspace_manager=mgr)
        assert result.turns_completed == 1

        # Verify artifacts
        output_data = json.loads((outcome / "turn_0_output.json").read_text())
        assert output_data["git_diff"] is not None
        assert "tests/test_calculator.py" in output_data["files_modified"]
        assert output_data["workspace_files"]["tests/test_calculator.py"] is not None
        assert "def test_" in output_data["workspace_files"]["tests/test_calculator.py"]

        # Score with rules
        turn_output = TurnOutput.model_validate(output_data)
        scorer = RuleBasedScorer()
        score_result = scorer.score(scenario, [turn_output])
        assert score_result is not None
        assert score_result.passed, f"Rules failed: {[c for c in score_result.data['checks'] if not c['passed']]}"

        # Original repo unchanged
        assert not (repo / "tests").exists()
        mgr.cleanup_all()

    def test_fix_bug_scenario_passes_rules(self, tmp_path: Path):
        repo = tmp_path / "repo"
        _init_fixture_repo(repo)

        scenario = Scenario(
            name="fix_divide_bug",
            description="Fix division by zero",
            turns=[
                Turn(
                    message="Fix divide bug",
                    expect=TurnExpectation(
                        no_errors=True,
                        has_reply=True,
                        files_modified_any=["src/calculator.py"],
                        files_not_modified=["README.md"],
                        git_diff_contains=["ValueError", "zero"],
                    ),
                    state_expect=StateExpectation(
                        files_contain={"src/calculator.py": "ValueError"},
                    ),
                ),
            ],
        )

        mgr = WorkspaceManager(repo)
        agent = CodeEditingAgentAdapter(action="fix_bug")
        config = build_agent_config(GroupConfig(agent="test"), scenario, None)
        outcome = tmp_path / "outcome"

        result = run_scenario_turns(agent, scenario, outcome, config, workspace_manager=mgr)
        assert result.turns_completed == 1

        output_data = json.loads((outcome / "turn_0_output.json").read_text())
        assert "ValueError" in output_data["git_diff"]
        assert "src/calculator.py" in output_data["files_modified"]
        assert "README.md" not in output_data["files_modified"]

        turn_output = TurnOutput.model_validate(output_data)
        scorer = RuleBasedScorer()
        score_result = scorer.score(scenario, [turn_output])
        assert score_result is not None
        assert score_result.passed, f"Rules failed: {[c for c in score_result.data['checks'] if not c['passed']]}"

        # Original unchanged
        original_calc = (repo / "src" / "calculator.py").read_text()
        assert "ValueError" not in original_calc
        mgr.cleanup_all()


class TestE2EParallelIsolation:
    """Verify parallel workers don't interfere with each other."""

    def test_two_scenarios_parallel_isolated(self, tmp_path: Path):
        repo = tmp_path / "repo"
        _init_fixture_repo(repo)

        def _run_scenario(action: str, name: str) -> dict:
            scenario = Scenario(
                name=name,
                description=f"Parallel {action}",
                turns=[Turn(message=f"Do {action}", expect=TurnExpectation(no_errors=True, has_reply=True))],
            )
            mgr = WorkspaceManager(repo)
            agent = CodeEditingAgentAdapter(action=action)
            config = build_agent_config(GroupConfig(agent="test"), scenario, None)
            outcome = tmp_path / "outcomes" / name

            run_scenario_turns(agent, scenario, outcome, config, workspace_manager=mgr)
            data = json.loads((outcome / "turn_0_output.json").read_text())
            mgr.cleanup_all()
            return data

        with ThreadPoolExecutor(max_workers=2) as ex:
            f1 = ex.submit(_run_scenario, "add_tests", "parallel_add")
            f2 = ex.submit(_run_scenario, "fix_bug", "parallel_fix")
            d1 = f1.result()
            d2 = f2.result()

        # Each scenario saw different modifications
        assert "tests/test_calculator.py" in d1["files_modified"]
        assert "src/calculator.py" in d2["files_modified"]

        # They didn't interfere - add_tests didn't modify calculator.py
        assert "src/calculator.py" not in d1["files_modified"]
        # fix_bug didn't create tests
        assert "tests/test_calculator.py" not in d2["files_modified"]

        # Original repo pristine
        assert not (repo / "tests").exists()
        original = (repo / "src" / "calculator.py").read_text()
        assert "ValueError" not in original
