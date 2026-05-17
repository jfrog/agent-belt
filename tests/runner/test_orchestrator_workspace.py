# (c) JFrog Ltd. (2026)

"""Tests for orchestrator + WorkspaceManager integration."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

from belt.agent.base import BaseAgentAdapter
from belt.entities import AgentConfig, GroupConfig, Scenario, StateExpectation, Turn, TurnOutput
from belt.runner.orchestrator import build_agent_config, run_scenario_turns
from belt.runner.workspace import WorkspaceManager


def _init_git_repo(path: Path, files: dict[str, str] | None = None) -> None:
    path.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "init"], cwd=path, capture_output=True, check=True)
    subprocess.run(["git", "config", "user.email", "test@test.com"], cwd=path, capture_output=True, check=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=path, capture_output=True, check=True)
    for name, content in (files or {"README.md": "# Test"}).items():
        (path / name).write_text(content)
    subprocess.run(["git", "add", "."], cwd=path, capture_output=True, check=True)
    subprocess.run(["git", "commit", "-m", "initial"], cwd=path, capture_output=True, check=True)


class EditingAgentAdapter(BaseAgentAdapter):
    """Agent that modifies files in its workspace_dir during execute()."""

    def __init__(self, edits: dict[str, str] | None = None, new_files: dict[str, str] | None = None):
        self._edits = edits or {}
        self._new_files = new_files or {}
        self._workspace_dir: str | None = None

    def setup(self, config: AgentConfig) -> None:
        self._workspace_dir = config.workspace_dir

    def execute(self, message: str, flags: list[str]) -> str:
        if self._workspace_dir:
            ws = Path(self._workspace_dir)
            for path, content in self._edits.items():
                (ws / path).write_text(content)
            for path, content in self._new_files.items():
                (ws / path).parent.mkdir(parents=True, exist_ok=True)
                (ws / path).write_text(content)
        return f"edited files in {self._workspace_dir}"

    def fetch_results(self, raw_output: str) -> TurnOutput:
        return TurnOutput(raw_cli=raw_output, reply_text="done", has_reply=True)

    def teardown(self) -> None:
        pass


class TestOrchestratorWithWorkspace:
    def test_workspace_dir_passed_to_agent(self, tmp_path: Path):
        repo = tmp_path / "repo"
        _init_git_repo(repo, {"main.py": "old"})
        mgr = WorkspaceManager(repo)

        agent = EditingAgentAdapter(edits={"main.py": "new"})
        scenario = Scenario(
            name="edit_test",
            description="Tests workspace isolation",
            turns=[Turn(message="edit main.py")],
        )
        config = build_agent_config(GroupConfig(agent="test"), scenario, None)
        outcome = tmp_path / "outcome"

        result = run_scenario_turns(agent, scenario, outcome, config, workspace_manager=mgr)

        assert result.turns_completed == 1
        assert agent._workspace_dir is not None
        assert agent._workspace_dir != str(repo)

        # Original repo should be unchanged
        assert (repo / "main.py").read_text() == "old"
        mgr.cleanup_all()

    def test_git_diff_captured_in_output(self, tmp_path: Path):
        repo = tmp_path / "repo"
        _init_git_repo(repo, {"code.py": "def original(): pass"})
        mgr = WorkspaceManager(repo)

        agent = EditingAgentAdapter(edits={"code.py": "def modified(): pass"})
        scenario = Scenario(
            name="diff_capture",
            description="Verify git diff appears in turn output",
            turns=[Turn(message="modify code.py")],
        )
        config = build_agent_config(GroupConfig(agent="test"), scenario, None)
        outcome = tmp_path / "outcome"

        run_scenario_turns(agent, scenario, outcome, config, workspace_manager=mgr)

        output_data = json.loads((outcome / "turn_0_output.json").read_text())
        assert output_data["git_diff"] is not None
        assert "modified" in output_data["git_diff"]
        assert "code.py" in output_data["files_modified"]
        mgr.cleanup_all()

    def test_new_files_captured_in_files_modified(self, tmp_path: Path):
        repo = tmp_path / "repo"
        _init_git_repo(repo)
        mgr = WorkspaceManager(repo)

        agent = EditingAgentAdapter(new_files={"tests/test_new.py": "def test_it(): pass"})
        scenario = Scenario(
            name="new_file_capture",
            description="New files appear in files_modified",
            turns=[Turn(message="add tests")],
        )
        config = build_agent_config(GroupConfig(agent="test"), scenario, None)
        outcome = tmp_path / "outcome"

        run_scenario_turns(agent, scenario, outcome, config, workspace_manager=mgr)

        output_data = json.loads((outcome / "turn_0_output.json").read_text())
        assert "tests/test_new.py" in output_data["files_modified"]
        mgr.cleanup_all()

    def test_state_expect_uses_worktree_path(self, tmp_path: Path):
        repo = tmp_path / "repo"
        _init_git_repo(repo)
        mgr = WorkspaceManager(repo)

        agent = EditingAgentAdapter(new_files={"output.txt": "result data"})
        scenario = Scenario(
            name="state_expect_ws",
            description="StateExpectation resolves against worktree",
            turns=[
                Turn(
                    message="create output.txt",
                    state_expect=StateExpectation(
                        files_exist=["output.txt"],
                        files_contain={"output.txt": "result"},
                    ),
                ),
            ],
        )
        config = build_agent_config(GroupConfig(agent="test"), scenario, None)
        outcome = tmp_path / "outcome"

        run_scenario_turns(agent, scenario, outcome, config, workspace_manager=mgr)

        output_data = json.loads((outcome / "turn_0_output.json").read_text())
        assert output_data["workspace_files"]["output.txt"] is not None
        assert "result" in output_data["workspace_files"]["output.txt"]
        mgr.cleanup_all()

    def test_worktree_released_after_run(self, tmp_path: Path):
        repo = tmp_path / "repo"
        _init_git_repo(repo)
        mgr = WorkspaceManager(repo)

        agent = EditingAgentAdapter()
        scenario = Scenario(
            name="cleanup_test",
            description="Worktree cleaned up after run",
            turns=[Turn(message="do nothing")],
        )
        config = build_agent_config(GroupConfig(agent="test"), scenario, None)
        outcome = tmp_path / "outcome"

        run_scenario_turns(agent, scenario, outcome, config, workspace_manager=mgr)

        assert len(mgr._worktrees) == 0
        mgr.cleanup_all()

    def test_no_workspace_manager_preserves_old_behavior(self, tmp_path: Path):
        agent = EditingAgentAdapter()
        scenario = Scenario(
            name="no_isolation",
            description="Without workspace_manager, behaves like before",
            turns=[Turn(message="no workspace")],
        )
        config = build_agent_config(GroupConfig(agent="test"), scenario, None)
        outcome = tmp_path / "outcome"

        result = run_scenario_turns(agent, scenario, outcome, config)

        assert result.turns_completed == 1
        output_data = json.loads((outcome / "turn_0_output.json").read_text())
        assert output_data["git_diff"] is None
        assert output_data["files_modified"] == []
