# (c) JFrog Ltd. (2026)

"""Tests for the agent-agnostic scenario orchestrator."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from belt.agent.base import BaseAgentAdapter
from belt.entities import AgentConfig, GroupConfig, Scenario, StateExpectation, Turn, TurnExpectation, TurnOutput
from belt.runner.orchestrator import (
    _build_sandbox_for_scenario,
    _capture_workspace_state,
    _write_runtime_info_sidecar,
    build_agent_config,
    run_scenario_turns,
)


class StubAgentAdapter(BaseAgentAdapter):
    """Minimal agent that records calls and returns canned TurnOutput."""

    def __init__(self, replies: list[str] | None = None, raw_state: str | None = None, fail_on_turn: int | None = None):
        self._replies = replies or ["hello"]
        self._raw_state = raw_state
        self._fail_on_turn = fail_on_turn
        self.calls: list[dict[str, Any]] = []
        self._setup_called = False
        self._teardown_called = False
        self._turn_idx = 0

    def setup(self, config: AgentConfig) -> None:
        self._setup_called = True
        self.calls.append({"method": "setup", "scenario_name": config.scenario_name})

    def execute(self, message: str, flags: list[str]) -> str:
        idx = self._turn_idx
        self._turn_idx += 1
        if self._fail_on_turn is not None and idx == self._fail_on_turn:
            raise RuntimeError(f"Simulated failure on turn {idx}")
        self.calls.append({"method": "execute", "message": message, "flags": flags})
        reply_idx = min(idx, len(self._replies) - 1)
        return f"raw output for: {self._replies[reply_idx]}"

    def fetch_results(self, raw_output: str) -> TurnOutput:
        self.calls.append({"method": "fetch_results", "raw_output": raw_output})
        return TurnOutput(
            raw_cli=raw_output,
            raw_state=self._raw_state,
            reply_text=raw_output.removeprefix("raw output for: "),
            has_reply=True,
        )

    def teardown(self) -> None:
        self._teardown_called = True
        self.calls.append({"method": "teardown"})

    def metadata(self) -> dict[str, Any] | None:
        return {"stub": True, "turns_executed": self._turn_idx}


def _make_scenario(name: str = "test_scenario", turns: int = 2) -> Scenario:
    return Scenario(
        name=name,
        description=f"Test scenario with {turns} turns",
        turns=[Turn(message=f"Turn {i} message", expect=TurnExpectation()) for i in range(turns)],
    )


class TestRunScenarioTurns:
    def test_executes_all_turns(self, tmp_path: Path):
        agent = StubAgentAdapter(replies=["r0", "r1"])
        scenario = _make_scenario(turns=2)
        config = build_agent_config(GroupConfig(agent="test"), scenario, None)

        result = run_scenario_turns(agent, scenario, tmp_path / "outcome", config)

        assert result.turns_completed == 2
        assert result.scenario_name == "test_scenario"
        assert result.error is None
        assert result.agent_metadata == {"stub": True, "turns_executed": 2}

    def test_writes_cli_artifacts(self, tmp_path: Path):
        agent = StubAgentAdapter()
        scenario = _make_scenario(turns=1)
        outcome = tmp_path / "outcome"
        config = build_agent_config(GroupConfig(agent="test"), scenario, None)

        run_scenario_turns(agent, scenario, outcome, config)

        cli_file = outcome / "turn_0_cli.txt"
        assert cli_file.exists()
        assert "raw output for:" in cli_file.read_text()

    def test_writes_output_json(self, tmp_path: Path):
        agent = StubAgentAdapter()
        scenario = _make_scenario(turns=1)
        outcome = tmp_path / "outcome"
        config = build_agent_config(GroupConfig(agent="test"), scenario, None)

        run_scenario_turns(agent, scenario, outcome, config)

        output_file = outcome / "turn_0_output.json"
        assert output_file.exists()
        data = json.loads(output_file.read_text())
        assert data["has_reply"] is True
        assert "raw_cli" in data

    def test_writes_state_when_present(self, tmp_path: Path):
        state_json = json.dumps({"values": {"messages": []}})
        agent = StubAgentAdapter(raw_state=state_json)
        scenario = _make_scenario(turns=1)
        outcome = tmp_path / "outcome"
        config = build_agent_config(GroupConfig(agent="test"), scenario, None)

        run_scenario_turns(agent, scenario, outcome, config)

        state_file = outcome / "turn_0_state.json"
        assert state_file.exists()
        parsed = json.loads(state_file.read_text())
        assert "values" in parsed

    def test_no_state_file_when_state_is_none(self, tmp_path: Path):
        agent = StubAgentAdapter(raw_state=None)
        scenario = _make_scenario(turns=1)
        outcome = tmp_path / "outcome"
        config = build_agent_config(GroupConfig(agent="test"), scenario, None)

        run_scenario_turns(agent, scenario, outcome, config)

        assert not (outcome / "turn_0_state.json").exists()

    def test_stops_on_execute_failure(self, tmp_path: Path):
        agent = StubAgentAdapter(fail_on_turn=1)
        scenario = _make_scenario(turns=3)
        outcome = tmp_path / "outcome"
        config = build_agent_config(GroupConfig(agent="test"), scenario, None)

        result = run_scenario_turns(agent, scenario, outcome, config)

        assert result.turns_completed == 1
        sentinel = outcome / "turn_1_cli.txt"
        assert sentinel.exists()
        assert "RuntimeError" in sentinel.read_text()

    def test_calls_setup_and_teardown(self, tmp_path: Path):
        agent = StubAgentAdapter()
        scenario = _make_scenario(turns=1)
        config = build_agent_config(GroupConfig(agent="test"), scenario, None)

        run_scenario_turns(agent, scenario, tmp_path / "out", config)

        methods = [c["method"] for c in agent.calls]
        assert methods[0] == "setup"
        assert methods[-1] == "teardown"

    def test_teardown_called_even_on_failure(self, tmp_path: Path):
        agent = StubAgentAdapter(fail_on_turn=0)
        scenario = _make_scenario(turns=1)
        config = build_agent_config(GroupConfig(agent="test"), scenario, None)

        result = run_scenario_turns(agent, scenario, tmp_path / "out", config)

        assert result.turns_completed == 0
        assert agent._teardown_called

    def test_metadata_captured_before_teardown(self, tmp_path: Path):
        agent = StubAgentAdapter()
        scenario = _make_scenario(turns=2)
        config = build_agent_config(GroupConfig(agent="test"), scenario, None)

        result = run_scenario_turns(agent, scenario, tmp_path / "out", config)

        assert result.agent_metadata is not None
        assert result.agent_metadata["turns_executed"] == 2

    def test_creates_outcome_dir(self, tmp_path: Path):
        agent = StubAgentAdapter()
        scenario = _make_scenario(turns=1)
        outcome = tmp_path / "deep" / "nested" / "outcome"
        config = build_agent_config(GroupConfig(agent="test"), scenario, None)

        run_scenario_turns(agent, scenario, outcome, config)

        assert outcome.is_dir()

    def test_flags_passed_through(self, tmp_path: Path):
        agent = StubAgentAdapter()
        scenario = Scenario(
            name="flagged",
            description="Scenario with flags",
            turns=[Turn(message="msg", flags=["-rd", "accept"])],
        )
        config = build_agent_config(GroupConfig(agent="test"), scenario, None)

        run_scenario_turns(agent, scenario, tmp_path / "out", config)

        execute_calls = [c for c in agent.calls if c["method"] == "execute"]
        assert execute_calls[0]["flags"] == ["-rd", "accept"]


class TestWorkspaceCapture:
    def test_captures_existing_file(self, tmp_path: Path):
        (tmp_path / "hello.txt").write_text("world")
        se = StateExpectation(files_exist=["hello.txt"])
        to = TurnOutput(raw_cli="")
        _capture_workspace_state(to, se, tmp_path)
        assert to.workspace_files["hello.txt"] == "world"

    def test_captures_missing_file_as_none(self, tmp_path: Path):
        se = StateExpectation(files_exist=["nope.txt"])
        to = TurnOutput(raw_cli="")
        _capture_workspace_state(to, se, tmp_path)
        assert to.workspace_files["nope.txt"] is None

    def test_captures_files_contain_targets(self, tmp_path: Path):
        (tmp_path / "code.py").write_text("def fixed_function(): pass")
        se = StateExpectation(files_contain={"code.py": "fixed_function"})
        to = TurnOutput(raw_cli="")
        _capture_workspace_state(to, se, tmp_path)
        assert "fixed_function" in to.workspace_files["code.py"]

    def test_captures_files_not_exist_targets(self, tmp_path: Path):
        se = StateExpectation(files_not_exist=["deleted.txt"])
        to = TurnOutput(raw_cli="")
        _capture_workspace_state(to, se, tmp_path)
        assert to.workspace_files["deleted.txt"] is None

    def test_truncates_large_files(self, tmp_path: Path):
        (tmp_path / "big.txt").write_text("x" * 20_000)
        se = StateExpectation(files_exist=["big.txt"])
        to = TurnOutput(raw_cli="")
        _capture_workspace_state(to, se, tmp_path)
        assert len(to.workspace_files["big.txt"]) < 20_000
        assert "truncated" in to.workspace_files["big.txt"]

    def test_no_capture_when_no_state_expect(self, tmp_path: Path):
        (tmp_path / "file.txt").write_text("data")
        se = StateExpectation()
        to = TurnOutput(raw_cli="")
        _capture_workspace_state(to, se, tmp_path)
        assert to.workspace_files == {}

    def test_git_diff_captured(self, tmp_path: Path):
        import subprocess

        subprocess.run(["git", "init"], cwd=tmp_path, capture_output=True)
        subprocess.run(["git", "config", "user.email", "test@test.com"], cwd=tmp_path, capture_output=True)
        subprocess.run(["git", "config", "user.name", "Test"], cwd=tmp_path, capture_output=True)
        (tmp_path / "file.txt").write_text("initial")
        subprocess.run(["git", "add", "."], cwd=tmp_path, capture_output=True)
        subprocess.run(["git", "commit", "-m", "init"], cwd=tmp_path, capture_output=True)
        (tmp_path / "file.txt").write_text("modified")

        se = StateExpectation(capture_git_diff=True)
        to = TurnOutput(raw_cli="")
        _capture_workspace_state(to, se, tmp_path)
        assert to.raw_state is not None
        assert "modified" in to.raw_state

    def test_git_diff_not_captured_when_flag_off(self, tmp_path: Path):
        se = StateExpectation(capture_git_diff=False)
        to = TurnOutput(raw_cli="")
        _capture_workspace_state(to, se, tmp_path)
        assert to.raw_state is None

    def test_full_scenario_with_state_expect(self, tmp_path: Path):
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        agent = StubAgentAdapter()
        scenario = Scenario(
            name="state_test",
            description="Tests workspace state capture",
            turns=[
                Turn(
                    message="Create a file",
                    state_expect=StateExpectation(files_not_exist=["output.txt"]),
                ),
            ],
        )
        outcome = tmp_path / "outcome"
        config = build_agent_config(GroupConfig(agent="test"), scenario, None)
        result = run_scenario_turns(agent, scenario, outcome, config, workspace=workspace)

        assert result.turns_completed == 1
        output_data = json.loads((outcome / "turn_0_output.json").read_text())
        assert "output.txt" in output_data["workspace_files"]


class TestBuildAgentConfig:
    def test_basic_config(self):
        gc = GroupConfig(agent="test")
        scenario = _make_scenario()
        config = build_agent_config(gc, scenario, {"key": "value"})

        assert config.group_config == gc
        assert config.scenario_name == "test_scenario"
        assert config.shared_state == {"key": "value"}
        assert config.scenario_options == {}


class TestWriteRuntimeInfoSidecar:
    """``_write_runtime_info_sidecar`` projects the adapter-author flat
    ``runtime_info()`` dict into the persisted two-level shape
    (``agent.{name,adapter_class,args,auth_signals}`` /
    ``cli.{binary_path,version}``).

    This is the contract boundary between adapter authors (who write
    flat dicts) and the framework (which owns the persisted layout).
    The boundary is exercised end-to-end by ``build_card`` tests, but
    pinning it as a unit test catches a regression in the projection
    itself - e.g. a future refactor that swaps ``cli.binary_path`` and
    ``cli.version`` or that accidentally adds a ``schema_version`` key
    (the sidecar is intentionally unversioned; the benchmark card it
    feeds is versioned).
    """

    class _FlatAgent(BaseAgentAdapter):
        """Adapter that returns the canonical flat ``runtime_info()`` shape."""

        def __init__(self, captured: dict[str, Any] | None = None) -> None:
            self._captured_agent_args = captured or {}

        @classmethod
        def runtime_info(cls) -> dict[str, Any]:
            return {
                "adapter_class": cls.__name__,
                "cli_binary_path": "/opt/myagent/bin",
                "cli_version": "3.1.4",
                "auth_signals": ["env:MYAGENT_TOKEN"],
            }

        def setup(self, config: AgentConfig) -> None: ...
        def execute(self, message: str, flags: Any) -> str:
            return ""

        def fetch_results(self, raw: str) -> Any: ...
        def teardown(self) -> None: ...

    def test_flat_runtime_info_projects_to_nested_sidecar(self, tmp_path: Path) -> None:
        from belt.constants import RUNTIME_INFO_FILE

        outcome = tmp_path / "g" / "scn"
        outcome.mkdir(parents=True)
        agent = self._FlatAgent(captured={"model": "gpt-4"})
        config = AgentConfig(
            group_config=GroupConfig(agent="myagent"),
            scenario_name="scn",
        )

        _write_runtime_info_sidecar(outcome, agent, config, group_name="g")

        sidecar = json.loads((outcome / RUNTIME_INFO_FILE).read_text())
        assert sidecar == {
            "group": "g",
            "agent": {
                "name": "myagent",
                "adapter_class": "_FlatAgent",
                "args": {"model": "gpt-4"},
                "auth_signals": ["env:MYAGENT_TOKEN"],
            },
            "cli": {"binary_path": "/opt/myagent/bin", "version": "3.1.4"},
        }
        assert "schema_version" not in sidecar

    def test_secret_in_captured_args_is_redacted_at_projection(self, tmp_path: Path) -> None:
        # The boundary writes already-redacted args. A future refactor
        # that bypasses ``safe_agent_args`` would surface as a literal
        # secret value here.
        from belt.constants import RUNTIME_INFO_FILE

        outcome = tmp_path / "g" / "scn"
        outcome.mkdir(parents=True)
        agent = self._FlatAgent(captured={"api_key": "sk-MUST-NOT-LEAK", "model": "gpt-4"})
        config = AgentConfig(
            group_config=GroupConfig(agent="myagent"),
            scenario_name="scn",
        )

        _write_runtime_info_sidecar(outcome, agent, config, group_name="g")

        sidecar = json.loads((outcome / RUNTIME_INFO_FILE).read_text())
        assert sidecar["agent"]["args"]["api_key"] == "<set>"
        assert "sk-MUST-NOT-LEAK" not in (outcome / RUNTIME_INFO_FILE).read_text()
        assert sidecar["agent"]["args"]["model"] == "gpt-4"

    def test_group_name_falls_back_to_outcome_parent(self, tmp_path: Path) -> None:
        # Documented fallback when the caller omits ``group_name``: the
        # parent directory's basename is used. Inside the framework all
        # callers pass ``group_name``, but the contract is preserved.
        from belt.constants import RUNTIME_INFO_FILE

        outcome = tmp_path / "fallback_group" / "scn"
        outcome.mkdir(parents=True)
        agent = self._FlatAgent()
        config = AgentConfig(
            group_config=GroupConfig(agent="myagent"),
            scenario_name="scn",
        )

        _write_runtime_info_sidecar(outcome, agent, config)

        sidecar = json.loads((outcome / RUNTIME_INFO_FILE).read_text())
        assert sidecar["group"] == "fallback_group"


class TestMultiTurnTemplating:
    """End-to-end check that ``Turn.message`` placeholders are rendered before
    they reach ``agent.execute`` and the per-turn NDJSON stream.
    Unit-level coverage of the renderer lives in ``test_render_turn_message.py``.
    """

    def test_prev_reply_text_rendered_into_next_turn(self, tmp_path: Path):
        agent = StubAgentAdapter(replies=["world", "echoed"])
        scenario = Scenario(
            name="templated",
            description="Turn 1 echoes turn 0's reply",
            turns=[
                Turn(message="say world", expect=TurnExpectation()),
                Turn(message="Echo: {{prev.reply_text}}", expect=TurnExpectation()),
            ],
        )
        outcome = tmp_path / "outcome"
        config = build_agent_config(GroupConfig(agent="test"), scenario, None)

        result = run_scenario_turns(agent, scenario, outcome, config)

        assert result.turns_completed == 2
        execute_calls = [c for c in agent.calls if c["method"] == "execute"]
        assert execute_calls[0]["message"] == "say world"
        # Turn 1's message contains turn 0's reply (``world``), not the literal
        # placeholder.
        assert execute_calls[1]["message"] == "Echo: world"

        # Stream's user_input event reflects the rendered message too.
        stream_lines = (outcome / "turn_1_stream.ndjson").read_text().splitlines()
        user_event = json.loads(stream_lines[0])
        assert user_event == {"type": "user_input", "message": "Echo: world"}

    def test_no_placeholders_passthrough(self, tmp_path: Path):
        agent = StubAgentAdapter(replies=["a", "b"])
        scenario = _make_scenario(turns=2)
        outcome = tmp_path / "outcome"
        config = build_agent_config(GroupConfig(agent="test"), scenario, None)

        run_scenario_turns(agent, scenario, outcome, config)

        execute_calls = [c for c in agent.calls if c["method"] == "execute"]
        assert execute_calls[0]["message"] == "Turn 0 message"
        assert execute_calls[1]["message"] == "Turn 1 message"

    def test_bad_template_writes_sentinel_and_breaks(self, tmp_path: Path):
        agent = StubAgentAdapter(replies=["a"])
        scenario = Scenario(
            name="bad_template",
            description="Future-turn reference",
            turns=[
                Turn(message="say a", expect=TurnExpectation()),
                Turn(message="{{turn_5.reply_text}}", expect=TurnExpectation()),
            ],
        )
        outcome = tmp_path / "outcome"
        config = build_agent_config(GroupConfig(agent="test"), scenario, None)

        result = run_scenario_turns(agent, scenario, outcome, config)

        # Turn 0 ran fine, turn 1 aborted on the rendering error.
        assert result.turns_completed == 1
        sentinel = (outcome / "turn_1_cli.txt").read_text()
        assert "turn 5" in sentinel
        # Agent.execute was never called for turn 1.
        execute_messages = [c["message"] for c in agent.calls if c["method"] == "execute"]
        assert len(execute_messages) == 1


class TestBuildSandboxForScenario:
    """Provider-capability validation runs before any subprocess work.

    The orchestrator helper instantiates the provider, calls
    ``validate_profile`` (which raises :class:`SandboxConfigError` when the
    chosen provider cannot enforce the profile), and only then either
    short-circuits to the local spawner (host) or runs ``setup`` (sandbox).
    These tests pin that order so a future refactor cannot reintroduce the
    silent-downgrade footgun this module exists to prevent.
    """

    def test_host_default_returns_local_spawner_no_provider(self, tmp_path: Path) -> None:
        from belt.runner.process.spawner import LocalSpawner
        from belt.scenario import SandboxProfile

        agent = StubAgentAdapter()
        spawner, provider, handle = _build_sandbox_for_scenario(
            SandboxProfile(),
            agent,
            tmp_path,
            "scn",
        )
        assert isinstance(spawner, LocalSpawner)
        assert provider is None
        assert handle is None

    def test_host_with_network_policy_none_aborts_scenario(self, tmp_path: Path) -> None:
        # The footgun: provider=host plus network_policy=none must NOT
        # silently run with the host's open network. The orchestrator
        # surfaces SandboxConfigError before any spawner is built.
        import pytest

        from belt.runner.sandbox.base import SandboxConfigError
        from belt.scenario import SandboxProfile

        agent = StubAgentAdapter()
        profile = SandboxProfile(provider="host", network_policy="none")
        with pytest.raises(SandboxConfigError) as exc:
            _build_sandbox_for_scenario(profile, agent, tmp_path, "footgun_scn")
        msg = str(exc.value)
        assert "footgun_scn" in msg
        assert "--sandbox docker" in msg

    def test_docker_with_network_policy_none_passes_validation(self, tmp_path: Path, monkeypatch) -> None:
        # Symmetric positive case: the same network_policy=none profile is
        # accepted under provider=docker because docker can enforce it via
        # --network=none. ``setup`` is patched to skip the daemon check so
        # the test does not depend on docker being installed on the host.
        from belt.runner.sandbox.base import SandboxHandle
        from belt.scenario import SandboxProfile

        agent = StubAgentAdapter()
        profile = SandboxProfile(provider="docker", image="img:tag", network_policy="none")

        def _fake_setup(self, profile, ctx):
            return SandboxHandle(profile=profile, context=ctx, state={})

        monkeypatch.setattr(
            "belt.runner.sandbox.docker.DockerSandboxProvider.setup",
            _fake_setup,
        )
        spawner, provider, handle = _build_sandbox_for_scenario(profile, agent, tmp_path, "scn")
        from belt.runner.process.spawner import SandboxedSpawner
        from belt.runner.sandbox.docker import DockerSandboxProvider

        assert isinstance(spawner, SandboxedSpawner)
        assert isinstance(provider, DockerSandboxProvider)
        assert handle is not None
        assert handle.profile.network_policy == "none"

    def test_docker_with_missing_image_aborts_before_setup(self, tmp_path: Path) -> None:
        # Profile-coherence failure must surface before the daemon-availability
        # check so it works on hosts without docker installed. Pin that
        # validate_profile fires first by leaving _docker_available alone --
        # if validate_profile ran AFTER setup() (or not at all), this would
        # raise DockerSandboxError ("docker binary not on PATH") on a host
        # without docker, masking the real config bug.
        import pytest

        from belt.runner.sandbox.base import SandboxConfigError
        from belt.scenario import SandboxProfile

        agent = StubAgentAdapter()
        profile = SandboxProfile(provider="docker", image=None)
        with pytest.raises(SandboxConfigError) as exc:
            _build_sandbox_for_scenario(profile, agent, tmp_path, "scn")
        assert "image" in str(exc.value)
