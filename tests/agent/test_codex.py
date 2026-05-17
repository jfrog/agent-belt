# (c) JFrog Ltd. (2026)

"""Tests for ``CodexAgentAdapter`` against ``codex-cli 0.130+`` exec surface.

Covers the rewritten contract: ``codex exec --json`` for the first turn,
``codex exec resume <session_id>`` for follow-ups, JSONL event grammar
(``thread.started``, ``turn.started``, ``item.{started,completed}``,
``turn.{completed,failed}``, ``error``), and reply-text sourcing from the
``--output-last-message`` file. Fixtures are committed under
``tests/agent/fixtures/codex/`` and exercised through a fake spawner so
no real codex binary is invoked.
"""

from __future__ import annotations

import subprocess
from io import StringIO
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from belt.agent.base import AgentNotAvailableError
from belt.agent.codex import _MIN_VERSION, CodexAgentAdapter, _parse_version
from belt.agent.error_types import RATE_LIMITED, UNKNOWN
from belt.runner.entities import AgentConfig
from belt.scenario import GroupConfig

_FIXTURES = Path(__file__).parent / "fixtures" / "codex"


def _read_fixture(name: str) -> str:
    return (_FIXTURES / name).read_text(encoding="utf-8")


class _FakeSpawner:
    """Records the last ``popen`` call and replays a canned JSONL stream.

    When ``reply_text`` is provided, the spawner writes it to the
    ``-o``/``--output-last-message`` path declared on the captured
    command, mimicking the codex CLI's behaviour so the adapter's
    file-based reply path can be exercised.
    """

    def __init__(self, stdout: str = "", stderr: str = "", returncode: int = 0, reply_text: str | None = "") -> None:
        self.stdout = stdout
        self.stderr = stderr
        self.returncode = returncode
        self.reply_text = reply_text
        self.last_cmd: list[str] | None = None
        self.last_kwargs: dict | None = None

    def popen(self, cmd, **kwargs):
        self.last_cmd = list(cmd)
        self.last_kwargs = dict(kwargs)
        if self.reply_text is not None:
            reply_file = self._extract_reply_file(cmd)
            if reply_file:
                Path(reply_file).write_text(self.reply_text, encoding="utf-8")
        proc = MagicMock()
        proc.stdout = StringIO(self.stdout)
        proc.stderr = StringIO(self.stderr)
        proc.returncode = self.returncode
        proc.wait = MagicMock(return_value=self.returncode)
        proc.pid = 99999
        return proc

    @staticmethod
    def _extract_reply_file(cmd: list[str]) -> str | None:
        for marker in ("-o", "--output-last-message"):
            if marker in cmd:
                idx = cmd.index(marker)
                if idx + 1 < len(cmd):
                    return cmd[idx + 1]
        return None


@pytest.fixture
def agent() -> CodexAgentAdapter:
    return CodexAgentAdapter()


@pytest.fixture
def config(tmp_path: Path) -> AgentConfig:
    return AgentConfig(
        group_config=GroupConfig(agent="codex"),
        scenario_name="t",
        workspace_dir=str(tmp_path),
    )


# ── _parse_version helper ──


class TestParseVersion:
    def test_parses_canonical_banner(self):
        assert _parse_version("codex-cli 0.130.0\n") == (0, 130, 0)

    def test_parses_higher_minor(self):
        assert _parse_version("codex-cli 0.140.7") == (0, 140, 7)

    def test_returns_none_for_garbage(self):
        assert _parse_version("garbage banner") is None
        assert _parse_version("") is None

    def test_min_version_is_0_130_0(self):
        assert _MIN_VERSION == (0, 130, 0)


# ── check_available ──


class TestCheckAvailable:
    def test_missing_binary(self):
        with patch("belt.agent.codex.resolve_binary", return_value=None):
            with pytest.raises(AgentNotAvailableError, match="codex CLI not found"):
                CodexAgentAdapter.check_available()

    def test_version_below_minimum_rejected(self):
        with patch("belt.agent.codex.resolve_binary", return_value="/usr/local/bin/codex"):
            with patch(
                "subprocess.run",
                return_value=subprocess.CompletedProcess(
                    args=[], returncode=0, stdout="codex-cli 0.129.5\n", stderr=""
                ),
            ):
                with pytest.raises(AgentNotAvailableError, match="0.129.5 is older than the required 0.130.0"):
                    CodexAgentAdapter.check_available()

    def test_minimum_version_accepted(self):
        with patch("belt.agent.codex.resolve_binary", return_value="/usr/local/bin/codex"):
            with patch(
                "subprocess.run",
                return_value=subprocess.CompletedProcess(
                    args=[], returncode=0, stdout="codex-cli 0.130.0\n", stderr=""
                ),
            ):
                CodexAgentAdapter.check_available()

    def test_higher_version_accepted(self):
        with patch("belt.agent.codex.resolve_binary", return_value="/usr/local/bin/codex"):
            with patch(
                "subprocess.run",
                return_value=subprocess.CompletedProcess(
                    args=[], returncode=0, stdout="codex-cli 0.142.3\n", stderr=""
                ),
            ):
                CodexAgentAdapter.check_available()

    def test_unparseable_version_rejected(self):
        with patch("belt.agent.codex.resolve_binary", return_value="/usr/local/bin/codex"):
            with patch(
                "subprocess.run",
                return_value=subprocess.CompletedProcess(args=[], returncode=0, stdout="who knows\n", stderr=""),
            ):
                with pytest.raises(AgentNotAvailableError, match="unparseable"):
                    CodexAgentAdapter.check_available()

    def test_subprocess_timeout_rejected(self):
        with patch("belt.agent.codex.resolve_binary", return_value="/usr/local/bin/codex"):
            with patch("subprocess.run", side_effect=subprocess.TimeoutExpired(cmd="codex", timeout=10)):
                with pytest.raises(AgentNotAvailableError, match="codex --version failed"):
                    CodexAgentAdapter.check_available()


# ── Auth signals ──


class TestAuthSignals:
    def test_credential_env_includes_openai_and_azure(self):
        assert "OPENAI_API_KEY" in CodexAgentAdapter.CREDENTIAL_ENV
        assert "AZURE_OPENAI_API_KEY" in CodexAgentAdapter.CREDENTIAL_ENV

    def test_credential_path_points_at_codex_auth(self):
        paths = [str(p) for p in CodexAgentAdapter.CREDENTIAL_PATHS]
        assert any(p.endswith(".codex/auth.json") for p in paths)

    def test_env_signal_detected(self, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.setenv("AZURE_OPENAI_API_KEY", "azure-key")
        signals = CodexAgentAdapter.auth_signals()
        assert any("AZURE_OPENAI_API_KEY" in s for s in signals)


# ── required_env_vars / cli_options ──


class TestEnvAndOptions:
    def test_required_env_vars_includes_azure_set(self):
        names = CodexAgentAdapter.required_env_vars()
        for var in (
            "AZURE_OPENAI_API_KEY",
            "AZURE_OPENAI_ENDPOINT",
            "AZURE_OPENAI_BASE_URL",
            "OPENAI_BASE_URL",
            "CODEX_HOME",
        ):
            assert var in names, f"{var} missing from required_env_vars"

    def test_required_env_vars_keeps_default_provider_set(self):
        names = CodexAgentAdapter.required_env_vars()
        assert "OPENAI_API_KEY" in names

    def test_cli_options_declares_model_and_profile(self):
        options = {opt.name: opt for opt in CodexAgentAdapter.cli_options()}
        assert set(options.keys()) == {"model", "profile"}
        assert options["model"].env_var == "CODEX_DEFAULT_MODEL"
        assert options["profile"].env_var == "CODEX_DEFAULT_PROFILE"


# ── execute (first turn) ──


class TestExecuteFirstTurn:
    def test_first_turn_argv_shape(self, agent: CodexAgentAdapter, config: AgentConfig):
        agent.setup(config)
        spawner = _FakeSpawner(stdout=_read_fixture("single_turn.jsonl"), reply_text="pong\n")
        agent._spawner = spawner

        agent.execute("Reply with pong", [])

        cmd = spawner.last_cmd
        assert cmd is not None
        assert cmd[:3] == ["codex", "exec", "--json"]
        assert "--skip-git-repo-check" in cmd
        assert "-o" in cmd
        assert "-C" in cmd and config.workspace_dir in cmd
        # Last two argv entries are always the ``--`` separator and the message itself.
        assert cmd[-2:] == ["--", "Reply with pong"]

    def test_first_turn_includes_model_and_profile(self, agent: CodexAgentAdapter, config: AgentConfig):
        agent = CodexAgentAdapter(model="gpt-5-codex", profile="azure")
        agent.setup(config)
        spawner = _FakeSpawner(stdout=_read_fixture("single_turn.jsonl"), reply_text="pong\n")
        agent._spawner = spawner

        agent.execute("ping", [])
        cmd = spawner.last_cmd
        assert cmd is not None
        assert "-m" in cmd and cmd[cmd.index("-m") + 1] == "gpt-5-codex"
        assert "-p" in cmd and cmd[cmd.index("-p") + 1] == "azure"

    def test_first_turn_does_not_include_resume(self, agent: CodexAgentAdapter, config: AgentConfig):
        agent.setup(config)
        spawner = _FakeSpawner(stdout=_read_fixture("single_turn.jsonl"), reply_text="pong\n")
        agent._spawner = spawner

        agent.execute("ping", [])
        assert spawner.last_cmd is not None
        assert "resume" not in spawner.last_cmd

    def test_first_turn_defaults_to_workspace_write_sandbox(self, agent: CodexAgentAdapter, config: AgentConfig):
        # codex's own default is ``read-only``, which silently blocks every
        # file write a scenario asks the agent to perform and surfaces as a
        # generic refusal in the reply. Belt's per-scenario git worktree
        # already provides the next outer isolation tier, so the adapter
        # promotes the default to ``workspace-write``.
        agent.setup(config)
        spawner = _FakeSpawner(stdout=_read_fixture("single_turn.jsonl"), reply_text="pong\n")
        agent._spawner = spawner

        agent.execute("ping", [])
        cmd = spawner.last_cmd or []
        assert "-s" in cmd, f"expected -s default in {cmd!r}"
        assert cmd[cmd.index("-s") + 1] == "workspace-write"

    def test_scenario_can_override_default_sandbox(self, agent: CodexAgentAdapter, config: AgentConfig):
        agent.setup(config)
        spawner = _FakeSpawner(stdout=_read_fixture("single_turn.jsonl"), reply_text="pong\n")
        agent._spawner = spawner

        agent.execute("ping", ["--sandbox", "read-only"])
        cmd = spawner.last_cmd or []
        # The default must NOT be appended on top of the explicit override.
        assert cmd.count("--sandbox") + cmd.count("-s") == 1, f"got {cmd!r}"
        assert "read-only" in cmd

    def test_resume_does_not_inject_sandbox_default(self, agent: CodexAgentAdapter, config: AgentConfig):
        # ``codex exec resume`` inherits the sandbox of the original
        # session; reasserting it is unnecessary and would conflict with
        # any session-scoped configuration the user set up via TOML.
        agent.setup(config)
        agent._session_id = "00000000-0000-0000-0000-000000000001"
        spawner = _FakeSpawner(stdout=_read_fixture("resume_state.jsonl"), reply_text="ok\n")
        agent._spawner = spawner

        agent.execute("recall", [])
        cmd = spawner.last_cmd or []
        assert "-s" not in cmd
        assert "--sandbox" not in cmd

    def test_extra_flags_passthrough_after_filter(self, agent: CodexAgentAdapter, config: AgentConfig):
        agent.setup(config)
        spawner = _FakeSpawner(stdout=_read_fixture("single_turn.jsonl"), reply_text="pong\n")
        agent._spawner = spawner

        agent.execute("ping", ["--ignore-rules"])
        assert "--ignore-rules" in (spawner.last_cmd or [])

    def test_denied_flag_stripped(self, agent: CodexAgentAdapter, config: AgentConfig):
        agent.setup(config)
        spawner = _FakeSpawner(stdout=_read_fixture("single_turn.jsonl"), reply_text="pong\n")
        agent._spawner = spawner

        agent.execute("ping", ["--dangerously-bypass-approvals-and-sandbox"])
        assert "--dangerously-bypass-approvals-and-sandbox" not in (spawner.last_cmd or [])


# ── execute (resume turn) ──


class TestExecuteResume:
    def test_resume_argv_shape(self, agent: CodexAgentAdapter, config: AgentConfig):
        agent = CodexAgentAdapter(model="gpt-5-codex", profile="azure")
        agent.setup(config)
        agent._session_id = "00000000-0000-0000-0000-000000000001"

        spawner = _FakeSpawner(stdout=_read_fixture("resume_state.jsonl"), reply_text="47\n")
        agent._spawner = spawner

        agent.execute("recall", [])
        cmd = spawner.last_cmd
        assert cmd is not None
        assert cmd[:4] == ["codex", "exec", "resume", "00000000-0000-0000-0000-000000000001"]
        # Resume must NOT include ``-p`` (codex rejects it) or ``--skip-git-repo-check``.
        assert "-p" not in cmd
        assert "--skip-git-repo-check" not in cmd
        # Profile is replayed via -c model_provider="<profile>".
        assert any(arg.startswith("model_provider=") for arg in cmd)
        # Workspace is whitelisted via projects.<cwd>.trust_level="trusted".
        assert any('trust_level="trusted"' in arg for arg in cmd)
        # Model is replayed explicitly so resume does not fall back to the global default.
        assert "-m" in cmd and cmd[cmd.index("-m") + 1] == "gpt-5-codex"

    def test_session_id_captured_from_thread_started(self, agent: CodexAgentAdapter, config: AgentConfig):
        agent.setup(config)
        spawner = _FakeSpawner(stdout=_read_fixture("single_turn.jsonl"), reply_text="pong\n")
        agent._spawner = spawner

        agent.execute("ping", [])
        agent.fetch_results(spawner.stdout)  # exercise the parse path

        assert agent._session_id == "00000000-0000-0000-0000-000000000001"


# ── fetch_results: event grammar mapping ──


class TestFetchResultsSingleTurn:
    def test_single_turn_reply_from_output_file(self, agent: CodexAgentAdapter, config: AgentConfig):
        agent.setup(config)
        spawner = _FakeSpawner(stdout=_read_fixture("single_turn.jsonl"), reply_text="pong\n")
        agent._spawner = spawner

        raw = agent.execute("ping", [])
        out = agent.fetch_results(raw)
        assert out.reply_text == "pong"
        assert out.has_reply is True
        assert out.has_error is False
        assert out.error_type is None
        assert out.llm_turn_count == 1

    def test_timing_populated(self, agent: CodexAgentAdapter, config: AgentConfig):
        agent.setup(config)
        spawner = _FakeSpawner(stdout=_read_fixture("single_turn.jsonl"), reply_text="pong\n")
        agent._spawner = spawner

        agent.execute("ping", [])
        out = agent.fetch_results(spawner.stdout)
        assert out.timing is not None
        assert out.timing.ttfe is not None
        assert out.timing.ttft is not None
        assert out.timing.ttlt is not None

    def test_empty_output_produces_no_reply(self, agent: CodexAgentAdapter):
        out = agent.fetch_results("")
        assert out.reply_text == ""
        assert out.has_reply is False
        assert out.has_error is False

    def test_raw_cli_preserved(self, agent: CodexAgentAdapter):
        raw = "some raw event stream"
        out = agent.fetch_results(raw)
        assert out.raw_cli == raw


class TestFetchResultsToolUse:
    def test_command_execution_extracted_to_tool_call(self, agent: CodexAgentAdapter, config: AgentConfig):
        agent.setup(config)
        spawner = _FakeSpawner(stdout=_read_fixture("tool_use.jsonl"), reply_text="done")
        agent._spawner = spawner

        agent.execute("run echo", [])
        out = agent.fetch_results(spawner.stdout)

        assert len(out.tool_calls) == 1
        tc = out.tool_calls[0]
        assert tc.name == "shell"
        assert tc.call_id == "item_1"
        assert tc.args["command"] == "/bin/zsh -lc 'echo hello'"
        assert tc.args["exit_code"] == 0
        assert tc.args["status"] == "completed"
        assert tc.args["aggregated_output"] == "hello\n"
        assert out.tool_sequence == ["shell"]

    def test_multiple_agent_message_items_collapsed(self, agent: CodexAgentAdapter, config: AgentConfig):
        # Tool-use turn emits both pre-tool narration and post-tool conclusion
        # as separate ``agent_message`` items. The reply file contains only
        # the final one, but ``llm_turn_count`` should reflect both.
        agent.setup(config)
        spawner = _FakeSpawner(stdout=_read_fixture("tool_use.jsonl"), reply_text="done")
        agent._spawner = spawner

        agent.execute("run echo", [])
        out = agent.fetch_results(spawner.stdout)
        assert out.reply_text == "done"
        assert out.llm_turn_count == 2

    def test_fallback_to_event_stream_when_reply_file_empty(self, agent: CodexAgentAdapter, config: AgentConfig):
        agent.setup(config)
        spawner = _FakeSpawner(stdout=_read_fixture("tool_use.jsonl"), reply_text="")
        agent._spawner = spawner

        agent.execute("run echo", [])
        out = agent.fetch_results(spawner.stdout)
        # Empty reply file -> concatenated agent_message text from the stream.
        assert "I will run echo hello" in out.reply_text
        assert "done" in out.reply_text


class TestFetchResultsErrors:
    def test_turn_failed_sets_has_error(self, agent: CodexAgentAdapter, config: AgentConfig):
        agent.setup(config)
        spawner = _FakeSpawner(stdout=_read_fixture("retry_error.jsonl"), reply_text="")
        agent._spawner = spawner

        agent.execute("ping", [])
        out = agent.fetch_results(spawner.stdout)
        assert out.has_error is True
        assert out.error_type == RATE_LIMITED
        assert out.has_reply is False

    def test_turn_failed_with_unrecognised_message_falls_back_to_unknown(self, agent: CodexAgentAdapter):
        raw = (
            '{"type":"thread.started","thread_id":"00000000-0000-0000-0000-000000000004"}\n'
            '{"type":"turn.failed","error":{"message":"surprise"}}\n'
        )
        out = agent.fetch_results(raw)
        assert out.has_error is True
        assert out.error_type == UNKNOWN

    def test_transient_error_events_alone_do_not_flip_has_error(self, agent: CodexAgentAdapter):
        # Reconnects without a terminal ``turn.failed`` are not authoritative;
        # the adapter must wait for a terminal event before declaring failure.
        raw = (
            '{"type":"thread.started","thread_id":"00000000-0000-0000-0000-000000000005"}\n'
            '{"type":"turn.started"}\n'
            '{"type":"error","message":"Reconnecting... 1/5 (transient)"}\n'
            '{"type":"item.completed","item":{"id":"item_0","type":"agent_message","text":"ok"}}\n'
            '{"type":"turn.completed","usage":{}}\n'
        )
        agent._last_message = "ok"
        out = agent.fetch_results(raw)
        assert out.has_error is False
        assert out.reply_text == "ok"


# ── Lifecycle ──


class TestLifecycle:
    def test_setup_resets_state(self, agent: CodexAgentAdapter, config: AgentConfig):
        agent._session_id = "old-session"
        agent._ttfe = 1.0
        agent._ttft = 2.0
        agent._ttlt = 3.0
        agent._last_message = "old"

        agent.setup(config)

        assert agent._session_id is None
        assert agent._ttfe is None
        assert agent._ttft is None
        assert agent._ttlt is None
        assert agent._last_message == ""

    def test_teardown_resets_session(self, agent: CodexAgentAdapter):
        agent._session_id = "active"
        agent._last_message = "anything"
        agent.teardown()
        assert agent._session_id is None
        assert agent._last_message == ""

    def test_metadata_with_session_id(self, agent: CodexAgentAdapter):
        agent._session_id = "00000000-0000-0000-0000-000000000001"
        assert agent.metadata() == {"session_id": "00000000-0000-0000-0000-000000000001"}

    def test_metadata_without_session_id(self, agent: CodexAgentAdapter):
        assert agent.metadata() is None

    def test_setup_group_returns_none(self, agent: CodexAgentAdapter, tmp_path: Path):
        gc = GroupConfig(agent="codex")
        assert agent.setup_group(gc, tmp_path) is None


# ── make_subprocess_env ──


class TestSubprocessEnv:
    def test_passes_azure_keys_through_scrubbed_env(self, agent: CodexAgentAdapter, monkeypatch):
        monkeypatch.setenv("AZURE_OPENAI_API_KEY", "azure-secret")
        monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://example.openai.azure.com/")
        monkeypatch.setenv("CODEX_HOME", "/tmp/codex-home")
        env = agent.make_subprocess_env()
        assert env.get("AZURE_OPENAI_API_KEY") == "azure-secret"
        assert env.get("AZURE_OPENAI_ENDPOINT") == "https://example.openai.azure.com/"
        assert env.get("CODEX_HOME") == "/tmp/codex-home"

    def test_strips_unrelated_secrets(self, agent: CodexAgentAdapter, monkeypatch):
        monkeypatch.setenv("UNRELATED_SECRET", "dont-leak")
        monkeypatch.delenv("BELT_ALLOW_FULL_ENV", raising=False)
        env = agent.make_subprocess_env()
        assert "UNRELATED_SECRET" not in env


# ── Display / runtime info ──


class TestDisplayInfo:
    def test_display_info_with_version(self):
        with patch("belt.agent.codex.resolve_binary", return_value="/usr/local/bin/codex"):
            with patch(
                "subprocess.run",
                return_value=subprocess.CompletedProcess(
                    args=[], returncode=0, stdout="codex-cli 0.130.0\n", stderr=""
                ),
            ):
                info = CodexAgentAdapter.display_info()
                assert "codex-cli 0.130.0" in info

    def test_display_info_when_missing(self):
        with patch("belt.agent.codex.resolve_binary", return_value=None):
            assert "not found" in CodexAgentAdapter.display_info()

    def test_runtime_info_populates_cli_fields(self):
        with patch("belt.agent.codex.resolve_binary", return_value="/usr/local/bin/codex"):
            with patch.object(CodexAgentAdapter, "_capture_cli_version", return_value="codex-cli 0.130.0"):
                info = CodexAgentAdapter.runtime_info()
                assert info["cli_binary_path"] == "/usr/local/bin/codex"
                assert info["cli_version"] == "codex-cli 0.130.0"
                assert "node_version" not in info
