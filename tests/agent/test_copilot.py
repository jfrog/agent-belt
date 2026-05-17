# (c) JFrog Ltd. (2026)

"""Tests for CopilotAgentAdapter - check_available, JSONL parsing,
Popen streaming execution with env, error events, auth detection,
timing metrics, cost/tool_sequence/thinking extraction, full lifecycle."""

from __future__ import annotations

import json
from io import StringIO
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from belt.agent.base import AgentNotAvailableError
from belt.agent.copilot import CopilotAgentAdapter
from belt.entities import AgentConfig, GroupConfig
from belt.parser.ndjson import parse_ndjson


@pytest.fixture
def agent() -> CopilotAgentAdapter:
    return CopilotAgentAdapter()


@pytest.fixture
def config() -> AgentConfig:
    return AgentConfig(
        group_config=GroupConfig(agent="copilot"),
        scenario_name="test",
    )


def _mock_popen(stdout_text: str, returncode: int = 0, stderr_text: str = "") -> MagicMock:
    """Create a mock Popen object with given stdout/stderr."""
    proc = MagicMock()
    proc.stdout = StringIO(stdout_text)
    proc.stderr = StringIO(stderr_text)
    proc.returncode = returncode
    proc.wait = MagicMock(return_value=returncode)
    proc.pid = 99999
    return proc


# ── check_available ──


class TestCheckAvailable:
    def test_not_available_when_missing(self):
        with patch("shutil.which", return_value=None):
            with pytest.raises(AgentNotAvailableError, match="copilot CLI not found"):
                CopilotAgentAdapter.check_available()

    def test_available_when_present(self):
        with patch("shutil.which", return_value="/usr/local/bin/copilot"):
            CopilotAgentAdapter.check_available()


class TestAuthSignals:
    def test_declares_copilot_github_token(self):
        assert "COPILOT_GITHUB_TOKEN" in CopilotAgentAdapter.CREDENTIAL_ENV
        assert "GH_TOKEN" in CopilotAgentAdapter.CREDENTIAL_ENV
        assert "GITHUB_TOKEN" in CopilotAgentAdapter.CREDENTIAL_ENV

    def test_declares_copilot_home_path(self):
        # Per Copilot CLI docs - login state is stored under ~/.copilot/.
        paths = [str(p) for p in CopilotAgentAdapter.CREDENTIAL_PATHS]
        assert any(p.endswith(".copilot") for p in paths)

    def test_env_signal_detected(self, monkeypatch):
        monkeypatch.setenv("COPILOT_GITHUB_TOKEN", "ghu_test")
        signals = CopilotAgentAdapter.auth_signals()
        assert "env COPILOT_GITHUB_TOKEN" in signals

    def test_gh_token_signal_detected(self, monkeypatch):
        monkeypatch.delenv("COPILOT_GITHUB_TOKEN", raising=False)
        monkeypatch.setenv("GH_TOKEN", "ghp_test")
        signals = CopilotAgentAdapter.auth_signals()
        assert "env GH_TOKEN" in signals


# ── execute (Popen streaming) ──


class TestExecute:
    def test_successful_execution(self, agent: CopilotAgentAdapter):
        events = json.dumps({"type": "message", "role": "assistant", "content": "2"})
        proc = _mock_popen(events + "\n")
        with patch("subprocess.Popen", return_value=proc) as mock_popen:
            raw = agent.execute("What is 1+1?", [])
            args = mock_popen.call_args[0][0]
            assert args[0] == "copilot"
            assert "--output-format" in args
            assert args[args.index("--output-format") + 1] == "json"
            assert "--allow-all-tools" in args
            assert "-p" in args
            assert args[args.index("-p") + 1] == "What is 1+1?"
            assert events in raw

    def test_passes_session_id_on_second_turn(self, agent: CopilotAgentAdapter):
        agent._session_id = "sess-123"
        proc = _mock_popen("{}\n")
        with patch("subprocess.Popen", return_value=proc) as mock_popen:
            agent.execute("follow up", [])
            args = mock_popen.call_args[0][0]
            assert "--resume" in args
            idx = args.index("--resume")
            assert args[idx + 1] == "sess-123"

    def test_no_resume_on_first_turn(self, agent: CopilotAgentAdapter):
        agent._session_id = None
        proc = _mock_popen("{}\n")
        with patch("subprocess.Popen", return_value=proc) as mock_popen:
            agent.execute("hello", [])
            args = mock_popen.call_args[0][0]
            assert "--resume" not in args

    def test_passes_model_when_set(self):
        a = CopilotAgentAdapter(model="claude-sonnet-4.5")
        proc = _mock_popen("{}\n")
        with patch("subprocess.Popen", return_value=proc) as mock_popen:
            a.execute("hi", [])
            args = mock_popen.call_args[0][0]
            assert "--model" in args
            idx = args.index("--model")
            assert args[idx + 1] == "claude-sonnet-4.5"

    def test_no_model_when_unset(self, agent: CopilotAgentAdapter):
        proc = _mock_popen("{}\n")
        with patch("subprocess.Popen", return_value=proc) as mock_popen:
            agent.execute("hi", [])
            args = mock_popen.call_args[0][0]
            assert "--model" not in args

    def test_nonzero_exit_appends_stderr(self, agent: CopilotAgentAdapter):
        proc = _mock_popen("", returncode=1, stderr_text="auth failed")
        with patch("subprocess.Popen", return_value=proc):
            raw = agent.execute("fail", [])
            assert "auth failed" in raw

    def test_passes_extra_flags(self, agent: CopilotAgentAdapter):
        proc = _mock_popen("ok\n")
        with patch("subprocess.Popen", return_value=proc) as mock_popen:
            agent.execute("test", ["--add-dir", "/tmp/foo"])
            args = mock_popen.call_args[0][0]
            assert "--add-dir" in args

    def test_message_argv_safety_with_dash_prefix(self, agent: CopilotAgentAdapter):
        """A message starting with `--` must not be reparsed as a flag.

        Shape B: -p is the prior argv element so `--evil` is its value.
        """
        proc = _mock_popen("{}\n")
        with patch("subprocess.Popen", return_value=proc) as mock_popen:
            agent.execute("--evil", [])
            args = mock_popen.call_args[0][0]
            assert "--evil" in args
            idx = args.index("--evil")
            assert args[idx - 1] == "-p"

    def test_subprocess_env_includes_copilot_token(self, agent: CopilotAgentAdapter, monkeypatch):
        monkeypatch.setenv("COPILOT_GITHUB_TOKEN", "ghu_test_token")
        proc = _mock_popen("{}\n")
        with patch("subprocess.Popen", return_value=proc) as mock_popen:
            agent.execute("hi", [])
            env = mock_popen.call_args[1].get("env", {})
            assert env.get("COPILOT_GITHUB_TOKEN") == "ghu_test_token"
            assert env.get("COPILOT_ALLOW_ALL") == "true"

    def test_workspace_dir_propagated_to_cwd(self):
        a = CopilotAgentAdapter()
        a.setup(
            AgentConfig(
                group_config=GroupConfig(agent="copilot"),
                scenario_name="test",
                workspace_dir="/tmp/copilot-ws-test",
            )
        )
        proc = _mock_popen("{}\n")
        with patch("subprocess.Popen", return_value=proc) as mock_popen:
            a.execute("hi", [])
            cwd = mock_popen.call_args[1].get("cwd")
            assert cwd == "/tmp/copilot-ws-test"

    def test_timing_metrics_captured(self, agent: CopilotAgentAdapter):
        lines = (
            json.dumps({"type": "message", "role": "assistant", "content": "hello"})
            + "\n"
            + json.dumps({"type": "result", "session_id": "s1"})
            + "\n"
        )
        proc = _mock_popen(lines)
        with patch("subprocess.Popen", return_value=proc):
            agent.execute("test", [])
            assert agent._ttfe is not None
            assert agent._ttlt is not None


# ── fetch_results - real Copilot schema (verified against copilot 1.0.40) ──


class TestFetchResultsRealSchema:
    """Tests against Copilot's namespaced event schema as observed live."""

    def test_simple_assistant_message(self, agent: CopilotAgentAdapter):
        events = json.dumps(
            {
                "type": "assistant.message",
                "data": {"messageId": "m1", "content": "Four", "toolRequests": [], "turnId": "0"},
            }
        )
        to = agent.fetch_results(events)
        assert to.reply_text == "Four"
        assert to.has_reply is True

    def test_turn_start_increments_turn_count(self, agent: CopilotAgentAdapter):
        events = "\n".join(
            [
                json.dumps({"type": "assistant.turn_start", "data": {}}),
                json.dumps({"type": "assistant.message", "data": {"content": "first"}}),
                json.dumps({"type": "assistant.turn_start", "data": {}}),
                json.dumps({"type": "assistant.message", "data": {"content": "second"}}),
            ]
        )
        to = agent.fetch_results(events)
        assert to.llm_turn_count == 2

    def test_tool_requests_extracted_from_message(self, agent: CopilotAgentAdapter):
        events = json.dumps(
            {
                "type": "assistant.message",
                "data": {
                    "content": "",
                    "toolRequests": [
                        {
                            "toolCallId": "call_1",
                            "name": "report_intent",
                            "arguments": {"intent": "Running bash command"},
                            "type": "function",
                        },
                        {
                            "toolCallId": "call_2",
                            "name": "bash",
                            "arguments": {"command": "echo hello"},
                            "type": "function",
                        },
                    ],
                },
            }
        )
        to = agent.fetch_results(events)
        assert to.tool_sequence == ["report_intent", "bash"]
        assert len(to.tool_calls) == 2
        assert to.tool_calls[1].name == "bash"
        assert to.tool_calls[1].args == {"command": "echo hello"}
        assert to.tool_calls[1].call_id == "call_2"

    def test_tool_execution_start_dedup_against_message(self, agent: CopilotAgentAdapter):
        """toolRequests + tool.execution_start with same toolCallId must dedup."""
        events = "\n".join(
            [
                json.dumps(
                    {
                        "type": "assistant.message",
                        "data": {
                            "content": "",
                            "toolRequests": [
                                {"toolCallId": "call_x", "name": "bash", "arguments": {"command": "ls"}},
                            ],
                        },
                    }
                ),
                json.dumps(
                    {
                        "type": "tool.execution_start",
                        "data": {"toolCallId": "call_x", "toolName": "bash", "arguments": {"command": "ls"}},
                    }
                ),
            ]
        )
        to = agent.fetch_results(events)
        assert to.tool_sequence == ["bash"]
        assert len(to.tool_calls) == 1

    def test_tool_execution_start_picked_up_when_no_toolrequests(self, agent: CopilotAgentAdapter):
        events = json.dumps(
            {
                "type": "tool.execution_start",
                "data": {"toolCallId": "call_y", "toolName": "view", "arguments": {"path": "README.md"}},
            }
        )
        to = agent.fetch_results(events)
        assert to.tool_sequence == ["view"]
        assert to.tool_calls[0].args == {"path": "README.md"}

    def test_user_message_skipped(self, agent: CopilotAgentAdapter):
        events = "\n".join(
            [
                json.dumps({"type": "user.message", "data": {"content": "What is 2+2?"}}),
                json.dumps({"type": "assistant.message", "data": {"content": "Four"}}),
            ]
        )
        to = agent.fetch_results(events)
        assert to.reply_text == "Four"
        assert "What is 2+2" not in to.reply_text

    def test_session_events_ignored(self, agent: CopilotAgentAdapter):
        events = "\n".join(
            [
                json.dumps({"type": "session.mcp_servers_loaded", "data": {"servers": []}}),
                json.dumps({"type": "session.skills_loaded", "data": {"skills": []}}),
                json.dumps({"type": "session.tools_updated", "data": {"model": "gpt-4.1"}}),
                json.dumps({"type": "assistant.message", "data": {"content": "hello"}}),
            ]
        )
        to = agent.fetch_results(events)
        assert to.reply_text == "hello"
        assert to.tool_sequence == []

    def test_result_event_captures_session_id(self, agent: CopilotAgentAdapter):
        events = "\n".join(
            [
                json.dumps({"type": "assistant.message", "data": {"content": "ok"}}),
                json.dumps(
                    {
                        "type": "result",
                        "sessionId": "sess-uuid-999",
                        "exitCode": 0,
                        "usage": {"totalApiDurationMs": 2147, "sessionDurationMs": 6687, "premiumRequests": 0},
                    }
                ),
            ]
        )
        to = agent.fetch_results(events)
        assert agent._session_id == "sess-uuid-999"
        assert to.timing is not None
        assert to.timing.total == 2.147

    def test_result_exit_code_zero_means_no_error(self, agent: CopilotAgentAdapter):
        events = "\n".join(
            [
                json.dumps({"type": "assistant.message", "data": {"content": "ok"}}),
                json.dumps({"type": "result", "sessionId": "s1", "exitCode": 0}),
            ]
        )
        to = agent.fetch_results(events)
        assert to.has_error is False

    def test_result_nonzero_exit_means_error(self, agent: CopilotAgentAdapter):
        events = "\n".join(
            [
                json.dumps({"type": "assistant.message", "data": {"content": "ok"}}),
                json.dumps({"type": "result", "sessionId": "s1", "exitCode": 1}),
            ]
        )
        to = agent.fetch_results(events)
        assert to.has_error is True

    def test_message_delta_does_not_contribute_to_reply(self, agent: CopilotAgentAdapter):
        """The consolidated assistant.message owns reply_text; deltas would double-count."""
        events = "\n".join(
            [
                json.dumps({"type": "assistant.message_delta", "data": {"deltaContent": "Fo"}}),
                json.dumps({"type": "assistant.message_delta", "data": {"deltaContent": "ur"}}),
                json.dumps({"type": "assistant.message", "data": {"content": "Four"}}),
            ]
        )
        to = agent.fetch_results(events)
        assert to.reply_text == "Four"

    def test_full_real_world_simple_run(self, agent: CopilotAgentAdapter):
        """End-to-end: simulates a full simple Copilot session, no tool calls."""
        events = "\n".join(
            [
                json.dumps({"type": "session.mcp_servers_loaded", "data": {"servers": []}}),
                json.dumps({"type": "user.message", "data": {"content": "What is 2+2?"}}),
                json.dumps({"type": "assistant.turn_start", "data": {}}),
                json.dumps({"type": "assistant.message_start", "data": {"messageId": "m1"}}),
                json.dumps({"type": "assistant.message_delta", "data": {"deltaContent": "Four"}}),
                json.dumps(
                    {
                        "type": "assistant.message",
                        "data": {"messageId": "m1", "content": "Four", "toolRequests": [], "turnId": "0"},
                    }
                ),
                json.dumps({"type": "assistant.turn_end", "data": {"turnId": "0"}}),
                json.dumps(
                    {
                        "type": "result",
                        "sessionId": "sess-1",
                        "exitCode": 0,
                        "usage": {"totalApiDurationMs": 1500, "premiumRequests": 0},
                    }
                ),
            ]
        )
        to = agent.fetch_results(events)
        assert to.reply_text == "Four"
        assert to.has_reply is True
        assert to.has_error is False
        assert to.llm_turn_count == 1
        assert to.tool_sequence == []
        assert agent._session_id == "sess-1"
        assert to.timing is not None
        assert to.timing.total == 1.5

    def test_full_real_world_tool_use(self, agent: CopilotAgentAdapter):
        """End-to-end: simulates a Copilot session with bash tool call."""
        events = "\n".join(
            [
                json.dumps({"type": "user.message", "data": {"content": "echo hi"}}),
                json.dumps({"type": "assistant.turn_start", "data": {}}),
                json.dumps(
                    {
                        "type": "assistant.message",
                        "data": {
                            "content": "",
                            "toolRequests": [
                                {
                                    "toolCallId": "call_a",
                                    "name": "report_intent",
                                    "arguments": {"intent": "Running bash"},
                                    "type": "function",
                                },
                                {
                                    "toolCallId": "call_b",
                                    "name": "bash",
                                    "arguments": {"command": "echo hi"},
                                    "type": "function",
                                },
                            ],
                        },
                    }
                ),
                json.dumps(
                    {
                        "type": "tool.execution_start",
                        "data": {"toolCallId": "call_a", "toolName": "report_intent", "arguments": {}},
                    }
                ),
                json.dumps(
                    {
                        "type": "tool.execution_start",
                        "data": {"toolCallId": "call_b", "toolName": "bash", "arguments": {"command": "echo hi"}},
                    }
                ),
                json.dumps({"type": "assistant.turn_start", "data": {}}),
                json.dumps(
                    {
                        "type": "assistant.message",
                        "data": {"content": "The command printed: hi", "toolRequests": []},
                    }
                ),
                json.dumps(
                    {
                        "type": "result",
                        "sessionId": "sess-tool",
                        "exitCode": 0,
                        "usage": {"totalApiDurationMs": 3463},
                    }
                ),
            ]
        )
        to = agent.fetch_results(events)
        assert to.reply_text == "The command printed: hi"
        assert to.tool_sequence == ["report_intent", "bash"]
        assert len(to.tool_calls) == 2
        assert to.tool_calls[1].args == {"command": "echo hi"}
        assert to.llm_turn_count == 2
        assert agent._session_id == "sess-tool"

    def test_empty_output(self, agent: CopilotAgentAdapter):
        to = agent.fetch_results("")
        assert to.reply_text == ""
        assert to.has_reply is False
        assert to.tool_sequence == []
        assert to.thinking_text is None

    def test_raw_cli_preserved(self, agent: CopilotAgentAdapter):
        to = agent.fetch_results("some output")
        assert to.raw_cli == "some output"


class TestFetchResultsDefensiveLegacy:
    """Defensive paths for claude/codex-style shapes - kept so future schema
    drift in Copilot doesn't silently break the agent."""

    def test_legacy_message_event(self, agent: CopilotAgentAdapter):
        events = json.dumps({"type": "message", "role": "assistant", "content": "Forty-two."})
        to = agent.fetch_results(events)
        assert to.reply_text == "Forty-two."
        assert to.llm_turn_count == 1

    def test_legacy_top_level_tool_use(self, agent: CopilotAgentAdapter):
        events = json.dumps({"type": "tool_use", "name": "bash", "id": "t1", "input": {"cmd": "ls"}})
        to = agent.fetch_results(events)
        assert to.tool_sequence == ["bash"]

    def test_legacy_function_call(self, agent: CopilotAgentAdapter):
        events = json.dumps(
            {
                "type": "function_call",
                "name": "shell",
                "call_id": "call_z",
                "arguments": '{"cmd": "ls"}',
            }
        )
        to = agent.fetch_results(events)
        assert to.tool_sequence == ["shell"]
        assert to.tool_calls[0].args == {"cmd": "ls"}

    def test_legacy_thinking_block(self, agent: CopilotAgentAdapter):
        events = json.dumps(
            {
                "type": "assistant",
                "role": "assistant",
                "content": [
                    {"type": "thinking", "thinking": "Reasoning..."},
                    {"type": "text", "text": "Answer."},
                ],
            }
        )
        to = agent.fetch_results(events)
        assert to.thinking_text == "Reasoning..."
        assert to.reply_text == "Answer."


# ── Lifecycle ──


class TestLifecycle:
    def test_setup_resets_state(self, agent: CopilotAgentAdapter, config: AgentConfig):
        agent._session_id = "old"
        agent._ttfe = 1.0
        agent._ttft = 2.0
        agent._ttlt = 3.0
        agent.setup(config)
        assert agent._session_id is None
        assert agent._ttfe is None
        assert agent._ttft is None
        assert agent._ttlt is None

    def test_setup_captures_workspace_dir(self, agent: CopilotAgentAdapter):
        config = AgentConfig(
            group_config=GroupConfig(agent="copilot"),
            scenario_name="test",
            workspace_dir="/tmp/abc",
        )
        agent.setup(config)
        assert agent._workspace_dir == "/tmp/abc"

    def test_teardown_resets_session_id(self, agent: CopilotAgentAdapter):
        agent._session_id = "active"
        agent.teardown()
        assert agent._session_id is None

    def test_metadata_with_session_id(self, agent: CopilotAgentAdapter):
        agent._session_id = "sess-1"
        assert agent.metadata() == {"session_id": "sess-1"}

    def test_metadata_without(self, agent: CopilotAgentAdapter):
        assert agent.metadata() is None

    def test_group_setup_returns_none(self, agent: CopilotAgentAdapter):
        gc = GroupConfig(agent="copilot")
        assert agent.setup_group(gc, Path("/tmp")) is None


# ── Parsers ──


class TestParsers:
    def test_parse_ndjson_mixed(self):
        raw = 'not json\n{"type": "msg"}\n\n{"type": "result"}\n'
        events = parse_ndjson(raw)
        assert len(events) == 2

    def test_parse_ndjson_empty(self):
        assert parse_ndjson("") == []


# ── Interface methods ──


class TestInterfaceMethods:
    def test_cli_options_has_model(self):
        options = CopilotAgentAdapter.cli_options()
        assert len(options) == 1
        assert options[0].name == "model"
        assert options[0].env_var == "COPILOT_MODEL"

    def test_required_env_vars_includes_copilot_keys(self):
        names = CopilotAgentAdapter.required_env_vars()
        assert "COPILOT_GITHUB_TOKEN" in names
        assert "GH_TOKEN" in names
        assert "GITHUB_TOKEN" in names
        assert "COPILOT_MODEL" in names
        assert "COPILOT_HOME" in names
        assert "COPILOT_ALLOW_ALL" in names

    def test_supported_output_fields(self):
        fields = CopilotAgentAdapter.supported_output_fields()
        assert "tool_sequence" in fields
        assert "thinking_text" in fields
        assert "llm_turn_count" in fields

    def test_display_info_with_version(self):
        import subprocess as sp

        with patch("shutil.which", return_value="/usr/local/bin/copilot"):
            with patch("subprocess.run") as mock_run:
                mock_run.return_value = sp.CompletedProcess(args=[], returncode=0, stdout="0.1.5\n", stderr="")
                info = CopilotAgentAdapter.display_info()
                assert "copilot 0.1.5" in info

    def test_display_info_when_missing(self):
        with patch("shutil.which", return_value=None):
            info = CopilotAgentAdapter.display_info()
            assert "not found" in info

    def test_health_check_noop(self):
        CopilotAgentAdapter().health_check()

    def test_scoring_strategy(self):
        strategy = CopilotAgentAdapter().scoring_strategy()
        assert strategy is not None


class TestConstructor:
    def test_no_args(self):
        a = CopilotAgentAdapter()
        assert a._model is None
        assert a._session_id is None

    def test_model_stored(self):
        a = CopilotAgentAdapter(model="gpt-5.4-mini")
        assert a._model == "gpt-5.4-mini"

    def test_rejects_unknown_kwargs(self):
        with pytest.raises(TypeError):
            CopilotAgentAdapter(unknown_kwarg=1)  # type: ignore[call-arg]


class TestDeniedFlags:
    """Block scenario flags that broaden Copilot's permissions or expose the
    session for external steering. Selective ``--allow-tool``/``--allow-url``
    remain permitted so scenarios can grant narrow capabilities."""

    def test_blocks_allow_all(self):
        assert "--allow-all" in CopilotAgentAdapter.denied_flags()

    def test_blocks_yolo(self):
        assert "--yolo" in CopilotAgentAdapter.denied_flags()

    def test_blocks_allow_all_paths_and_urls(self):
        denied = CopilotAgentAdapter.denied_flags()
        assert "--allow-all-paths" in denied
        assert "--allow-all-urls" in denied

    def test_blocks_remote_session_flags(self):
        denied = CopilotAgentAdapter.denied_flags()
        assert "--remote" in denied
        assert "--connect" in denied

    def test_does_not_block_selective_allow_tool(self):
        assert "--allow-tool" not in CopilotAgentAdapter.denied_flags()

    def test_filter_flags_strips_denied(self, agent: CopilotAgentAdapter):
        clean = agent.filter_flags(["--allow-all", "--no-banner", "--yolo"])
        assert "--allow-all" not in clean
        assert "--yolo" not in clean
        assert "--no-banner" in clean

    def test_filter_flags_strips_denied_with_equals_value(self, agent: CopilotAgentAdapter):
        clean = agent.filter_flags(["--connect=session-123"])
        assert clean == []


class TestParseStreamEvent:
    """Live-progress renderer used by the TUI ``--progress rich`` mode."""

    def test_tool_execution_start(self):
        ev = {"type": "tool.execution_start", "data": {"toolName": "view", "arguments": {"path": "x.py"}}}
        assert CopilotAgentAdapter.parse_stream_event(ev) == ("🔧", "view(path=x.py)")

    def test_tool_execution_no_args(self):
        ev = {"type": "tool.execution_start", "data": {"toolName": "report_intent"}}
        assert CopilotAgentAdapter.parse_stream_event(ev) == ("🔧", "report_intent()")

    def test_tool_execution_long_args_truncated(self):
        long_value = "x" * 200
        ev = {"type": "tool.execution_start", "data": {"toolName": "bash", "arguments": {"cmd": long_value}}}
        result = CopilotAgentAdapter.parse_stream_event(ev)
        assert result is not None
        assert result[0] == "🔧"
        assert "…" in result[1]
        assert result[1].startswith("bash(") and result[1].endswith(")")
        assert len(result[1]) <= 90

    def test_assistant_message_string_content(self):
        ev = {"type": "assistant.message", "data": {"content": "Hello world"}}
        assert CopilotAgentAdapter.parse_stream_event(ev) == ("💬", "Hello world")

    def test_assistant_message_list_content(self):
        ev = {
            "type": "assistant.message",
            "data": {"content": [{"type": "text", "text": "Part one"}, {"type": "text", "text": "Part two"}]},
        }
        assert CopilotAgentAdapter.parse_stream_event(ev) == ("💬", "Part one Part two")

    def test_assistant_message_empty(self):
        ev = {"type": "assistant.message", "data": {"content": ""}}
        assert CopilotAgentAdapter.parse_stream_event(ev) is None

    def test_assistant_message_truncates_long(self):
        ev = {"type": "assistant.message", "data": {"content": "a " * 200}}
        result = CopilotAgentAdapter.parse_stream_event(ev)
        assert result is not None
        assert result[1].endswith("…")
        assert len(result[1]) <= 120

    def test_assistant_message_strips_newlines(self):
        ev = {"type": "assistant.message", "data": {"content": "line one\nline two"}}
        assert CopilotAgentAdapter.parse_stream_event(ev) == ("💬", "line one line two")

    def test_unknown_event_returns_none(self):
        assert CopilotAgentAdapter.parse_stream_event({"type": "session.start"}) is None

    def test_missing_data_field(self):
        assert CopilotAgentAdapter.parse_stream_event({"type": "tool.execution_start"}) == ("🔧", "?()")
